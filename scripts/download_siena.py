"""Download the Siena Scalp EEG Database (adults, PhysioNet, CC BY 4.0) and parse its seizure lists.

Run on the VM:  python scripts/download_siena.py            (metadata, parse, then all EDFs)
                python scripts/download_siena.py parse      (only rebuild the two csv files)

Writes under data/siena/:
  <PNxx>/<file>.edf, <PNxx>/Seizures-list-<PNxx>.txt, RECORDS, subject_info.csv, SHA256SUMS.txt, LICENSE.txt
  seizures.csv   case,file,start_s,end_s     (seconds from the start of that EDF)
  files.csv      case,file,n_seizures,downloaded
  parse_notes.txt  every place where a seizure list was messy and what we did about it

Source: https://physionet.org/content/siena-scalp-eeg/1.0.0/ . We read the same files from PhysioNet's
AWS open-data copy because physionet.org itself is very slow. Finished files are skipped on re-run.
"""
import csv
import hashlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3 = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "siena"
THREADS = 6
MIN_FREE_AFTER_GB = 2.5   # never fill the shared VM disk: stop if a file would leave less than this

# Hand-checked corrections for obvious typos in the seizure lists (each one is also written to parse_notes.txt).
FIX_FILE = {"PNO6-1.edf": "PN06-1.edf", "PNO6-2.edf": "PN06-2.edf", "PNO6-4.edf": "PN06-4.edf",
            "PN11-.edf": "PN11-1.edf", "PN01.edf": "PN01-1.edf"}
# PN00-3: list says seizure 18.28.29 -> 19.29.29 (61 min) but the recording ends at 18.57.13, so the end
# hour is a typo; 18.29.29 (a 60 s seizure, like its neighbours) is the only reading that fits the file.
FIX_END = {("PN00-3.edf", "19.29.29"): "18.29.29"}


def get(rel, timeout=60, retries=5):
    err = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(S3 + rel, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{rel}: {err}")


def clock(text):
    """First h.m.s time in a messy string ('16:13.23', '1 6.49.25', '15.43.53 (CLINICAL ONSET); ...')."""
    t = re.sub(r"\s+", "", text)
    m = re.search(r"(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})", t)
    if not m:
        raise ValueError(f"no time in {text!r}")
    h, mi, s = map(int, m.groups())
    return h * 3600 + mi * 60 + s, m.group(0)


def parse_list(path, records, notes):
    """-> (seizures [(file, start_s, end_s)], regs {file: (start_clock_s, end_clock_s)})"""
    lines = [ln.rstrip("\r\n") for ln in path.read_text(errors="replace").splitlines()]
    cur, regs, seiz, pending = None, {}, [], None
    for ln in lines:
        key, _, val = ln.partition(":")
        k = key.strip().lower()
        if k == "file name":
            name = val.strip()
            fixed = FIX_FILE.get(name, name)
            if fixed != name:
                notes.append(f"{path.name}: file name {name!r} -> {fixed!r}")
            cur = fixed
            if cur not in records:
                raise ValueError(f"{path.name}: {cur} not in RECORDS")
        elif k == "registration start time" and cur:
            regs.setdefault(cur, [None, None])[0] = clock(val)[0]
            if "  " in val.strip() or " " in val.strip():
                notes.append(f"{path.name}: odd registration start {val.strip()!r} read as {clock(val)[1]}")
        elif k == "registration end time" and cur:
            regs.setdefault(cur, [None, None])[1] = clock(val)[0]
        elif k in ("seizure start time", "start time"):
            s, raw = clock(val)
            pending = (cur, s)
            if len(re.findall(r"\d{1,2}[.:]\d{1,2}[.:]\d{1,2}", val)) > 1 or re.search(r"[A-Za-z]", val):
                notes.append(f"{path.name}: start {val.strip()!r}: used the first time {raw}")
        elif k in ("seizure end time", "end time") and pending:
            e, raw = clock(val)
            f, s = pending
            if (f, raw) in FIX_END:
                notes.append(f"{path.name}: {f} end {raw} is a typo (recording ends at "
                             f"{regs[f][1] // 3600:02d}.{regs[f][1] // 60 % 60:02d}.{regs[f][1] % 60:02d}), "
                             f"used {FIX_END[(f, raw)]}")
                e = clock(FIX_END[(f, raw)])[0]
            if len(re.findall(r"\d{1,2}[.:]\d{1,2}[.:]\d{1,2}", val)) > 1 or re.search(r"[A-Za-z]", val):
                notes.append(f"{path.name}: end {val.strip()!r}: used the first time {raw}")
            seiz.append((f, s, e))
            pending = None
    out = []
    for f, s, e in seiz:
        r0 = regs[f][0]
        st, en = (s - r0) % 86400, (e - r0) % 86400   # clock times -> seconds since registration start
        if en < st:
            en += 86400
        out.append((f, st, en))
    return out, regs


def edf_header(path):
    """(start clock in seconds after midnight, duration in seconds) read straight from the EDF header."""
    with open(path, "rb") as f:
        h = f.read(256)
    hh, mm, ss = map(int, h[176:184].decode().strip().split("."))
    return hh * 3600 + mm * 60 + ss, int(h[236:244]) * float(h[244:252])


def hms(sec):
    return f"{int(sec) // 3600:02d}.{int(sec) // 60 % 60:02d}.{int(sec) % 60:02d}"


def parse_all(records, notes):
    """Seizure rows with times in seconds from the start of each EDF.

    The seizure lists give wall-clock times. The reference for the clock is the start time written in the
    EDF header (a machine value); the typed 'Registration start time' in the list is only used if the EDF
    is not on disk. They agree in 40 of 41 files; PN05-3 is 10 s apart and PN14-3 is 3 h apart (typo in the list).
    """
    rows, hdr = [], {}
    for p in sorted({r.split("/")[0] for r in records}):
        names = {r.split("/")[1] for r in records if r.startswith(p + "/")}
        sz, regs = parse_list(OUT / p / f"Seizures-list-{p}.txt", names, notes)
        for f in names:
            if (OUT / p / f).exists():
                hdr[f] = edf_header(OUT / p / f)
        for f, st, en in sz:
            list_start = regs[f][0]
            # parse_list already subtracted the list's registration start; undo that to get the clock time
            clock_s, clock_e = list_start + st, list_start + en
            ref = hdr[f][0] if f in hdr else list_start
            a, b = (clock_s - ref) % 86400, (clock_e - ref) % 86400
            if b < a:
                b += 86400
            rows.append((p, f, a, b))
        for f in names:
            if f in hdr and f in regs and hdr[f][0] != regs[f][0]:
                notes.append(f"{p}: {f} list says registration starts {hms(regs[f][0])}, EDF header says "
                             f"{hms(hdr[f][0])}; used the EDF header")
    return rows, {f: h[1] for f, h in hdr.items()}


def write_csvs(records):
    notes = []
    rows, durations = parse_all(records, notes)
    # sanity: every seizure must fall inside the recording
    for c, f, st, en in rows:
        if en > durations.get(f, 10**9):
            notes.append(f"WARNING {f}: seizure {st}-{en}s is outside the {durations[f]:.0f}s recording")
    with open(OUT / "seizures.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "file", "start_s", "end_s"])
        w.writerows(rows)
    counts = {}
    for c, f, _, _ in rows:
        counts[f] = counts.get(f, 0) + 1
    with open(OUT / "files.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "file", "n_seizures", "downloaded"])
        for rec in records:
            c, f = rec.split("/")
            w.writerow([c, f, counts.get(f, 0), int((OUT / c / f).exists())])
    (OUT / "parse_notes.txt").write_text("\n".join(notes) + "\n")
    print(f"parsed {len(rows)} seizures in {len(counts)} files (dataset page states 47 seizures, 14 patients)")
    for n in notes:
        print("  note:", n)
    return rows, counts


def fetch(rel, size, sha):
    """Download one EDF to a .part file, check size and sha256, then rename. Returns seconds taken."""
    dest = OUT / rel
    if dest.exists() and dest.stat().st_size == size:
        return 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    free = shutil.disk_usage(OUT).free / 1e9
    if free - size / 1e9 < MIN_FREE_AFTER_GB:
        raise RuntimeError(f"{rel}: skipped, only {free:.1f} GB free")
    t0 = time.time()
    err = None
    for a in range(5):
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(S3 + rel, timeout=90) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
            if part.stat().st_size != size:
                raise RuntimeError(f"size {part.stat().st_size} != {size}")
            if sha and h.hexdigest() != sha:
                raise RuntimeError("sha256 mismatch")
            part.replace(dest)
            return time.time() - t0
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{rel}: {err}")


def head_size(rel):
    for a in range(5):
        try:
            req = urllib.request.Request(S3 + rel, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers["Content-Length"])
        except Exception:  # noqa: BLE001
            time.sleep(2 ** a)
    raise RuntimeError(f"HEAD {rel} failed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for meta in ["RECORDS", "subject_info.csv", "SHA256SUMS.txt", "LICENSE.txt"]:
        if not (OUT / meta).exists():
            (OUT / meta).write_bytes(get(meta))
    records = [ln.strip() for ln in (OUT / "RECORDS").read_text().splitlines() if ln.strip()]
    for p in sorted({r.split("/")[0] for r in records}):
        dest = OUT / p / f"Seizures-list-{p}.txt"
        if not dest.exists():
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(get(f"{p}/Seizures-list-{p}.txt"))
    _, counts = write_csvs(records)
    if sys.argv[1:] == ["parse"]:
        return
    shas = {}
    for ln in (OUT / "SHA256SUMS.txt").read_text().splitlines():
        parts = ln.split()
        if len(parts) == 2:
            shas[parts[1].lstrip("*")] = parts[0]
    with ThreadPoolExecutor(THREADS) as ex:
        sizes = dict(zip(records, ex.map(head_size, records)))
    total = sum(sizes.values())
    print(f"{len(records)} EDFs, {total / 1e9:.1f} GB, disk free {shutil.disk_usage(OUT).free / 1e9:.1f} GB", flush=True)
    # seizure files first (in case disk or time runs out), smallest first so features can start early
    order = sorted(records, key=lambda r: (counts.get(r.split("/")[1], 0) == 0, sizes[r]))
    done_bytes = 0
    with ThreadPoolExecutor(THREADS) as ex:
        futs = {ex.submit(fetch, r, sizes[r], shas.get(r)): r for r in order}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                secs = fut.result()
                done_bytes += sizes[r]
                print(f"{time.time() - t_start:7.0f}s  done {r} ({sizes[r] / 1e6:.0f} MB, {secs:.0f}s)  "
                      f"total {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", r, e, flush=True)
    write_csvs(records)  # refresh the downloaded flags
    print(f"download finished in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
