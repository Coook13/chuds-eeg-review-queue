"""Build a small local subset of CHB-MIT (PhysioNet, open access).

Run from anywhere:  python scripts/download_chbmit.py

Writes data/seizures.csv and data/files.csv for ALL 24 cases (see CONTRACT.md),
then downloads the EDFs of the 5 seizure-richest cases that use the standard
23-channel bipolar montage. Safe to re-run: finished files are skipped.
"""
import csv
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/chbmit/1.0.0/"
# PhysioNet's own AWS Open Data copy of the same files (same names, same byte sizes).
# From here physionet.org serves ~20-50 KB/s, S3 serves several MB/s, so S3 goes first.
S3 = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"
MIRRORS = [S3, BASE]
DATA = Path(__file__).resolve().parent.parent / "data"
CASES = [f"chb{i:02d}" for i in range(1, 25)]
FULL = os.environ.get("FULL") == "1"  # FULL=1: every seizure file of every case (for LinuxONE)
N_CASES = 24 if FULL else 5          # how many cases to download
N_FREE = int(os.environ.get("N_FREE", 2))  # seizure-free files per case (more = more honest false-alarm rate)
THREADS = 8 if FULL else 6
MAX_BYTES = (40 if FULL else 6) * 1024**3
MAX_SECONDS = (180 if FULL else 40) * 60
MIN_STD_FRAC = 0.0 if FULL else 0.8  # FULL keeps odd montages; src/detector.py derives bipolar channels
SKIP_ODD_FILES = not FULL


def fetch(rel, dest, retries=4):
    """Download <rel> to dest (via a .part file), trying each mirror in turn.

    Returns True, or False if every mirror says 404.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    err = None
    for base in MIRRORS:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(base + rel, timeout=60) as r, open(part, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                part.replace(dest)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # not on this mirror, try the next
                err = e
            except Exception as e:  # timeouts, resets
                err = e
            time.sleep(2 ** attempt)
    if err:
        raise RuntimeError(f"{rel}: {err}")
    return False


def remote_size(rel):
    """Content-Length from a HEAD request to the first mirror that answers, or None."""
    for base in MIRRORS:
        try:
            req = urllib.request.Request(base + rel, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers["Content-Length"])
        except Exception:
            continue
    return None


def parse_summary(text):
    """Parse one chbNN-summary.txt into a list of file dicts.

    Handles "Seizure Start Time" and "Seizure 1 Start Time" variants, files
    with no clock times (chb24), and "Channels changed:" blocks (the channel
    list that is current when a file is listed gets attached to that file).
    """
    files, channels, in_channels, cur = [], [], False, None
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if line.startswith(("Channels in EDF Files", "Channels changed")):
            channels, in_channels = [], True
        elif m := re.match(r"Channel \d+:\s*(\S+)", line):
            if in_channels:
                channels.append(m.group(1))
        elif m := re.match(r"File Name:\s*(\S+)", line):
            in_channels = False
            cur = dict(file=m.group(1), start_clock="", end_clock="", declared=0,
                       starts=[], ends=[], channels=tuple(channels))
            files.append(cur)
        elif cur is None:
            continue
        elif m := re.match(r"File Start Time:\s*(\S+)", line):
            cur["start_clock"] = m.group(1)
        elif m := re.match(r"File End Time:\s*(\S+)", line):
            cur["end_clock"] = m.group(1)
        elif m := re.match(r"Number of Seizures in File:\s*(\d+)", line):
            cur["declared"] = int(m.group(1))
        elif m := re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line):
            cur["starts"].append(int(m.group(1)))
        elif m := re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line):
            cur["ends"].append(int(m.group(1)))
    return files


def is_standard(channels, standard):
    """File has every channel of the reference 23-channel bipolar montage.

    Extra channels (ECG, VNS, '-'/'.' placeholders, Ref channels) are fine because
    downstream code picks channels by name; a different order is fine too.
    """
    return standard <= set(channels)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    # 1. summaries
    for case in CASES:
        dest = DATA / "chbmit" / case / f"{case}-summary.txt"
        if dest.exists() or fetch(f"{case}/{case}-summary.txt", dest):
            continue
        print(f"{case}: summary 404, skipped")

    # 2. parse
    info = {}  # case -> list of file dicts
    problems = []
    for case in CASES:
        path = DATA / "chbmit" / case / f"{case}-summary.txt"
        if not path.exists():
            continue
        info[case] = parse_summary(path.read_text())
        for f in info[case]:
            n = len(f["starts"])
            if n != len(f["ends"]) or n != f["declared"]:
                problems.append(f"{case}/{f['file']}: declared {f['declared']}, "
                                f"parsed {n} starts / {len(f['ends'])} ends")
    n_files = sum(len(v) for v in info.values())
    n_seiz = sum(len(f["starts"]) for v in info.values() for f in v)
    print(f"parsed {n_files} files, {n_seiz} seizures (PhysioNet states 198)")
    for p in problems:
        print("PARSE PROBLEM:", p)

    # 3. CSVs (downloaded flag is filled in at the end from what is on disk)
    def edf_path(case, name):
        return DATA / "chbmit" / case / name

    def write_files_csv():
        write_csv(DATA / "files.csv",
                  ["case", "file", "start_clock", "end_clock", "n_seizures", "downloaded"],
                  [(c, f["file"], f["start_clock"], f["end_clock"], len(f["starts"]),
                    int(edf_path(c, f["file"]).exists()))
                   for c, v in info.items() for f in v])

    write_csv(DATA / "seizures.csv", ["case", "file", "start_s", "end_s"],
              [(c, f["file"], s, e) for c, v in info.items() for f in v
               for s, e in zip(f["starts"], f["ends"])])
    write_files_csv()

    # 4. rank cases (standard montage only)
    standard = set(info["chb01"][0]["channels"])  # 22 unique names (T8-P8 appears twice)
    ranking = []
    for case, v in info.items():
        frac = sum(is_standard(f["channels"], standard) for f in v) / len(v)
        ranking.append((case, sum(len(f["starts"]) for f in v), frac, len(v)))
    ranking.sort(key=lambda r: (-r[1], r[3], r[0]))  # most seizures, then fewer files
    print(f"\n{'case':6} {'seizures':>8} {'std_montage':>11} {'files':>6}")
    for case, n, frac, nf in ranking:
        flag = "" if frac >= MIN_STD_FRAC else "  (unusual montage, excluded)"
        print(f"{case:6} {n:8d} {frac:11.0%} {nf:6d}{flag}")
    chosen = [r[0] for r in ranking if r[2] >= MIN_STD_FRAC][:N_CASES]
    print("\nchosen:", ", ".join(f"{c} ({n} seizures)" for c, n, _, _ in ranking if c in chosen))

    # 5. build the queue: seizure files first, then the first N_FREE seizure-free files
    queue = []
    for case in chosen:
        usable = [f for f in info[case] if is_standard(f["channels"], standard) or not SKIP_ODD_FILES]
        for f in info[case]:
            if f["starts"] and f not in usable:
                print(f"skipping {case}/{f['file']}: seizure file without the standard channels")
        seiz = [f["file"] for f in usable if f["starts"]]
        free = [f["file"] for f in usable if not f["starts"]][:N_FREE]
        queue += [(case, n, True) for n in seiz] + [(case, n, False) for n in free]
    queue.sort(key=lambda q: not q[2])  # stable: seizure files go first

    print(f"\nqueue: {len(queue)} files; checking sizes ...")
    with ThreadPoolExecutor(THREADS) as ex:
        sizes = list(ex.map(lambda q: remote_size(f"{q[0]}/{q[1]}"), queue))
    kept, total = [], 0
    for q, s in zip(queue, sizes):
        if total + (s or 0) > MAX_BYTES:
            print(f"over 6 GB budget, dropping {q[0]}/{q[1]}")
            continue
        total += s or 0
        kept.append((q, s))
    print(f"estimated download: {total / 1024**3:.2f} GB for {len(kept)} files")

    for stale in (DATA / "chbmit").glob("*/*.part"):  # leftovers from an interrupted run
        stale.unlink()
    start, failed = time.time(), []

    def get(item):
        (case, name, _), size = item
        dest = edf_path(case, name)
        if dest.exists() and (size is None or dest.stat().st_size == size):
            return
        if time.time() - start > MAX_SECONDS:
            failed.append((case, name, "time limit"))
            return
        try:
            if not fetch(f"{case}/{name}", dest):
                failed.append((case, name, "404"))
        except Exception as e:
            failed.append((case, name, str(e)))
        else:
            print(f"  got {case}/{name}", flush=True)

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(get, kept))
    write_files_csv()
    got = [DATA / "chbmit" / c / n for (c, n, _), _ in kept if edf_path(c, n).exists()]
    print(f"\ndownloaded files on disk: {len(got)}/{len(kept)}, "
          f"{sum(p.stat().st_size for p in got) / 1024**3:.2f} GB, "
          f"{time.time() - start:.0f}s")
    for fl in failed:
        print("FAILED:", fl)

    # 6. sanity check with mne
    import mne
    print()
    for case in chosen:
        paths = sorted(p for p in got if p.parent.name == case)
        if not paths:
            print(f"{case}: no EDF to check")
            continue
        raw = mne.io.read_raw_edf(paths[0], preload=False, verbose="ERROR")
        print(f"{case} {paths[0].name}: sfreq={raw.info['sfreq']} Hz, "
              f"n_channels={len(raw.ch_names)}, duration={raw.times[-1]:.0f}s")
        if case == chosen[0]:
            print("  channels:", raw.ch_names)


if __name__ == "__main__":
    sys.exit(main())
