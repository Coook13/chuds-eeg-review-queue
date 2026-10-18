"""Download the Helsinki neonatal EEG seizure dataset (Zenodo record 2547147, CC BY 4.0, 79 neonates).

Run on the VM:  python scripts/download_helsinki.py
Writes data/helsinki/: eeg1.edf ... eeg79.edf, annotations_2017_A/B/C.csv, annotations_2017.mat,
clinical_information.csv, Ethics_approval.pdf, record.json (the Zenodo API answer: file list, sizes, md5).
Files are fetched from the Zenodo API file links in parallel, via .part files, md5-checked against the API list;
finished files are skipped on re-run.

Source: Stevenson NJ, Tapani K, Lauronen L, Vanhatalo S (2019). A dataset of neonatal EEG recordings with seizure
annotations. Scientific Data 6:190039.  https://zenodo.org/records/2547147
"""
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RECORD = "https://zenodo.org/api/records/2547147"
OUT = Path(__file__).resolve().parent.parent / "data" / "helsinki"
THREADS = 4               # Zenodo throttles aggressive clients; 4 is polite and still fast
MIN_FREE_AFTER_GB = 3.0   # never fill the shared VM disk


def fetch_json(url, retries=6):
    err = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{url}: {err}")


def fetch(f):
    """Download one file object from the API list. Returns seconds taken (0 if it was already there)."""
    dest = OUT / f["key"]
    if dest.exists() and dest.stat().st_size == f["size"]:
        return 0.0
    free = shutil.disk_usage(OUT).free / 1e9
    if free - f["size"] / 1e9 < MIN_FREE_AFTER_GB:
        raise RuntimeError(f"{f['key']}: skipped, only {free:.1f} GB free")
    part = dest.with_name(dest.name + ".part")
    want = f["checksum"].split(":", 1)[1]
    t0, err = time.time(), None
    for a in range(8):
        h = hashlib.md5()
        try:
            req = urllib.request.Request(f["links"]["self"], headers={"User-Agent": "eeg-review-queue/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    h.update(chunk)
            if part.stat().st_size != f["size"]:
                raise RuntimeError(f"size {part.stat().st_size} != {f['size']}")
            if h.hexdigest() != want:
                raise RuntimeError("md5 mismatch")
            part.replace(dest)
            return time.time() - t0
        except urllib.error.HTTPError as e:
            err = e
            time.sleep(min(60, float(e.headers.get("Retry-After", 2 ** a))) if e.code == 429 else 2 ** a)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{f['key']}: {err}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = fetch_json(RECORD)
    (OUT / "record.json").write_text(json.dumps(rec, indent=1))
    files = sorted(rec["files"], key=lambda f: (f["key"].endswith(".edf"), f["size"]))  # small metadata first
    total = sum(f["size"] for f in files)
    print(f"{rec['metadata']['title']}: {len(files)} files, {total / 1e9:.2f} GB, license {rec['metadata']['license']['id']}, "
          f"disk free {shutil.disk_usage(OUT).free / 1e9:.1f} GB", flush=True)
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(THREADS) as ex:
        futs = {ex.submit(fetch, f): f for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                secs = fut.result()
                done += f["size"]
                print(f"{time.time() - t0:7.0f}s  done {f['key']} ({f['size'] / 1e6:.0f} MB, {secs:.0f}s)  "
                      f"total {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(f["key"])
                print("FAILED", f["key"], e, flush=True)
    print(f"download finished in {time.time() - t0:.0f}s, {len(failed)} failed {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
