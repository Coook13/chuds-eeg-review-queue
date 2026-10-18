"""Download the Emotiv epilepsy EEG dataset from Guinea-Bissau and Nigeria (Zenodo record 1252141, CC BY 4.0).

Run on the VM:  python scripts/download_africa.py
Writes data/africa/: the two zips, the two metadata CSVs, record.json (Zenodo API answer with md5s), and unzips
into data/africa/guinea_bissau/ and data/africa/nigeria/. Files are md5-checked against the API list; finished
files are skipped on re-run.

Source: Durongbhan et al. / Watson et al., bioRxiv 10.1101/324954 (Emotiv EPOC, 14 channels, 128 Hz).
"""
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/1252141"
OUT = Path(__file__).resolve().parent.parent / "data" / "africa"
WANT = ["EEGs_Guinea-Bissau.zip", "EEGs_Nigeria.zip", "metadata_guineabissau.csv", "metadata_nigeria.csv"]
MIN_FREE_AFTER_GB = 3.0


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
    dest = OUT / f["key"]
    if dest.exists() and dest.stat().st_size == f["size"]:
        return 0.0
    free = shutil.disk_usage(OUT).free / 1e9
    if free - f["size"] / 1e9 < MIN_FREE_AFTER_GB:
        raise RuntimeError(f"{f['key']}: skipped, only {free:.1f} GB free")
    part = dest.with_name(dest.name + ".part")
    want = f["checksum"].split(":", 1)[1]
    t0, err = time.time(), None
    for a in range(6):
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
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{f['key']}: {err}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = fetch_json(RECORD)
    (OUT / "record.json").write_text(json.dumps(rec, indent=1))
    files = {f["key"]: f for f in rec["files"]}
    print("files in record:", {k: v["size"] for k, v in files.items()})
    for name in WANT:
        secs = fetch(files[name])
        print(f"  {name}: {'already there' if secs == 0 else f'{secs:.0f}s'}", flush=True)
    for name, sub in [("EEGs_Guinea-Bissau.zip", "guinea_bissau"), ("EEGs_Nigeria.zip", "nigeria")]:
        dest = OUT / sub
        if dest.exists() and any(dest.rglob("*.csv*")):
            print(f"  {sub}: already unzipped")
            continue
        with zipfile.ZipFile(OUT / name) as z:
            z.extractall(dest)
        print(f"  {sub}: unzipped {len(list(dest.rglob('*.csv*')))} csv")


if __name__ == "__main__":
    sys.exit(main())
