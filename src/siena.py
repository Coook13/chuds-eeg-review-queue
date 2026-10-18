"""Adult scalp EEG: the Siena Scalp EEG Database (PhysioNet, CC BY 4.0), read into the SAME format as CHB-MIT.

Same 18 bipolar channels (src.features.CHANNELS), same 256 Hz, same microvolts, so the CHB-MIT-trained
models and the band-power features apply unchanged. Siena is a referential recording with old 10-20 names
and 512 Hz, so: derive each pair as A minus B, map T3/T5/T4/T6 -> T7/P7/T8/P8, and resample 512 -> 256 Hz
(band-power features depend on the sample count, so this must match CHB-MIT).

Data is expected under data/siena/ (scripts/download_siena.py). Features are cached in runs/siena_features/.
"""
import re
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from src.features import CHANNELS, file_features

ROOT = Path(__file__).resolve().parents[1]
SIENA = ROOT / "data" / "siena"
FEAT_DIR = ROOT / "runs" / "siena_features"
TARGET_FS = 256.0
OLD_TO_NEW = {"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"}   # old 10-20 names -> the names CHB-MIT uses
CHUNK_S, PAD_S = 900, 4   # read 15 min at a time with 4 s of context each side (keeps RAM low, result exact)

_SZ = None


def seizures():
    """data/siena/seizures.csv (case,file,start_s,end_s)."""
    global _SZ
    if _SZ is None:
        _SZ = pd.read_csv(SIENA / "seizures.csv")
    return _SZ


def index():
    """Downloaded Siena recordings, in order within each patient (file name order, as the lists number them)."""
    files = pd.read_csv(SIENA / "files.csv")
    files = files[files.downloaded == 1].copy()
    files["_k"] = files.file.map(lambda f: "-".join(f"{int(x):03d}" for x in re.findall(r"\d+", f)[1:]))
    files = files.sort_values(["case", "_k"]).drop(columns="_k").reset_index(drop=True)
    return files, seizures()


def electrode(ch_name):
    """'EEG Fp1' -> 'FP1', 'EEG T3' -> 'T7', 'EEG FP2' -> 'FP2'. Non-EEG labels ('EKG EKG', 'SPO2') come back as-is."""
    n = re.sub(r"^\s*EEG[\s_-]+", "", ch_name.strip(), flags=re.I).upper()
    return OLD_TO_NEW.get(n, n)


def load_file(case, fname):
    """-> (data, 256.0): data is (18, n_samples) float32 microvolts in the order of src.features.CHANNELS.
    Raises ValueError if the recording lacks one of the electrodes."""
    import mne
    raw = mne.io.read_raw_edf(SIENA / case / fname, preload=False, verbose="ERROR")
    fs, n = float(raw.info["sfreq"]), raw.n_times
    pos = {}
    for i, name in enumerate(raw.ch_names):
        pos.setdefault(electrode(name), i)   # first match wins (some files list a name twice)
    pairs = [c.split("-") for c in CHANNELS]
    need = sorted({e for p in pairs for e in p})
    missing = [e for e in need if e not in pos]
    if missing:
        raise ValueError(f"{fname}: electrodes missing: {missing}")
    picks = [pos[e] for e in need]
    col = {e: k for k, e in enumerate(need)}
    ratio = Fraction(int(TARGET_FS), int(round(fs)))   # 512 Hz -> 1/2
    up, down = ratio.numerator, ratio.denominator
    chunk, pad = int(CHUNK_S * fs), int(PAD_S * fs)
    chunk -= chunk % down
    pad -= pad % down                                  # keeps output samples aligned with input samples
    n_out = -(-n * up // down)
    out = np.empty((len(CHANNELS), n_out), dtype=np.float32)
    for k in range(0, n, chunk):
        a, b = max(0, k - pad), min(n, k + chunk + pad)
        x = raw.get_data(picks=picks, start=a, stop=b) * 1e6          # volts -> microvolts
        if (up, down) != (1, 1):
            x = resample_poly(x, up, down, axis=1)
        o0 = (k - a) * up // down
        o1 = o0 + -(-(min(n, k + chunk) - k) * up // down)
        ok = k * up // down
        seg = x[:, o0:o1]
        for j, (p, q) in enumerate(pairs):
            out[j, ok:ok + seg.shape[1]] = seg[col[p]] - seg[col[q]]
    return out, TARGET_FS


def features(case, fname, sz=None):
    """Cached (X, centres, y) for one Siena recording, same layout as src.detector.features."""
    sz = seizures() if sz is None else sz
    out = FEAT_DIR / f"{fname}.npz"
    if not out.exists():
        FEAT_DIR.mkdir(parents=True, exist_ok=True)
        data, fs = load_file(case, fname)
        X, tc = file_features(data, fs)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(tmp, X=X, tc=tc, n_samples=data.shape[1], fs=fs)
        tmp.replace(out)
    d = np.load(out)
    X, tc = d["X"], d["tc"]
    y = np.zeros(len(tc), dtype=np.int8)
    for _, r in sz[sz.file == fname].iterrows():
        y[(tc >= r.start_s) & (tc <= r.end_s)] = 1
    return X, tc, y


def xy(case, fname, sz=None):
    X, _, y = features(case, fname, sz)
    return X, y


if __name__ == "__main__":
    import sys, time
    # python -m src.siena [file ...]  -> build the feature cache for the named files (default: every downloaded one)
    files, sz = index()
    todo = files if len(sys.argv) == 1 else files[files.file.isin(sys.argv[1:])]
    t0 = time.time()
    for r in todo.itertuples():
        t1 = time.time()
        try:
            X, tc, y = features(r.case, r.file, sz)
            print(f"{r.file}: {len(tc)} windows, {int(y.sum())} seizure windows, {time.time() - t1:.0f}s "
                  f"(total {time.time() - t0:.0f}s)", flush=True)
        except ValueError as e:
            print("SKIP", e, flush=True)
