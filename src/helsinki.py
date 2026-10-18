"""Newborn scalp EEG: the Helsinki neonatal EEG seizure dataset (Zenodo 2547147, CC BY 4.0), in the SAME format
as CHB-MIT: the 18 bipolar channels of src.features.CHANNELS, 256 Hz, microvolts.

Facts checked in the files (see data/helsinki/record.json and the first file headers):
- 79 term neonates, one EDF each (eeg1.edf ... eeg79.edf), 256 Hz, 19 EEG channels named like 'EEG Fp1-Ref'
  (referential montage, 10-20 system, old names T3/T4/T5/T6), plus ECG and respiration channels to ignore.
- Annotations: annotations_2017_A/B/C.csv, one column per neonate (1..79), one row per second, 1 = seizure,
  0 = not seizure, blank = after the end of that annotation. Three experts (A, B, C).

Like src/siena.py: derive each bipolar pair as (A - ref) - (B - ref), map T3/T5/T4/T6 -> T7/P7/T8/P8.
Seizure events: CONSENSUS = all three experts mark the second; MAJORITY = at least two. Runs of seconds become
events (start_s, end_s) and runs separated by less than GAP_S seconds are merged.

  python -m src.helsinki annotations    # writes data/helsinki/seizures.csv, files.csv (+ majority and extra tables)
  python -m src.helsinki features       # builds runs/helsinki_features/ (v1 features, cached)
"""
import csv
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from src.features import CHANNELS, file_features

ROOT = Path(__file__).resolve().parents[1]
HEL = ROOT / "data" / "helsinki"
FEAT_DIR = ROOT / "runs" / "helsinki_features"
TARGET_FS = 256.0
OLD_TO_NEW = {"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"}
GAP_S = 10          # seizure runs closer than this are one event
CHUNK_S = 1800      # read 30 min at a time (RAM)

_SZ = {}


def seizures(kind="consensus"):
    """data/helsinki/seizures.csv (consensus) or seizures_majority.csv: case,file,start_s,end_s."""
    if kind not in _SZ:
        name = "seizures.csv" if kind == "consensus" else f"seizures_{kind}.csv"
        _SZ[kind] = pd.read_csv(HEL / name)
    return _SZ[kind]


def index():
    files = pd.read_csv(HEL / "files.csv")
    files = files[files.downloaded == 1].copy()
    files["_n"] = files.case.str.extract(r"(\d+)")[0].astype(int)
    return files.sort_values("_n").drop(columns="_n").reset_index(drop=True), seizures()


def electrode(ch_name):
    """'EEG Fp1-Ref' -> 'FP1', 'EEG T3-Ref' -> 'T7'. ECG / respiration labels come back as they are."""
    n = re.sub(r"^\s*EEG[\s_-]+", "", ch_name.strip(), flags=re.I).upper()
    n = re.sub(r"[-_\s]*REF$", "", n)
    return OLD_TO_NEW.get(n, n)


def load_file(case, fname):
    """-> (data, 256.0): (18, n_samples) float32 microvolts in the order of src.features.CHANNELS."""
    import mne
    raw = mne.io.read_raw_edf(HEL / fname, preload=False, verbose="ERROR")
    fs, n = float(raw.info["sfreq"]), raw.n_times
    pos = {}
    for i, name in enumerate(raw.ch_names):
        pos.setdefault(electrode(name), i)
    pairs = [c.split("-") for c in CHANNELS]
    need = sorted({e for p in pairs for e in p})
    missing = [e for e in need if e not in pos]
    if missing:
        raise ValueError(f"{fname}: electrodes missing: {missing}")
    picks = [pos[e] for e in need]
    col = {e: k for k, e in enumerate(need)}
    ratio = Fraction(int(TARGET_FS), int(round(fs)))
    up, down = ratio.numerator, ratio.denominator
    chunk, pad = int(CHUNK_S * fs), int(4 * fs)
    chunk -= chunk % down
    pad -= pad % down
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
    """Cached v1 (X, centres, y) for one newborn recording; y = 1 inside consensus seizure events."""
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


# ---------------------------------------------------------------- annotations -> events
def edf_duration_s(path):
    """Recording length in seconds from the EDF header (no EEG data read)."""
    with open(path, "rb") as f:
        h = f.read(256)
    return int(h[236:244]) * float(h[244:252])


def runs_to_events(mask, gap=GAP_S):
    """Boolean per-second mask -> [(start_s, end_s)], second i covers [i, i+1); runs closer than gap are merged."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = idx[np.r_[0, cuts + 1]]
    ends = idx[np.r_[cuts, len(idx) - 1]] + 1
    out = [[int(starts[0]), int(ends[0])]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - out[-1][1] < gap:
            out[-1][1] = int(e)
        else:
            out.append([int(s), int(e)])
    return [tuple(x) for x in out]


def build_annotations():
    ann = {k: pd.read_csv(HEL / f"annotations_2017_{k}.csv") for k in "ABC"}
    clin = pd.read_csv(HEL / "clinical_information.csv").set_index("ID")
    rows_c, rows_m, files, extra = [], [], [], []
    for n in range(1, 80):
        fname = f"eeg{n}.edf"
        have = (HEL / fname).exists()
        dur = edf_duration_s(HEL / fname) if have else float("nan")
        L = int(dur) if have else 0
        marks = []
        n_len = {}
        for k in "ABC":
            col = ann[k][str(n)].to_numpy(dtype=float)
            n_len[k] = int(np.sum(~np.isnan(col)))
            m = np.zeros(L, dtype=bool)
            m[:min(L, len(col))] = col[:L] == 1
            marks.append(m)
        votes = np.sum(marks, axis=0) if L else np.zeros(0)
        ev_c, ev_m = runs_to_events(votes >= 3), runs_to_events(votes >= 2)
        rows_c += [(f"eeg{n}", fname, s, e) for s, e in ev_c]
        rows_m += [(f"eeg{n}", fname, s, e) for s, e in ev_m]
        files.append((f"eeg{n}", fname, len(ev_c), int(have)))
        extra.append({"case": f"eeg{n}", "file": fname, "duration_s": dur,
                      "annotated_s_A": n_len["A"], "annotated_s_B": n_len["B"], "annotated_s_C": n_len["C"],
                      "consensus_seizure_s": int((votes >= 3).sum()), "majority_seizure_s": int((votes >= 2).sum()),
                      "events_consensus": len(ev_c), "events_majority": len(ev_m),
                      "reviewers_marking_seizure_clinical_sheet": int(clin.loc[n, "Number of Reviewers Annotating Seizure"])})
    for name, rows in [("seizures.csv", rows_c), ("seizures_majority.csv", rows_m)]:
        with open(HEL / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "file", "start_s", "end_s"])
            w.writerows(rows)
    with open(HEL / "files.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "file", "n_seizures", "downloaded"])
        w.writerows(files)
    ex = pd.DataFrame(extra)
    ex.to_csv(HEL / "files_extra.csv", index=False)
    return ex, rows_c, rows_m


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["annotations"]:
        ex, rc, rm = build_annotations()
        dur = np.array([e - s for _, _, s, e in rc])
        print(f"{len(ex)} neonates, {ex.duration_s.sum() / 3600:.1f} h of EEG")
        print(f"CONSENSUS (3 of 3): {len(rc)} events in {int((ex.events_consensus > 0).sum())} neonates, "
              f"{int(ex.consensus_seizure_s.sum())} seizure seconds; event length min/median/max "
              f"{dur.min()}/{np.median(dur):.0f}/{dur.max()} s; {int((dur < 10).sum())} shorter than 10 s")
        print(f"MAJORITY (2 of 3):  {len(rm)} events in {int((ex.events_majority > 0).sum())} neonates, "
              f"{int(ex.majority_seizure_s.sum())} seizure seconds")
        print("neonates with 3 / 2 / 1 / 0 reviewers marking seizure (clinical sheet):",
              [int((ex.reviewers_marking_seizure_clinical_sheet == k).sum()) for k in (3, 2, 1, 0)])
        bad = ex[(ex.annotated_s_B - ex.duration_s).abs() > 2]
        print("neonates whose annotator-B length differs from the EDF by >2 s:", len(bad))
        print(bad[["case", "duration_s", "annotated_s_A", "annotated_s_B", "annotated_s_C"]].to_string(index=False))
    elif cmd == ["features"]:
        files, sz = index()
        t0 = time.time()
        for r in files.itertuples():
            t1 = time.time()
            try:
                X, tc, y = features(r.case, r.file, sz)
                print(f"{r.file}: {len(tc)} windows, {int(y.sum())} seizure windows, {time.time() - t1:.0f}s "
                      f"(total {time.time() - t0:.0f}s)", flush=True)
            except ValueError as e:
                print("SKIP", e, flush=True)
