"""Seizure detector: a general model (other patients) + a personal model (this patient's other
recordings), scores averaged. Evaluation is event-level and honest:
- general model never sees the test patient (leave-one-patient-out)
- personal model never sees the test file (leave-one-file-out)
Run:  python -m src.detector eval
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import CHANNELS, file_features, pick_channels
from src.rules import EventTracker

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data", ROOT / "runs"
FEAT_DIR = RUNS / "features"
THRESHOLD = 0.8          # declared before looking at results; the sweep is reported too
MAX_NEG = 30000          # negatives sampled per training set (positives all kept)
RNG = np.random.default_rng(0)

# Model: IBM Snap ML boosting machine on LinuxONE. Scoring runs on the IBM Integrated Accelerator
# for AI (zDNN) by default; TREE_FORMAT=compress_trees scores on the CPU instead (same outputs).
# Without snapml (e.g. a laptop) we fall back to scikit-learn so the code still runs.
try:
    from snapml import BoostingMachineClassifier
    MODEL_KIND = os.environ.get("MODEL_KIND", "snap")
except ImportError:
    MODEL_KIND = "hgb"
    print("WARNING: snapml not installed, using scikit-learn HistGradientBoosting (official numbers come from LinuxONE)")
TREE_FORMAT = os.environ.get("TREE_FORMAT", "zdnn_tensors")
SNAP_JOBS = int(os.environ.get("SNAP_JOBS", 2))
MODEL_TAG = (f"snap-{'zdnn' if TREE_FORMAT == 'zdnn_tensors' else 'cpu'}" if MODEL_KIND == "snap" else "hgb")
SCORING = ("IBM Integrated Accelerator for AI (Snap ML + zDNN)" if MODEL_TAG == "snap-zdnn"
           else "CPU (Snap ML)" if MODEL_KIND == "snap" else "CPU (scikit-learn)")


SAME_PATIENT = {"chb21": "chb01"}  # chb21 is chb01 recorded 1.5 years later: always one fold


def patient(case):
    return SAME_PATIENT.get(case, case)


def index():
    files = pd.read_csv(DATA / "files.csv")
    files = files[files.downloaded == 1].copy()
    files["patient"] = files.case.map(patient)
    # chronological within a patient: chb01 files, then chb21 files, then by file number
    files = files.sort_values(["patient", "case", "file"]).reset_index(drop=True)
    sz = pd.read_csv(DATA / "seizures.csv")
    return files, sz


def load_file(case, fname):
    """18 standard bipolar channels in microvolts. Files recorded with a referential montage
    (e.g. 'FP1-CS2') get the bipolar pair derived as (A-ref) - (B-ref)."""
    import mne
    raw = mne.io.read_raw_edf(DATA / "chbmit" / case / fname, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    try:
        return raw.get_data(picks=pick_channels(raw.ch_names)).astype(np.float32) * 1e6, fs
    except ValueError:
        pass
    ref = {}  # electrode -> channel index, for names like FP1-CS2 / FP1-REF
    for i, n in enumerate(raw.ch_names):
        parts = n.upper().split("-")
        if len(parts) == 2 and parts[1] in ("CS2", "REF", "LE", "AVG"):
            ref[parts[0]] = i
    pairs = [c.split("-") for c in CHANNELS]
    if not all(a in ref and b in ref for a, b in pairs):
        raise ValueError(f"{fname}: cannot build the standard channels")
    allx = raw.get_data(picks=sorted(set(ref.values()))).astype(np.float32) * 1e6
    pos = {i: k for k, i in enumerate(sorted(set(ref.values())))}
    return np.stack([allx[pos[ref[a]]] - allx[pos[ref[b]]] for a, b in pairs]), fs


def usable(files, sz):
    """Compute/cache features for every file; drop files whose channels can't be built."""
    ok = []
    for r in files.itertuples():
        try:
            features(r.case, r.file, sz)
            ok.append(True)
        except ValueError as e:
            print("skip:", e)
            ok.append(False)
    return files[ok]


def features(case, fname, sz):
    """Cached (X, centres, y) for one file."""
    out = FEAT_DIR / f"{fname}.npz"
    if not out.exists():
        FEAT_DIR.mkdir(parents=True, exist_ok=True)
        data, fs = load_file(case, fname)
        X, tc = file_features(data, fs)
        np.savez_compressed(out, X=X, tc=tc)
    d = np.load(out)
    X, tc = d["X"], d["tc"]
    y = np.zeros(len(tc), dtype=np.int8)
    for _, r in sz[sz.file == fname].iterrows():
        y[(tc >= r.start_s) & (tc <= r.end_s)] = 1
    return X, tc, y


def fit(parts, max_neg=MAX_NEG):
    """parts: list of (X, y). Keeps all seizure windows, samples non-seizure ones."""
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0:
        return None
    neg = RNG.choice(neg, min(max_neg, len(neg)), replace=False)
    keep = np.concatenate([pos, neg])
    if MODEL_KIND == "snap":
        m = BoostingMachineClassifier(num_round=200, learning_rate=0.1, class_weight="balanced",
                                      random_state=0, n_jobs=SNAP_JOBS)
        m.fit(X[keep], y[keep].astype(np.float32))
        m.optimize_trees(tree_format=TREE_FORMAT)  # once per model; zdnn_tensors = on-chip AI accelerator
        return m
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                       random_state=0)
    return m.fit(X[keep], y[keep])


def xy(case, fname, sz):
    X, _, y = features(case, fname, sz)
    return X, y


def general_model(files, sz, exclude_case):
    rows = files[files.case.map(patient) != patient(exclude_case)]
    return fit([xy(r.case, r.file, sz) for r in rows.itertuples()])


def personal_model(files, sz, case, exclude_file):
    rows = files[(files.case == case) & (files.file != exclude_file)]
    return fit([xy(r.case, r.file, sz) for r in rows.itertuples()])


def score(models, X):
    ps = [m.predict_proba(X)[:, 1] for m in models if m is not None]
    return np.mean(ps, axis=0)


def events_from_scores(tc, s, threshold, per_window=None):
    """per_window: optional array of thresholds, one per window (overrides threshold)."""
    tr = EventTracker(threshold=threshold)
    for i, (t, v) in enumerate(zip(tc, s)):
        tr.update(float(t), float(v), None if per_window is None else float(per_window[i]))
    return tr.events


def match(events, truth, dur_s):
    """Event-level metrics for one file."""
    hit, delays = 0, []
    for _, r in truth.iterrows():
        over = [e for e in events if e["start_s"] <= r.end_s and e["end_s"] >= r.start_s]
        if over:
            hit += 1
            delays.append(max(0.0, over[0].get("alert_s", over[0]["start_s"]) - r.start_s))  # time the alert appeared
    false = [e for e in events if not any(e["start_s"] <= r.end_s + 120 and e["end_s"] >= r.start_s - 60
                                          for _, r in truth.iterrows())]
    return {"seizures": len(truth), "detected": hit, "false_alerts": len(false),
            "hours": dur_s / 3600, "delays": delays}


def evaluate():
    files, sz = index()
    cases = sorted(files.case.unique())
    print(f"{len(files)} files, cases {cases}")
    t0 = time.time()
    scored = []  # (case, file, mode, tc, s)
    for case in cases:
        g = general_model(files, sz, case)
        for r in files[files.case == case].itertuples():
            X, tc, y = features(case, r.file, sz)
            p = personal_model(files, sz, case, r.file)
            scored.append((case, r.file, "general", tc, score([g], X)))
            if p is not None:
                scored.append((case, r.file, "personal", tc, score([p], X)))
                scored.append((case, r.file, "combined", tc, score([g, p], X)))
            else:
                scored.append((case, r.file, "combined", tc, score([g], X)))
        print(f"  {case} done ({time.time() - t0:.0f}s)")

    results = {}
    for mode in ["general", "personal", "combined"]:
        sweep = {}
        for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            tot = {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
            for case, fname, m, tc, s in scored:
                if m != mode:
                    continue
                res = match(events_from_scores(tc, s, th), sz[sz.file == fname], tc[-1] + 1)
                for k in tot:
                    tot[k] += res[k]
            if tot["hours"] == 0:
                continue
            sweep[th] = {"seizures": tot["seizures"], "detected": tot["detected"],
                         "sensitivity": round(tot["detected"] / max(tot["seizures"], 1), 3),
                         "false_alerts_per_24h": round(tot["false_alerts"] / tot["hours"] * 24, 2),
                         "median_delay_s": float(np.median(tot["delays"])) if tot["delays"] else None,
                         "hours": round(tot["hours"], 1)}
        results[mode] = sweep
    RUNS.mkdir(exist_ok=True)
    (RUNS / "metrics.json").write_text(json.dumps(results, indent=2))
    for mode, sweep in results.items():
        print(f"\n== {mode} ==")
        for th, v in sweep.items():
            flag = "  <- declared threshold" if th == THRESHOLD else ""
            print(f"  th={th}: caught {v['detected']}/{v['seizures']} ({v['sensitivity']:.0%}), "
                  f"{v['false_alerts_per_24h']} false alerts/24h, median delay {v['median_delay_s']}s{flag}")
    return results


if __name__ == "__main__":
    if sys.argv[1:] == ["eval"]:
        evaluate()
