"""Scale test: stream N beds at once and measure the cost of one live scoring step.
Each bed = a real recording replayed from its own offset, with its own streaming features.
Two ways of scoring the same windows:
  sequential: features + model call per bed (simple)
  batched:    features per bed, then ONE model call for all beds (how a ward server should do it)
Run:  python scripts/scale_test.py 50 100      (env FSET=v2, STEPS=60, DISTINCT_FILES=10)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detector as D
from src.experiments import train_rows
from src.features import WIN_S, LiveFeatures, window_features

FSET = os.environ.get("FSET", "v1")
STEPS = int(os.environ.get("STEPS", 60))
DISTINCT = int(os.environ.get("DISTINCT_FILES", 10))  # recordings held in RAM, shared by beds


def main(ns):
    files, sz = D.index()
    files = D.usable(files, sz)
    seiz = files[files.n_seizures > 0]
    model = D.fit([train_rows(FSET, r.case, r.file, sz) for r in seiz.head(20).itertuples()])
    recs = []
    for r in seiz.drop_duplicates("patient").head(DISTINCT).itertuples():
        data, fs = D.load_file(r.case, r.file)
        recs.append((r.case, r.file, data, fs, D.features(r.case, r.file, sz)[0]))
    results = []
    for n in ns:
        beds = []
        for i in range(n):
            case, fname, data, fs, X1 = recs[i % len(recs)]
            start = 700 + 37 * (i // len(recs))  # different offsets for beds sharing a recording
            beds.append({"data": data, "fs": fs, "t": start,
                         "lf": LiveFeatures(FSET, X1[start - 2 - 660:start - 2], first_index=start - 2)})
        seq, bat = [], []
        for step in range(STEPS):
            t0 = time.perf_counter()
            xs = []
            for b in beds:
                w, e = int(WIN_S * b["fs"]), int(b["t"] * b["fs"])
                xs.append(b["lf"].step(window_features(b["data"][:, e - w:e], b["fs"])))
                b["t"] += 1
            t_feat = time.perf_counter() - t0
            t1 = time.perf_counter()
            model.predict_proba(np.stack(xs))
            bat.append((t_feat + time.perf_counter() - t1) * 1000)
            if step < 10:  # sequential model calls on the same windows, fewer reps (slow by design)
                t2 = time.perf_counter()
                for x in xs:
                    model.predict_proba(x[None])
                seq.append((t_feat + time.perf_counter() - t2) * 1000)
        res = {"beds": n, "feature_set": FSET, "steps": STEPS,
               "batched_ms_per_step_all_beds": round(float(np.median(bat)), 2),
               "batched_ms_per_window": round(float(np.median(bat)) / n, 3),
               "batched_p95_ms_per_step": round(float(np.percentile(bat, 95)), 2),
               "sequential_ms_per_step_all_beds": round(float(np.median(seq)), 2),
               "sequential_ms_per_window": round(float(np.median(seq)) / n, 3),
               "real_time_ok": bool(np.percentile(bat, 95) < 1000),  # each bed needs one step per second
               "max_beds_real_time_batched_est": int(1000 / (np.median(bat) / n))}
        results.append(res)
        print(json.dumps(res), flush=True)
    import platform
    out = {"machine": platform.machine(), "cpus": os.cpu_count(), "results": results}
    (D.RUNS / f"scale_test{os.environ.get('OUT_TAG', '')}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def retrain_timing(n_patients=8):
    """Time exactly what the dashboard's Confirm button does: train a personal model from the
    confirmed seizure + this recording so far + the patient's earlier recordings."""
    from src.experiments import NEG_PER_FILE
    files, sz = D.index()
    files = D.usable(files, sz)
    out = []
    for pat in files[files.n_seizures > 0].patient.unique():
        own = files[files.patient == pat]
        seiz = own[own.n_seizures > 0]
        if len(seiz) < 2:
            continue
        r = seiz.iloc[1]                     # the patient's second seizure recording
        s = sz[sz.file == r.file].sort_values("start_s").iloc[0]
        t0 = time.perf_counter()
        X, tc, y = D.features(r.case, r.file, sz)
        now = s.end_s + 60
        idx = np.where(tc < now)[0]
        rng = np.random.default_rng(0)
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        neg = rng.choice(neg, min(NEG_PER_FILE * 4, len(neg)), replace=False)
        keep = np.concatenate([pos, neg])
        earlier = list(own.file)[:list(own.file).index(r.file)]
        parts = [(X[keep], y[keep])] + [train_rows(FSET, own[own.file == f].case.iloc[0], f, sz) for f in earlier]
        D.fit(parts)
        secs = time.perf_counter() - t0
        out.append({"patient": pat, "recording": r.file, "earlier_recordings": len(earlier),
                    "training_rows": int(sum(len(p[1]) for p in parts)), "seconds": round(secs, 2)})
        print(json.dumps(out[-1]), flush=True)
        if len(out) >= n_patients:
            break
    import platform
    summary = {"machine": platform.machine(), "cpus": os.cpu_count(), "runs": out,
               "median_seconds": float(np.median([o["seconds"] for o in out])),
               "max_seconds": float(max(o["seconds"] for o in out))}
    (D.RUNS / f"retrain_timing{os.environ.get('OUT_TAG', '')}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}))


def one_live_step(n_beds=1):
    """Exactly one live scoring step for n beds with a cached general model, then exit.
    Run under the zDNN call-count shim (LD_PRELOAD=runs/accel_shim.so ACCEL_SHIM_OUT=...) to prove
    the scores come from the on-chip AI accelerator."""
    import joblib
    files, sz = D.index()
    r = files[files.n_seizures > 0].iloc[0]
    model = joblib.load(sorted(D.RUNS.glob(f"models/{FSET}_{D.MODEL_TAG}_general_*.joblib"))[0])
    data, fs = D.load_file(r.case, r.file)
    X1 = D.features(r.case, r.file, sz)[0]
    w, t = int(WIN_S * fs), 900
    xs = [LiveFeatures(FSET, X1[t - 2 - 660:t - 2], first_index=t - 2).step(
          window_features(data[:, t * int(fs) - w:t * int(fs)], fs)) for _ in range(n_beds)]
    p = model.predict_proba(np.stack(xs))[:, 1]
    print(json.dumps({"beds": n_beds, "model_tag": D.MODEL_TAG, "scoring": D.SCORING, "score_bed1": float(p[0])}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--retrain"]:
        retrain_timing()
    elif sys.argv[1:2] == ["--one-step"]:
        one_live_step(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        main([int(a) for a in sys.argv[1:]] or [50, 100])
