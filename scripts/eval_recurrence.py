"""Evaluate the Predict layer: risk of ANOTHER seizure within the next 60 min, from the seizures seen so far.

Run (VM):  nice -n 10 .venv/bin/python scripts/eval_recurrence.py
Writes:    runs/recurrence_results.json, runs/recurrence_results.txt, runs/recurrence_model.joblib

Design (fixed before looking at results):
- Timeline per CHB-MIT case from the summary clock times (H:MM:SS, may pass 24:00 or wrap to 00:xx -> next
  day). chb01 and chb21 are one patient (one LOPO fold) but two monitoring sessions 1.5 years apart, so each
  case is its own timeline. chb24's summary has no clock times: we read start time + duration from the EDF
  headers when they are on disk; otherwise files are ordered by number and assumed back to back.
- Gaps <= GAP_TOL_S between files count as continuous recording; a longer gap ends a recorded segment.
- Prediction points: each seizure end + 60 s, and every 15 min inside each recorded segment.
  Dropped: points while a seizure is under way or not yet seen (start < t < end + 60 s), and points whose
  next 60 min is not fully inside one recorded segment (censoring).
- Label: a new seizure starts in (t, t + 60 min]. Features: src.recurrence.features (past only).
- Models: logistic regression (log1p + standardise) and a small Snap ML boosting machine scored with
  optimize_trees(tree_format="zdnn_tensors"). Leave-one-patient-out. Baseline: seizures in last 6 h > 0.
- Final model: logistic regression unless the tree's pooled AUC is >= 0.02 higher AND its Brier is not worse.
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recurrence import FEATURES, HORIZON_S, SEEN_DELAY_S, features, predict_logreg  # noqa: E402

DATA, RUNS = ROOT / "data", ROOT / "runs"
OUT = Path(os.environ.get("RECURRENCE_OUT", RUNS))   # override only for dry runs; official outputs go to runs/
MODEL_OUT = OUT / "recurrence_model.joblib"
CHECK_EVERY_S = 900
GAP_TOL_S = 300
WRAP_TOL_S = 600       # a file may start up to 10 min before the previous one "ends" (clock jitter) before we add a day
SAME_PATIENT = {"chb21": "chb01"}   # same as src.detector.patient()
N_BOOT = 1000
RNG = np.random.default_rng(0)

try:
    from snapml import BoostingMachineClassifier
    TREE_KIND = "snapml BoostingMachineClassifier, optimize_trees(tree_format='zdnn_tensors')"
except ImportError:
    BoostingMachineClassifier = None
    TREE_KIND = "sklearn HistGradientBoostingClassifier (snapml not installed; NOT the official number)"


def patient(case):
    return SAME_PATIENT.get(case, case)


# ---------------------------------------------------------------- timelines
def clock_s(c):
    h, m, s = str(c).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def chb_key(fname):
    m = re.match(r"chb\d+([a-z]?)_(\d+)(\+?)\.edf", fname)
    return (m.group(1), int(m.group(2)), m.group(3))


def edf_header(path):
    """(start clock seconds after midnight, duration s) from an EDF header, or None."""
    try:
        with open(path, "rb") as fh:
            h = fh.read(256).decode("ascii", "replace")
        hh, mm, ss = map(int, h[176:184].strip().split("."))
        return hh * 3600 + mm * 60 + ss, int(h[236:244]) * float(h[244:252])
    except (OSError, ValueError):
        return None


def place(raw):
    """raw: ordered [(file, start_clock_s, duration_s)] -> [(file, abs_start, abs_end)], first start = 0."""
    out, prev_end = [], None
    for f, s, dur in raw:
        if prev_end is not None:
            while s < prev_end - WRAP_TOL_S:
                s += 86400
        out.append([f, s, s + dur])
        prev_end = s + dur
    t0 = out[0][1]
    return [(f, a - t0, b - t0) for f, a, b in out]


def chb_timelines(notes):
    files = pd.read_csv(DATA / "files.csv")
    sz = pd.read_csv(DATA / "seizures.csv")
    tl = {}
    for case, g in files.groupby("case"):
        g = g.set_index("file", drop=False).loc[sorted(g.file, key=chb_key)]
        if g.start_clock.isna().any():
            hdr = {f: edf_header(DATA / "chbmit" / case / f) for f in g.file}
            if all(hdr.values()):
                raw = [(f, hdr[f][0], hdr[f][1]) for f in g.file]
                notes.append(f"{case}: no clock times in the summary; used start time + duration from the EDF "
                             f"headers ({len(raw)} files). Missing file numbers become unrecorded gaps (censored).")
            else:
                raw, t = [], 0
                for f in g.file:
                    raw.append((f, t, 3600)); t += 3600
                notes.append(f"{case}: no clock times and no EDF headers; ordered by file number and ASSUMED "
                             f"back-to-back 1 h files.")
        else:
            raw = []
            for r in g.itertuples():
                s, e = clock_s(r.start_clock), clock_s(r.end_clock)
                dur = (e - s) % 86400
                raw.append((r.file, s, dur))
        placed = place(raw)
        start = {f: a for f, a, _ in placed}
        ev = sorted((start[r.file] + r.start_s, start[r.file] + r.end_s) for r in sz[sz.case == case].itertuples())
        tl[case] = {"patient": patient(case), "files": placed, "seizures": ev}
    return tl


def siena_timelines(notes):
    """Siena adults: file clock times from the per-patient seizure lists (Registration start/end time),
    with the two EDF-header corrections recorded in data/siena/parse_notes.txt."""
    sdir = DATA / "siena"
    lists = sorted(sdir.glob("PN*/Seizures-list-*.txt"))
    if not lists:
        notes.append("Siena: seizure lists not on disk, external check skipped.")
        return {}
    from scripts.download_siena import FIX_FILE
    override = {}
    pn = (sdir / "parse_notes.txt").read_text() if (sdir / "parse_notes.txt").exists() else ""
    for m in re.finditer(r"(PN\d+-\d+\.edf) list says registration starts [\d.]+, EDF header says (\d+)\.(\d+)\.(\d+)", pn):
        override[m.group(1)] = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    files = pd.read_csv(sdir / "files.csv")
    sz = pd.read_csv(sdir / "seizures.csv")
    regs = {}
    for p in lists:
        cur = None
        for ln in p.read_text(errors="replace").splitlines():
            k, _, v = ln.partition(":")
            k = k.strip().lower()
            if k == "file name":
                cur = FIX_FILE.get(v.strip(), v.strip())
            elif k in ("registration start time", "registration end time") and cur:
                t = re.sub(r"\s+", "", v)
                mm = re.search(r"(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})", t)
                regs.setdefault(cur, [None, None])[0 if "start" in k else 1] = \
                    int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + int(mm.group(3))
    tl = {}
    for case, g in files.groupby("case"):
        g = g.set_index("file", drop=False).loc[sorted(g.file, key=lambda f: tuple(int(x) for x in re.findall(r"\d+", f)[1:]))]
        raw = []
        for f in g.file:
            s, e = regs[f]
            dur = (e - s) % 86400            # the list's own start/end gives the duration
            if f in override:
                s = override[f]              # EDF header start (seizure times are relative to it)
            raw.append((f, s, dur))
        placed = place(raw)
        start = {f: a for f, a, _ in placed}
        ev = sorted((start[r.file] + r.start_s, start[r.file] + r.end_s) for r in sz[sz.case == case].itertuples())
        tl[case] = {"patient": case, "files": placed, "seizures": ev}
    notes.append(f"Siena: {len(tl)} patients, file times from the seizure lists' registration times; "
                 f"EDF-header start overrides for {sorted(override)}.")
    return tl


def segments(files):
    segs = []
    for _, a, b in files:
        if segs and a - segs[-1][1] <= GAP_TOL_S:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return segs


def monitored(files, t):
    return sum(max(0.0, min(b, t) - a) for _, a, b in files)


# ---------------------------------------------------------------- prediction points
def build_points(tl):
    rows, drops = [], {"post_seizure": {"in_or_unseen_seizure": 0, "not_recorded": 0, "censored_next_60min": 0},
                       "checkpoint": {"in_or_unseen_seizure": 0, "not_recorded": 0, "censored_next_60min": 0}}
    gaps = []
    for case, d in tl.items():
        segs, ev = segments(d["files"]), d["seizures"]
        gaps += [s2[0] - s1[1] for s1, s2 in zip(segs, segs[1:])]
        cand = [(e + SEEN_DELAY_S, "post_seizure") for _, e in ev]
        for a, b in segs:
            k = 1
            while a + k * CHECK_EVERY_S <= b:
                cand.append((a + k * CHECK_EVERY_S, "checkpoint")); k += 1
        starts = np.array([s for s, _ in ev]) if ev else np.zeros(0)
        for t, typ in cand:
            seg = next((sg for sg in segs if sg[0] <= t <= sg[1]), None)
            if seg is None:
                drops[typ]["not_recorded"] += 1; continue
            if any(s < t < e + SEEN_DELAY_S for s, e in ev):
                drops[typ]["in_or_unseen_seizure"] += 1; continue
            if t + HORIZON_S > seg[1]:
                drops[typ]["censored_next_60min"] += 1; continue
            seen = [(s, e) for s, e in ev if e + SEEN_DELAY_S <= t]
            f = features(seen, t, monitored(d["files"], t))
            y = int(np.any((starts > t) & (starts <= t + HORIZON_S)))
            rows.append({"case": case, "patient": d["patient"], "t": t, "type": typ, "y": y, **f})
    return pd.DataFrame(rows), drops, gaps


# ---------------------------------------------------------------- models
def make_lr():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000))])


def lr_params(pipe):
    sc, lr = pipe.named_steps["sc"], pipe.named_steps["lr"]
    return {"mean": sc.mean_.copy(), "scale": sc.scale_.copy(), "coef": lr.coef_[0].copy(),
            "intercept": float(lr.intercept_[0])}


def fit_tree(X, y):
    if BoostingMachineClassifier is not None:
        m = BoostingMachineClassifier(num_round=100, max_depth=3, learning_rate=0.1, random_state=0, n_jobs=1)
        m.fit(X.astype(np.float32), y.astype(np.float32))
        m.optimize_trees(tree_format="zdnn_tensors")
        return m
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)


def tree_proba(m, X):
    return m.predict_proba(X.astype(np.float32))[:, 1]


# ---------------------------------------------------------------- metrics
def auc(y, s):
    """Mann-Whitney AUC with ties counted as 1/2."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calib(y, p, n_bins=10):
    order = np.argsort(p, kind="stable")
    out = []
    for idx in np.array_split(order, n_bins):
        out.append({"n": int(len(idx)), "pred_mean": round(float(np.mean(p[idx])), 4),
                    "observed": round(float(np.mean(y[idx])), 4),
                    "pred_range": [round(float(p[idx].min()), 4), round(float(p[idx].max()), 4)]})
    return out


def binary_stats(y, b):
    y, b = np.asarray(y), np.asarray(b).astype(bool)
    tp, fp = int(np.sum(b & (y == 1))), int(np.sum(b & (y == 0)))
    fn, tn = int(np.sum(~b & (y == 1))), int(np.sum(~b & (y == 0)))
    return {"flagged": int(b.sum()), "sensitivity": round(tp / max(tp + fn, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4), "ppv": round(tp / max(tp + fp, 1), 4),
            "npv": round(tn / max(tn + fn, 1), 4)}


def boot_ci(df, cols, n=N_BOOT):
    """Patient-level (cluster) bootstrap of pooled AUC for each score column and the differences vs baseline."""
    pats = df.patient.unique()
    groups = {p: df.index[df.patient == p].to_numpy() for p in pats}
    res = {c: [] for c in cols}
    res.update({f"{c}_minus_baseline": [] for c in cols if c != "baseline"})
    y = df.y.to_numpy()
    for _ in range(n):
        idx = np.concatenate([groups[p] for p in RNG.choice(pats, len(pats), replace=True)])
        a = {c: auc(y[idx], df[c].to_numpy()[idx]) for c in cols}
        for c in cols:
            res[c].append(a[c])
            if c != "baseline":
                res[f"{c}_minus_baseline"].append(a[c] - a["baseline"])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]
            for k, v in res.items()}


def summarize(df, cols):
    y = df.y.to_numpy()
    out = {"n_points": int(len(df)), "n_positive": int(y.sum()), "base_rate": round(float(y.mean()), 4),
           "n_patients": int(df.patient.nunique())}
    for c in cols:
        out[f"auc_{c}"] = round(auc(y, df[c]), 4)
    for c in cols:
        if c != "baseline":
            out[f"brier_{c}"] = round(brier(y, df[c]), 4)
    return out


def within_patient(df, cols):
    per = {}
    for p, g in df.groupby("patient"):
        if 0 < g.y.sum() < len(g):
            per[p] = {c: round(auc(g.y, g[c]), 3) for c in cols}
            per[p].update({"n": int(len(g)), "pos": int(g.y.sum())})
    mean = {c: round(float(np.nanmean([v[c] for v in per.values()])), 4) for c in cols}
    return {"n_patients_with_both_classes": len(per), "mean_auc": mean, "per_patient": per}


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    notes = []
    tl = chb_timelines(notes)
    pts, drops, gaps = build_points(tl)
    X, y = pts[FEATURES].to_numpy(float), pts.y.to_numpy()
    pts["baseline"] = (pts.n_6h > 0).astype(float)

    # leave-one-patient-out
    pts["lr"] = np.nan; pts["tree"] = np.nan; pts["climatology"] = np.nan
    for p in sorted(pts.patient.unique()):
        te = (pts.patient == p).to_numpy(); tr = ~te
        lr = make_lr().fit(X[tr], y[tr])
        pts.loc[te, "lr"] = lr.predict_proba(X[te])[:, 1]
        pts.loc[te, "tree"] = tree_proba(fit_tree(X[tr], y[tr]), X[te])
        pts.loc[te, "climatology"] = y[tr].mean()
    cols = ["lr", "tree", "baseline"]
    # POST-HOC ablation (added after seeing that hours_monitored dominates the logreg): same logreg without it.
    # Reported for information only; it does not change the final model choice.
    abl_feats = [f for f in FEATURES if f != "hours_monitored"]
    XA = pts[abl_feats].to_numpy(float)
    pts["lr_no_hours"] = np.nan
    for p in sorted(pts.patient.unique()):
        te = (pts.patient == p).to_numpy()
        pts.loc[te, "lr_no_hours"] = make_lr().fit(XA[~te], y[~te]).predict_proba(XA[te])[:, 1]

    res = {"task": "risk of another seizure starting within 60 min of the prediction time",
           "tree_model": TREE_KIND,
           "design": {"check_every_s": CHECK_EVERY_S, "horizon_s": HORIZON_S, "seen_delay_s": SEEN_DELAY_S,
                      "gap_tolerance_s": GAP_TOL_S, "features": FEATURES, "cv": "leave-one-patient-out (chb01+chb21 one fold)",
                      "baseline": "seizures in the last 6 h > 0"},
           "data_notes": notes}
    n_cand = {k: int((pts.type == k).sum()) + sum(v.values()) for k, v in drops.items()}
    res["points"] = {"candidates": n_cand, "dropped": drops, "kept": pts.type.value_counts().to_dict(),
                     "recorded_hours": round(sum(monitored(d["files"], 1e12) for d in tl.values()) / 3600, 1),
                     "n_seizures": int(sum(len(d["seizures"]) for d in tl.values())),
                     "segment_gaps_over_tolerance": {"n": len(gaps),
                                                     "median_h": round(float(np.median(gaps)) / 3600, 2) if gaps else None}}
    pooled = summarize(pts, cols)
    pooled["brier_climatology"] = round(brier(y, pts.climatology), 4)
    for c in ("lr", "tree"):
        pooled[f"brier_skill_{c}"] = round(1 - pooled[f"brier_{c}"] / pooled["brier_climatology"], 4)
    res["pooled_lopo"] = pooled
    res["bootstrap_95ci_patient_level"] = boot_ci(pts, cols)
    res["by_point_type"] = {k: summarize(g, cols) for k, g in pts.groupby("type")}
    res["without_chb24"] = summarize(pts[pts.case != "chb24"], cols)
    res["within_patient"] = within_patient(pts, cols)
    res["baseline_flag"] = binary_stats(y, pts.baseline)
    res["post_hoc_ablation_lr_without_hours_monitored"] = {
        "pooled_auc": round(auc(y, pts.lr_no_hours), 4), "brier": round(brier(y, pts.lr_no_hours), 4),
        "within_patient_mean_auc": within_patient(pts, ["lr_no_hours"])["mean_auc"]["lr_no_hours"],
        "note": "added after seeing results; information only, final model unchanged"}
    better = "tree" if (pooled["auc_tree"] >= pooled["auc_lr"] + 0.02 and pooled["brier_tree"] <= pooled["brier_lr"]) else "lr"
    res["better_model"] = better
    res["calibration_deciles"] = {better: calib(y, pts[better].to_numpy())}
    other = "lr" if better == "tree" else "tree"
    res["calibration_deciles"][other] = calib(y, pts[other].to_numpy())
    # same flag rate as the baseline, but chosen by the better model: who would you watch?
    k = int(pts.baseline.sum())
    thr = np.sort(pts[better].to_numpy())[::-1][k - 1]
    res["better_model_at_baseline_flag_rate"] = {"threshold": round(float(thr), 4),
                                                 **binary_stats(y, pts[better] >= thr)}

    # final model on all CHB-MIT patients
    lr_all = make_lr().fit(X, y)
    params = lr_params(lr_all)
    assert np.allclose(predict_logreg(params, X), lr_all.predict_proba(X)[:, 1])
    res["final_lr_coefficients_std_log1p"] = {f: round(float(c), 4) for f, c in zip(FEATURES, params["coef"])}
    res["final_lr_coefficients_std_log1p"]["intercept"] = round(params["intercept"], 4)
    import joblib
    art = {"kind": "logreg" if better == "lr" else "snap", "params": params, "features": FEATURES,
           "base_rate": float(y.mean()), "trained_on": f"CHB-MIT, {pts.patient.nunique()} patients, {len(pts)} points",
           "horizon_s": HORIZON_S, "lopo_auc": pooled[f"auc_{better}"], "lopo_brier": pooled[f"brier_{better}"]}
    tree_all = fit_tree(X, y)
    if better == "tree":
        art["model"] = tree_all
    joblib.dump(art, MODEL_OUT)
    res["final_model"] = {"path": str(MODEL_OUT), "kind": art["kind"]}

    # external check: Siena adults, models trained on all CHB-MIT
    stl = siena_timelines(res["data_notes"])
    if stl:
        sp, sdrops, _ = build_points(stl)
        if len(sp):
            SX = sp[FEATURES].to_numpy(float)
            sp["lr"] = predict_logreg(params, SX)
            sp["tree"] = tree_proba(tree_all, SX)
            sp["baseline"] = (sp.n_6h > 0).astype(float)
            s = summarize(sp, cols)
            s["dropped"] = sdrops
            s["kept"] = sp.type.value_counts().to_dict()
            s["by_point_type"] = {k: summarize(g, cols) for k, g in sp.groupby("type")}
            s["calibration_deciles_" + better] = calib(sp.y.to_numpy(), sp[better].to_numpy(), n_bins=5)
            res["siena_external"] = s

    # usage example
    from src import recurrence
    recurrence._MODEL, recurrence.MODEL_PATH = None, MODEL_OUT
    res["example"] = {
        "two_recent": recurrence.risk([(1200, 1260), (5400, 5445)], now_s=5505),
        "none_in_6h": recurrence.risk([], now_s=6 * 3600),
    }
    res["runtime_s"] = round(time.time() - t_start, 1)
    pts.drop(columns=["climatology"]).to_csv(OUT / "recurrence_points.csv", index=False)
    (OUT / "recurrence_results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    (OUT / "recurrence_results.txt").write_text(report(res))
    print(report(res))


def report(r):
    p, ci, bt = r["pooled_lopo"], r["bootstrap_95ci_patient_level"], r["better_model"]
    L = ["PREDICT: risk of another seizure within 60 min (CHB-MIT, leave-one-patient-out)",
         f"tree model: {r['tree_model']}",
         f"recorded hours {r['points']['recorded_hours']}, seizures {r['points']['n_seizures']}",
         f"prediction points kept: {r['points']['kept']}  (candidates {r['points']['candidates']})",
         f"dropped: {json.dumps(r['points']['dropped'])}",
         f"n points {p['n_points']}, positives {p['n_positive']}, base rate {p['base_rate']:.3f}, patients {p['n_patients']}",
         "",
         f"pooled AUC  logreg {p['auc_lr']:.3f} {ci['lr']}   tree {p['auc_tree']:.3f} {ci['tree']}   "
         f"baseline(6h>0) {p['auc_baseline']:.3f} {ci['baseline']}",
         f"AUC minus baseline: logreg {ci['lr_minus_baseline']}  tree {ci['tree_minus_baseline']} (95% patient bootstrap)",
         f"Brier logreg {p['brier_lr']:.4f}  tree {p['brier_tree']:.4f}  climatology {p['brier_climatology']:.4f}  "
         f"(skill logreg {p['brier_skill_lr']:.3f}, tree {p['brier_skill_tree']:.3f})",
         f"better model: {bt}", ""]
    for k, s in r["by_point_type"].items():
        L.append(f"{k:13s} n={s['n_points']:5d} base={s['base_rate']:.3f} AUC lr={s['auc_lr']:.3f} "
                 f"tree={s['auc_tree']:.3f} baseline={s['auc_baseline']:.3f}")
    w = r["without_chb24"]
    L.append(f"without chb24  n={w['n_points']} base={w['base_rate']:.3f} AUC lr={w['auc_lr']:.3f} tree={w['auc_tree']:.3f} baseline={w['auc_baseline']:.3f}")
    wp = r["within_patient"]
    L.append(f"within-patient AUC (mean over {wp['n_patients_with_both_classes']} patients): {wp['mean_auc']}")
    L.append(f"baseline flag: {r['baseline_flag']}")
    L.append(f"post-hoc ablation, logreg without hours_monitored: {r['post_hoc_ablation_lr_without_hours_monitored']}")
    L.append(f"{bt} at the same flag rate: {r['better_model_at_baseline_flag_rate']}")
    L += ["", f"calibration deciles ({bt}): pred_mean -> observed (n)"]
    for c in r["calibration_deciles"][bt]:
        L.append(f"  {c['pred_mean']:.3f} -> {c['observed']:.3f} ({c['n']})")
    L += ["", f"final logreg coefficients (standardised log1p features): {r['final_lr_coefficients_std_log1p']}",
          f"final model: {r['final_model']}"]
    if "siena_external" in r:
        s = r["siena_external"]
        L += ["", f"Siena external (trained on all CHB-MIT): n={s['n_points']} pos={s['n_positive']} base={s['base_rate']:.3f} "
              f"patients={s['n_patients']} AUC lr={s['auc_lr']:.3f} tree={s['auc_tree']:.3f} baseline={s['auc_baseline']:.3f} "
              f"Brier lr={s['brier_lr']:.4f} tree={s['brier_tree']:.4f}", f"  kept {s['kept']} dropped {json.dumps(s['dropped'])}"]
    L += ["", "notes:"] + [f"  - {n}" for n in r["data_notes"]]
    L += ["", "example risk():", f"  {r['example']['two_recent']}", f"  {r['example']['none_in_6h']}",
          f"runtime {r['runtime_s']} s"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
