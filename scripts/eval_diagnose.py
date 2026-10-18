"""Evaluate and train the epilepsy DIAGNOSIS model (src/diagnose.py) on the Guinea-Bissau + Nigeria Emotiv data.

Honest set-up (one row = one person, so a person is never in both train and test):
  a) within country : stratified 10-fold CV, repeated 5x with different seeds. AUC is reported two ways:
                      pooled out-of-fold AUC per repeat (mean +- sd over the 5 repeats) and per-fold AUC
                      (mean +- sd over the 50 folds). Sensitivity/specificity at the Youden threshold of the
                      pooled out-of-fold scores (optimistic: the threshold is picked on those same scores).
  b) cross country  : train on all of one country, test on all of the other. Also a "site-normalised"
                      variant where each country's features are z-scored with that country's own (unlabelled)
                      statistics, i.e. a new clinic calibrates on its first recordings without labels.
  c) both countries : same CV on everyone, stratified by country x label, with and without a country flag.
  d) confound check : the same CV using only recording-quality numbers (length, contact quality, cleaning,
                      wrapped channels). If these alone separate the groups, the EEG result is suspect.
  e) final model    : trained on all people; saved to runs/diagnose_model.joblib for src.diagnose.predict_person.

Models: LR = standardised L2 logistic regression (C chosen by inner CV on training people only);
        SNAP = IBM Snap ML BoostingMachine, trees compiled with optimize_trees("zdnn_tensors") so scoring runs
        on the IBM Integrated Accelerator for AI. Hyper-parameters are fixed in src/diagnose.py (not tuned).

Run on the VM:  OMP_NUM_THREADS=1 nice -n 10 python scripts/eval_diagnose.py
Writes runs/diagnose_results.json and runs/diagnose_results.txt (rewritten after every stage).
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import diagnose as G  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

OUT_JSON, OUT_TXT = G.RUNS / "diagnose_results.json", G.RUNS / "diagnose_results.txt"
SEEDS = [0, 1, 2, 3, 4]
KINDS = {"lr": "Logistic regression (sklearn, standardised, CPU)",
         "snap": "Snap ML BoostingMachine, scored on IBM Integrated Accelerator for AI (zdnn_tensors)"}
PAPER = {"GB": 0.85, "NG": 0.78, "cross": "0.62-0.64"}
QUALITY_COLS = ["seconds", "contact_quality_mean", "clean_frac", "wrapped_channels", "interpolated_frac"]
R = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(), "machine": platform.machine(),
     "benchmark_paper_auc": PAPER, "timings_s": {}}
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def save():
    R["finished_or_last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT_JSON.write_text(json.dumps(R, indent=2, default=float))
    OUT_TXT.write_text("\n".join(LINES) + "\n")


def youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    return float(th[j]), float(tpr[j]), float(1 - fpr[j])


def cv(X, y, kind, strat=None):
    """Repeated stratified 10-fold CV over people."""
    reps, folds, tf = [], [], set()
    for s in SEEDS:
        oof = np.zeros(len(y))
        skf = StratifiedKFold(10, shuffle=True, random_state=s)
        for k, (tr, te) in enumerate(skf.split(X, y if strat is None else strat)):
            m = G.make_model(kind, seed=s * 100 + k).fit(X[tr], y[tr])
            tf.add(getattr(m, "tree_format_used", None))
            oof[te] = m.predict_proba(X[te])[:, 1]
            folds.append(roc_auc_score(y[te], oof[te]))
        th, se, sp = youden(y, oof)
        reps.append({"auc": roc_auc_score(y, oof), "sens": se, "spec": sp, "thr": th})
    a = np.array([r["auc"] for r in reps])
    return {"auc_pooled_mean": float(a.mean()), "auc_pooled_sd": float(a.std(ddof=1)),
            "auc_fold_mean": float(np.mean(folds)), "auc_fold_sd": float(np.std(folds, ddof=1)),
            "sens_youden_mean": float(np.mean([r["sens"] for r in reps])),
            "spec_youden_mean": float(np.mean([r["spec"] for r in reps])),
            "youden_thr_median": float(np.median([r["thr"] for r in reps])),
            "n": int(len(y)), "n_epilepsy": int(y.sum()), "repeats": reps,
            "tree_format_used": sorted(str(t) for t in tf if t is not None)}


def fmt_cv(name, r):
    return (f"  {name:<34} AUC {r['auc_pooled_mean']:.3f} +- {r['auc_pooled_sd']:.3f} (pooled OOF, 5 repeats) | "
            f"per-fold {r['auc_fold_mean']:.3f} +- {r['auc_fold_sd']:.3f} | sens {r['sens_youden_mean']:.2f} "
            f"spec {r['spec_youden_mean']:.2f} @Youden | n={r['n']} ({r['n_epilepsy']} epilepsy)")


def site_norm(X, c):
    Z = X.copy()
    for v in np.unique(c):
        mu, sd = X[c == v].mean(0), X[c == v].std(0) + 1e-9
        Z[c == v] = (X[c == v] - mu) / sd
    return Z


def main():
    t0 = time.time()
    X, meta = G.build_table(log=log)
    R["timings_s"]["features_all_people"] = round(time.time() - t0, 1)
    y, c = meta.label.to_numpy().astype(int), meta.country.to_numpy().astype(str)
    data = {}
    for k in ["GB", "NG"]:
        mk = meta[c == k]
        data[k] = {"people": int(len(mk)), "epilepsy": int(mk.label.sum()), "control": int((1 - mk.label).sum()),
                   "median_seconds": float(mk.seconds.median()),
                   "median_clean_frac": float(mk.clean_frac.median()),
                   "people_with_wrapped_channels": int((mk.wrapped_channels > 0).sum()),
                   "cleaning_skipped_(too_few_clean_windows)": int((~mk.cleaning_applied.astype(bool)).sum())}
    R["data"] = data
    log(f"features: {X.shape} in {R['timings_s']['features_all_people']}s | {json.dumps(data)}")
    assert np.isfinite(X).all()
    save()

    # a) within country
    R["within_country"] = {}
    log("\n== a) within country: stratified 10-fold CV over people, 5 repeats ==")
    for k in ["GB", "NG"]:
        for kind in KINDS:
            t = time.time()
            r = cv(X[c == k], y[c == k], kind)
            r["seconds"] = round(time.time() - t, 1)
            R["within_country"][f"{k}_{kind}"] = r
            log(fmt_cv(f"{k} {kind} (paper {PAPER[k]})", r) + f" | {r['seconds']}s")
            save()

    # d) confound check: recording-quality numbers only
    log("\n== d) confound check: quality numbers only (no EEG content) ==")
    Q = meta[QUALITY_COLS].to_numpy(float)
    R["quality_only"] = {}
    for k in ["GB", "NG"]:
        r = cv(Q[c == k], y[c == k], "lr")
        R["quality_only"][k] = r
        log(fmt_cv(f"{k} quality-only LR", r))
    save()

    # b) cross country
    log("\n== b) cross country (train on one, test on the other) ==")
    R["cross_country"] = {}
    for norm in ["plain", "site_normalised"]:
        XX = X if norm == "plain" else site_norm(X, c)
        for tr_c, te_c in [("GB", "NG"), ("NG", "GB")]:
            for kind in KINDS:
                t = time.time()
                aucs, sss = [], []
                for s in (SEEDS if kind == "snap" else [0]):
                    m = G.make_model(kind, seed=s).fit(XX[c == tr_c], y[c == tr_c])
                    p = m.predict_proba(XX[c == te_c])[:, 1]
                    aucs.append(roc_auc_score(y[c == te_c], p))
                    sss.append(youden(y[c == te_c], p)[1:])
                key = f"{tr_c}->{te_c}_{kind}_{norm}"
                R["cross_country"][key] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                                           "sens_youden": float(np.mean([a for a, _ in sss])),
                                           "spec_youden": float(np.mean([b for _, b in sss])),
                                           "runs": len(aucs), "seconds": round(time.time() - t, 1)}
                log(f"  {key:<34} AUC {np.mean(aucs):.3f}" + (f" +- {np.std(aucs):.3f} ({len(aucs)} seeds)"
                                                               if len(aucs) > 1 else "") + f" (paper {PAPER['cross']})")
                save()

    # c) both countries pooled, with / without a country flag
    log("\n== c) both countries together: 10-fold CV x5, stratified by country x label ==")
    R["pooled"] = {}
    strat = np.char.add(c, y.astype(str))
    flag = (c == "NG").astype(float)[:, None]
    for use_flag in [False, True]:
        XX = np.hstack([X, flag]) if use_flag else X
        for kind in KINDS:
            t = time.time()
            r = cv(XX, y, kind, strat=strat)
            r["seconds"] = round(time.time() - t, 1)
            key = f"{kind}_{'with' if use_flag else 'no'}_country_flag"
            R["pooled"][key] = r
            log(fmt_cv(key, r) + f" | {r['seconds']}s")
            save()

    # e) final model: best pooled config WITHOUT the country flag (a new clinic has no valid flag value);
    #    the flag result is reported above so the reader can see what it would add.
    best = max(KINDS, key=lambda kd: R["pooled"][f"{kd}_no_country_flag"]["auc_pooled_mean"])
    gain = {kd: R["pooled"][f"{kd}_with_country_flag"]["auc_pooled_mean"]
            - R["pooled"][f"{kd}_no_country_flag"]["auc_pooled_mean"] for kd in KINDS}
    t = time.time()
    model = G.make_model(best, seed=0).fit(X, y)
    train_s = time.time() - t
    imp = model.feature_importances_ if best == "snap" else np.ones(X.shape[1])
    final = {"model": model, "kind": best, "kind_label": KINDS[best], "feature_names": G.FEATURE_NAMES,
             "mu": X.mean(0), "sd": X.std(0) + 1e-9, "importance": imp, "n_train": int(len(y)),
             "med_epi": np.median(X[y == 1], 0), "med_ctl": np.median(X[y == 0], 0),
             "threshold": R["pooled"][f"{best}_no_country_flag"]["youden_thr_median"],
             "cv_auc_pooled": R["pooled"][f"{best}_no_country_flag"]["auc_pooled_mean"],
             "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node()}
    import joblib
    joblib.dump(final, G.MODEL_PATH)
    G._FINAL = None
    ex = int(np.where(c == "NG")[0][0])
    ppl = G.people()
    exf = ppl[(ppl.country == "NG") & (ppl.subject == int(meta.subject[ex]))].files.iloc[0][0]
    t = time.time()
    demo = G.predict_person(exf)
    pred_s = time.time() - t
    R["final"] = {"kind": best, "label": KINDS[best], "country_flag_used": False,
                  "country_flag_auc_gain_in_pooled_cv": gain, "threshold_youden_from_cv": final["threshold"],
                  "n_train": int(len(y)), "train_seconds": round(train_s, 2),
                  "tree_format_used": getattr(model, "tree_format_used", None),
                  "predict_person_seconds_first_call_incl_load": round(pred_s, 2),
                  "demo_file": str(Path(exf).relative_to(ROOT)), "demo_true_label": int(y[ex]), "demo_output": demo}
    log(f"\n== e) final model: {best} ({KINDS[best]}) on all {len(y)} people, {train_s:.1f}s; "
        f"country flag NOT used (pooled-CV AUC gain with flag: "
        + ", ".join(f"{k} {v:+.3f}" for k, v in gain.items()) + ")")
    log(f"  threshold (median Youden over CV repeats) = {final['threshold']:.3f}; saved {G.MODEL_PATH.name}")
    log(f"  predict_person demo on {R['final']['demo_file']} (true label {int(y[ex])}): {json.dumps(demo)}")
    checks(X, meta, y, c, model)
    R["timings_s"]["total"] = round(time.time() - t0, 1)
    log(f"\ntotal {R['timings_s']['total']}s on {platform.node()} ({platform.machine()}), 1 thread, nice 10")
    save()


def checks(X, meta, y, c, model):
    """f) Is the model reading epilepsy, or where/when people were recorded?"""
    import pandas as pd
    log("\n== f) confound checks ==")
    R["checks"] = {}
    # 1. recording dates: were the two groups recorded on the same days?
    md = {"GB": pd.read_csv(G.DATA / "metadata_guineabissau.csv"), "NG": pd.read_csv(G.DATA / "metadata_nigeria.csv")}
    for k, df in md.items():
        grp = df.columns[[col.lower() == "group" for col in df.columns]][0]
        day = pd.to_datetime(df.startTime, dayfirst=True).dt.date
        days = {g: set(day[df[grp].str.lower() == g]) for g in ["epilepsy", "control"]}
        both = days["epilepsy"] & days["control"]
        n_both = int(df[day.isin(both)].shape[0])
        R["checks"][f"{k}_recording_days"] = {"epilepsy_days": len(days["epilepsy"]), "control_days": len(days["control"]),
                                             "days_with_both_groups": len(both), "rows_on_shared_days": n_both,
                                             "rows": int(len(df))}
        log(f"  {k} recording days: epilepsy {len(days['epilepsy'])}, control {len(days['control'])}, "
            f"shared {len(both)} ({n_both}/{len(df)} recordings on a shared day)")
    # 2. ablation: drop gamma (30-45 Hz, where mains hum / muscle / environment noise live) and line length
    keep = np.array([not (("gamma" in n) or ("linelen" in n)) for n in G.FEATURE_NAMES])
    R["checks"]["no_gamma_no_linelen"] = {}
    for k in ["GB", "NG"]:
        for kind in KINDS:
            r = cv(X[c == k][:, keep], y[c == k], kind)
            R["checks"]["no_gamma_no_linelen"][f"{k}_{kind}"] = r
            log(fmt_cv(f"{k} {kind} without gamma+linelen", r))
            save()
    # 3. what the final model leans on
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        top = np.argsort(-imp)[:10]
        R["checks"]["final_top10_features"] = [{"feature": G.FEATURE_NAMES[j], "importance": float(imp[j]),
                                                "plain": G.describe(G.FEATURE_NAMES[j]),
                                                "median_epilepsy": float(np.median(X[y == 1, j])),
                                                "median_control": float(np.median(X[y == 0, j]))} for j in top]
        log("  final model top-10 features: " + "; ".join(f"{G.FEATURE_NAMES[j]} ({imp[j]:.3f}, "
            f"epi {np.median(X[y == 1, j]):.2f} vs ctl {np.median(X[y == 0, j]):.2f})" for j in top))
        bands = {}
        for j, v in enumerate(imp):
            b = next((bd for bd in list(G.BANDS) + ["linelen", "mobility", "complexity"] if bd in G.FEATURE_NAMES[j]), "?")
            bands[b] = bands.get(b, 0) + float(v)
        tot = sum(bands.values()) or 1
        R["checks"]["final_importance_by_feature_family"] = {b: round(v / tot, 3) for b, v in bands.items()}
        log("  importance share by family: " + json.dumps(R["checks"]["final_importance_by_feature_family"]))
    save()


if __name__ == "__main__":
    main()
