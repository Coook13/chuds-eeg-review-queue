"""Metrics from cached scores (runs/scores/<tag>/), no retraining.

Event level (what a ward experiences):
  sensitivity = seizures with an overlapping alert / all seizures
  false alarms per day = alerts overlapping no seizure / hours * 24
  median seconds from seizure onset to alert
Window level (each 2-second window labelled seizure or not, on the raw model score):
  sensitivity, specificity, AUC.  Specificity looks very high because non-seizure windows
  outnumber seizure windows roughly 1000:1, so always read it next to false alarms per day.

Run:  python scripts/metrics_from_scores.py v1_snap [--th 0.8] [--first-confirm-th 0.95] [--on 5]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detector as D
from src import rules


def window_metrics(y, s, th):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(bool)
    p = np.asarray(s) >= th
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    tn, fp = int((~p & ~y).sum()), int((p & ~y).sum())
    auc = float(roc_auc_score(y, s)) if y.any() and (~y).any() else None
    return {"window_sensitivity": round(tp / max(tp + fn, 1), 4), "window_specificity": round(tn / max(tn + fp, 1), 4),
            "auc": round(auc, 4) if auc is not None else None, "seizure_windows": tp + fn, "other_windows": tn + fp}


def event_metrics(items, th, per_window_fn=None):
    """items: (file, tc, scores, extra, t_min). Only seizures starting at/after t_min count
    (confirm mode: after the first confirmation); hours = number of 1-second windows scored."""
    tot = {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
    sz = D.index()[1]
    for f, tc, s, extra, t_min in items:
        pw = per_window_fn(extra, th) if per_window_fn else None
        truth = sz[(sz.file == f) & (sz.start_s >= t_min)] if t_min is not None else sz[sz.file == f]
        res = D.match(D.events_from_scores(tc, s, th, pw), truth, len(tc))
        for k in tot:
            tot[k] += res[k]
    return {"caught": f"{tot['detected']}/{tot['seizures']}",
            "event_sensitivity": round(tot["detected"] / max(tot["seizures"], 1), 4),
            "false_alarms_per_day": round(tot["false_alerts"] / max(tot["hours"], 1e-9) * 24, 2),
            "median_delay_s": float(np.median(tot["delays"])) if tot["delays"] else None,
            "hours": round(tot["hours"], 1)}


def load(sdir, mode):
    out = []
    for p in sorted(sdir.glob(f"{mode}__*.npz")):
        d = np.load(p, allow_pickle=True)
        out.append((p.name.split("__", 1)[1][:-4], d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--th", type=float, default=D.THRESHOLD)
    ap.add_argument("--first-confirm-th", type=float, default=None)
    ap.add_argument("--on", type=int, default=None, help="override rules.ON_COUNT")
    a = ap.parse_args()
    if a.on:
        rules.ON_COUNT = a.on
    sdir = D.RUNS / "scores" / a.tag
    report = {"tag": a.tag, "threshold": a.th, "first_confirm_threshold": a.first_confirm_th, "on_count": rules.ON_COUNT}
    for mode in ["general", "personal", "past"]:
        rows = load(sdir, mode)
        if not rows:
            continue
        ys, ss = np.concatenate([d["y"] for _, d in rows]), np.concatenate([d["s"] for _, d in rows])
        report[mode] = {**event_metrics([(f, d["tc"], d["s"], None, None) for f, d in rows], a.th), **window_metrics(ys, ss, a.th)}
    rows = load(sdir, "confirm")
    if rows:
        zone = [(f, d) for f, d in rows if (d["stage"] > 0).any()]
        # compare only time after the first confirmation (same windows for both models)
        def cut(d, key):
            m = d["stage"] > 0
            return d["tc"][m], d[key][m], d["y"][m], d["stage"][m]
        res = {}
        for key in ["general", "personalised"]:
            items, ys, ss = [], [], []
            for f, d in zone:
                tc, s, y, st = cut(d, key)
                items.append((f, tc, s, st, float(tc[0]))); ys.append(y); ss.append(s)
            fn = None
            if key == "personalised" and a.first_confirm_th:
                fn = lambda st, th: np.where(st == 1, max(th, a.first_confirm_th), th)
            res[key] = {**event_metrics(items, a.th, fn), **window_metrics(np.concatenate(ys), np.concatenate(ss), a.th)}
        report["confirm_same_time"] = res
    print(json.dumps(report, indent=2))
    out = D.RUNS / f"metrics_{a.tag}_th{a.th}{'_fc' + str(a.first_confirm_th) if a.first_confirm_th else ''}{'_on' + str(a.on) if a.on else ''}.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
