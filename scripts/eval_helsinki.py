"""Does the seizure detector work on NEWBORNS? Event-level evaluation on the Helsinki neonatal EEG dataset
(79 NICU neonates, mostly 1 to 2 h of continuous recording), using the project's own detector code unchanged.

  (a) children + adults -> newborns : ONE model trained on CHB-MIT + all Siena, tested on every newborn
                          ("a brand-new newborn, no newborn training data").
  (b) mixed, newborn held out : model trained on CHB-MIT + Siena + all OTHER newborns; the test newborn is never in
                          training. Folds hold out FOLDS groups of newborns (default 20, about 4 newborns each, seizure
                          newborns spread evenly) instead of one fit per newborn, to fit the time box.
Truth = CONSENSUS events (all 3 experts mark the second). The same scores are also matched against MAJORITY events.
v1 = cached features; v2 = src.experiments.transform(X, 'v2') computed in memory ("change vs last 10 min").

Run on the VM (one process per feature set, 1 thread each):
  OMP_NUM_THREADS=1 python scripts/eval_helsinki.py v1
  OMP_NUM_THREADS=1 python scripts/eval_helsinki.py v2
  python scripts/eval_helsinki.py merge      -> runs/helsinki_results.json / .txt
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import detector as D            # noqa: E402  (fit, events_from_scores, match, xy, index: unchanged)
from src import helsinki as H            # noqa: E402
from src import siena as S               # noqa: E402
from src.experiments import transform    # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
NEG_PER_FILE = int(os.environ.get("NEG_PER_FILE", 500))   # RAM guard; D.fit then samples MAX_NEG of the pool
FOLDS = int(os.environ.get("FOLDS", 20))
RUNS = D.RUNS


def sample_rows(Z, y, fname):
    rng = np.random.default_rng(zlib.crc32(fname.encode()))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep = np.sort(np.concatenate([pos, rng.choice(neg, min(NEG_PER_FILE, len(neg)), replace=False)]))
    return Z[keep], y[keep]


def summarise(rows, th, sz):
    tot = {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
    for fname, tc, s in rows:
        res = D.match(D.events_from_scores(tc, s, th), sz[sz.file == fname], tc[-1] + 1)
        for k in tot:
            tot[k] += res[k]
    return {"seizures": tot["seizures"], "detected": tot["detected"],
            "sensitivity": round(tot["detected"] / max(tot["seizures"], 1), 3),
            "false_alerts": tot["false_alerts"],
            "false_alerts_per_24h": round(tot["false_alerts"] / max(tot["hours"], 1e-9) * 24, 2),
            "median_delay_s": float(np.median(tot["delays"])) if tot["delays"] else None,
            "hours": round(tot["hours"], 1)}


def sweep(rows, sz):
    return {str(th): summarise(rows, th, sz) for th in THRESHOLDS}


def run(fset):
    t0 = time.time()
    hfiles, sz = H.index()
    sz_maj = H.seizures("majority")
    extra = pd.read_csv(H.HEL / "files_extra.csv").set_index("file")
    sfiles, ssz = S.index()
    cfiles, csz = D.index()

    # ---- training pools (all rows sampled the same way: all seizure windows + up to NEG_PER_FILE others)
    chb, skipped = [], 0
    for r in cfiles.itertuples():
        if not (D.FEAT_DIR / f"{r.file}.npz").exists():
            skipped += 1
            continue
        X, y = D.xy(r.case, r.file, csz)
        chb.append(sample_rows(transform(X, fset), y, r.file))
    sie = []
    for r in sfiles.itertuples():
        X, _, y = S.features(r.case, r.file, ssz)
        sie.append(sample_rows(transform(X, fset), y, r.file))
    nb = {}
    for r in hfiles.itertuples():
        X, _, y = H.features(r.case, r.file, sz)
        nb[r.file] = sample_rows(transform(X, fset), y, r.file)
    print(f"[{fset}] pools: CHB {len(chb)} recordings ({skipped} skipped), Siena {len(sie)}, newborns {len(nb)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- (a) children + adults -> newborns
    model_a = D.fit(chb + sie)
    print(f"[{fset}] model (a) fitted ({time.time() - t0:.0f}s)", flush=True)

    # ---- folds: newborns with the most seizure seconds first, dealt round-robin so every fold has some
    order = sorted(hfiles.file, key=lambda f: -extra.loc[f, "consensus_seizure_s"])
    fold_of = {f: i % FOLDS for i, f in enumerate(order)}
    rows_a, rows_b = [], []
    for k in range(FOLDS):
        test = [f for f in hfiles.file if fold_of[f] == k]
        mixed = D.fit(chb + sie + [nb[f] for f in hfiles.file if fold_of[f] != k])
        for f in test:
            case = hfiles.loc[hfiles.file == f, "case"].iloc[0]
            X, tc, _ = H.features(case, f, sz)
            Z = transform(X, fset)
            rows_a.append((f, tc, model_a.predict_proba(Z)[:, 1]))
            rows_b.append((f, tc, mixed.predict_proba(Z)[:, 1]))
        print(f"[{fset}]   fold {k + 1}/{FOLDS} done: {len(test)} newborns ({time.time() - t0:.0f}s)", flush=True)

    free = [r for r in rows_a if extra.loc[r[0], "reviewers_marking_seizure_clinical_sheet"] == 0]
    free_b = [r for r in rows_b if extra.loc[r[0], "reviewers_marking_seizure_clinical_sheet"] == 0]
    out = {"fset": fset, "folds": FOLDS, "neg_per_file_cap": NEG_PER_FILE,
           "chb_recordings_used": len(chb), "chb_recordings_skipped_no_features": skipped,
           "siena_recordings": len(sie), "newborn_recordings": len(nb),
           "a_children_adults_to_newborns": sweep(rows_a, sz),
           "b_mixed_newborn_held_out": sweep(rows_b, sz),
           "a_vs_majority_truth": sweep(rows_a, sz_maj),
           "b_vs_majority_truth": sweep(rows_b, sz_maj),
           "no_reviewer_marked_seizure_neonates": {
               "n": len(free), "a": sweep(free, sz), "b": sweep(free_b, sz)},
           "per_newborn_at_declared_threshold": {}}
    for f, tc, sb in rows_b:
        sa = [r for r in rows_a if r[0] == f][0][2]
        out["per_newborn_at_declared_threshold"][f] = {
            "a": summarise([(f, tc, sa)], D.THRESHOLD, sz), "b": summarise([(f, tc, sb)], D.THRESHOLD, sz)}
    out["wall_clock_s"] = round(time.time() - t0)
    (RUNS / f"helsinki_results_{fset}.json").write_text(json.dumps(out, indent=2))
    print(f"[{fset}] done in {out['wall_clock_s']} s", flush=True)


# ------------------------------------------------------------------ merge / report
def table(title, sw):
    lines = [f"\n== {title} =="]
    for th, v in sw.items():
        flag = "  <- declared threshold" if float(th) == D.THRESHOLD else ""
        lines.append(f"  th={th}: caught {v['detected']}/{v['seizures']} ({v['sensitivity']:.0%}), "
                     f"{v['false_alerts_per_24h']} false alerts/24h ({v['false_alerts']} alerts), "
                     f"median delay {v['median_delay_s']}s, {v['hours']} h of EEG{flag}")
    return "\n".join(lines).replace("Nones", "n/a")


def merge():
    r = {k: json.loads((RUNS / f"helsinki_results_{k}.json").read_text()) for k in ("v1", "v2")}
    ex = pd.read_csv(H.HEL / "files_extra.csv")
    sz, szm = H.seizures(), H.seizures("majority")
    hours = ex.duration_s.sum() / 3600
    with_ev = ex[ex.events_consensus > 0]
    dur = (sz.end_s - sz.start_s)
    res = {"dataset": {"name": "Helsinki neonatal EEG seizure dataset", "url": "https://zenodo.org/records/2547147",
                       "neonates": len(ex), "hours": round(hours, 1),
                       "consensus_events": len(sz), "consensus_neonates_with_events": int(len(with_ev)),
                       "majority_events": len(szm), "majority_neonates_with_events": int((ex.events_majority > 0).sum()),
                       "consensus_event_length_s": {"min": int(dur.min()), "median": float(dur.median()),
                                                    "max": int(dur.max()), "n_under_10s": int((dur < 10).sum())},
                       "neonates_no_reviewer_marked_seizure": int((ex.reviewers_marking_seizure_clinical_sheet == 0).sum())},
           "v1": r["v1"], "v2": r["v2"]}
    pn = r["v2"]["per_newborn_at_declared_threshold"]
    tb = []
    for f in with_ev.file:
        a, b = pn[f]["a"], pn[f]["b"]
        tb.append({"file": f, "eeg_hours": b["hours"], "consensus_seizures": b["seizures"],
                   "caught_a_v2": a["detected"], "caught_b_v2": b["detected"],
                   "false_alerts_b_v2": b["false_alerts"]})
    res["per_newborn_with_consensus_seizures_v2_th0.8"] = tb
    bs = [x for x in tb]
    res["summary_c"] = {
        "newborns_with_consensus_seizures": len(tb),
        "newborns_all_seizures_caught_b_v2": sum(1 for x in bs if x["caught_b_v2"] == x["consensus_seizures"]),
        "newborns_some_caught_b_v2": sum(1 for x in bs if 0 < x["caught_b_v2"] < x["consensus_seizures"]),
        "newborns_none_caught_b_v2": sum(1 for x in bs if x["caught_b_v2"] == 0),
        "caught_b_v2": sum(x["caught_b_v2"] for x in bs), "seizures": sum(x["consensus_seizures"] for x in bs)}
    res["wall_clock_s"] = {"v1": r["v1"]["wall_clock_s"], "v2": r["v2"]["wall_clock_s"]}
    t = {}
    for name in ("download", "annotations", "features"):
        s, e = RUNS / f"helsinki_t_{name}_start", RUNS / f"helsinki_t_{name}_end"
        if s.exists() and e.exists():
            t[f"{name}_s"] = int(e.read_text()) - int(s.read_text())
    res["timings_s"] = t
    (RUNS / "helsinki_results.json").write_text(json.dumps(res, indent=2))

    d = res["dataset"]
    L = [f"Helsinki neonatal EEG (Zenodo 2547147): {d['neonates']} neonates, {d['hours']} h of EEG, 256 Hz, 19 electrodes",
         f"CONSENSUS (3 of 3 experts) truth: {d['consensus_events']} events in {d['consensus_neonates_with_events']} neonates "
         f"(event length min/median/max {d['consensus_event_length_s']['min']}/{d['consensus_event_length_s']['median']:.0f}/"
         f"{d['consensus_event_length_s']['max']} s, {d['consensus_event_length_s']['n_under_10s']} under 10 s)",
         f"MAJORITY (2 of 3) truth: {d['majority_events']} events in {d['majority_neonates_with_events']} neonates",
         f"{d['neonates_no_reviewer_marked_seizure']} neonates were not marked as having seizures by any of the 3 experts (clinical sheet)",
         f"training: CHB-MIT {r['v1']['chb_recordings_used']} recordings + Siena {r['v1']['siena_recordings']} recordings"
         f"{' + other newborns (b)' }; (b) folds: {r['v1']['folds']} groups of newborns held out"]
    for fset in ("v1", "v2"):
        x = res[fset]
        L.append(f"\n##### {fset} features ({'as cached' if fset == 'v1' else '+ change vs last 10 min'}), run time {x['wall_clock_s']} s")
        L.append(table(f"(a) CHILDREN + ADULTS -> NEWBORNS, no newborn training data [{fset}]", x["a_children_adults_to_newborns"]))
        L.append(table(f"(b) MIXED: CHB-MIT + Siena + other newborns, test newborn held out [{fset}]", x["b_mixed_newborn_held_out"]))
        L.append(table(f"   extra: (b) scores matched against MAJORITY events [{fset}]", x["b_vs_majority_truth"]))
        nf = x["no_reviewer_marked_seizure_neonates"]
        L.append(table(f"   extra: (a) on the {nf['n']} newborns NO expert marked (every alert is false) [{fset}]", nf["a"]))
        L.append(table(f"   extra: (b) on the same {nf['n']} newborns [{fset}]", nf["b"]))
    c = res["summary_c"]
    L.append(f"\n== (c) per-newborn summary, (b) v2 at th {D.THRESHOLD} ==")
    L.append(f"  {c['newborns_with_consensus_seizures']} newborns have consensus seizures; (b) v2 caught {c['caught_b_v2']}/{c['seizures']}: "
             f"all seizures caught in {c['newborns_all_seizures_caught_b_v2']} newborns, some in {c['newborns_some_caught_b_v2']}, "
             f"none in {c['newborns_none_caught_b_v2']}")
    L.append("  newborn: EEG hours, consensus seizures, caught (a) v2 / (b) v2, false alerts (b) v2")
    for x in tb:
        L.append(f"    {x['file']}: {x['eeg_hours']} h, {x['consensus_seizures']} seizures, caught {x['caught_a_v2']} / {x['caught_b_v2']}, "
                 f"{x['false_alerts_b_v2']} false alerts")
    L.append(f"\ntimings (wall clock, 1 thread per process, shared VM): {t}; eval v1 {res['wall_clock_s']['v1']} s, v2 {res['wall_clock_s']['v2']} s")
    (RUNS / "helsinki_results.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge()
    elif sys.argv[1:] in (["v1"], ["v2"]):
        run(sys.argv[1])
    else:
        sys.exit(__doc__)
