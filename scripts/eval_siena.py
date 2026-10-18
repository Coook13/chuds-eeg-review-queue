"""Does the seizure detector work on ADULTS? Two honest, event-level evaluations on the Siena Scalp EEG
Database (14 adults, 41 recordings, 47 seizures), using the project's own detector code unchanged.

  a) children -> adults : one general model trained on ALL CHB-MIT recordings (children), tested on every
                          Siena recording. This is the "brand-new adult patient" number.
  b) adults, personal   : per Siena patient, recordings in file order; each recording is tested with a model
                          trained ONLY on that patient's EARLIER recordings that contain a seizure
                          (skipped if there is none). Nothing from later recordings is ever used.
  c) extra (hybrid)     : on the recordings that (b) can test, average (a)'s score and (b)'s score.

Run on the VM:  OMP_NUM_THREADS=1 python scripts/eval_siena.py
Writes runs/siena_results.json and runs/siena_results.txt.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import detector as D   # noqa: E402  (fit, events_from_scores, match, xy, index: used unchanged)
from src import siena as S      # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
NEG_PER_CHB_FILE = int(os.environ.get("NEG_PER_CHB_FILE", 1000))   # RAM guard; D.fit then samples MAX_NEG of them
OUT_JSON, OUT_TXT = D.RUNS / "siena_results.json", D.RUNS / "siena_results.txt"


def chb_training_parts():
    """(X, y) for every CHB-MIT recording whose features are already cached (read-only: nothing is written)."""
    files, sz = D.index()
    rng = np.random.default_rng(0)
    parts, skipped, n_pos = [], 0, 0
    for r in files.itertuples():
        if not (D.FEAT_DIR / f"{r.file}.npz").exists():
            skipped += 1
            continue
        X, y = D.xy(r.case, r.file, sz)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = np.sort(np.concatenate([pos, rng.choice(neg, min(NEG_PER_CHB_FILE, len(neg)), replace=False)]))
        parts.append((X[keep], y[keep]))
        n_pos += len(pos)
    return parts, len(files), skipped, n_pos


def summarise(rows, th):
    """rows: list of (fname, tc, scores). Event-level totals over all of them with the project's match()."""
    sz = S.seizures()
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


def table(title, sweep, per_patient=None):
    lines = [f"\n== {title} =="]
    for th, v in sweep.items():
        flag = "  <- declared threshold" if float(th) == D.THRESHOLD else ""
        lines.append(f"  th={th}: caught {v['detected']}/{v['seizures']} ({v['sensitivity']:.0%}), "
                     f"{v['false_alerts_per_24h']} false alerts/24h ({v['false_alerts']} alerts), "
                     f"median delay {v['median_delay_s']}s, {v['hours']} h of EEG{flag}")
    if per_patient:
        lines.append(f"  per patient at th={D.THRESHOLD}:")
        for p, v in per_patient.items():
            lines.append(f"    {p}: caught {v['detected']}/{v['seizures']}, {v['false_alerts']} false alerts "
                         f"in {v['hours']} h, median delay {v['median_delay_s']}s")
    return "\n".join(lines)


def main():
    t_all = time.time()
    files, sz = S.index()
    n_sz_total = len(sz)
    print(f"Siena: {len(files)} recordings, {n_sz_total} seizures", flush=True)

    # ---- load Siena features once (cached by src/siena.py)
    t0 = time.time()
    data = {}
    for r in files.itertuples():
        X, tc, y = S.features(r.case, r.file, sz)
        data[r.file] = {"case": r.case, "X": X, "tc": tc, "y": y, "n_sz": int(r.n_seizures)}
    hours_all = sum((d["tc"][-1] + 1) / 3600 for d in data.values())
    print(f"loaded Siena features: {sum(len(d['tc']) for d in data.values())} windows, {hours_all:.1f} h "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- (a) children -> adults
    t_a = time.time()
    parts, n_chb, skipped, n_pos = chb_training_parts()
    print(f"CHB-MIT training: {len(parts)}/{n_chb} recordings used ({skipped} had no cached features), "
          f"{n_pos} seizure windows", flush=True)
    general = D.fit(parts)
    del parts
    print(f"general model fitted ({time.time() - t_a:.0f}s)", flush=True)
    gscore = {f: general.predict_proba(d["X"])[:, 1] for f, d in data.items()}
    rows_a = [(f, d["tc"], gscore[f]) for f, d in data.items()]
    sweep_a = {str(th): summarise(rows_a, th) for th in THRESHOLDS}
    pp_a = {}
    for case in sorted(files.case.unique()):
        pp_a[case] = summarise([r for r in rows_a if data[r[0]]["case"] == case], D.THRESHOLD)
    time_a = time.time() - t_a
    print(f"(a) done ({time_a:.0f}s)", flush=True)

    # ---- (b) adults, personal, earlier seizures only
    t_b = time.time()
    rows_b, rows_c, untested, tested_sz = [], [], [], 0
    for case in sorted(files.case.unique()):
        names = files[files.case == case].file.tolist()   # already in file order
        for i, f in enumerate(names):
            earlier = [g for g in names[:i] if data[g]["n_sz"] > 0]
            if not earlier:
                untested.append(f)
                continue
            m = D.fit([(data[g]["X"], data[g]["y"]) for g in earlier])
            if m is None:
                untested.append(f)
                continue
            d = data[f]
            ps = m.predict_proba(d["X"])[:, 1]
            rows_b.append((f, d["tc"], ps))
            rows_c.append((f, d["tc"], (ps + gscore[f]) / 2))
            tested_sz += d["n_sz"]
        print(f"  {case} done ({time.time() - t_b:.0f}s)", flush=True)
    sweep_b = {str(th): summarise(rows_b, th) for th in THRESHOLDS}
    sweep_c = {str(th): summarise(rows_c, th) for th in THRESHOLDS}
    # the same recordings scored by (a) alone, so (b) can be compared like for like
    rows_a_same = [(f, tc, gscore[f]) for f, tc, _ in rows_b]
    sweep_a_same = {str(th): summarise(rows_a_same, th) for th in THRESHOLDS}
    pp_b = {}
    for case in sorted(files.case.unique()):
        sub = [r for r in rows_b if data[r[0]]["case"] == case]
        if sub:
            pp_b[case] = summarise(sub, D.THRESHOLD)
    time_b = time.time() - t_b
    print(f"(b) done ({time_b:.0f}s)", flush=True)

    total = time.time() - t_all
    results = {
        "dataset": {"name": "Siena Scalp EEG Database v1.0.0", "url": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
                    "patients": int(files.case.nunique()), "recordings": len(files), "seizures": n_sz_total,
                    "hours": round(hours_all, 1)},
        "children_to_adults": {"all_recordings": sweep_a, "per_patient_at_declared_threshold": pp_a},
        "adults_personal_earlier_only": {
            "recordings_tested": len(rows_b), "recordings_not_testable": untested, "seizures_tested": tested_sz,
            "tested_recordings": sweep_b, "per_patient_at_declared_threshold": pp_b,
            "children_model_on_the_same_recordings": sweep_a_same},
        "extra_hybrid_on_tested_recordings": sweep_c,
        "chb_training": {"recordings_used": n_chb - skipped, "recordings_skipped_no_features": skipped,
                         "seizure_windows": n_pos, "neg_per_file_cap": NEG_PER_CHB_FILE, "fit_max_neg": D.MAX_NEG},
        "declared_threshold": D.THRESHOLD,
        "wall_clock_s": {"eval_total": round(total), "mode_a": round(time_a), "mode_b_and_c": round(time_b)},
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    txt = [f"Siena (adults): {len(files)} recordings, {files.case.nunique()} patients, {n_sz_total} seizures, "
           f"{hours_all:.1f} h of EEG",
           f"CHB-MIT training set for (a): {n_chb - skipped} recordings, {n_pos} seizure windows",
           table("a) CHILDREN -> ADULTS: general model trained on CHB-MIT, tested on all Siena recordings",
                 sweep_a, pp_a),
           table(f"b) ADULTS, PERSONAL (earlier seizures only): {len(rows_b)} testable recordings, "
                 f"{tested_sz} seizures; {len(untested)} recordings untestable (no earlier seizure file)",
                 sweep_b, pp_b),
           table("   reference: the children-trained model (a) on exactly those same recordings", sweep_a_same),
           table("c) EXTRA hybrid = mean(a, b) on the same testable recordings", sweep_c),
           f"\nwall clock: eval {total:.0f}s (a {time_a:.0f}s, b+c {time_b:.0f}s)"]
    OUT_TXT.write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
