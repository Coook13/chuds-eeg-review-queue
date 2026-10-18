"""Does adding other ADULTS to the training set help a brand-new adult patient?  (leave-one-adult-out)

  children-only : model trained on CHB-MIT only (control, same training rows as the mixed model)
  mixed         : for each Siena patient, train on ALL CHB-MIT recordings + ALL OTHER Siena patients' recordings,
                  test on every recording of that patient (the patient is never seen in training)
Both are run with the v1 features (src.features, as cached) and the v2 features (src.experiments.transform(X, 'v2'),
which adds "change vs this patient's last 10 minutes"). v2 is computed here from the cached v1 features, in memory:
no other cache is read or written beyond the existing runs/features and runs/siena_features (read only).

Training rows: per recording, all seizure windows + up to NEG_PER_FILE random other windows, then src.detector.fit
(which samples MAX_NEG non-seizure windows from the pool). Same rule for CHB-MIT and Siena recordings.

Run on the VM:  OMP_NUM_THREADS=1 python scripts/eval_siena_mixed.py
Writes runs/siena_results_mixed.json and runs/siena_results_mixed.txt.
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import detector as D            # noqa: E402  (fit, xy, index: unchanged)
from src import siena as S               # noqa: E402
from src.experiments import transform    # noqa: E402  (v2 features; import has no side effects)
import eval_siena as E                   # noqa: E402  (summarise / table, same metric code as the first round)

NEG_PER_FILE = int(os.environ.get("NEG_PER_FILE", 1000))
THRESHOLDS = E.THRESHOLDS
OUT_JSON, OUT_TXT = D.RUNS / "siena_results_mixed.json", D.RUNS / "siena_results_mixed.txt"


def sample_rows(Z, y, fname):
    rng = np.random.default_rng(zlib.crc32(fname.encode()))   # same rows for v1 and v2
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep = np.sort(np.concatenate([pos, rng.choice(neg, min(NEG_PER_FILE, len(neg)), replace=False)]))
    return Z[keep], y[keep]


def build_pool(fset, files, sz):
    chb, sie, skipped = [], {}, 0
    cfiles, csz = D.index()
    for r in cfiles.itertuples():
        if not (D.FEAT_DIR / f"{r.file}.npz").exists():
            skipped += 1
            continue
        X, y = D.xy(r.case, r.file, csz)
        chb.append(sample_rows(transform(X, fset), y, r.file))
    for r in files.itertuples():
        X, _, y = S.features(r.case, r.file, sz)
        sie.setdefault(r.case, []).append(sample_rows(transform(X, fset), y, r.file))
    return chb, sie, skipped


def run_stage(fset, files, sz, same27):
    t0 = time.time()
    chb, sie, skipped = build_pool(fset, files, sz)
    n_pos_chb = int(sum(p[1].sum() for p in chb))
    print(f"[{fset}] pools built: CHB {len(chb)} recordings ({skipped} skipped, no cached features), "
          f"{n_pos_chb} seizure windows; Siena {sum(len(v) for v in sie.values())} recordings "
          f"({time.time() - t0:.0f}s)", flush=True)
    ctrl = D.fit(chb)
    print(f"[{fset}] children-only model fitted ({time.time() - t0:.0f}s)", flush=True)
    rows = {"children_only": [], "mixed": []}
    case_of = {}
    for case in sorted(files.case.unique()):
        mixed = D.fit(chb + [p for c, v in sie.items() if c != case for p in v])
        for r in files[files.case == case].itertuples():
            X, tc, _ = S.features(r.case, r.file, sz)
            Z = transform(X, fset)
            rows["children_only"].append((r.file, tc, ctrl.predict_proba(Z)[:, 1]))
            rows["mixed"].append((r.file, tc, mixed.predict_proba(Z)[:, 1]))
            case_of[r.file] = case
        print(f"[{fset}]   {case} done ({time.time() - t0:.0f}s)", flush=True)
    out = {"chb_recordings_used": len(chb), "chb_recordings_skipped_no_features": skipped,
           "chb_seizure_windows": n_pos_chb, "wall_clock_s": round(time.time() - t0)}
    for name, rs in rows.items():
        sub = [r for r in rs if r[0] in same27]
        pp = {}
        for case in sorted(files.case.unique()):
            pp[case] = E.summarise([r for r in rs if case_of[r[0]] == case], D.THRESHOLD)
        out[name] = {"all_41_recordings": {str(th): E.summarise(rs, th) for th in THRESHOLDS},
                     "same_27_recordings": {str(th): E.summarise(sub, th) for th in THRESHOLDS},
                     "per_patient_at_declared_threshold": pp}
    return out


def line(tag, v):
    return (f"{tag}: caught {v['detected']}/{v['seizures']}, {v['false_alerts_per_24h']} false alerts/24h, "
            f"median delay {v['median_delay_s']}s")


def write(res, txt_extra=""):
    OUT_JSON.write_text(json.dumps(res, indent=2))
    prev = res.get("previous_round", {})
    t = [f"Mixed general model on Siena adults (leave-one-adult-out), {res['dataset']['recordings']} recordings, "
         f"{res['dataset']['seizures']} seizures, {res['dataset']['hours']} h"]
    for fset, title in [("v1", "v1 features (as cached)"), ("v2", "v2 features (+ change vs last 10 min)")]:
        if fset not in res:
            continue
        s = res[fset]
        t.append(f"\n##### {title}: CHB-MIT {s['chb_recordings_used']} recordings, {s['chb_seizure_windows']} seizure windows; "
                 f"{s['wall_clock_s']} s")
        t.append(E.table(f"children only (CHB-MIT), all 41 recordings [{fset}]", s["children_only"]["all_41_recordings"],
                         s["children_only"]["per_patient_at_declared_threshold"]))
        t.append(E.table(f"MIXED (CHB-MIT + other adults), all 41 recordings [{fset}]", s["mixed"]["all_41_recordings"],
                         s["mixed"]["per_patient_at_declared_threshold"]))
        t.append(E.table(f"same 27 recordings / 31 seizures: children only [{fset}]", s["children_only"]["same_27_recordings"]))
        t.append(E.table(f"same 27 recordings / 31 seizures: MIXED [{fset}]", s["mixed"]["same_27_recordings"]))
    if prev:
        t.append(E.table("same 27 recordings: PERSONAL after earlier seizures only [v1, first round]",
                         prev["personal_after_earlier_same_27"]))
    if "pitch_line" in res:
        t.append("\nPITCH LINE (th 0.8, same 27 recordings / 31 seizures)\n  " + res["pitch_line"])
    t.append(txt_extra)
    OUT_TXT.write_text("\n".join(t) + "\n")
    return "\n".join(t)


def main():
    t_all = time.time()
    files, sz = S.index()
    prev_json = json.loads((D.RUNS / "siena_results.json").read_text())
    untestable = set(prev_json["adults_personal_earlier_only"]["recordings_not_testable"])
    same27 = set(files.file) - untestable
    assert len(same27) == 27, len(same27)
    res = {"dataset": prev_json["dataset"],
           "design": {"mixed": "train on all CHB-MIT + all other Siena patients, test on the held-out Siena patient",
                      "neg_per_file_cap": NEG_PER_FILE, "fit_max_neg": D.MAX_NEG,
                      "declared_threshold": D.THRESHOLD},
           "previous_round": {
               "children_only_first_round_same_27": prev_json["adults_personal_earlier_only"]["children_model_on_the_same_recordings"],
               "personal_after_earlier_same_27": prev_json["adults_personal_earlier_only"]["tested_recordings"]}}
    for fset in ["v1", "v2"]:
        res[fset] = run_stage(fset, files, sz, same27)
        write(res)    # partial results are saved after each stage
    th = str(D.THRESHOLD)
    p = res["previous_round"]
    v1, v2 = res["v1"], res["v2"]
    res["pitch_line"] = (
        f"children only {line('', v1['children_only']['same_27_recordings'][th]).lstrip(': ')} | "
        f"mixed (children + other adults) {line('', v1['mixed']['same_27_recordings'][th]).lstrip(': ')} | "
        f"personal after earlier seizures {line('', p['personal_after_earlier_same_27'][th]).lstrip(': ')} "
        f"| with v2 features: children only {v2['children_only']['same_27_recordings'][th]['detected']}/31, "
        f"mixed {v2['mixed']['same_27_recordings'][th]['detected']}/31")
    res["wall_clock_s"] = round(time.time() - t_all)
    print(write(res, f"\nwall clock: {res['wall_clock_s']} s (v1 {v1['wall_clock_s']} s, v2 {v2['wall_clock_s']} s)"))
    OUT_JSON.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
