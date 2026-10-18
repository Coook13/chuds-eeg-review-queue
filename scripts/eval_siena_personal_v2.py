"""Fills the one missing cell of the Siena comparison: PERSONAL (trained only on the patient's EARLIER seizure
recordings) with the v2 features. Same protocol as mode (b) in scripts/eval_siena.py, v2 computed in memory from the
cached v1 features via src.experiments.transform(X, 'v2'), one patient at a time (RAM).

Adds a "personal_after_earlier" block to v2 in runs/siena_results_mixed.json and appends a table to
runs/siena_results_mixed.txt (both written by scripts/eval_siena_mixed.py).
Run on the VM:  OMP_NUM_THREADS=1 python scripts/eval_siena_personal_v2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import detector as D            # noqa: E402
from src import siena as S               # noqa: E402
from src.experiments import transform    # noqa: E402
import eval_siena as E                   # noqa: E402

OUT_JSON, OUT_TXT = D.RUNS / "siena_results_mixed.json", D.RUNS / "siena_results_mixed.txt"


def main():
    t0 = time.time()
    files, sz = S.index()
    res = json.loads(OUT_JSON.read_text())
    rows, untested, case_of = [], [], {}
    for case in sorted(files.case.unique()):
        sub = files[files.case == case]
        names, n_sz = sub.file.tolist(), dict(zip(sub.file, sub.n_seizures))
        d = {}
        for f in names:
            X, tc, y = S.features(case, f, sz)
            d[f] = (transform(X, "v2"), tc, y)
        for i, f in enumerate(names):
            earlier = [g for g in names[:i] if n_sz[g] > 0]
            m = D.fit([(d[g][0], d[g][2]) for g in earlier]) if earlier else None
            if m is None:
                untested.append(f)
                continue
            rows.append((f, d[f][1], m.predict_proba(d[f][0])[:, 1]))
            case_of[f] = case
        del d
        print(f"  {case} done ({time.time() - t0:.0f}s)", flush=True)
    prev_untested = set(json.loads((D.RUNS / "siena_results.json").read_text())
                        ["adults_personal_earlier_only"]["recordings_not_testable"])
    assert set(untested) == prev_untested, "different test set than the first round"
    sweep = {str(th): E.summarise(rows, th) for th in E.THRESHOLDS}
    pp = {c: E.summarise([r for r in rows if case_of[r[0]] == c], D.THRESHOLD) for c in sorted(set(case_of.values()))}
    res["v2"]["personal_after_earlier"] = {"same_27_recordings": sweep, "per_patient_at_declared_threshold": pp,
                                           "wall_clock_s": round(time.time() - t0)}
    th = str(D.THRESHOLD)
    g, m_, p_ = (res["v2"]["children_only"]["same_27_recordings"][th], res["v2"]["mixed"]["same_27_recordings"][th], sweep[th])
    f = lambda v: f"caught {v['detected']}/{v['seizures']}, {v['false_alerts_per_24h']} false alerts/24h, median delay {v['median_delay_s']}s"
    res["pitch_line_v2"] = f"children only {f(g)} | mixed {f(m_)} | personal after earlier seizures {f(p_)}"
    OUT_JSON.write_text(json.dumps(res, indent=2))
    with open(OUT_TXT, "a") as fh:
        fh.write(E.table("same 27 recordings: PERSONAL after earlier seizures only [v2 features]", sweep, pp))
        fh.write(f"\n\nPITCH LINE, v2 features (th 0.8, same 27 recordings / 31 seizures)\n  {res['pitch_line_v2']}\n")
        fh.write(f"(personal v2 run: {time.time() - t0:.0f} s)\n")
    print(E.table("PERSONAL [v2]", sweep, pp))
    print(res["pitch_line_v2"])


if __name__ == "__main__":
    main()
