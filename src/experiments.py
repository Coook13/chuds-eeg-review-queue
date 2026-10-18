"""Model-architecture experiments with the same honest event-level protocol as detector.py.
Memory-light so it runs on the 7.8 GB LinuxONE VM: each recording's training rows are sampled
once and cached to disk; folds load only what they need; each test recording is transformed once.

Feature sets:  v1 = 234 per-channel features
               v2 = v1 + 'change vs this patient's last 10 min' + cross-channel max/mean (520)
               v3 = v2 + cross-channel aggregates 2 s and 4 s ago (context)          (624)
Modes:  general  = leave-one-PATIENT-out (chb01 and chb21 are the same patient: one fold)
        personal = same patient, all other recordings (leave-one-file-out, optimistic)
        past     = same patient, ONLY earlier recordings (what a hospital really has)
        hybrid   = general until the patient's first recorded seizure, then past
Run:  python -m src.experiments v1 v2 v3       env: MODES=general,personal,past  NEG_PER_FILE=800
"""
import json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np
from src import detector as D
from src.features import augment, PER_CH

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
# Feature set for the GENERAL model only (personal models use the run's feature set). v1 is a
# prefix of v2 and v2 of v3, so one live feature vector can feed both models.
GENERAL_FSET = os.environ.get("GENERAL_FSET")
NEG_PER_FILE = int(os.environ.get("NEG_PER_FILE", 800))
GENERAL_MAX_NEG = int(os.environ.get("GENERAL_MAX_NEG", 120000))
MODES = os.environ.get("MODES", "general,personal,past,confirm").split(",")


def transform(X, fset):
    if fset == "v1":
        return X
    A = augment(X)
    if fset == "v2":
        return A
    agg = A[:, -4 * len(PER_CH):]           # the cross-channel aggregates at the end of v2
    lag = lambda k: np.vstack([np.repeat(agg[:1], k, 0), agg[:-k]])
    return np.concatenate([A, lag(2), lag(4)], axis=1)


def train_rows(fset, case, fname, sz):
    """All seizure windows + a fixed random sample of other windows, cached on disk."""
    path = D.RUNS / "train_rows_r30" / fset / f"{fname}.npz"  # r30: baseline refreshed every 30 s
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        X, _, y = D.features(case, fname, sz)
        Z = transform(X, fset)
        rng = np.random.default_rng(zlib.crc32(fname.encode()))
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = np.sort(np.concatenate([pos, rng.choice(neg, min(NEG_PER_FILE, len(neg)), replace=False)]))
        np.savez_compressed(path, Z=Z[keep], y=y[keep])
    d = np.load(path)
    return d["Z"], d["y"]


GENERAL_EXTRA = [s for s in os.environ.get("GENERAL_EXTRA", "").split(",") if s]  # e.g. siena,helsinki
GENERAL_TAG = os.environ.get("GENERAL_TAG", "")  # model filename tag, so test models never overwrite shipped ones


def extra_rows(fset):
    """Training rows from other datasets (Siena adults, Helsinki newborns), added to every general
    model. They are never tested on, so the CHB-MIT leave-one-patient-out test stays honest."""
    import importlib
    parts = []
    for name in GENERAL_EXTRA:
        M = importlib.import_module(f"src.{name}")
        files, sz = M.index()
        for r in files.itertuples():
            path = D.RUNS / "train_rows_r30" / fset / f"extra_{name}_{r.file}.npz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                X, _, y = M.features(r.case, r.file, sz)
                Z = transform(X, fset)
                rng = np.random.default_rng(zlib.crc32(r.file.encode()))
                pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
                keep = np.sort(np.concatenate([pos, rng.choice(neg, min(NEG_PER_FILE, len(neg)), replace=False)]))
                np.savez_compressed(path, Z=Z[keep].astype(np.float32), y=y[keep])
            d = np.load(path)
            parts.append((d["Z"], d["y"]))
        print(f"extra {name}: {len(files)} recordings, {sum(int(p[1].sum()) for p in parts)} seizure windows so far", flush=True)
    return parts


CONFIRM_DELAY_S = 60  # the clinician confirms a seizure about a minute after it ends
CONFIRM_MIX = int(os.environ.get("CONFIRM_MIX", 0))  # rows of OTHER patients' data added as background


def background_rows(fset, other_files, case_of, sz, n):
    """A fixed sample of other patients' training rows (seizures and not), so a personal model
    trained on only one confirmed seizure still knows what ordinary non-seizure EEG looks like."""
    if n <= 0:
        return []
    rng = np.random.default_rng(1)
    Zs, ys = zip(*[train_rows(fset, case_of[f], f, sz) for f in other_files])
    Z, y = np.concatenate(Zs), np.concatenate(ys)
    pick = rng.choice(len(y), min(n, len(y)), replace=False)
    return [(Z[pick], y[pick])]


def confirm_protocol(fset, own, case_of, sz, g, background=(), gfset=None):
    """Clinician-confirmed personalisation, simulated honestly (past-only).
    The patient starts on the general model g. After each seizure ends (+ confirmation delay),
    a personal model is retrained on ALL of this patient's data up to that moment and used
    until the next confirmation. Returns per-file (file, tc, general_scores, confirm_scores,
    stage) where stage = number of confirmed seizures in force at each window."""
    rng = np.random.default_rng(0)
    recs = []
    for f in own:
        X, tc, y = D.features(case_of[f], f, sz)
        recs.append({"f": f, "Z": transform(X, fset), "tc": tc, "y": y,
                     "sz": sz[sz.file == f].sort_values("start_s")})
    # confirmation points in patient time: (file index, seconds into that file)
    points = [(i, float(s.end_s) + CONFIRM_DELAY_S) for i, r in enumerate(recs) for s in r["sz"].itertuples()]
    out = []
    for r in recs:
        Zg = r["Z"] if not gfset or gfset == fset else transform(D.features(case_of[r["f"]], r["f"], sz)[0], gfset)
        r["gs"] = g.predict_proba(Zg)[:, 1]
        r["cs"], r["stage"] = r["gs"].copy(), np.zeros(len(r["tc"]), dtype=int)
    for k, (fi, t_conf) in enumerate(points, start=1):
        parts = []
        for j, r in enumerate(recs[:fi + 1]):
            m = r["tc"] < t_conf if j == fi else np.ones(len(r["tc"]), bool)
            idx = np.where(m)[0]
            pos, neg = idx[r["y"][idx] == 1], idx[r["y"][idx] == 0]
            neg = rng.choice(neg, min(NEG_PER_FILE, len(neg)), replace=False)
            keep = np.concatenate([pos, neg])
            parts.append((r["Z"][keep], r["y"][keep]))
        model = D.fit(parts + list(background))
        if model is None:
            continue
        nxt = points[k] if k < len(points) else (len(recs), 0.0)
        for j in range(fi, min(nxt[0], len(recs) - 1) + 1):
            r = recs[j]
            lo = t_conf if j == fi else -1.0
            hi = nxt[1] if j == nxt[0] else np.inf
            m = (r["tc"] >= lo) & (r["tc"] < hi)
            if m.any():
                r["cs"][m] = model.predict_proba(r["Z"][m])[:, 1]
                r["stage"][m] = k
    return [(r["f"], r["tc"], r["gs"], r["cs"], r["stage"]) for r in recs]


def compare_zone(rows, sz, th):
    """General vs personalised on IDENTICAL stretches: only time after the patient's first
    confirmation counts. Also splits the personalised result by confirmed-seizure count."""
    bucket = lambda k: "1" if k == 1 else ("2-3" if k <= 3 else "4+")
    new = lambda: {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
    res = {name: new() for name in ["general", "personalised"]}
    curve = {b: {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0} for b in ["1", "2-3", "4+"]}
    per_patient = {}
    for row in rows:
        pat, (f, tc, gs, cs, stage) = row[0], row[1:]
        pp = per_patient.setdefault(pat, {"general": new(), "personalised": new()})
        zone = stage > 0
        if not zone.any():
            continue
        truth = sz[sz.file == f]
        hours = zone.sum() / 3600
        from src.rules import FIRST_CONFIRM_THRESHOLD as FCT
        for name, s in [("general", gs), ("personalised", cs)]:
            pw = np.where(stage == 1, max(th, FCT), th) if (name == "personalised" and FCT) else None
            evs = D.events_from_scores(tc, s, th, pw)
            t_of = lambda t: stage[min(np.searchsorted(tc, t), len(tc) - 1)]
            res[name]["hours"] += hours
            pp[name]["hours"] += hours
            for t in truth.itertuples():
                if t_of(t.start_s) == 0:
                    continue  # seizure before personalisation: not part of the comparison
                over = [e for e in evs if e["start_s"] <= t.end_s and e["end_s"] >= t.start_s]
                hit = bool(over)
                for acc in (res[name], pp[name]):
                    acc["seizures"] += 1; acc["detected"] += hit
                    if hit:
                        acc["delays"].append(max(0.0, over[0].get("alert_s", over[0]["start_s"]) - t.start_s))
                if name == "personalised":
                    b = curve[bucket(t_of(t.start_s))]; b["seizures"] += 1; b["detected"] += hit
            for e in evs:
                if t_of(e["start_s"]) == 0:
                    continue
                false = not any(e["start_s"] <= t.end_s + 120 and e["end_s"] >= t.start_s - 60 for t in truth.itertuples())
                res[name]["false_alerts"] += false
                pp[name]["false_alerts"] += false
                if name == "personalised" and false:
                    curve[bucket(t_of(e["start_s"]))]["false_alerts"] += 1
        for k in np.unique(stage[zone]):
            curve[bucket(int(k))]["hours"] += (stage == k).sum() / 3600
    fmt = lambda d: {"caught": f"{d['detected']}/{d['seizures']}",
                     "sensitivity": round(d["detected"] / max(d["seizures"], 1), 3),
                     "false_alerts_per_24h": round(d["false_alerts"] / max(d["hours"], 1e-9) * 24, 2),
                     "median_delay_s": float(np.median(d["delays"])) if d.get("delays") else None,
                     "hours": round(d["hours"], 1)}
    return {"general_same_time": fmt(res["general"]), "personalised": fmt(res["personalised"]),
            "by_confirmed_seizures": {b: fmt(v) for b, v in curve.items()},
            "by_patient": {p: {"general": fmt(v["general"]), "personalised": fmt(v["personalised"])}
                           for p, v in per_patient.items() if v["general"]["hours"] > 0}}


def run(fset):
    files, sz = D.index()
    files = D.usable(files, sz)
    case_of = dict(zip(files.file, files.case))
    fit = lambda names, **kw: D.fit([train_rows(fset, case_of[n], n, sz) for n in names], **kw)
    scored = {m: [] for m in MODES + ["hybrid"]}
    t0, n_models = time.time(), 0
    extra = extra_rows(GENERAL_FSET or fset) if GENERAL_EXTRA else []
    for pat in files.patient.unique()[:int(os.environ.get("LIMIT_PATIENTS", 999))]:
        tp = time.time()
        import joblib  # the live engine reuses these leave-this-patient-out models
        gfs = GENERAL_FSET or fset
        gpath = D.RUNS / "models" / f"{gfs}{GENERAL_TAG}_{D.MODEL_TAG}_general_{pat}.joblib"
        if os.environ.get("REUSE_GENERAL") == "1" and gpath.exists():
            g = joblib.load(gpath)
        else:
            g = D.fit([train_rows(gfs, case_of[n], n, sz) for n in files[files.patient != pat].file] + extra,
                      max_neg=GENERAL_MAX_NEG)
            n_models += 1
            gpath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(g, gpath)
        own = list(files[files.patient == pat].file)
        if "confirm" in MODES and g is not None:
            bg = background_rows(fset, list(files[files.patient != pat].file), case_of, sz, CONFIRM_MIX)
            scored["confirm"] += [(pat,) + row for row in confirm_protocol(fset, own, case_of, sz, g, bg, gfs)]
            n_models += int(sz[sz.file.isin(own)].shape[0])
        for i, f in enumerate(own):
            X, tc, _ = D.features(case_of[f], f, sz)
            Z = transform(X, fset)
            Zg = Z if gfs == fset else transform(X, gfs)
            pg = g.predict_proba(Zg)[:, 1] if g is not None else None
            if "general" in MODES:
                scored["general"].append((f, tc, pg))
            if "personal" in MODES:
                p = fit([o for o in own if o != f]); n_models += 1
                if p is not None:
                    scored["personal"].append((f, tc, p.predict_proba(Z)[:, 1]))
            if "past" in MODES:
                pp = fit(own[:i]) if i else None; n_models += 1
                if pp is not None:
                    s = pp.predict_proba(Z)[:, 1]
                    scored["past"].append((f, tc, s))
                    scored["hybrid"].append((f, tc, s))
                elif pg is not None:
                    scored["hybrid"].append((f, tc, pg))
        print(f"[{fset}] patient {pat}: {len(own)} recordings, {time.time() - tp:.0f}s "
              f"(total {time.time() - t0:.0f}s)", flush=True)

    # Cache every per-second score, so thresholds / event logic / sensitivity / specificity can be
    # recomputed later in seconds without retraining (scripts/metrics_from_scores.py).
    sdir = D.RUNS / "scores" / f"{fset}{os.environ.get('OUT_TAG', '')}"
    sdir.mkdir(parents=True, exist_ok=True)
    for mode, rows in scored.items():
        for row in rows:
            if mode == "confirm":
                pat, f, tc, gs, cs, stage = row
                np.savez_compressed(sdir / f"confirm__{f}.npz", tc=tc, general=gs.astype(np.float32),
                                    personalised=cs.astype(np.float32), stage=stage.astype(np.int16),
                                    y=D.features(case_of[f], f, sz)[2], patient=pat)
            else:
                f, tc, s = row
                np.savez_compressed(sdir / f"{mode}__{f}.npz", tc=tc, s=np.asarray(s, np.float32),
                                    y=D.features(case_of[f], f, sz)[2], patient=D.patient(case_of[f]))
    out = {"_seconds": round(time.time() - t0), "_models_trained": n_models,
           "_n_features": int(transform(D.features(files.case.iloc[0], files.file.iloc[0], sz)[0][:700], fset).shape[1]),
           "_patients": int(files.patient.nunique()), "_recordings": len(files),
           "_seizures": int(sz[sz.file.isin(files.file)].shape[0])}
    for mode, rows in scored.items():
        if not rows:
            continue
        if mode == "confirm":
            out["confirm"] = {th: compare_zone(rows, sz, th) for th in THRESHOLDS}
            continue
        out[mode] = {}
        for th in THRESHOLDS:
            tot = {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
            for f, tc, s in rows:
                res = D.match(D.events_from_scores(tc, s, th), sz[sz.file == f], tc[-1] + 1)
                for k in tot:
                    tot[k] += res[k]
            out[mode][th] = {"caught": f"{tot['detected']}/{tot['seizures']}",
                             "sensitivity": round(tot["detected"] / max(tot["seizures"], 1), 3),
                             "false_alerts_per_24h": round(tot["false_alerts"] / max(tot["hours"], 1e-9) * 24, 2),
                             "median_delay_s": float(np.median(tot["delays"])) if tot["delays"] else None,
                             "hours": round(tot["hours"], 1)}
    (D.RUNS / f"exp_{fset}{os.environ.get('OUT_TAG', '')}.json").write_text(json.dumps(out, indent=2))
    return fset, out


def report(fset, out):
    print(f"\n######## {fset}: {out['_patients']} patients, {out['_recordings']} recordings, "
          f"{out['_seizures']} seizures, {out['_n_features']} features, "
          f"{out['_models_trained']} models, {out['_seconds']} s")
    for mode in ["general", "personal", "past", "hybrid"]:
        for th, v in out.get(mode, {}).items():
            print(f"  {mode:8s} th={th}: caught {v['caught']} ({v['sensitivity']:.0%}), "
                  f"{v['false_alerts_per_24h']} false/24h, delay {v['median_delay_s']}s")
    for th, c in out.get("confirm", {}).items():
        g, p = c["general_same_time"], c["personalised"]
        curve = ", ".join(f"after {b}: {v['caught']} / {v['false_alerts_per_24h']} false/24h"
                          for b, v in c["by_confirmed_seizures"].items() if v["caught"] != "0/0" or v["hours"])
        print(f"  confirm  th={th}: SAME TIME general {g['caught']} ({g['sensitivity']:.0%}) "
              f"{g['false_alerts_per_24h']} false/24h delay {g.get('median_delay_s')}s  ->  personalised "
              f"{p['caught']} ({p['sensitivity']:.0%}) {p['false_alerts_per_24h']} false/24h delay "
              f"{p.get('median_delay_s')}s over {p['hours']} h | {curve}")


if __name__ == "__main__":
    sets = sys.argv[1:] or ["v1", "v2", "v3"]
    if len(sets) == 1:
        report(*run(sets[0]))
    else:
        with Pool(len(sets)) as pool:
            for res in pool.imap_unordered(run, sets):
                report(*res)
