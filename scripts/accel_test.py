"""Benchmark the seizure model on the IBM Z on-chip AI accelerator (Telum NNPA) via IBM Snap ML.

What it does
  1. Builds the training set from cached CHB-MIT features (all seizure windows + ~30k random normal
     windows from every patient except the held-out one) and trains
       - sklearn HistGradientBoostingClassifier (our current model, same params as src/detector.fit)
       - Snap ML BoostingMachineClassifier and RandomForestClassifier (comparable settings)
  2. Scores the held-out patient: window-level ROC AUC, event-level caught / false alerts per 24 h
     (src.detector.events_from_scores + src.detector.match) at thresholds 0.5-0.9.
  3. Times predict_proba: 1 window, batch of 100, batch of 10,000 for sklearn CPU, Snap ML CPU
     (tree_format='compress_trees') and Snap ML accelerator (tree_format='zdnn_tensors').
  4. Proves (or disproves) accelerator use: an LD_PRELOAD shim counts calls to libzdnn's
     zdnn_matmul_op and records their status codes, with CPU-format and SNAPML_ZDNN_LIB-disabled controls.
  5. Writes runs/accel_results.json and runs/accel_results.txt.

Run (from the project root, inside the venv):
    .venv/bin/python scripts/accel_test.py                 # full run, holdout = patient with most seizures
    .venv/bin/python scripts/accel_test.py --holdout chb15
    .venv/bin/python scripts/accel_test.py --quick         # smoke test, a minute or two

Only creates new files: scripts/accel_test.py and runs/accel_* . Does not touch existing code.
"""
import argparse, json, os, pickle, platform, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MAX_NEG = 30000
NEG_PRESAMPLE = 0.10     # keep 10% of normal windows per file, then trim to MAX_NEG (uniform overall)
SEED = 0
TRAIN_THREADS = 2        # the VM has 3 shared CPUs
SHIM_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
static unsigned long calls = 0, nnpa_checks = 0, nonzero = 0;
static unsigned long long ns_total = 0;
static int first_status = -1, last_status = -1;
static bool last_installed = false;
/* libzdnn is loaded RTLD_LOCAL by python/ctypes, so RTLD_NEXT cannot see it: look it up by handle. */
static void *real_sym(const char *name) {
    void *h = dlopen("libzdnn.so.0", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libzdnn.so.0", RTLD_LAZY);
    void *p = h ? dlsym(h, name) : 0;
    if (!p) p = dlsym(RTLD_NEXT, name);
    return p;
}
static unsigned long long now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (unsigned long long)t.tv_sec*1000000000ull + t.tv_nsec; }
int zdnn_matmul_op(const void *a, const void *b, const void *c, int op, void *out) {
    static int (*real)(const void*,const void*,const void*,int,void*) = 0;
    if (!real) real = real_sym("zdnn_matmul_op");
    unsigned long long t0 = now_ns();
    int st = real(a,b,c,op,out);
    unsigned long long dt = now_ns() - t0;
    __atomic_fetch_add(&calls,1,__ATOMIC_RELAXED);
    __atomic_fetch_add(&ns_total,dt,__ATOMIC_RELAXED);
    if (st != 0) __atomic_fetch_add(&nonzero,1,__ATOMIC_RELAXED);
    if (first_status == -1) first_status = st;
    last_status = st;
    return st;
}
bool zdnn_is_nnpa_installed(void) {
    static bool (*real)(void) = 0;
    if (!real) real = real_sym("zdnn_is_nnpa_installed");
    bool r = real();
    __atomic_fetch_add(&nnpa_checks,1,__ATOMIC_RELAXED);
    last_installed = r;
    return r;
}
__attribute__((destructor)) static void report(void) {
    const char *p = getenv("ACCEL_SHIM_OUT");
    FILE *f = p ? fopen(p, "w") : stderr;
    if (!f) return;
    fprintf(f, "{\"zdnn_matmul_op_calls\": %lu, \"nonzero_status_calls\": %lu, \"first_status\": %d, \"last_status\": %d, \"ms_inside_matmul\": %.3f, \"is_nnpa_installed_calls\": %lu, \"is_nnpa_installed_last\": %s}\n",
        calls, nonzero, first_status, last_status, ns_total/1e6, nnpa_checks, last_installed?"true":"false");
    if (p) fclose(f);
}
'''


# ----------------------------------------------------------------------------- small helpers
def log(*a):
    print(*a, flush=True)


def stats(ts):
    a = np.asarray(ts) * 1000.0
    return {"median_ms": round(float(np.median(a)), 4), "p95_ms": round(float(np.percentile(a, 95)), 4),
            "min_ms": round(float(a.min()), 4), "n": int(len(a))}


def run_round_robin(backends, X_for, reps, warm=2):
    """backends: {name: fn(X)}. X_for(i) -> input of rep i. Interleaves backends so drifting load on
    the shared VM hits every backend equally. Returns {name: [seconds]}."""
    for _ in range(warm):
        for fn in backends.values():
            fn(X_for(0))
    out = {k: [] for k in backends}
    for i in range(reps):
        x = X_for(i)
        for k, fn in backends.items():
            t = time.perf_counter()
            fn(x)
            out[k].append(time.perf_counter() - t)
    return out


# ----------------------------------------------------------------------------- child (proof) mode
def child(fmt, reps, batch):
    """Runs inside a subprocess, possibly under the LD_PRELOAD shim / with SNAPML_ZDNN_LIB overridden."""
    import ctypes
    caught = []
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import snapml  # noqa: F401  (import triggers the zdnn / cpu library selection)
        caught.extend(f"{x.category.__name__}: {x.message}" for x in w)
    uniq = {}
    for c in caught:
        uniq[c] = uniq.get(c, 0) + 1
    res = {"requested_tree_format": fmt, "batch": batch, "reps": reps,
           "warnings": [f"{k} (x{v})" for k, v in uniq.items()]}
    try:
        X = np.load(RUNS / "accel_testX.npy")
        m = pickle.loads((RUNS / "accel_bm.pkl").read_bytes())
        t0 = time.perf_counter()
        if fmt != "default":
            m.optimize_trees(fmt)
        res["optimize_s"] = round(time.perf_counter() - t0, 4)
        res["optimized_tree_format_"] = getattr(m.booster_, "optimized_tree_format_", None)
        res["zdnn_snapml_library_loaded"] = "snapml.libsnapmllocal3_zdnn" in sys.modules
        m.set_params(n_jobs=1)
        X = np.ascontiguousarray(X[:batch])
        tw = time.perf_counter()
        m.predict_proba(X)
        total = time.perf_counter() - tw
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            m.predict_proba(X)
            ts.append(time.perf_counter() - t)
        total += sum(ts)
        res["predict_median_ms"] = round(float(np.median(ts)) * 1000, 4)
        res["total_predict_s"] = round(total, 4)
        res["predict_calls"] = reps + 1
    except BaseException as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    print("CHILD_JSON " + json.dumps(res), flush=True)


# ----------------------------------------------------------------------------- parent pieces
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def environment():
    import importlib.metadata as md
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    for pkg in ("snapml", "pandas", "scipy"):
        try:
            v[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            v[pkg] = "n/a"
    v["libzdnn_deb"] = sh("dpkg-query -W -f='${Package} ${Version}' libzdnn0")
    v["kernel"] = sh("uname -r")
    v["machine_type"] = sh("grep -m1 -o 'machine = [0-9]*' /proc/cpuinfo")
    v["cpu_flags_nnpa"] = "nnpa" in sh("grep -m1 features /proc/cpuinfo").split()
    v["hypervisor"] = sh("lscpu | grep -i '^Hypervisor:' | sed 's/Hypervisor: *//'")
    v["cpus"] = os.cpu_count()
    v["loadavg_start"] = [round(x, 2) for x in os.getloadavg()]
    return v


def nnpa_probe():
    import ctypes
    out = {}
    try:
        lib = ctypes.CDLL("libzdnn.so.0")
        lib.zdnn_is_nnpa_installed.restype = ctypes.c_bool
        out["zdnn_is_nnpa_installed"] = bool(lib.zdnn_is_nnpa_installed())
    except Exception as e:  # noqa: BLE001
        out["zdnn_is_nnpa_installed"] = f"error: {e}"
    pe = sh("sudo -n perf stat -x, -e pai_ext/NNPA_ALL/ -- true 2>&1 | head -3")
    out["perf_pai_ext_NNPA_ALL"] = pe.replace("\n", " | ")[:200]
    out["pai_ext_counters_usable"] = ("not supported" not in pe) and ("NNPA_ALL" in pe or pe.strip()[:1].isdigit())
    return out


def load_data(holdout, quick):
    from src import detector as det
    files, sz = det.index()
    files = files[files.file.map(lambda f: (det.FEAT_DIR / f"{f}.npz").exists())]  # cached only
    sz = sz[sz.file.isin(files.file)]
    per = sz.case.map(det.patient).value_counts()
    if holdout == "auto":
        holdout = str(per.index[0])
    log(f"cached recordings: {len(files)}; seizures per patient (top): {per.head(6).to_dict()}; holdout={holdout}")
    if quick:
        keep = files[files.case.map(det.patient) == holdout].file.tolist()[:6]
        others = files[files.case.map(det.patient) != holdout]
        files = files[files.file.isin(keep) | files.file.isin(others.file.iloc[::6])]
    rng = np.random.default_rng(SEED)
    pos, neg, test = [], [], []
    for r in files.itertuples():
        X, tc, y = det.features(r.case, r.file, sz)
        if det.patient(r.case) == holdout:
            test.append((r.case, r.file, X, tc, y))
            continue
        pos.append(X[y == 1])
        ni = np.where(y == 0)[0]
        neg.append(X[ni[rng.random(len(ni)) < NEG_PRESAMPLE]])
    Xp, Xn = np.concatenate(pos), np.concatenate(neg)
    if len(Xn) > MAX_NEG:
        Xn = Xn[rng.choice(len(Xn), MAX_NEG, replace=False)]
    Xtr = np.concatenate([Xp, Xn]).astype(np.float32)
    ytr = np.concatenate([np.ones(len(Xp), np.int32), np.zeros(len(Xn), np.int32)])
    p = rng.permutation(len(ytr))
    Xtr, ytr = np.ascontiguousarray(Xtr[p]), ytr[p]
    info = {"holdout": holdout, "train_windows": int(len(ytr)), "train_seizure_windows": int(ytr.sum()),
            "train_normal_windows": int((ytr == 0).sum()), "n_features": int(Xtr.shape[1]),
            "test_files": len(test), "test_windows": int(sum(len(t[3]) for t in test)),
            "test_seizure_windows": int(sum(int(t[4].sum()) for t in test)),
            "test_seizures": int(sum(len(sz[sz.file == t[1]]) for t in test)),
            "test_hours": round(sum(t[3][-1] + 1 for t in test) / 3600, 1)}
    log("data:", info)
    return Xtr, ytr, test, sz, info


def fit_models(Xtr, ytr, quick):
    import snapml
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    models, train = {}, {}
    n_round = 30 if quick else 200
    n_tree = 30 if quick else 200

    with threadpool_limits(limits=TRAIN_THREADS):
        t = time.perf_counter()
        hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                             random_state=0).fit(Xtr, ytr)
        train["sklearn_hgb"] = {"fit_s": round(time.perf_counter() - t, 2), "threads": TRAIN_THREADS,
                                "params": "max_iter=200, lr=0.1, class_weight=balanced (src/detector.fit)",
                                "iterations_used": int(hgb.n_iter_)}
    models["sklearn_hgb"] = hgb
    log("trained sklearn_hgb", train["sklearn_hgb"])

    t = time.perf_counter()
    bm = snapml.BoostingMachineClassifier(num_round=n_round, learning_rate=0.1, class_weight="balanced",
                                          random_state=0, n_jobs=TRAIN_THREADS).fit(Xtr, ytr)
    train["snapml_bm"] = {"fit_s": round(time.perf_counter() - t, 2), "threads": TRAIN_THREADS,
                          "params": f"num_round={n_round}, lr=0.1, class_weight=balanced, depth range 1-5 (default)"}
    log("trained snapml_bm", train["snapml_bm"])
    models["snapml_bm"] = bm

    w = np.where(ytr == 1, len(ytr) / (2.0 * ytr.sum()), len(ytr) / (2.0 * (ytr == 0).sum())).astype(np.float32)
    t = time.perf_counter()
    rf = snapml.RandomForestClassifier(n_estimators=n_tree, max_depth=10, use_histograms=True,
                                       random_state=0, n_jobs=TRAIN_THREADS).fit(Xtr, ytr, sample_weight=w)
    train["snapml_rf"] = {"fit_s": round(time.perf_counter() - t, 2), "threads": TRAIN_THREADS,
                          "params": f"n_estimators={n_tree}, max_depth=10, histograms, balanced sample_weight"}
    log("trained snapml_rf", train["snapml_rf"])
    models["snapml_rf"] = rf
    return models, train


def variants(models):
    """sklearn model + two optimised copies of each Snap ML model (a model can only be optimised once)."""
    out, opt = {}, {}
    out["sklearn_hgb_cpu"] = models["sklearn_hgb"]
    for key in ("snapml_bm", "snapml_rf"):
        raw = pickle.dumps(models[key])
        for tag, fmt in (("cpu", "compress_trees"), ("accel", "zdnn_tensors")):
            m = pickle.loads(raw)
            t = time.perf_counter()
            m.optimize_trees(fmt)
            opt[f"{key}_{tag}"] = {"tree_format_requested": fmt, "optimize_s": round(time.perf_counter() - t, 3),
                                   "optimized_tree_format_": (m.booster_ if key == "snapml_bm" else m).optimized_tree_format_}
            out[f"{key}_{tag}"] = m
        if key == "snapml_bm":
            out["snapml_bm_default"] = pickle.loads(raw)  # straight out of fit(): no optimize_trees call
            (RUNS / "accel_bm.pkl").write_bytes(raw)
    return out, opt


def predict_fn(name, m, threads):
    if not name.startswith("sklearn"):
        m.set_params(n_jobs=threads)  # per-call n_jobs is deprecated in Snap ML
    return lambda X: m.predict_proba(X)[:, 1]


def score_chunked(name, m, X, threads, chunk=10000):
    f = predict_fn(name, m, threads)
    return np.concatenate([f(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def quality(variant_models, test, sz):
    from sklearn.metrics import roc_auc_score
    from src import detector as det
    Xall = np.ascontiguousarray(np.concatenate([t[2] for t in test]).astype(np.float32))
    yall = np.concatenate([t[4] for t in test])
    cuts = np.cumsum([len(t[3]) for t in test])[:-1]
    res, scores = {}, {}
    for name, m in variant_models.items():
        if name == "snapml_bm_default":
            continue
        t = time.perf_counter()
        s = score_chunked(name, m, Xall, 2)
        scores[name] = s
        r = {"auc": round(float(roc_auc_score(yall, s)), 4) if 0 < yall.sum() < len(yall) else None,
             "score_all_windows_s": round(time.perf_counter() - t, 2), "events": {}}
        parts = np.split(s, cuts)
        for th in THRESHOLDS:
            tot = {"seizures": 0, "detected": 0, "false_alerts": 0, "hours": 0.0, "delays": []}
            for (case, fname, X, tc, y), sp in zip(test, parts):
                e = det.match(det.events_from_scores(tc, sp, th), sz[sz.file == fname], tc[-1] + 1)
                for k in tot:
                    tot[k] += e[k]
            r["events"][str(th)] = {"caught": tot["detected"], "seizures": tot["seizures"],
                                    "false_alerts": tot["false_alerts"], "hours": round(tot["hours"], 1),
                                    "false_alerts_per_24h": round(tot["false_alerts"] / tot["hours"] * 24, 2),
                                    "median_delay_s": float(np.median(tot["delays"])) if tot["delays"] else None}
        res[name] = r
        log(f"quality {name}: AUC={r['auc']}", {k: f"{v['caught']}/{v['seizures']} & {v['false_alerts_per_24h']}/24h"
                                                 for k, v in r["events"].items()})
    agree = {}
    for key in ("snapml_bm", "snapml_rf"):
        if f"{key}_cpu" in scores:
            d = np.abs(scores[f"{key}_cpu"] - scores[f"{key}_accel"])
            agree[key] = {"max_abs_prob_diff_accel_vs_cpu": float(d.max()), "mean_abs": float(d.mean()),
                          "n_windows": int(len(d))}
    return res, agree, Xall


def speed(variant_models, Xall, quick):
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(SEED + 1)
    idx10k = rng.choice(len(Xall), 10000, replace=len(Xall) < 10000)
    X10k = np.ascontiguousarray(Xall[idx10k])
    np.save(RUNS / "accel_testX.npy", X10k)
    singles = [np.ascontiguousarray(Xall[i:i + 1]) for i in rng.choice(len(Xall), 64, replace=False)]
    batch100 = [np.ascontiguousarray(Xall[rng.choice(len(Xall), 100, replace=False)]) for _ in range(8)]
    reps = {"single": 40 if quick else 300, "b100": 8 if quick else 50, "b10k": 2 if quick else 7}
    out = {"config": {"reps": reps, "note": "median of repeated calls; backends interleaved each rep; "
                                              "VM CPUs are shared with other jobs"}, "results": {}}
    for threads in (1, 2):
        with threadpool_limits(limits=threads):
            fns = {n: predict_fn(n, m, threads) for n, m in variant_models.items()}
            plan = [("b10k", lambda i: X10k, reps["b10k"])]
            if threads == 1:
                plan = [("single", lambda i: singles[i % len(singles)], reps["single"]),
                        ("b100", lambda i: batch100[i % len(batch100)], reps["b100"])] + plan
            for label, xf, n in plan:
                t = run_round_robin(fns, xf, n, warm=1 if label == "b10k" else 2)
                size = {"single": 1, "b100": 100, "b10k": 10000}[label]
                for name, ts in t.items():
                    s = stats(ts)
                    s["windows_per_s"] = round(size / (s["median_ms"] / 1000.0), 1)
                    out["results"].setdefault(name, {})[f"{label}_threads{threads}"] = s
                log(f"speed {label} threads={threads}:",
                    {k: out["results"][k][f"{label}_threads{threads}"]["median_ms"] for k in out["results"]})
    out["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    return out


def sweep(Xtr, ytr, test, sz, Xall, quick):
    """Does the accelerator ever beat Snap ML's CPU engine? Same data, bigger/deeper boosting ensembles."""
    import snapml
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(SEED + 2)
    X10k = np.ascontiguousarray(Xall[rng.choice(len(Xall), 10000, replace=len(Xall) < 10000)])
    X100 = np.ascontiguousarray(X10k[:100])
    cfgs = [(30, 5), (30, 8)] if quick else [(200, 8), (200, 12), (500, 10)]
    rows = []
    for n_round, depth in cfgs:
        row = {"num_round": n_round, "max_depth": depth}
        try:
            t = time.perf_counter()
            m = snapml.BoostingMachineClassifier(num_round=n_round, max_depth=depth, learning_rate=0.1,
                                                 class_weight="balanced", random_state=0,
                                                 n_jobs=TRAIN_THREADS).fit(Xtr, ytr)
            row["fit_s"] = round(time.perf_counter() - t, 2)
            raw = pickle.dumps(m)
            cpu, acc = pickle.loads(raw), pickle.loads(raw)
            cpu.optimize_trees("compress_trees")
            acc.optimize_trees("zdnn_tensors")
            q, _, _ = quality({f"bm_{n_round}x{depth}_cpu": cpu}, test, sz)
            qq = next(iter(q.values()))
            row["auc"] = qq["auc"]
            row["events"] = qq["events"]
            fns = {"cpu": predict_fn("snapml_bm_cpu", cpu, 1), "accel": predict_fn("snapml_bm_accel", acc, 1)}
            with threadpool_limits(limits=1):
                for label, X, n in (("b100", X100, 8 if quick else 30), ("b10k", X10k, 2 if quick else 5)):
                    tt = run_round_robin(fns, lambda i, X=X: X, n, warm=1)
                    row[label] = {k: stats(v)["median_ms"] for k, v in tt.items()}
                    row[label]["accel_over_cpu"] = round(row[label]["accel"] / row[label]["cpu"], 2)
        except BaseException as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        log("sweep", row.get("num_round"), row.get("max_depth"), {k: v for k, v in row.items() if k != "events"})
        rows.append(row)
    return rows


def proof(quick):
    """Hard evidence: count libzdnn.zdnn_matmul_op calls (+ status codes) with an LD_PRELOAD shim."""
    src, so = RUNS / "accel_shim.c", RUNS / "accel_shim.so"
    src.write_text(SHIM_C)
    cc = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(so), str(src), "-ldl"],
                        capture_output=True, text=True)
    if cc.returncode != 0:
        return {"error": "could not compile shim: " + cc.stderr[-400:]}
    py = sys.executable
    reps = "2" if quick else "5"
    off = {"SNAPML_ZDNN_LIB": "/nonexistent/libzdnn.so"}
    cases = [
        ("A. accelerator ON: tree_format=zdnn_tensors, 10k-window batches", {}, "zdnn_tensors", 10000),
        ("A2. accelerator ON, 100-window batches", {}, "zdnn_tensors", 100),
        ("A3. accelerator ON, single-window calls", {}, "zdnn_tensors", 1),
        ("B. CPU control: tree_format=compress_trees, 10k batches", {}, "compress_trees", 10000),
        ("C. no optimize_trees() call (fit() default), 10k batches", {}, "default", 10000),
        ("D. tree_format=auto, 10k batches", {}, "auto", 10000),
        ("E. toggle OFF: SNAPML_ZDNN_LIB=/nonexistent, tree_format=auto, 10k batches", off, "auto", 10000),
        ("F. toggle OFF + force zdnn_tensors", off, "zdnn_tensors", 10000),
    ]
    rows = []
    for label, extra, fmt, batch in cases:
        outf = RUNS / "accel_shim_out.json"
        if outf.exists():
            outf.unlink()
        env = dict(os.environ, LD_PRELOAD=str(so), ACCEL_SHIM_OUT=str(outf), **extra)
        n_rep = reps if batch > 1 else ("20" if quick else "200")
        p = subprocess.run([py, str(Path(__file__).resolve()), "--child", fmt, "--child-reps", n_rep,
                            "--child-batch", str(batch)],
                           env=env, capture_output=True, text=True, timeout=900)
        row = {"case": label, "env": extra, "exit_code": p.returncode}
        for line in p.stdout.splitlines():
            if line.startswith("CHILD_JSON "):
                row.update(json.loads(line[len("CHILD_JSON "):]))
        if outf.exists():
            row["shim"] = json.loads(outf.read_text())
        if p.returncode != 0 or "error" in row:
            row["stderr_tail"] = p.stderr.strip()[-300:]
        rows.append(row)
        log("proof", label, "->", {k: row.get(k) for k in ("optimized_tree_format_", "predict_median_ms", "error")},
            row.get("shim"))
    if outf.exists():
        outf.unlink()
    return {"shim_source": str(src.relative_to(ROOT)), "cases": rows}


def write_text(res, path):
    L = []
    e, d = res["environment"], res["data"]
    L += ["ACCELERATOR TEST: seizure model on IBM Z Telum (Snap ML + zDNN)", "=" * 70, ""]
    L += ["ENVIRONMENT"] + [f"  {k}: {v}" for k, v in e.items()]
    L += [f"  nnpa probe: {res['nnpa_probe']}", ""]
    L += ["DATA"] + [f"  {k}: {v}" for k, v in d.items()] + [""]
    L += ["TRAINING (same training matrix for all models)"]
    for k, v in res["train"].items():
        L.append(f"  {k:<14} fit {v['fit_s']:>8.2f} s  threads={v['threads']}  {v['params']}"
                 + (f"  iterations_used={v['iterations_used']}" if 'iterations_used' in v else ""))
    L += ["", "ONE-TIME optimize_trees() COST"] + [
        f"  {k:<18} {v['tree_format_requested']:<15} -> {v['optimized_tree_format_']:<15} {v['optimize_s']} s"
        for k, v in res["optimize"].items()]
    L += ["", f"QUALITY on held-out patient {d['holdout']} (model never saw this patient)"]
    for name, r in res["quality"].items():
        L.append(f"  {name:<18} window AUC {r['auc']}")
        for th, v in r["events"].items():
            L.append(f"      th={th}: caught {v['caught']}/{v['seizures']}, {v['false_alerts']} false alerts "
                     f"({v['false_alerts_per_24h']}/24h over {v['hours']} h), median delay {v['median_delay_s']} s")
    L += ["", "ACCEL vs CPU score agreement (same trained model, two inference engines)"]
    for k, v in res["agreement"].items():
        L.append(f"  {k}: max |p_accel - p_cpu| = {v['max_abs_prob_diff_accel_vs_cpu']:.3g} "
                 f"(mean {v['mean_abs']:.3g}) over {v['n_windows']} windows")
    L += ["", "SPEED (median; ms per call; windows/s)  [threads=1 unless noted]"]
    hdr = f"  {'backend':<20}{'1 window ms':>12}{'b100 ms':>10}{'b100 win/s':>12}{'b10k ms':>10}{'b10k win/s':>12}{'b10k@2thr ms':>14}{'win/s':>10}"
    L.append(hdr)
    for name, r in res["speed"]["results"].items():
        g = lambda k, f: r[k][f] if k in r else float("nan")  # noqa: E731
        L.append(f"  {name:<20}{g('single_threads1','median_ms'):>12.3f}{g('b100_threads1','median_ms'):>10.2f}"
                 f"{g('b100_threads1','windows_per_s'):>12.0f}{g('b10k_threads1','median_ms'):>10.1f}"
                 f"{g('b10k_threads1','windows_per_s'):>12.0f}{g('b10k_threads2','median_ms'):>14.1f}"
                 f"{g('b10k_threads2','windows_per_s'):>10.0f}")
    L.append(f"  config: {res['speed']['config']}; loadavg at end {res['speed']['loadavg_end']}")
    L += ["", "ACCELERATOR PROOF (LD_PRELOAD shim around libzdnn.zdnn_matmul_op; one child process per case)"]
    pr = res["proof"]
    if "error" in pr:
        L.append("  " + pr["error"])
    else:
        for c in pr["cases"]:
            sh_ = c.get("shim", {})
            calls = sh_.get("zdnn_matmul_op_calls")
            L.append(f"  {c['case']}")
            L.append(f"      optimized_tree_format_={c.get('optimized_tree_format_')}  "
                     f"zdnn snapml lib loaded={c.get('zdnn_snapml_library_loaded')}  "
                     f"predict median={c.get('predict_median_ms')} ms (batch {c.get('batch')}, "
                     f"{c.get('predict_calls')} calls)  exit={c['exit_code']}")
            share = ""
            if calls and c.get("total_predict_s"):
                share = f"  = {100.0 * sh_['ms_inside_matmul'] / 1000.0 / c['total_predict_s']:.0f}% of predict wall time"
            inst = sh_.get("is_nnpa_installed_last") if sh_.get("is_nnpa_installed_calls") else "not called"
            L.append(f"      zdnn_matmul_op calls={calls}  non-OK status={sh_.get('nonzero_status_calls')}  "
                     f"first/last status={sh_.get('first_status')}/{sh_.get('last_status')}  "
                     f"ms inside matmul={sh_.get('ms_inside_matmul')}{share}  zdnn_is_nnpa_installed -> {inst}")
            if c.get("warnings"):
                L.append(f"      warnings: {c['warnings']}")
            if c.get("error"):
                L.append(f"      error: {c['error']}")
    if res.get("sweep"):
        L += ["", "SWEEP: does the accelerator ever beat Snap ML CPU? (BoostingMachine, threads=1, median ms per call)",
              f"  {'rounds x depth':<16}{'fit s':>8}{'AUC':>8}{'b100 cpu':>10}{'b100 accel':>12}{'accel/cpu':>11}"
              f"{'b10k cpu':>10}{'b10k accel':>12}{'accel/cpu':>11}"]
        for r in res["sweep"]:
            if "error" in r:
                L.append(f"  {r['num_round']} x {r['max_depth']:<10} error: {r['error']}")
                continue
            L.append(f"  {str(r['num_round']) + ' x ' + str(r['max_depth']):<16}{r['fit_s']:>8.1f}{r['auc']:>8}"
                     f"{r['b100']['cpu']:>10.3f}{r['b100']['accel']:>12.3f}{r['b100']['accel_over_cpu']:>11}"
                     f"{r['b10k']['cpu']:>10.1f}{r['b10k']['accel']:>12.1f}{r['b10k']['accel_over_cpu']:>11}")
    L += ["", "NOTES"] + [f"  - {n}" for n in res["notes"]]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--child", default=None)
    ap.add_argument("--child-reps", type=int, default=5)
    ap.add_argument("--child-batch", type=int, default=10000)
    ap.add_argument("--no-sweep", action="store_true", help="skip the bigger-ensemble scaling sweep")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.child_reps, a.child_batch)

    RUNS.mkdir(exist_ok=True)
    t_all = time.time()
    res = {"environment": environment(), "nnpa_probe": nnpa_probe()}
    log("env:", res["environment"], res["nnpa_probe"])
    Xtr, ytr, test, sz, info = load_data(a.holdout, a.quick)
    res["data"] = info
    models, res["train"] = fit_models(Xtr, ytr, a.quick)
    vm, res["optimize"] = variants(models)
    res["quality"], res["agreement"], Xall = quality(vm, test, sz)
    res["speed"] = speed(vm, Xall, a.quick)
    res["proof"] = proof(a.quick)
    if not a.no_sweep:
        res["sweep"] = sweep(Xtr, ytr, test, sz, Xall, a.quick)
    res["notes"] = [
        "Snap ML only runs its accelerator path for tree ensembles (BoostingMachine, RandomForest, DecisionTree, "
        "imported xgboost/lightgbm/onnx/pmml); sklearn HistGradientBoosting cannot be moved onto it directly.",
        "A model can be optimised once; tree_format is chosen with optimize_trees() / import_model(). A freshly "
        "fitted model that is never optimised does not use the accelerator.",
        "Hardware PAI counters (perf pai_ext NNPA_*) are exposed by the kernel but report <not supported> in this "
        "z/VM guest, so accelerator use is shown with the zdnn_matmul_op call-count shim instead.",
        "Probabilities are on each model's own scale (class_weight=balanced); event metrics at a fixed threshold "
        "are therefore not directly comparable across model families.",
        f"total wall time {time.time() - t_all:.0f} s; quick={a.quick}",
    ]
    out = RUNS / ("accel_results_quick" if a.quick else
                  "accel_results" if a.holdout == "auto" else f"accel_results_{a.holdout}")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str))
    write_text(res, out.with_suffix(".txt"))
    log("wrote", out.with_suffix(".json"), out.with_suffix(".txt"))


if __name__ == "__main__":
    main()
