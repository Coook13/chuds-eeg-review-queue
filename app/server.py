"""Flask server for the EEG review queue.

    python app/server.py          # fake data (app/mock_engine.py)
    python app/server.py --real   # real engine (src/engine.py)
"""
import argparse
import logging
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HOST = "IBM LinuxONE (s390x)" if platform.machine() == "s390x" else f"this computer ({platform.machine()})"
COUNTRY = {"GB": "Guinea-Bissau", "NG": "Nigeria"}


def results_numbers():
    """Accuracy figures quoted on the screens, parsed from the runs/ results files (a key is missing if its file is)."""
    out = {}
    try:
        t = (ROOT / "runs" / "diagnose_results.txt").read_text()
        cross = [float(x) for x in re.findall(r"(?:GB->NG|NG->GB)_\w+\s+AUC ([0-9.]+)", t)]
        out["diagnose"] = {"gb_auc": float(re.search(r"GB snap .*?AUC ([0-9.]+)", t).group(1)),
                           "ng_auc": float(re.search(r"NG snap .*?AUC ([0-9.]+)", t).group(1)),
                           "cross_min": min(cross), "cross_max": max(cross)}
    except Exception:
        pass
    try:
        t = (ROOT / "runs" / "recurrence_results.txt").read_text()
        m = re.search(r"pooled AUC\s+logreg ([0-9.]+).*?baseline\(6h>0\) ([0-9.]+)", t)
        out["predict"] = {"auc": float(m.group(1)), "baseline_auc": float(m.group(2))}
    except Exception:
        pass
    return out


class Intake:
    """Diagnose at intake: src/diagnose.predict_person on a short, fixed list of people, run live on this machine.

    The people come from the Guinea-Bissau / Nigeria resting-EEG data (data/africa). They were all in the
    final model's training data, so the screen shows the workflow; honest accuracy is the cross-validated AUC."""
    RULE = "In each country, the first 2 people with epilepsy and the first 2 without, by subject ID."

    def __init__(self):
        from src import diagnose as G
        self.G, self.lock = G, threading.Lock()
        p = G.people()
        p = p[p.files.map(len) > 0]
        self.people = {}
        for c in ("GB", "NG"):
            for label in (1, 0):
                for r in p[(p.country == c) & (p.label == label)].sort_values("subject").head(2).itertuples():
                    self.people[f"{c}-{r.subject}"] = r

    def list(self):
        return [{"id": k, "country": COUNTRY[r.country], "subject": int(r.subject), "recordings": len(r.files)}
                for k, r in self.people.items()]

    def _load_model(self):
        G = self.G
        if G._FINAL is not None:
            return
        import joblib
        import numpy as np
        f = joblib.load(G.MODEL_PATH)
        f["_path"] = str(G.MODEL_PATH)
        if "med_epi" not in f:
            # ponytail: runs/diagnose_model.joblib was saved before top_reasons needed med_epi/med_ctl;
            # rebuilt here exactly as scripts/eval_diagnose.py does, from the same 314-person feature table.
            # Drop this once the model file is re-saved.
            d = np.load(G.FEAT_CACHE, allow_pickle=True)
            X, y = d["X"], d["meta_label"].astype(int)
            if not np.allclose(X.mean(0), f["mu"]):
                raise RuntimeError("feature cache does not match the saved model")
            f["med_epi"], f["med_ctl"] = np.median(X[y == 1], 0), np.median(X[y == 0], 0)
        G._FINAL = f

    def run(self, pid):
        r = self.people[pid]                              # KeyError -> 404
        with self.lock:                                   # predict_person keeps the model in a module global
            self._load_model()
            tic = time.perf_counter()
            out = self.G.predict_person(r.files)
            secs = time.perf_counter() - tic
        return {"id": pid, "country": COUNTRY[r.country], "subject": int(r.subject), **out,
                "dataset_label": "epilepsy" if r.label else "control",
                "compute_s": round(secs, 2), "host": HOST}


def bed_risk(engine, bed):
    """Next-hour seizure risk for one bed from src/recurrence.risk, using the seizures detected so far."""
    from src.recurrence import risk
    # ponytail: reads the real engine's event tracker and replay position directly; no Engine method for it yet
    with engine.lock:
        b = engine._find(bed)
        events = list(b["tracker"].events)
        rejected = {a for a, _ in b.get("rejected", [])}
        now_s = b["t"]                                    # seconds into this recording
    if events and events[-1]["ongoing"]:
        return {"p": None, "pending": True, "reason": "Updates when the current suspected seizure ends."}
    # alerts the clinician marked "not a seizure" are left out; the replay starts before the file's first
    # annotated seizure, so seconds-into-recording is the monitoring time the model expects
    ended = [(e["start_s"], e["end_s"]) for e in events if e["start_s"] not in rejected]
    r = risk(ended, now_s)
    return {"p": r["risk_60min"], "pending": False, "reason": r["reason"]}


def _json_default(o):
    """Let numpy scalars/arrays from the real engine through jsonify."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _sh(cmd):
    import subprocess
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"({e.__class__.__name__})"


def _json(path):
    import json
    try:
        return json.loads((ROOT / "runs" / path).read_text())
    except Exception:
        return None


def system_page(engine):
    """IBM Z system view for technical judges: the machine, the AI on it and what it measured.
    Every machine fact is read live; every number comes from the engine or a runs/ results file."""
    import html, json, statistics, urllib.request
    from importlib.metadata import version
    e = html.escape
    m = engine.metrics()
    beds = engine.get_beds()
    try:
        snap = "snapml " + version("snapml")
    except Exception:
        snap = "not installed"
    try:
        gran = ", ".join(x["name"] for x in json.load(urllib.request.urlopen("http://127.0.0.1:8081/v1/models", timeout=2))["models"])
    except Exception:
        gran = "not reachable"
    machine = [
        ("Chip architecture", _sh("uname -m")),
        ("Manufacturer / machine type", " ".join(_sh("grep -E '^(Manufacturer|Type):' /proc/sysinfo").split())),
        ("On-chip AI accelerator (nnpa)", "present" if " nnpa" in _sh("grep -m1 '^features' /proc/cpuinfo") else "not found"),
        ("IBM zDNN library", _sh("dpkg -l libzdnn0 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f2,3") or "not installed"),
        ("IBM Snap ML", snap),
        ("CPUs / memory", _sh("nproc") + " CPUs, " + _sh("free -h | awk '/Mem/ {print $2}'") + " RAM"),
        ("Operating system", _sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'")),
    ]
    live = [
        ("Seizure model scoring", m.get("scoring", "CPU")),
        ("Patients streaming now", m.get("beds")),
        ("Processing per 1-second window (live, median)", f"{m.get('processing_ms_per_window')} ms"),
        ("Personal models retrained on this machine", m.get("personalised_beds")),
        ("Replay speed", f"{m.get('replay_speed')}x"),
        ("Handover notes written by", f"IBM Granite on this machine: {gran}"),
    ]
    measured = []
    sc = _json("scale_test_snap.json")
    for r in (sc or {}).get("results", []):
        measured.append((f"Scale test: {r['beds']} beds at once",
                         f"{r['batched_ms_per_window']} ms per window, {r['batched_ms_per_step_all_beds']} ms per second of EEG for all beds "
                         f"(p95 {r['batched_p95_ms_per_step']} ms); real time holds up to about {r['max_beds_real_time_batched_est']} beds"))
    rt = _json("retrain_timing_snap.json")
    if rt and rt.get("runs"):
        secs = [r["seconds"] for r in rt["runs"]]
        measured.append(("Retrain one patient's personal model", f"median {statistics.median(secs):.2f} s, max {max(secs):.2f} s over {len(secs)} patients"))
    for n in (1, 100):
        z = _json(f"zdnn_step_{n}.json")
        if z:
            measured.append((f"AI accelerator calls in one live step ({n} bed{'s' if n > 1 else ''})",
                             f"{z['zdnn_matmul_op_calls']} zDNN matmul calls, {z['nonzero_status_calls']} errors, {z['ms_inside_matmul']} ms on the accelerator"))
    res = results_numbers()
    if "diagnose" in res:
        d = res["diagnose"]
        measured.append(("Diagnose: epilepsy screen from 5 min of resting EEG (cross-validated, each person scored by a model that never saw them)",
                         f"AUC {d['gb_auc']:.3f} Guinea-Bissau, {d['ng_auc']:.3f} Nigeria (within country); "
                         f"{d['cross_min']:.2f}-{d['cross_max']:.2f} when trained in one country and tested in the other"))
    if "predict" in res:
        p = res["predict"]
        measured.append(("Predict: a seizure in the next hour (leave-one-patient-out, CHB-MIT)",
                         f"pooled AUC {p['auc']:.3f}, vs {p['baseline_auc']:.3f} for the simple rule 'a seizure in the last 6 h'"))
    rows = lambda items: "".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in items)
    bed_rows = "".join(f"<tr><td>{b['bed']}</td><td>{e(b['patient_id'])}</td><td>{e(b['model'])}</td>"
                       f"<td>{b['score_now']:.2f}</td><td>{e(b['status'])}</td></tr>" for b in beds)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>IBM Z system view</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f7f8fa;color:#1d2530;margin:0}}
header{{display:flex;align-items:center;gap:12px;padding:12px 20px;background:#fff;border-bottom:1px solid #e3e7ec}}
.brand{{font-weight:650}} nav a{{font-size:12.5px;color:#5c6672;text-decoration:none;padding:4px 10px;border-radius:6px;border:1px solid transparent}}
nav a.on{{color:#2b5c8a;border-color:#a9c3dc;background:#eef4fa}} main{{max-width:1180px;margin:18px auto;padding:0 20px;display:grid;grid-template-columns:1fr 1fr;gap:16px}}
section{{background:#fff;border:1px solid #e3e7ec;border-radius:10px;padding:14px 16px}} section.wide{{grid-column:1/-1}}
h2{{margin:0 0 10px;font-size:12.5px;letter-spacing:.04em;text-transform:uppercase;color:#5c6672}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 4px;border-bottom:1px solid #eef0f3;vertical-align:top}}
th{{font-weight:600;width:42%}} .note{{font-size:12px;color:#5c6672;margin:8px 0 0}}</style></head><body>
<header><span class="brand">EEG Review Queue</span><nav><a href="/intake">Intake</a><a href="/">Clinician view</a><a class="on" href="/system">IBM Z system view</a></nav></header>
<main>
<section><h2>The machine (read live from this server)</h2><table>{rows(machine)}</table></section>
<section><h2>AI running on it now</h2><table>{rows(live)}</table></section>
<section class="wide"><h2>Measured on this mainframe</h2><table>{rows(measured) or '<tr><td>No results files yet.</td></tr>'}</table>
<p class="note">On the AI accelerator and Snap ML on CPU the scores are identical (difference under 1e-15). For this model size the accelerator is slower per window than Snap ML on CPU, and both stay far inside the real-time budget.</p></section>
<section class="wide"><h2>Beds and the model each one uses</h2><table><tr><th>Bed</th><th>Patient</th><th>Model</th><th>Activity now</th><th>Status</th></tr>{bed_rows}</table></section>
</main></body></html>"""


def create_app(engine, intake=None):
    """Build the Flask app around any object with the Engine interface (CONTRACT.md).
    intake: optional object with list() and run(id) for the Intake (diagnose) screen.

    The caller is responsible for engine.start()."""
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False
    app.json.default = _json_default

    @app.errorhandler(KeyError)
    @app.errorhandler(IndexError)
    @app.errorhandler(StopIteration)      # the real engine looks beds up with next(...)
    def not_found(e):
        return jsonify(error=str(e.args[0]) if e.args else "unknown bed"), 404

    @app.errorhandler(ValueError)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.after_request
    def no_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    @app.get("/intake")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/system")
    def system():
        return system_page(engine)

    @app.get("/api/capabilities")
    def capabilities():
        return jsonify(reject=hasattr(engine, "reject"), intake=intake is not None, host=HOST,
                       results=results_numbers())

    @app.get("/api/beds")
    def beds():
        out = engine.get_beds()
        if hasattr(engine, "_find") and hasattr(engine, "lock"):      # real engine; the mock fills "risk" itself
            for d in out:
                try:
                    d["risk"] = bed_risk(engine, d["bed"])
                except Exception as e:                                  # never let the risk chip break the queue
                    d["risk"] = {"p": None, "pending": False, "reason": f"Risk not available ({e.__class__.__name__})"}
        return jsonify(out)

    @app.get("/api/intake")
    def intake_list():
        if intake is None:
            return jsonify(error="intake screen not available"), 501
        return jsonify(rule=intake.RULE, people=intake.list())

    @app.post("/api/diagnose/<pid>")
    def diagnose(pid):
        """Runs the epilepsy screen on one listed person's recording, live (about 0.2 s on LinuxONE)."""
        if intake is None:
            return jsonify(error="intake screen not available"), 501
        try:
            return jsonify(intake.run(pid))
        except (KeyError, StopIteration):
            return jsonify(error=f"no such person: {pid}"), 404
        except Exception as e:
            logging.getLogger(__name__).exception("diagnose %s failed", pid)
            return jsonify(error=f"screen failed: {e.__class__.__name__}: {e}"), 500

    @app.get("/api/evidence/<int:bed>")
    def evidence(bed):
        ev = engine.get_evidence(bed)
        # ponytail: reads the real engine's annotations directly; the mock has none, so it's skipped there
        truth = getattr(engine, "truth", None)
        if truth is not None:
            t0 = ev["t0_s"]
            t1 = t0 + len(ev["data"][0]) / ev["fs"] if ev["data"] else t0
            fname = engine._find(bed)["file"]
            rows = truth[(truth.file == fname) & (truth.end_s >= t0) & (truth.start_s <= t1)]
            ev["truth_spans"] = [[float(a), float(z)] for a, z in zip(rows.start_s, rows.end_s)]
        return jsonify(ev)

    @app.post("/api/ack/<int:bed>")
    def ack(bed):
        who = str((request.get_json(silent=True) or {}).get("who", "")).strip()[:80]
        if not who:
            return jsonify(error='body must be JSON like {"who": "Dr Name"}'), 400
        return jsonify(engine.ack(bed, who))

    @app.post("/api/confirm/<int:bed>")
    def confirm(bed):
        """Retrains the patient's personal model; can take a few seconds, other requests keep being served."""
        who = str((request.get_json(silent=True) or {}).get("who", "")).strip()[:80]
        if not who:
            return jsonify(error='body must be JSON like {"who": "Dr Name"}'), 400
        return jsonify(engine.confirm(bed, who))

    @app.post("/api/reject/<int:bed>")
    def reject(bed):
        """Doctor marks the latest alert as a false alarm; the engine learns it as a negative example."""
        who = str((request.get_json(silent=True) or {}).get("who", "")).strip()[:80]
        if not who:
            return jsonify(error='body must be JSON like {"who": "Dr Name"}'), 400
        if not hasattr(engine, "reject"):
            return jsonify(error="this engine has no reject"), 501
        return jsonify(engine.reject(bed, who))

    @app.get("/api/handover/<int:bed>")
    def handover(bed):
        return jsonify(bed=bed, note=engine.handover(bed))

    @app.get("/api/metrics")
    def metrics():
        return jsonify(engine.metrics())

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use src/engine.py instead of the mock")
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    if args.real:
        sys.path.insert(0, str(ROOT))
        from src.engine import Engine  # lazy: src/engine.py may not exist yet
        engine = Engine()
        try:
            intake = Intake()
        except Exception as e:          # e.g. data/africa not downloaded on this machine
            print(f"Intake screen off: {e.__class__.__name__}: {e}")
            intake = None
    else:
        from mock_engine import MockEngine, MockIntake
        engine, intake = MockEngine(), MockIntake()
    engine.start()

    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no line per 1 s poll
    port = int(os.environ.get("PORT", 8050))
    print(f"EEG Review Queue ({'real' if args.real else 'mock'} engine) on http://localhost:{port}")
    create_app(engine, intake).run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
