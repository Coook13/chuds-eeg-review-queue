"""Predict: risk that a patient has ANOTHER seizure within the next 60 minutes, from the seizures seen so far.

Column 3 of the product (Diagnose -> Detect -> Predict -> Learn). Input is only the seizure history the
detector has produced for one continuous monitoring timeline; no EEG and no patient ID.

    from src.recurrence import risk
    risk([(1200, 1260), (5400, 5445)], now_s=5505)
    -> {"risk_60min": 0.31, "reason": "...", "features": {...}}

Features (all from the past only; see features()):
  n_1h, n_6h, n_24h     seizures that started in the last 1 / 6 / 24 h
  min_since_last        minutes since the last seizure ended (capped at 24 h; 24 h if none yet)
  last_dur_s            duration of the last seizure in seconds (0 if none yet)
  mean_isi_min          mean minutes between consecutive seizure starts so far (capped at 24 h; 24 h if < 2)
  hours_monitored       hours of recording so far

Model: runs/recurrence_model.joblib, written by scripts/eval_recurrence.py (trained on all CHB-MIT patients).
The logistic-regression model is stored as plain numbers, so scoring needs only numpy (no sklearn/snapml).
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "recurrence_model.joblib"

HORIZON_S = 3600        # predict a new seizure start within the next 60 min
SEEN_DELAY_S = 60       # a clinician sees a detected seizure ~60 s after it ends
CAP_MIN = 24 * 60.0     # cap for "minutes since last seizure" and "mean inter-seizure interval"
FEATURES = ["n_1h", "n_6h", "n_24h", "min_since_last", "last_dur_s", "mean_isi_min", "hours_monitored"]

_MODEL = None


def features(events, now_s, monitored_s=None):
    """events: [(start_s, end_s)] on one continuous timeline; only seizures that ended by now_s are used.
    monitored_s: seconds of recording up to now_s (default: now_s, i.e. monitored since time 0)."""
    ev = sorted((float(s), float(e)) for s, e in events if e <= now_s)
    starts = np.array([s for s, _ in ev])
    f = {}
    for name, w in (("n_1h", 3600), ("n_6h", 6 * 3600), ("n_24h", 24 * 3600)):
        f[name] = float(np.sum(starts > now_s - w)) if len(ev) else 0.0
    if ev:
        s_last, e_last = ev[-1]
        f["min_since_last"] = min((now_s - e_last) / 60.0, CAP_MIN)
        f["last_dur_s"] = e_last - s_last
    else:
        f["min_since_last"], f["last_dur_s"] = CAP_MIN, 0.0
    f["mean_isi_min"] = min(float(np.mean(np.diff(starts))) / 60.0, CAP_MIN) if len(ev) >= 2 else CAP_MIN
    f["hours_monitored"] = (now_s if monitored_s is None else monitored_s) / 3600.0
    return f


def vector(f):
    return np.array([[f[k] for k in FEATURES]], dtype=np.float64)


def _load():
    global _MODEL
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def predict_logreg(params, X):
    """Same maths as the sklearn pipeline log1p -> standardise -> logistic regression, in numpy."""
    z = (np.log1p(np.asarray(X, dtype=np.float64)) - params["mean"]) / params["scale"]
    return 1.0 / (1.0 + np.exp(-(z @ params["coef"] + params["intercept"])))


def _plural(n, word):
    n = int(n)
    return f"{n} {word}" + ("" if n == 1 else "s")


def reason_text(f, p, base_rate):
    if f["n_24h"] == 0 and f["min_since_last"] >= CAP_MIN:
        hist = "No seizures in the last 24 hours"
    else:
        hist = (f"{_plural(f['n_1h'], 'seizure')} in the last hour, {int(f['n_6h'])} in the last 6 h, "
                f"{int(f['n_24h'])} in the last 24 h; the last one ended "
                + (f"{f['min_since_last']:.0f} min ago" if f["min_since_last"] < 120
                   else f"{f['min_since_last'] / 60:.1f} h ago"))
    ratio = p / base_rate if base_rate > 0 else float("nan")
    level = ("well above" if ratio >= 2 else "above" if ratio >= 1.2
             else "about the same as" if ratio > 0.8 else "below")
    return (f"{hist}. Estimated {p:.0%} chance of another seizure in the next hour, "
            f"{level} the {base_rate:.0%} average for a monitored hour in the training data.")


def risk(events_so_far, now_s, monitored_s=None):
    """events_so_far: [(start_s, end_s)] seizures detected so far on one continuous patient timeline (seconds).
    now_s: current time on that timeline. monitored_s: seconds actually recorded so far (default now_s).
    -> {"risk_60min": p, "reason": plain English, "features": the inputs used}"""
    m = _load()
    f = features(events_so_far, now_s, monitored_s)
    X = vector(f)
    if m["kind"] == "logreg":
        p = float(predict_logreg(m["params"], X)[0])
    else:   # Snap ML tree (needs snapml, i.e. the LinuxONE VM)
        p = float(m["model"].predict_proba(X.astype(np.float32))[0, 1])
    return {"risk_60min": round(p, 4), "reason": reason_text(f, p, m["base_rate"]), "features": f}
