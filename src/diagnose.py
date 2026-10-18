"""Diagnosis model: "does this person have epilepsy?" from ~5 minutes of resting EEG at intake.

Data: Zenodo 1252141 (Emotiv EPOC, 14 channels, 128 Hz; Guinea-Bissau and Nigeria; CC BY 4.0),
downloaded by scripts/download_africa.py into data/africa/. Evaluation: scripts/eval_diagnose.py.

Per person:
  1. the 14 EEG channels of every usable recording of that person (in microvolts)
  2. band-pass 1-45 Hz (4th-order Butterworth, zero phase) + 50 Hz notch
  3. 4 s windows with 50% overlap; windows with gross artefacts are dropped (see clean_windows)
  4. src.features.windows_features on each window (13 features x 14 channels)
  5. median and IQR of every feature across the person's windows -> one row of 364 numbers

Eyes open / closed: we use the WHOLE recording. In Guinea-Bissau all 51 people with epilepsy did
eyes-closed first while 13 of 46 controls did eyes-open first, and in Nigeria only the first condition
is recorded, with many free-text deviations ("eyes closed at 2:40", "did not close his eyes").
The eyes-closed part is therefore not reliably marked. Median/IQR over windows do not depend on the
order of the windows, so the order of eyes-open/closed cannot leak into the features.

App usage (on the VM):
    from src.diagnose import predict_person
    predict_person("data/africa/nigeria/EEGs_Nigeria/signal-6-1.csv.gz")
    -> {"epilepsy_probability": 0.31, "top_reasons": ["...", "...", "..."], ...}
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from src.features import BANDS, PER_CH, windows_features

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data" / "africa", ROOT / "runs"
FEAT_CACHE = RUNS / "diagnose_features.npz"
MODEL_PATH = RUNS / "diagnose_model.joblib"

FS = 128
CH14 = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]
WIN_S, STEP_S = 4.0, 2.0
PTP_MAX_UV = 500.0      # a window is dropped if any channel swings more than this (gross artefact)
MIN_WINDOWS = 20        # if cleaning leaves fewer, keep all windows of the person (and flag it)
WIN_FEATS = [f"{c}_{f}" for c in CH14 for f in PER_CH]
FEATURE_NAMES = [f"{n}_{agg}" for agg in ("med", "iqr") for n in WIN_FEATS]
DROP_REMARK = re.compile(r"remove from dataset|delete this file", re.I)  # the recorders' own exclusions

_SOS = butter(4, [1.0, 45.0], btype="bandpass", fs=FS, output="sos")
_NB, _NA = iirnotch(50.0, 30.0, fs=FS)


# ----------------------------------------------------------------------------- signal -> features
def load_signal(path):
    """One Emotiv CSV (optionally .gz) -> (14, n_samples) float64 microvolts, quality dict."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh)
    missing = [c for c in CH14 if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing channels {missing}")
    x = df[CH14].to_numpy(np.float64).T
    cq = [c for c in df.columns if c.startswith("CQ_") and c[3:] in CH14]
    q = {"seconds": x.shape[1] / FS, "wrapped_channels": unwrap_channels(x)[1],
         "interpolated_frac": float((df["INTERPOLATED"] != 0).mean()) if "INTERPOLATED" in df else None,
         "contact_quality_mean": float(df[cq].to_numpy(float).mean()) if cq else None}
    return x, q


WRAP_LIMIT = 400.0


def unwrap_channels(x):
    """Nigeria files store 7 of the 14 channels (AF4 F4 F7 FC5 O2 P7 T8) wrapped into about +-342 uV:
    when the signal crosses the limit it jumps by ~684 uV in one sample (e.g. -342 -> +342). A channel
    whose values all sit inside +-WRAP_LIMIT and that has such jumps is unwrapped with the jump size as
    the period. Guinea-Bissau channels sit at ~4000 uV and are never touched."""
    x = x.copy()
    n_fixed = 0
    for c in range(len(x)):
        v = x[c]
        if np.abs(v).max() >= WRAP_LIMIT:
            continue
        d = np.abs(np.diff(v))
        big = d > WRAP_LIMIT
        if big.any():
            x[c] = np.unwrap(v, period=float(np.median(d[big])))
            n_fixed += 1
    return x, n_fixed


def preprocess(x):
    x, _ = unwrap_channels(x)
    x = x - np.median(x, axis=1, keepdims=True)      # Emotiv DC offsets (~4000 uV) before filtering
    x = sosfiltfilt(_SOS, x, axis=1)
    return filtfilt(_NB, _NA, x, axis=1)


def cut_windows(x):
    w, s = int(WIN_S * FS), int(STEP_S * FS)
    starts = np.arange(0, x.shape[1] - w + 1, s)
    return np.stack([x[:, a:a + w] for a in starts]) if len(starts) else np.empty((0, x.shape[0], w))


def clean_windows(wins):
    """Drop windows with any channel peak-to-peak > PTP_MAX_UV (movement, electrode pops) or a flat
    channel. Robust median/IQR does most of the work; this only removes gross garbage."""
    ptp = wins.max(2) - wins.min(2)
    ok = (ptp.max(1) <= PTP_MAX_UV) & (ptp.min(1) > 0.5)
    if ok.sum() < MIN_WINDOWS:
        return wins, False, float(ok.mean()) if len(ok) else 0.0
    return wins[ok], True, float(ok.mean())


def person_features(paths):
    """All recordings of one person -> (364-vector, info dict)."""
    wins, quals = [], []
    for p in paths:
        x, q = load_signal(p)
        wins.append(cut_windows(preprocess(x)))
        quals.append(q)
    wins = np.concatenate(wins)
    if len(wins) == 0:
        raise ValueError("recording shorter than one window")
    kept, cleaned, frac_ok = clean_windows(wins)
    F = windows_features(kept, FS).astype(np.float64)
    q75, q25 = np.percentile(F, [75, 25], axis=0)
    vec = np.concatenate([np.median(F, axis=0), q75 - q25])
    info = {"seconds": sum(q["seconds"] for q in quals), "windows": int(len(wins)),
            "windows_kept": int(len(kept)), "clean_frac": frac_ok, "cleaning_applied": cleaned,
            "wrapped_channels": int(max(q["wrapped_channels"] for q in quals)),
            "interpolated_frac": float(np.mean([q["interpolated_frac"] or 0 for q in quals])),
            "contact_quality_mean": float(np.mean([q["contact_quality_mean"] or 0 for q in quals]))}
    return vec, info


# ----------------------------------------------------------------------------- who is who
def people():
    """One row per person: country, subject, label (1 = epilepsy), list of signal files."""
    rows = []
    g = pd.read_csv(DATA / "metadata_guineabissau.csv")
    for r in g.itertuples():
        f = DATA / "guinea_bissau" / "EEGs_Guinea-Bissau" / f"signal-{r[1]}.csv.gz"
        rows.append({"country": "GB", "subject": int(r[1]), "label": int(r.Group.lower() == "epilepsy"),
                     "files": [str(f)] if f.exists() else [], "dropped_by_remark": False})
    n = pd.read_csv(DATA / "metadata_nigeria.csv")
    n["remarks"] = n["remarks"].fillna("")
    for sid, grp in n.groupby("subject.id", sort=True):
        keep = grp[~grp.remarks.str.contains(DROP_REMARK)]
        files = [DATA / "nigeria" / "EEGs_Nigeria" / f for f in keep["csv.file"]]
        files = [str(f) for f in files if f.exists()]
        labels = grp.Group.str.lower().unique()
        assert len(labels) == 1, (sid, labels)
        rows.append({"country": "NG", "subject": int(sid), "label": int(labels[0] == "epilepsy"),
                     "files": files, "dropped_by_remark": len(keep) < len(grp)})
    return pd.DataFrame(rows)


def build_table(force=False, log=print):
    """Features for every person, cached in runs/diagnose_features.npz."""
    if FEAT_CACHE.exists() and not force:
        d = np.load(FEAT_CACHE, allow_pickle=False)
        meta = pd.DataFrame({k[5:]: d[k] for k in d.files if k.startswith("meta_")})
        return d["X"], meta
    ppl = people()
    X, meta = [], []
    for i, r in enumerate(ppl.itertuples()):
        if not r.files:
            log(f"  skip {r.country}-{r.subject}: no usable file (dropped_by_remark={r.dropped_by_remark})")
            continue
        try:
            v, info = person_features(r.files)
        except Exception as e:  # noqa: BLE001
            log(f"  skip {r.country}-{r.subject}: {e}")
            continue
        if info["seconds"] < 60:
            log(f"  skip {r.country}-{r.subject}: only {info['seconds']:.0f}s")
            continue
        X.append(v)
        meta.append({"country": r.country, "subject": r.subject, "label": r.label, "n_files": len(r.files),
                     **info})
        if i % 40 == 0:
            log(f"  {i}/{len(ppl)} people")
    X, meta = np.asarray(X), pd.DataFrame(meta)
    RUNS.mkdir(exist_ok=True)
    np.savez_compressed(FEAT_CACHE, X=X, **{f"meta_{c}": meta[c].to_numpy() for c in meta.columns})
    return X, meta


# ----------------------------------------------------------------------------- models
def make_lr():
    """Standardised L2 logistic regression; C picked by an inner 5-fold CV on the training people only."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=5, scoring="roc_auc",
                                              solver="liblinear", max_iter=5000, n_jobs=1))


SNAP_PARAMS = dict(num_round=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                   class_weight="balanced", n_jobs=1)   # declared before any result; not tuned


class SnapBoost:
    """IBM Snap ML BoostingMachine; after fit, trees are compiled for the on-chip AI accelerator (zDNN)."""

    def __init__(self, seed=0, tree_format="zdnn_tensors"):
        self.seed, self.tree_format = seed, tree_format

    def fit(self, X, y):
        import pickle
        from snapml import BoostingMachineClassifier
        self.m = BoostingMachineClassifier(random_state=self.seed, **SNAP_PARAMS)
        self.m.fit(np.ascontiguousarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
        fi = getattr(self.m, "feature_importances_", None)
        self.feature_importances_ = np.asarray(fi if fi is not None else np.ones(X.shape[1]), float)
        self._raw = pickle.dumps(self.m)    # trees before compiling: what we save to disk
        self._optimize()
        return self

    def _optimize(self):
        self.m.optimize_trees(tree_format=self.tree_format)
        self.tree_format_used = getattr(self.m.booster_, "optimized_tree_format_", None)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "m"}

    def __setstate__(self, state):
        import pickle
        self.__dict__.update(state)
        self.m = pickle.loads(self._raw)
        self._optimize()                    # recompile for the accelerator on load

    def predict_proba(self, X):
        return self.m.predict_proba(np.ascontiguousarray(X, dtype=np.float32))


def make_model(kind, seed=0):
    return make_lr() if kind == "lr" else SnapBoost(seed)


# ----------------------------------------------------------------------------- explanations
PLACE = {"AF3": "left forehead", "AF4": "right forehead", "F3": "left front", "F4": "right front",
         "F7": "left front-side", "F8": "right front-side", "FC5": "left front-centre",
         "FC6": "right front-centre", "T7": "left temple", "T8": "right temple",
         "P7": "left back-side", "P8": "right back-side", "O1": "left back of head", "O2": "right back of head"}


def describe(name):
    """'T7_theta_rel_med' -> 'share of theta waves (4-8 Hz) at the left temple (T7)'."""
    ch, rest = name.split("_", 1)
    feat, agg = rest.rsplit("_", 1)
    if feat.endswith("_logpow") or feat.endswith("_rel"):
        band, kind = feat.split("_")
        lo, hi = BANDS[band]
        what = (f"{band} wave power ({max(lo, 1):g}-{hi:g} Hz)" if kind == "logpow"
                else f"share of {band} waves ({max(lo, 1):g}-{hi:g} Hz)")
    else:
        what = {"linelen": "overall signal activity (line length)",
                "mobility": "average signal frequency (Hjorth mobility)",
                "complexity": "signal irregularity (Hjorth complexity)"}[feat]
    text = f"{what} at the {PLACE[ch]} ({ch})"
    return text if agg == "med" else f"variability over the recording of the {text}"


def top_reasons(final, x, k=3):
    """Plain-English drivers of one prediction.
    LR: coefficient x standardised value, the k features pushing hardest towards the predicted side.
    Trees: importance x |standardised value|, keeping features whose value sits closer to the median of the
    predicted group (epilepsy or control) in the training data, so each reason agrees with the prediction."""
    z = (x - final["mu"]) / final["sd"]
    p_epi = float(final["model"].predict_proba(x[None])[0, 1]) >= final["threshold"]
    if final["kind"] == "lr":
        lr = final["model"].steps[-1][1]
        contrib = lr.coef_[0] * final["model"].steps[0][1].transform(x[None])[0]
        order = list(np.argsort(-contrib if p_epi else contrib))
        side = {j: contrib[j] > 0 for j in order}
    else:
        contrib = final["importance"] * np.abs(z)
        side = {j: abs(x[j] - final["med_epi"][j]) < abs(x[j] - final["med_ctl"][j]) for j in range(len(x))}
        ranked = list(np.argsort(-contrib))
        order = [j for j in ranked if side[j] == p_epi] + [j for j in ranked if side[j] != p_epi]
    out = []
    for j in order[:k]:
        d = describe(FEATURE_NAMES[j])
        group = "people with epilepsy" if side[j] else "people without epilepsy"
        out.append(f"{d[0].upper()}{d[1:]} is {'higher' if z[j] > 0 else 'lower'} than average "
                   f"({z[j]:+.1f} SD), a level more typical of {group} in the training data")
    return out


_FINAL = None


def predict_person(csv_path, model_path=MODEL_PATH):
    """One person's resting EEG (Emotiv 14-channel CSV, .csv or .csv.gz; several paths allowed as a list)
    -> {"epilepsy_probability": p, "top_reasons": [3 strings], ...}. Screening aid, not a diagnosis."""
    global _FINAL
    import joblib
    if _FINAL is None or _FINAL.get("_path") != str(model_path):
        _FINAL = joblib.load(model_path)
        _FINAL["_path"] = str(model_path)
    paths = csv_path if isinstance(csv_path, (list, tuple)) else [csv_path]
    x, info = person_features(paths)
    p = float(_FINAL["model"].predict_proba(x[None])[0, 1])
    return {"epilepsy_probability": round(p, 3),
            "flag": "refer for neurologist review" if p >= _FINAL["threshold"] else "no epilepsy pattern found",
            "threshold": round(float(_FINAL["threshold"]), 3),
            "top_reasons": top_reasons(_FINAL, x),
            "recording": {"seconds": round(info["seconds"]), "windows_used": info["windows_kept"],
                          "clean_frac": round(info["clean_frac"], 2)},
            "model": _FINAL["kind_label"]}
