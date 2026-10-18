"""Brain-wave features per short window. Same code runs offline (training) and live (replay)."""
import numpy as np

# Standard CHB-MIT bipolar "double banana" channels, present in nearly every file.
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
PER_CH = [f"{b}_logpow" for b in BANDS] + [f"{b}_rel" for b in BANDS] + ["linelen", "mobility", "complexity"]
FEATURE_NAMES = [f"{c}_{f}" for c in CHANNELS for f in PER_CH]
WIN_S, STEP_S = 2.0, 1.0


def pick_channels(raw_names):
    """Map our 18 standard channels to indices in a file. mne renames duplicates like T8-P8-0."""
    up = [n.upper() for n in raw_names]
    idx = []
    for c in CHANNELS:
        hits = [i for i, n in enumerate(up) if n == c or n.startswith(c + "-")]
        if not hits:
            raise ValueError(f"channel {c} missing")
        idx.append(hits[0])
    return idx


def window_features(win, fs):
    """win: (n_channels, n_samples) for one window -> 1-D feature vector."""
    return windows_features(win[None], fs)[0]


def windows_features(wins, fs):
    """wins: (n_windows, n_channels, n_samples) -> (n_windows, n_channels * len(PER_CH))."""
    wins = wins - wins.mean(axis=2, keepdims=True)
    n = wins.shape[2]
    spec = np.abs(np.fft.rfft(wins * np.hanning(n), axis=2)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fs)
    bp = np.stack([spec[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in BANDS.values()], -1)
    total = bp.sum(-1, keepdims=True) + 1e-12
    d1 = np.diff(wins, axis=2)
    d2 = np.diff(d1, axis=2)
    v0, v1, v2 = wins.var(2) + 1e-12, d1.var(2) + 1e-12, d2.var(2) + 1e-12
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    linelen = np.log(np.abs(d1).mean(2) + 1e-12)
    feats = np.concatenate([np.log(bp + 1e-12), bp / total,
                            linelen[..., None], mobility[..., None], complexity[..., None]], -1)
    return feats.reshape(len(wins), -1).astype(np.float32)


def file_features(data, fs):
    """data: (n_channels, n_samples) whole recording -> (X, window_centre_seconds)."""
    w, s = int(WIN_S * fs), int(STEP_S * fs)
    starts = np.arange(0, data.shape[1] - w + 1, s)
    X = np.concatenate([windows_features(np.stack([data[:, a:a + w] for a in starts[i:i + 600]]), fs)
                        for i in range(0, len(starts), 600)])
    return X, (starts + w / 2) / fs


BASELINE_WIN = 600  # windows (~10 min) of the patient's own recent history
BASELINE_REFRESH = 30  # recompute that baseline every 30 windows (s); cheap enough for 100+ beds
AUG_NAMES = (FEATURE_NAMES + [f"{n}_vsbase" for n in FEATURE_NAMES]
             + [f"{agg}_{f}{sfx}" for sfx in ["", "_vsbase"] for agg in ["max", "mean"] for f in PER_CH])


def augment(X):
    """v2 features: add 'change from this patient's last ~10 minutes' (causal rolling median,
    so it works live) and cross-channel max/mean so focal seizures anywhere look alike."""
    import pandas as pd
    base = pd.DataFrame(X).rolling(BASELINE_WIN, min_periods=30).median().shift(1).bfill().to_numpy()
    base = base[(np.arange(len(X)) // BASELINE_REFRESH) * BASELINE_REFRESH]  # hold between refreshes
    vs = (X - base).astype(np.float32)
    n_ch = len(CHANNELS)
    aggs = []
    for M in (X, vs):
        M3 = M.reshape(len(M), n_ch, len(PER_CH))
        aggs += [M3.max(1), M3.mean(1)]
    return np.concatenate([X, vs] + aggs, axis=1)


class LiveFeatures:
    """Streaming version of experiments.transform for one patient: same features, one window at a
    time, using only the past. Prefill with the v1 features of the minutes before the start."""

    def __init__(self, fset, history=(), first_index=None):
        """history: v1 features of the windows just before; first_index: absolute window index
        (seconds into the recording minus 2) of the first window that step() will receive."""
        history = list(history)
        warm = history[-5:]
        self.keep = BASELINE_WIN + BASELINE_REFRESH
        self.fset, self.hist, self.agg_hist, self.base = fset, history[:-5][-self.keep:], [], None
        self.k = (first_index if first_index is not None else len(history)) - len(warm)
        for x in warm:  # replay the last few windows so the context lags are warm
            self.step(x)

    def step(self, x1):
        if self.fset == "v1":
            return x1
        if self.base is None or self.k % BASELINE_REFRESH == 0:
            off = self.k % BASELINE_REFRESH  # starting mid-block: use the baseline from the block start
            h = (self.hist[:len(self.hist) - off] if off else self.hist)[-BASELINE_WIN:]
            self.base = np.median(h, axis=0) if len(h) >= 30 else x1
        self.k += 1
        vs = (x1 - self.base).astype(np.float32)
        n_ch = len(CHANNELS)
        aggs = []
        for v in (x1, vs):
            m3 = v.reshape(n_ch, len(PER_CH))
            aggs += [m3.max(0), m3.mean(0)]
        agg = np.concatenate(aggs)
        self.hist = (self.hist + [x1])[-self.keep:]
        v2 = np.concatenate([x1, vs, agg])
        if self.fset == "v2":
            return v2
        self.agg_hist = (self.agg_hist + [agg])[-5:]
        lag = lambda k: self.agg_hist[-1 - k] if len(self.agg_hist) > k else self.agg_hist[0]
        return np.concatenate([v2, lag(2), lag(4)])


if __name__ == "__main__":
    fs = 256
    t = np.arange(0, 10, 1 / fs)
    fake = np.stack([np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t)) for _ in CHANNELS])
    X, tc = file_features(fake, fs)
    assert X.shape == (9, len(FEATURE_NAMES)), X.shape
    # A 3 Hz sine should put most power in delta.
    rel_delta = X[0, PER_CH.index("delta_rel")]
    assert rel_delta > 0.8, rel_delta
    print("features ok", X.shape, "delta share", round(float(rel_delta), 3))
