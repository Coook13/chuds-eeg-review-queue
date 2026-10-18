"""Live engine with clinician-confirmed personalisation.

Every bed starts on the GENERAL model (trained only on other patients). When the clinician
confirms a seizure, the engine retrains that patient's PERSONAL model on IBM LinuxONE from the
confirmed event plus everything recorded so far (this recording up to now + the patient's
earlier recordings), and switches the bed to it. Before/after alert counts are kept per bed and
checked against the recording's expert seizure annotations.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-row predictions: threads only add overhead
import json, threading, time, urllib.request
import joblib
import numpy as np
from src import detector as D
from src.experiments import GENERAL_EXTRA, GENERAL_MAX_NEG, GENERAL_TAG, NEG_PER_FILE, extra_rows, train_rows, transform
from src.features import CHANNELS, WIN_S, LiveFeatures, window_features
from src.rules import FIRST_CONFIRM_THRESHOLD, EventTracker, status_for

VIEW_CH = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2"]
MODEL_DIR = D.RUNS / "models"
FSET = os.environ.get("FSET", "v1")              # personal models
GFSET = os.environ.get("GENERAL_FSET", FSET)    # general model (v2 copes better with new hospitals)
ORDER = ["v1", "v2", "v3"]                      # each set is a prefix of the next
TOP = max(FSET, GFSET, key=ORDER.index)
NFEAT = {"v1": 234, "v2": 520, "v3": 624}
CONFIRM_MARGIN_S = 60  # windows this close to the confirmed seizure are left out of training
import platform
HOST = "IBM LinuxONE (s390x)" if platform.machine() == "s390x" else f"a {platform.machine()} laptop"


def clock(start_clock, t_s):
    # ponytail: chb24 has no clock times in its summary, so it counts from 00:00:00
    if not isinstance(start_clock, str) or ":" not in start_clock:
        start_clock = "00:00:00"
    h, m, s = (int(x) for x in start_clock.split(":"))
    tot = (h * 3600 + m * 60 + s + int(t_s)) % 86400
    return f"{tot // 3600:02d}:{tot % 3600 // 60:02d}:{tot % 60:02d}"


def general_model(files, sz, pat):
    """Leave-this-patient-out model, cached (src/experiments.py saves the same models)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{GFSET}{GENERAL_TAG}_{D.MODEL_TAG}_general_{pat}.joblib"  # GENERAL_TAG=_s: CHB-MIT + Siena adults
    if path.exists():
        return joblib.load(path)
    rows = files[files.patient != pat]
    extra = extra_rows(GFSET) if GENERAL_EXTRA else []
    m = D.fit([train_rows(GFSET, r.case, r.file, sz) for r in rows.itertuples()] + extra, max_neg=GENERAL_MAX_NEG)
    joblib.dump(m, path)
    return m


def llm_rewrite(text):
    """Optional: polish the handover note with an OpenAI-compatible LLM endpoint running on
    LinuxONE (set LLM_URL, LLM_MODEL). Falls back to the template."""
    url = os.environ.get("LLM_URL")
    if not url:
        return text
    prompt = ("Rewrite this EEG review handover for a clinician in plain, calm clinical English, "
              "in at most 110 words. Keep every number exactly. Do not suggest any drug, dose or diagnosis. "
              "End with 'Flagged for clinician review.'\n\n" + text)
    body = json.dumps({"model": os.environ.get("LLM_MODEL", "default"), "max_tokens": 260,
                       "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}).encode()
    try:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        # ponytail: Granite 2B on 3 shared mainframe CPUs writes ~5 tokens/s, so a note takes ~40-70 s
        with urllib.request.urlopen(req, timeout=float(os.environ.get("LLM_TIMEOUT", 150))) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return text + f"\n\n(LLM unavailable: {e.__class__.__name__}; template shown.)"


def choose_beds(files, sz, n_beds):
    """Demo beds by a fixed, written rule (no hand-picking): the first n_beds-1 patients by ID that
    have a recording with 2 or more seizures (their first such recording), plus one seizure-free
    control recording from the next patient by ID."""
    multi = files[files.n_seizures >= 2].sort_values(["patient", "file"]).drop_duplicates("patient")
    picks = list(multi.head(n_beds - 1).itertuples())
    seen = {r.patient for r in picks}
    quiet = files[(files.n_seizures == 0) & ~files.patient.isin(seen)].sort_values(["patient", "file"]).head(1)
    return picks + list(quiet.itertuples())


class Engine:
    def __init__(self, speed=float(os.environ.get("SPEED", 10)), n_beds=int(os.environ.get("BEDS", 6))):
        self.speed, self.lock = speed, threading.Lock()
        files, sz = D.index()
        self.files, self.truth = D.usable(files, sz), sz
        self.beds, self.proc_ms = [], []
        for i, r in enumerate(choose_beds(self.files, sz, n_beds), start=1):
            data, fs = D.load_file(r.case, r.file)
            first = sz[sz.file == r.file].start_s.min() if r.n_seizures else 1500
            lead = float(os.environ.get("LEAD_S", 120)) + 60 * (i - 1)  # stagger so beds alert at different times
            start = max(WIN_S + 2, first - lead)  # v1 needs no warm-up history; v2/v3 use what exists
            X1 = D.features(r.case, r.file, sz)[0]
            w0 = int(start) - 2  # window index of the first live window
            self.beds.append({
                "bed": i, "case": r.case, "patient": r.patient, "file": r.file, "fs": fs, "data": data,
                "start_clock": r.start_clock, "t": float(start), "t_start": float(start),
                "model": general_model(self.files, sz, r.patient), "model_label": "general",
                "personalised": False, "t_personalised": None, "lf": LiveFeatures(TOP, X1[max(0, w0 - 660):w0], first_index=w0),
                "tracker": EventTracker(threshold=D.THRESHOLD), "score_now": 0.0,
                "alert_raised_at": None, "acknowledged_by": None, "acknowledged_at": None,
                "n_events_seen": 0, "confirmed": [], "rejected": []})

    # ---- replay loop ----
    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t0 = time.time()
            with self.lock:
                for b in self.beds:
                    self._step(b)
            time.sleep(max(0.0, 1.0 / self.speed - (time.time() - t0)))

    def _step(self, b):
        fs, w = b["fs"], int(WIN_S * b["fs"])
        end = int(b["t"] * fs)
        if end > b["data"].shape[1]:
            return  # recording finished; hold the last state
        tic = time.perf_counter()
        x = b["lf"].step(window_features(b["data"][:, end - w:end], fs))
        x = x[:NFEAT[FSET if b["personalised"] else GFSET]]
        s = float(b["model"].predict_proba(x[None])[0, 1])
        self.proc_ms = (self.proc_ms + [(time.perf_counter() - tic) * 1000])[-500:]
        strict = FIRST_CONFIRM_THRESHOLD if (b["personalised"] and len(b["confirmed"]) == 1) else None
        b["score_now"] = b["tracker"].update(b["t"] - WIN_S / 2, s,
                                             max(D.THRESHOLD, strict) if strict else None)
        n = len(b["tracker"].events)
        if n > b["n_events_seen"]:  # new event: raise (or re-raise) the alert
            b["n_events_seen"] = n
            b["alert_raised_at"] = clock(b["start_clock"], b["t"])
            b["acknowledged_by"] = b["acknowledged_at"] = None
        b["t"] += 1.0

    # ---- personalisation ----
    def _truth_span(self, b, ev):
        """The clinician marks the seizure on the EEG; we use the expert annotation that overlaps
        the confirmed alert (or the alert itself if none overlaps)."""
        t = self.truth[(self.truth.file == b["file"]) & (self.truth.start_s <= ev["end_s"]) & (self.truth.end_s >= ev["start_s"])]
        return (float(t.start_s.iloc[0]), float(t.end_s.iloc[0])) if len(t) else (ev["start_s"], ev["end_s"])

    def confirm(self, bed, who):
        """Doctor says YES, this alert is a seizure: learn it as a positive example and retrain."""
        with self.lock:
            b = self._find(bed)
            if not b["tracker"].events:
                raise ValueError("no suspected seizure on this bed to confirm")
            span = self._truth_span(b, b["tracker"].events[-1])
            if span not in b["confirmed"]:
                b["confirmed"].append(span)
            b["acknowledged_by"], b["acknowledged_at"] = who or "clinician", clock(b["start_clock"], b["t"])
        return self._retrain(b)

    def reject(self, bed, who):
        """Doctor says NO, this alert is not a seizure: keep every window of it as a negative example
        and retrain. With no seizure example for this patient yet (none confirmed, none in earlier
        recordings) the bed stays on the general model and the negatives wait for the next confirm."""
        with self.lock:
            b = self._find(bed)
            if not b["tracker"].events:
                raise ValueError("no alert on this bed to reject")
            ev = b["tracker"].events[-1]
            span = (ev["start_s"], ev["end_s"] if not ev["ongoing"] else b["t"])
            if span not in b["rejected"]:
                b["rejected"].append(span)
            b["acknowledged_by"], b["acknowledged_at"] = who or "clinician", clock(b["start_clock"], b["t"])
        return self._retrain(b)

    def _retrain(self, b):
        """Shared by confirm and reject. Training set = this recording so far (confirmed seizures as
        positives; rejected alerts kept in full as negatives; other windows sampled as negatives,
        skipping a margin around confirmed seizures) + the patient's earlier recordings."""
        with self.lock:
            now, case, fname, pat = b["t"], b["case"], b["file"], b["patient"]
            confirmed, rejected = list(b["confirmed"]), list(b["rejected"])
        tic = time.time()  # retrain outside the lock so the other beds keep streaming
        X, tc, _ = D.features(case, fname, self.truth)
        Z = transform(X, FSET)
        so_far = tc < now
        y = np.zeros(len(tc), dtype=np.int8)
        near = np.zeros(len(tc), bool)
        hard = np.zeros(len(tc), bool)
        for a, z in confirmed:
            y[(tc >= a) & (tc <= z)] = 1
            near |= (tc >= a - CONFIRM_MARGIN_S) & (tc <= z + CONFIRM_MARGIN_S)
        for a, z in rejected:
            hard |= (tc >= a) & (tc <= z)
        hard &= so_far & (y == 0)
        idx = np.where(so_far & ((y == 1) | ~near) & ~hard)[0]
        rng = np.random.default_rng(0)
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        neg = rng.choice(neg, min(NEG_PER_FILE * 4, len(neg)), replace=False)
        keep = np.concatenate([pos, neg, np.where(hard)[0]])  # rejected alerts: every window kept
        parts = [(Z[keep], y[keep])]
        own = list(self.files[self.files.patient == pat].file)
        earlier = own[:own.index(fname)]
        parts += [train_rows(FSET, self.files[self.files.file == f].case.iloc[0], f, self.truth) for f in earlier]
        has_pos = sum(int(p[1].sum()) for p in parts) > 0
        model = D.fit(parts) if has_pos else None
        secs = time.time() - tic
        with self.lock:
            note = []
            if confirmed:
                note.append(f"{len(confirmed)} confirmed seizure{'s' if len(confirmed) > 1 else ''}")
            if rejected:
                note.append(f"{len(rejected)} rejected alert{'s' if len(rejected) > 1 else ''}")
            if earlier:
                note.append(f"{len(earlier)} earlier recordings")
            if model is None:  # nothing to learn a seizure from yet: keep general, store the negatives
                b["model_label"] = f"general ({', '.join(note)} stored for the next confirm)"
                return self._bed_dict(b)
            if not b["personalised"]:
                b["t_personalised"] = b["t"]
            b["model"], b["personalised"] = model, True
            b["model_label"] = (f"personal: {' + '.join(note)}, retrained {clock(b['start_clock'], b['t'])} "
                                f"in {secs:.1f} s on {HOST}")
            return self._bed_dict(b)

    def _phase_counts(self, b, lo, hi):
        truth = self.truth[self.truth.file == b["file"]]
        evs = [e for e in b["tracker"].events if lo <= e["start_s"] < hi]
        false = sum(1 for e in evs if not ((truth.start_s - 60 <= e["end_s"]) & (truth.end_s + 120 >= e["start_s"])).any())
        return {"alerts": len(evs), "false_alerts": false, "minutes": round(max(0.0, min(hi, b["t"]) - lo) / 60, 1)}

    # ---- contract methods ----
    def _bed_dict(self, b):
        evs = b["tracker"].events
        # acknowledged_by is cleared on every new alert, so it means "the latest alert was resolved"
        status, reason = status_for(evs, now_s=b["t"], resolved=b["acknowledged_by"] is not None)
        ev_start = max(0.0, evs[-1]["start_s"] - 30) if evs else max(0.0, b["t"] - 120)
        tp = b["t_personalised"]
        return {
            "bed": b["bed"], "patient_id": b["case"], "status": status, "reason": reason,
            "score_now": round(b["score_now"], 3), "model": b["model_label"], "personalised": b["personalised"],
            "before_after": {"before": self._phase_counts(b, b["t_start"], tp if tp else float("inf")),
                             "after": self._phase_counts(b, tp, float("inf")) if tp else None},
            "events": [{"onset_clock": clock(b["start_clock"], e["start_s"]),
                        "duration_s": int(e["end_s"] - e["start_s"]),
                        "peak_score": round(e["peak"], 3), "ongoing": e["ongoing"]} for e in evs],
            "alert_raised_at": b["alert_raised_at"], "acknowledged_by": b["acknowledged_by"],
            "acknowledged_at": b["acknowledged_at"], "replay_clock": clock(b["start_clock"], b["t"]),
            "evidence": {"file": b["file"], "start_s": int(ev_start), "end_s": int(ev_start + 120)}}

    def _find(self, bed):
        return next(b for b in self.beds if b["bed"] == int(bed))

    def get_beds(self):
        with self.lock:
            return [self._bed_dict(b) for b in self.beds]

    def get_evidence(self, bed):
        with self.lock:
            b = self._find(bed)
            ev = self._bed_dict(b)["evidence"]
            fs = b["fs"]
            a, z = int(ev["start_s"] * fs), int(min(ev["end_s"], b["t"]) * fs)
            idx = [CHANNELS.index(c) for c in VIEW_CH]
            seg = b["data"][idx, a:z]
            step = int(fs // 64)
            seg = seg[:, :seg.shape[1] // step * step].reshape(len(idx), -1, step).mean(-1)
            spans = [[e["start_s"], e["end_s"]] for e in b["tracker"].events
                     if e["end_s"] >= ev["start_s"] and e["start_s"] <= ev["end_s"]]
            return {"bed": b["bed"], "fs": 64, "channels": VIEW_CH, "t0_s": ev["start_s"],
                    "data": np.round(seg, 1).tolist(), "event_spans": spans}

    def ack(self, bed, who):
        with self.lock:
            b = self._find(bed)
            b["acknowledged_by"] = who or "clinician"
            b["acknowledged_at"] = clock(b["start_clock"], b["t"])
            return self._bed_dict(b)

    def handover(self, bed):
        with self.lock:
            d = self._bed_dict(self._find(bed))
        lines = [f"Bed {d['bed']}, patient {d['patient_id']} (CHB-MIT recording {d['evidence']['file']}).",
                 f"Status: {d['status'].upper()}. Reason: {d['reason']}.", f"Model in use: {d['model']}."]
        for i, e in enumerate(d["events"], 1):
            lines.append(f"Suspected seizure {i}: onset {e['onset_clock']}, {e['duration_s'] // 60} min "
                         f"{e['duration_s'] % 60} s, peak model score {e['peak_score']}"
                         f"{', ongoing' if e['ongoing'] else ''}.")
        lines.append(f"Alert raised {d['alert_raised_at'] or 'n/a'}. "
                     + (f"Acknowledged by {d['acknowledged_by']} at {d['acknowledged_at']}."
                        if d["acknowledged_by"] else "Not yet acknowledged."))
        lines.append(f"Evidence: {d['evidence']['file']}, {d['evidence']['start_s']}-{d['evidence']['end_s']} s.")
        lines.append("These are model-detected suspected events, not confirmed seizures. "
                     "Flagged for clinician review.")
        return llm_rewrite("\n".join(lines))

    def metrics(self):
        with self.lock:
            delays, false, alerts = [], 0, 0
            for b in self.beds:
                truth = self.truth[self.truth.file == b["file"]]
                for e in b["tracker"].events:
                    alerts += 1
                    hit = truth[(truth.start_s <= e["end_s"]) & (truth.end_s >= e["start_s"])]
                    if len(hit):
                        delays.append(max(0.0, e.get("alert_s", e["start_s"]) - hit.start_s.min()))
                    else:
                        false += 1
            return {"host": HOST, "scoring": D.SCORING, "beds": len(self.beds), "alerts_raised": alerts, "false_alerts": false,
                    "personalised_beds": sum(b["personalised"] for b in self.beds),
                    "median_detection_delay_s": float(np.median(delays)) if delays else None,
                    "processing_ms_per_window": round(float(np.median(self.proc_ms)), 2) if self.proc_ms else None,
                    "replay_speed": self.speed}
