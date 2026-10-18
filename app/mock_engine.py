"""Fake engine with the same interface as src/engine.py (see CONTRACT.md).

Six beds replay a scripted story, then keep generating random events, so the
UI can be built and demoed without the model or the EDF files.
"""
import math
import random
import statistics
import threading
import time

SPEEDUP = 10                 # replay seconds per real second
TICK_S = 0.25                # background thread period, real seconds
ORIG_FS, OUT_HZ = 256, 64    # evidence "fs" (recording rate) vs rate of the data sent to the browser
EVIDENCE_S = 120
RETRAIN_S = 1.5              # stands in for retraining a personal model on LinuxONE
P_FALSE = {"general": 0.5, "personal": 0.05}   # chance that a short event is a false alert, per model
LONG_EVENT_S = 300           # rules from CONTRACT.md
REPEAT_COUNT, REPEAT_WINDOW_S = 2, 3600
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1"]

# patient, clock at replay start, position in the recording at replay start (s),
# earlier monitoring on the general model as (minutes, alerts, false alerts),
# events already scripted as (onset_s relative to replay start, duration_s, is_false_alert)
BEDS = [
    ("chb01", "01:12:00", 2200, (180, 2, 2), []),
    ("chb08", "14:40:30", 900, (150, 3, 2), [(-400, 70, True)]),       # review: one short event, already ended
    ("chb05", "02:10:20", 1400, (210, 4, 2), [(-100, 560, False)]),    # ongoing, turns urgent after ~20 real seconds
    ("chb10", "09:05:10", 3000, (120, 1, 1), []),
    ("chb14", "21:30:45", 2500, (240, 5, 3), [(-1800, 80, False), (300, 150, False)]),  # 2nd event within the hour -> urgent
    ("chb20", "05:48:00", 1750, (90, 1, 1), []),
]


def clock(sec):
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def secs(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def dur_text(s):
    s = int(s)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60} s"
    return f"{s // 3600} h {s % 3600 // 60} min"


def noise(i, ch, bed):
    """Deterministic pseudo-noise in [-1, 1], so repeated requests draw the same trace."""
    x = math.sin(i * 12.9898 + ch * 78.233 + bed * 37.719) * 43758.5453
    return (x - math.floor(x)) * 2 - 1


class MockEngine:
    def __init__(self, seed=7):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._started = None            # time.monotonic() at start(), None until then
        self._thread = None
        self._stop = threading.Event()
        self._alerts, self._delays, self._proc_ms, self._last_t = 0, [], 3.1, None
        self._beds = {}
        for i, (pid, clk, pos, (mins, alerts, false), scripted) in enumerate(BEDS, start=1):
            b = {"bed": i, "pid": pid, "c0": secs(clk), "f0": pos,
                 "all": [], "events": [], "ack": None, "score": 0.06,
                 "personal": None, "confirmed": set(), "rejected": set(),   # personal: {"n", "at", "took"} once retrained
                 "stats": {"general": {"alerts": alerts, "false": false, "sec": mins * 60.0}, "personal": None}}
            for onset, dur, is_false in scripted:
                self._add_event(b, onset, dur, past=onset < 0, scripted=True, false=is_false)
            self._beds[i] = b
        self._tick(0.0)

    # ---- Engine interface -------------------------------------------------

    def start(self):
        if self._thread:
            return
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_beds(self):
        with self._lock:
            t = self._now()
            return [self._bed_dict(b, t) for b in self._beds.values()]

    def ack(self, bed, who):
        with self._lock:
            b, t = self._bed(bed), self._now()
            b["ack"] = (who, clock(b["c0"] + t))
            return self._bed_dict(b, t)

    def confirm(self, bed, who):
        """Clinician confirms the latest event: retrain this patient's personal model and switch the bed to it."""
        with self._lock:
            b = self._bed(bed)
            if not b["events"]:
                raise ValueError("no suspected seizure on this bed to confirm")
            target = b["events"][-1]["onset"]
        tic = time.monotonic()
        time.sleep(RETRAIN_S)                                 # retraining happens here, outside the lock
        took = time.monotonic() - tic
        with self._lock:
            t = self._now()
            b["confirmed"].add(target)
            b["stats"]["personal"] = b["stats"]["personal"] or {"alerts": 0, "false": 0, "sec": 0.0}
            b["personal"] = {"n": len(b["confirmed"]), "at": clock(b["c0"] + t), "took": took}
            b["ack"] = b["ack"] or (who or "clinician", clock(b["c0"] + t))   # confirming implies it was reviewed
            return self._bed_dict(b, t)

    def reject(self, bed, who):
        """Clinician says the latest alert is not a seizure. Like the real engine, the bed stays on the general
        model until it has a seizure example; with one it 'retrains' (sleeps) as confirm does."""
        with self._lock:
            b = self._bed(bed)
            if not b["events"]:
                raise ValueError("no alert on this bed to reject")
            b["rejected"].add(b["events"][-1]["onset"])
            b["ack"] = (who or "clinician", clock(b["c0"] + self._now()))
            if not b["personal"]:
                return self._bed_dict(b, self._now())
        tic = time.monotonic()
        time.sleep(RETRAIN_S)
        took = time.monotonic() - tic
        with self._lock:
            t = self._now()
            b["personal"] = {"n": len(b["confirmed"]), "at": clock(b["c0"] + t), "took": took}
            return self._bed_dict(b, t)

    def handover(self, bed):
        with self._lock:
            d = self._bed_dict(self._bed(bed), self._now())
        lines = [f"HANDOVER - Bed {d['bed']} ({d['patient_id']}) - replay clock {d['replay_clock']}",
                 f"Status: {d['status'].upper()} - {d['reason']}",
                 f"Model: {d['model']}"]
        for n, e in enumerate(d["events"], start=1):
            state = "ongoing" if e["ongoing"] else "ended"
            lines.append(f"Event {n}: onset {e['onset_clock']}, {dur_text(e['duration_s'])}, "
                         f"peak score {e['peak_score']:.2f}, {state}")
        if d["events"]:
            lines.append(f"Alert raised {d['alert_raised_at']}. Model score now {d['score_now']:.2f}.")
            if d["acknowledged_by"]:
                lines.append(f"Acknowledged by {d['acknowledged_by']} at {d['acknowledged_at']}.")
            else:
                lines.append("Not yet acknowledged.")
            ev = d["evidence"]
            lines.append(f"Evidence: {ev['file']}, {ev['start_s']} s to {ev['end_s']} s. "
                         "Confirm or dismiss on the EEG; the model only flags.")
        else:
            lines.append("No suspected seizure activity in the last hour.")
        ba = d["before_after"]
        if ba["after"]:
            for name, x in (("General model before", ba["before"]), ("Personal model after", ba["after"])):
                lines.append(f"{name}: {x['alerts']} alerts, {x['false_alerts']} false, {x['minutes']} min monitored.")
        return "\n".join(lines)

    def metrics(self):
        with self._lock:
            return {"beds": len(self._beds), "alerts_raised": self._alerts,
                    "median_detection_delay_s": round(statistics.median(self._delays), 1) if self._delays else None,
                    "processing_ms_per_window": round(self._proc_ms, 1),
                    "personalised_beds": sum(1 for b in self._beds.values() if b["personal"])}

    def get_evidence(self, bed):
        with self._lock:
            b, t = self._bed(bed), self._now()
            start, end, spans = self._evidence_range(b, t)
            now_abs = b["f0"] + t
            n_s = min(EVIDENCE_S, max(30, end - start))
            spans = [(a, min(z, now_abs)) for a, z in spans if a < start + n_s and z > start]
            bed_no = b["bed"]
        shift = start // 3600 * 3600                     # report times relative to the file the window starts in
        data = [[round(v, 2) for v in self._synth(bed_no, c, start, n_s, spans)]
                for c in range(len(CHANNELS))]
        return {"bed": bed_no, "fs": ORIG_FS, "channels": CHANNELS, "t0_s": start - shift,
                "data": data, "event_spans": [[round(a - shift, 1), round(z - shift, 1)] for a, z in spans]}

    # ---- internals --------------------------------------------------------

    def _bed(self, bed):
        if bed not in self._beds:
            raise KeyError(f"no such bed: {bed}")
        return self._beds[bed]

    def _now(self):
        return 0.0 if self._started is None else (time.monotonic() - self._started) * SPEEDUP

    def _run(self):
        while not self._stop.wait(TICK_S):
            with self._lock:
                self._tick(self._now())

    def _model(self, b):
        return "personal" if b["personal"] else "general"

    def _add_event(self, b, onset, dur, past=False, scripted=False, false=False):
        peak = self._rng.uniform(0.82, 0.97)
        b["all"].append({"onset": onset, "end": onset + dur, "delay": self._rng.uniform(6, 20),
                         "peak": peak, "seen": peak if past else 0.0, "detected": False,
                         "scripted": scripted, "false": false})

    def _tick(self, t):
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        for b in self._beds.values():
            if all(t >= e["end"] for e in b["all"]):       # nothing pending: schedule the next random event
                dur = self._rng.uniform(320, 600) if self._rng.random() < 0.2 else self._rng.uniform(40, 150)
                self._add_event(b, t + self._rng.uniform(2400, 7200), dur)    # 4 to 12 real minutes away
            target = 0.04 + 0.1 * self._rng.random()
            for e in b["all"]:
                if e["onset"] <= t < e["end"]:
                    ramp = min(1.0, (t - e["onset"]) / 25)
                    target = max(target, 0.15 + (e["peak"] - 0.15) * ramp + self._rng.uniform(-0.04, 0.04))
                    e["seen"] = max(e["seen"], min(target, e["peak"]))
                if not e["detected"] and t >= e["onset"] + e["delay"]:
                    e["detected"] = True
                    if not e["scripted"]:      # only short events can be false alerts; the active model decides how often
                        e["false"] = e["end"] - e["onset"] < 200 and self._rng.random() < P_FALSE[self._model(b)]
                    st = b["stats"][self._model(b)]
                    st["alerts"] += 1
                    st["false"] += int(e["false"])
                    b["events"].append(e)
                    b["ack"] = None                          # a new event needs a fresh acknowledgement
                    self._alerts += 1
                    self._delays.append(e["delay"])
            b["score"] = max(0.0, min(1.0, b["score"] + 0.5 * (target - b["score"])))
            b["events"] = [e for e in b["events"] if e["onset"] >= t - REPEAT_WINDOW_S]
            b["all"] = [e for e in b["all"] if e["end"] > t - 2 * REPEAT_WINDOW_S]
            if not b["events"]:
                b["ack"] = None
            b["stats"][self._model(b)]["sec"] += dt
        self._proc_ms = 0.9 * self._proc_ms + 0.1 * self._rng.uniform(2.4, 4.2)

    def _evidence_range(self, b, t):
        """(start, end, spans) in absolute recording seconds."""
        now_abs = int(b["f0"] + t)
        evs = b["events"]
        if evs:
            last = evs[-1]
            start = max(0, int(b["f0"] + last["onset"]) - 60)
            end = min(now_abs, int(b["f0"] + last["end"]) + 30)
        else:
            start, end = max(0, now_abs - EVIDENCE_S), now_abs
        spans = [(b["f0"] + e["onset"], b["f0"] + e["end"]) for e in evs]
        return start, end, spans

    def _bed_dict(self, b, t):
        evs = [{"onset_clock": clock(b["c0"] + e["onset"]),
                "duration_s": int(min(t, e["end"]) - e["onset"]),
                "peak_score": round(e["seen"], 2), "ongoing": t < e["end"]} for e in b["events"]]
        urgent = any(e["duration_s"] >= LONG_EVENT_S for e in evs) or len(evs) >= REPEAT_COUNT
        status = "urgent" if urgent else "review" if evs else "stable"
        if evs:
            last = evs[-1]
            state = "still ongoing" if last["ongoing"] else "ended"
            if len(evs) > 1:
                reason = f"{len(evs)} suspected seizures in the last hour, latest {dur_text(last['duration_s'])}, {state}"
            else:
                reason = f"suspected seizure {dur_text(last['duration_s'])}, {state}"
        else:
            reason = "no suspected seizure activity"
        start, end, _ = self._evidence_range(b, t)
        idx = start // 3600
        ack = b["ack"]
        p, st, nrej = b["personal"], b["stats"], len(b["rejected"])
        rej = f"{nrej} rejected alert{'s' if nrej != 1 else ''}"
        if not p:
            label = f"general ({rej} stored for the next confirm)" if nrej else "general"
        else:
            label = (f"personal: {p['n']} confirmed seizure{'s' if p['n'] != 1 else ''}{f' + {rej}' if nrej else ''}, "
                     f"retrained {p['at']} in {p['took']:.1f} s on LinuxONE")
        counts = lambda x: None if x is None else {"alerts": x["alerts"], "false_alerts": x["false"],
                                                   "minutes": round(x["sec"] / 60, 1)}
        # fake next-hour risk (the real one comes from src/recurrence.py via app/server.py)
        kept = [e for e in b["events"] if e["onset"] not in b["rejected"]]
        if evs and evs[-1]["ongoing"]:
            risk = {"p": None, "pending": True, "reason": "Updates when the current suspected seizure ends."}
        else:
            risk = {"p": round(min(0.6, 0.07 + 0.16 * len(kept)), 2), "pending": False,
                    "reason": f"Mock value from {len(kept)} suspected seizure{'s' if len(kept) != 1 else ''} in the last hour."}
        return {"bed": b["bed"], "patient_id": b["pid"], "status": status, "reason": reason,
                "score_now": round(b["score"], 2), "events": evs,
                "alert_raised_at": clock(b["c0"] + b["events"][0]["onset"] + b["events"][0]["delay"]) if evs else None,
                "acknowledged_by": ack[0] if ack else None, "acknowledged_at": ack[1] if ack else None,
                "replay_clock": clock(b["c0"] + t),
                "model": label, "personalised": p is not None, "risk": risk,
                "before_after": {"before": counts(st["general"]), "after": counts(st["personal"])},
                "evidence": {"file": f"{b['pid']}_{idx + 1:02d}.edf", "start_s": start - idx * 3600,
                             "end_s": end - idx * 3600}}

    @staticmethod
    def _synth(bed, ch, start, n_s, spans):
        """One channel of fake EEG: alpha/theta/beta sines + noise, plus a 4-5 Hz rhythmic burst inside spans."""
        out, n = [], int(n_s * OUT_HZ)
        ph = ch * 1.3 + bed * 0.7
        gain = 0.75 + 0.5 * ((ch * 5 + bed) % 7) / 6          # per-channel amplitude, so lanes differ
        for i in range(n):
            ti = start + i / OUT_HZ
            v = (18 * math.sin(6.2832 * 10.0 * ti + ph) + 8 * math.sin(6.2832 * 6.0 * ti + 2 * ph)
                 + 4 * math.sin(6.2832 * 21.0 * ti + 3 * ph) + 5 * noise(int(ti * OUT_HZ), ch, bed))
            for a, z in spans:
                if a <= ti <= z:
                    env = min(1.0, (ti - a) / 3, (z - ti) / 3)
                    w = 6.2832 * 4.5 * (ti - a)
                    v += env * 70 * (math.sin(w + ph) + 0.35 * math.sin(2 * w + 2 * ph))
            out.append(v * gain)
        return out


class MockIntake:
    """Fake Intake (diagnose) screen with the same interface as app/server.py's Intake. Every value is made up."""
    RULE = "In each country, the first 2 people with epilepsy and the first 2 without, by subject ID."
    PEOPLE = [("GB-1", "Guinea-Bissau", 1, 1), ("GB-3", "Guinea-Bissau", 3, 1), ("GB-2", "Guinea-Bissau", 2, 0),
              ("GB-5", "Guinea-Bissau", 5, 0), ("NG-500", "Nigeria", 500, 1), ("NG-501", "Nigeria", 501, 1),
              ("NG-6", "Nigeria", 6, 0), ("NG-9", "Nigeria", 9, 0)]

    def list(self):
        return [{"id": k, "country": c, "subject": s, "recordings": 1} for k, c, s, _ in self.PEOPLE]

    def run(self, pid):
        k, c, s, label = next(p for p in self.PEOPLE if p[0] == pid)     # StopIteration -> 404 like the real one
        time.sleep(0.3)
        p = round(0.9 - 0.8 * (1 - label) + 0.05 * math.sin(s), 3)
        group = "people with epilepsy" if p >= 0.423 else "people without epilepsy"
        return {"id": k, "country": c, "subject": s, "epilepsy_probability": p,
                "flag": "refer for neurologist review" if p >= 0.423 else "no epilepsy pattern found",
                "threshold": 0.423,
                "top_reasons": [f"Mock reason {i}: a made-up feature, a level more typical of {group}" for i in (1, 2, 3)],
                "recording": {"seconds": 300, "windows_used": 120, "clean_frac": 0.9},
                "model": "mock (fake values for offline UI work)",
                "dataset_label": "epilepsy" if label else "control", "compute_s": 0.3, "host": "a laptop (mock)"}
