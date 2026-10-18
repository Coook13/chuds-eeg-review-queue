"""Clinician-editable escalation rules and the event logic that feeds them."""
from dataclasses import dataclass, field

# --- Escalation rules (a clinician can edit these numbers) ---
URGENT_DURATION_S = 300        # one suspected seizure lasting 5+ minutes
URGENT_REPEAT_COUNT = 2        # or this many suspected seizures...
URGENT_REPEAT_WINDOW_S = 3600  # ...within this many seconds

# While a personal model has learned from only ONE confirmed seizure it is least sure of itself,
# so it alerts only above this stricter threshold until a second seizure is confirmed.
# None = same threshold throughout.
import os
FIRST_CONFIRM_THRESHOLD = float(os.environ["FIRST_CONFIRM_TH"]) if os.environ.get("FIRST_CONFIRM_TH") else None

# --- Event logic (tuned on training data, see src/detector.py) ---
SMOOTH_N = 5         # average the last N window scores
ON_COUNT = 5         # consecutive smoothed windows above threshold to open an event
OFF_COUNT = 10       # consecutive windows below threshold to close it
WINDOW_HALF_S = 1.0  # windows are 2 s long; times passed to update() are window centres


@dataclass
class EventTracker:
    threshold: float = 0.5
    recent: list = field(default_factory=list)
    above: int = 0
    below: int = 0
    events: list = field(default_factory=list)  # dicts: start_s, end_s, peak, ongoing

    def update(self, t_s, score, threshold=None):
        """Feed one window score at time t_s (seconds into the file). Returns smoothed score.
        threshold overrides self.threshold for this window (e.g. stricter right after the first
        confirmed seizure, see FIRST_CONFIRM_THRESHOLD)."""
        th = self.threshold if threshold is None else threshold
        self.recent = (self.recent + [score])[-SMOOTH_N:]
        sm = sum(self.recent) / len(self.recent)
        cur = self.events[-1] if self.events and self.events[-1]["ongoing"] else None
        if sm >= th:
            self.above, self.below = self.above + 1, 0
            if cur:
                cur["end_s"], cur["peak"] = t_s, max(cur["peak"], sm)
            elif self.above >= ON_COUNT:
                # start_s is backdated to the first window above threshold (used for overlap tests);
                # alert_s is when the alert actually appears: the end of the window that completed
                # ON_COUNT, i.e. its centre t_s plus half a window. Delays are measured from alert_s.
                self.events.append({"start_s": t_s - ON_COUNT + 1, "end_s": t_s, "peak": sm, "ongoing": True,
                                    "alert_s": t_s + WINDOW_HALF_S})
        else:
            self.above, self.below = 0, self.below + 1
            if cur and self.below >= OFF_COUNT:
                cur["ongoing"] = False
        return sm


def status_for(events, now_s=None, resolved=False):
    """Apply the escalation rules. Returns (status, reason).

    A non-stable status is an UNRESOLVED REVIEW REQUEST. Acknowledge, Confirm and Not-a-seizure
    resolve it (resolved=True) until the next alert. Urgency is judged against now_s (the current
    replay time), not the last event, so an old cluster stops being urgent after the repeat window:
    - urgent: a suspected seizure of URGENT_DURATION_S+ that is ongoing or ended within the window,
      or URGENT_REPEAT_COUNT+ suspected seizures that ended within the last window;
    - review: any other unresolved alert, including old ones nobody has reviewed yet;
    - stable: no alerts, or every alert has been resolved and none is ongoing.
    now_s defaults to the last event's end (same as the old behaviour when called without a clock)."""
    if not events:
        return "stable", "no suspected seizure activity"
    last = events[-1]
    now = last["end_s"] if now_s is None else now_s
    if resolved and not last["ongoing"]:
        return "stable", "alert reviewed, no new suspected seizure since"
    win = URGENT_REPEAT_WINDOW_S
    recent = [e for e in events if e["ongoing"] or now - e["end_s"] <= win]
    dur = last["end_s"] - last["start_s"]
    live = "still ongoing" if last["ongoing"] else "ended"
    if last in recent and dur >= URGENT_DURATION_S:
        return "urgent", f"suspected seizure {int(dur // 60)} min {int(dur % 60)} s, {live}"
    if len(recent) >= URGENT_REPEAT_COUNT:
        return "urgent", f"{len(recent)} suspected seizures in the last {win // 60} min"
    if recent:
        return "review", f"suspected seizure {int(dur)} s, {live}"
    ago = int((now - last["end_s"]) // 60)
    return "review", f"alert not yet reviewed (last suspected seizure {ago} min ago)"


if __name__ == "__main__":
    tr = EventTracker(threshold=0.5)
    for t in range(60):
        tr.update(t, 0.9 if 10 <= t < 30 else 0.1)
    assert len(tr.events) == 1 and not tr.events[0]["ongoing"], tr.events
    assert status_for(tr.events)[0] == "review"
    assert status_for([{"start_s": 0, "end_s": 400, "peak": 1, "ongoing": True}])[0] == "urgent"
    two = [{"start_s": 0, "end_s": 30, "peak": 1, "ongoing": False},
           {"start_s": 900, "end_s": 930, "peak": 1, "ongoing": False}]
    assert status_for(two)[0] == "urgent"
    # the event alert time is when the 5th window above threshold completes, not the backdated start
    assert tr.events[0]["alert_s"] - tr.events[0]["start_s"] == ON_COUNT - 1 + WINDOW_HALF_S
    # an old cluster stops being urgent, stays as an unresolved review, and clears once resolved
    assert status_for(two, now_s=930 + 3601)[0] == "review"
    assert status_for(two, now_s=930 + 3601, resolved=True)[0] == "stable"
    assert status_for(two, now_s=1000)[0] == "urgent"
    ongoing = [{"start_s": 0, "end_s": 50, "peak": 1, "ongoing": True}]
    assert status_for(ongoing, now_s=50, resolved=True)[0] == "review"  # an ongoing seizure is never cleared
    print("rules ok", tr.events[0])
