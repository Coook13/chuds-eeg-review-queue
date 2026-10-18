# Interfaces (do not change without telling the lead)

## Data files (written by scripts/download_chbmit.py)

- `data/chbmit/<case>/<file>.edf`: raw CHB-MIT EDF files.
- `data/files.csv`: columns `case,file,start_clock,end_clock,n_seizures,downloaded`
- `data/seizures.csv`: columns `case,file,start_s,end_s` (seconds from the start of that file)

## Engine interface (src/engine.py implements it; app/mock_engine.py fakes it)

```python
class Engine:
    def start(self) -> None: ...               # begins replaying all beds in a background thread
    def get_beds(self) -> list[dict]: ...      # one dict per bed, schema below
    def get_evidence(self, bed: int) -> dict: ...
    def ack(self, bed: int, who: str) -> dict: ...  # returns updated bed dict
    def handover(self, bed: int) -> str: ...   # plain-text handover note
    def metrics(self) -> dict: ...             # live counters for the footer
```

### Bed dict

```json
{"bed": 3, "patient_id": "chb05", "status": "urgent",
 "reason": "suspected seizure 6 min 10 s, still ongoing",
 "score_now": 0.91,
 "events": [{"onset_clock": "02:14:05", "duration_s": 370, "peak_score": 0.93, "ongoing": true}],
 "alert_raised_at": "02:14:19", "acknowledged_by": null, "acknowledged_at": null,
 "replay_clock": "02:20:15",
 "evidence": {"file": "chb05_13.edf", "start_s": 1080, "end_s": 1500}}
```

- `status` is one of `"stable"`, `"review"`, `"urgent"`.
- Escalation rules (clinician-editable, in `src/rules.py`):
  - Any suspected seizure event moves the patient to `review`.
  - An event lasting at least 300 seconds, or 2 or more events within 3600 seconds, moves the patient to `urgent`.
- Urgent beds are always shown first and never hidden.

### Evidence dict

```json
{"bed": 3, "fs": 256, "channels": ["FP1-F7", "F7-T7"], "t0_s": 1080,
 "data": [[...], [...]], "event_spans": [[1140.0, 1510.0]]}
```

- `data` is a list of channels, each a list of floats, downsampled to at most 64 Hz for the browser.
- Return at most 8 channels and 120 seconds.
