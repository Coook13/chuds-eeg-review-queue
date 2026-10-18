# EEG Review Queue app

Needs Python 3 and Flask only (`pip install flask`); open http://localhost:8050 after starting.
`python app/server.py` runs with fake beds from `app/mock_engine.py` (6 beds, replay at 10x, events appear on their own).
`python app/server.py --real` runs with `src/engine.py` (`Engine()` with no arguments, interface in `CONTRACT.md`).
Set `PORT=9000` to change the port (default 8050); the server binds 0.0.0.0.
Endpoints: `/api/beds`, `/api/evidence/<bed>`, `/api/handover/<bed>`, `/api/metrics`, and POST `/api/ack/<bed>` or `/api/confirm/<bed>` (retrains, takes seconds) with `{"who": "..."}`.
