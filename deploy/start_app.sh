#!/usr/bin/env bash
# Start the live review queue on LinuxONE, with IBM Granite (llama.cpp, port 8081) writing handover notes.
# Waits until the general models are cached (src/experiments.py writes them).
# Run:  cd ~/eeg-review-queue && nohup bash deploy/start_app.sh > runs/app.log 2>&1 < /dev/null &
# View: on your laptop  ssh -i ~/.ssh/linuxone_ibmz -N -L 8050:localhost:8050 linux1@YOUR_LINUXONE_IP
#       then open http://localhost:8050
cd "$(dirname "$0")/.."
while ! ls runs/models/v1_*general_*.joblib > /dev/null 2>&1; do sleep 30; done  # wait only until general models exist
export LLM_URL=http://127.0.0.1:8081/v1/chat/completions LLM_MODEL=granite-3.3-2b-instruct
export TREE_FORMAT=zdnn_tensors  # Snap ML tree models scored on the on-chip IBM Integrated Accelerator for AI
export GENERAL_FSET=v2 GENERAL_EXTRA=siena GENERAL_TAG=_s  # v2 general model trained on children + Siena adults (17.2 vs 22.8 false alarms/day)  # general model on baseline-relative features (won on 321 recordings); personal models stay v1
export PORT=8050 SPEED="${SPEED:-10}" BEDS="${BEDS:-6}"
echo "[$(date '+%F %T')] starting app on $(uname -m)"
exec .venv/bin/python -u app/server.py --real
