#!/usr/bin/env bash
# Snap ML re-run ON LinuxONE (model lane). Scoring on the IBM Integrated Accelerator for AI (zDNN).
# Run:  cd ~/eeg-review-queue && nohup bash deploy/run_snap.sh > runs/run_snap.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export TREE_FORMAT=${TREE_FORMAT:-zdnn_tensors} SNAP_JOBS=${SNAP_JOBS:-2} OUT_TAG=_snap
TIMEFORMAT='%R s wall, %U s user'
stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "experiments v1 with Snap ML ($TREE_FORMAT)"
{ time $PY -u -m src.experiments v1 > runs/exp_v1_snap.log 2>&1 ; } 2>> runs/timings_snap.txt
stamp "scale test"
{ time $PY -u scripts/scale_test.py 50 100 > runs/scale_snap.log 2>&1 ; } 2>> runs/timings_snap.txt
stamp "retrain timing"
$PY -u scripts/scale_test.py --retrain > runs/retrain_snap.log 2>&1
stamp "zDNN call count for one live scoring step (1 bed, then 100 beds)"
for N in 1 100; do
  LD_PRELOAD=$PWD/runs/accel_shim.so ACCEL_SHIM_OUT=$PWD/runs/zdnn_step_$N.json $PY -u scripts/scale_test.py --one-step $N > runs/zdnn_step_$N.log 2>&1
done
stamp "SNAP DONE"
