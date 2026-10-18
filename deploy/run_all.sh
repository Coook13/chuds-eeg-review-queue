#!/usr/bin/env bash
# Full pipeline ON LinuxONE. Steps can be skipped: SKIP_DATA=1 bash deploy/run_all.sh
# Run:  cd ~/eeg-review-queue && nohup bash deploy/run_all.sh > runs/run_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
PY=.venv/bin/python
export OMP_NUM_THREADS=3
TIMEFORMAT='%R s wall, %U s user'
stamp() { echo "[$(date '+%F %T')] $*"; }

{ stamp "proof"; uname -a; uname -m; nproc; lscpu | head -20; head -3 /etc/os-release; free -h
  $PY -c "import sklearn, numpy, platform; print('python', platform.python_version(), 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
} > runs/proof_linuxone.txt 2>&1

if [ "${SKIP_DATA:-0}" != 1 ]; then
  stamp "download (all seizure recordings + 2 seizure-free per case, S3 mirror)"
  { time FULL=1 $PY -u scripts/download_chbmit.py > runs/download.log 2>&1 ; } 2>> runs/timings.txt
  stamp "features for every recording"
  { time $PY -u -c "from src import detector as D; f, s = D.index(); u = D.usable(f, s); print(len(u), 'usable of', len(f), 'recordings;', u.patient.nunique(), 'patients')" > runs/features.log 2>&1 ; } 2>> runs/timings.txt
fi

for FS in ${FSETS:-v1 v2 v3}; do
  stamp "experiments $FS"
  { time $PY -u -m src.experiments $FS > runs/exp_$FS.log 2>&1 ; } 2>> runs/timings.txt
  echo "experiments $FS: $(tail -1 runs/timings.txt)" >> runs/proof_linuxone.txt
  if [ "$FS" = v1 ]; then
    stamp "scale test"
    { time OMP_NUM_THREADS=1 FSET=v1 $PY -u scripts/scale_test.py 50 100 > runs/scale.log 2>&1 ; } 2>> runs/timings.txt
    echo "scale test: $(tail -1 runs/timings.txt)" >> runs/proof_linuxone.txt
  fi
done
stamp "ALL DONE"
