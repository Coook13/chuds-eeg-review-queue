#!/usr/bin/env bash
# One-time setup on the LinuxONE VM (s390x, Ubuntu 24.04).
# PyPI has no s390x wheels for numpy/scipy/pandas/scikit-learn, so use Ubuntu's own builds,
# then a venv that can see them, then pip only for pure-Python mne.
set -u
uname -m
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-numpy python3-scipy python3-pandas python3-sklearn \
  python3-joblib python3-flask python3-matplotlib python3-tqdm python3-pooch python3-decorator \
  python3-jinja2 python3-packaging
rm -rf .venv && python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q mne lazy_loader
.venv/bin/python -c "import numpy, scipy, pandas, sklearn, joblib, mne, flask; print('all imports ok', 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'mne', mne.__version__)"
.venv/bin/python -m src.features && .venv/bin/python -m src.rules
