# [Chuds_IBM Z Datathon 2026 Project] EEG Review Queue with clinician-confirmed personalisation

**Theme:** Real-Time AI for Critical Decisions. **Event code:** SSF2026. **Runs on:** IBM LinuxONE (s390x).

When several patients are on brain monitoring (EEG), one specialist can't watch every screen. Our system spots suspected seizure activity in each patient's live EEG as it happens, flags which patient needs a specialist now and why, and opens the exact stretch of EEG as evidence. It also logs the clinician's acknowledgement and writes a handover note. It never suggests treatment.

**Headline feature: clinician-confirmed personalisation.**
1. Every bed starts on a general model, trained only on other patients.
2. When the clinician confirms a real seizure, IBM LinuxONE retrains that patient's personal model in seconds. It uses the confirmed event, the recording so far, and the patient's earlier recordings.
3. The bed switches to the personal model, and the dashboard shows that bed's before/after alerts and false alerts.

## Why it matters
- Seizures often go unnoticed. In a hospital study, patients failed to record 55.5% of 582 seizures, and 85.8% of seizures during sleep (Hoppe et al., 2007).
- Up to 70% of people with epilepsy could live seizure-free if properly diagnosed and treated (WHO).

## How it works
1. **Features, every second** (`src/features.py`):
   - takes the last 2 seconds of 18 standard EEG channels;
   - per channel, computes band powers, line length and Hjorth parameters;
   - 234 features in total.
2. **General model** (`src/detector.py`, `src/experiments.py`): gradient boosting trained leave-one-patient-out. chb01 and chb21 are the same person, so they are always held out together.
3. **Event logic plus clinician-editable escalation rules** (`src/rules.py`):
   - any suspected seizure moves the bed to **review**;
   - 5+ minutes, or 2+ seizures within an hour, moves it to **urgent**;
   - urgent beds are never hidden.
4. **Confirm and retrain** (`src/engine.py`, `Engine.confirm`): trains the personal model on LinuxONE and switches the bed to it.
5. **Handover note:** written by IBM Granite 3.3 2B, running on the same LinuxONE VM through llama.cpp built for s390x with a big-endian model. If no LLM is reachable, a template is used.
6. **Review queue** (`app/`): live replay of real recordings for several beds, with an evidence viewer, acknowledge, confirm, a before/after table and the handover note.
   - **Demo beds follow a fixed rule** (`engine.choose_beds`): take the first BEDS-1 patients by ID that have a recording with 2+ seizures, using their first such recording, plus one seizure-free control recording from the next patient by ID. Beds are not hand-picked.
   - The live before/after panel compares unequal time spans, so it illustrates the workflow rather than proving anything. The evidence is the confirm-protocol comparison on the same later recordings (Results).
   - Live delays are measured in recording time. The replay runs at 10x.

Hybrid multi-model predictive AI = general model + personal model + rules. Predictive + generative AI = detector + Granite handover note.

## Results (all trained and evaluated on IBM LinuxONE)
Full details and caveats are in [results/RESULTS.md](results/RESULTS.md).

**Data and protocol:**
- Data: CHB-MIT, 23 patients, 321 recordings (477 hours), 191 seizures. That covers every seizure recording plus 8 seizure-free recordings per patient.
- Evaluation is event-level: a seizure counts as caught if an alert overlaps it. False alarms are counted per 24 hours of recording.
- Model: IBM Snap ML, scored on the on-chip IBM Integrated Accelerator for AI (zDNN).
- The threshold of 0.8 was fixed before any results were seen.

| Situation (threshold 0.8) | Seizures caught | False alarms per day | Median seconds to alert (from seizure start to when the alert appears) |
|---|---|---|---|
| **Brand-new patient, general model (shipped: v2 features, trained on children + Siena adults)** | **103 / 191 (54%)** | **17.2** | **13** |
| Brand-new patient, general model (v2, children only) | 102 / 191 (53%) | 22.8 | 12.5 |
| Brand-new patient, general model with v1 features (earlier) | 82 / 191 (43%) | 148.0 | 13.5 |
| Same 303 h after the doctor confirms seizures: general model (v2, children only) | 84 / 168 (50%) | 27.1 | 13 |
| **Same 303 h after the doctor confirms seizures: personalised** | **141 / 168 (84%)** | **12.1** | **11** |
| Personal model trained only on the patient's earlier recordings | 135 / 164 (82%) | 2.9 | 12 |

The v2 general model compares each channel with its own last 10 minutes, which cut false alarms for a brand-new patient from 148 to 22.8 per day (window AUC 0.74 to 0.86). Adding the 14 Siena adults to its training cut them again, to 17.2 per day (AUC 0.90), with the same seizures caught. Adding the Helsinki newborns cut false alarms to 9.1 per day but missed more seizures (91/191), so it was not shipped. Details are in results/RESULTS.md.

Personalising raised false alarms for 6 of 23 patients (per-patient table in results/RESULTS.md).

**On LinuxONE:**
- Scale test: 100 beds streamed at once at 0.31 ms per window ([results/scale_test_snap.json](results/scale_test_snap.json)).
- Retraining one patient's personal model takes about 1.3 to 3.1 s ([results/retrain_timing_snap.json](results/retrain_timing_snap.json)).

**Other parts:**
- **Diagnose:** epilepsy yes or no from a 5-minute EEG. Within-country cross-validated AUC is 0.898 (Guinea-Bissau) and 0.935 (Nigeria), against the original study's 0.85 and 0.78. See results/diagnose_results.txt.
- **Predict:** next-hour seizure risk. Pooled AUC is 0.754, against 0.716 for the simple rule "seizure in the last 6 h", so the gain is modest. See results/recurrence_results.txt.
- **Three populations:** children, adults and newborns. See results/ALL_POPULATIONS.md.

## Run it
```bash
pip install numpy scipy pandas scikit-learn joblib mne flask   # on LinuxONE: see deploy/setup.sh
FULL=1 python scripts/download_chbmit.py   # every seizure recording, via PhysioNet's AWS mirror
python -m src.experiments v1               # trains and evaluates every protocol; writes runs/exp_v1.json
python scripts/scale_test.py 50 100        # many beds at once
python app/server.py --real                # live queue at http://localhost:8050
```
- LinuxONE step by step: `deploy/linuxone.md`, `deploy/run_all.sh`, `deploy/start_app.sh`.
- Plain-English guide: `docs/HOW_IT_WORKS.md`.

## Data (open, cited)
- **CHB-MIT Scalp EEG Database v1.0.0:**
  - Shoeb A. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD thesis, MIT.
  - Goldberger A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23).
  - https://physionet.org/content/chbmit/1.0.0/ (Open Data Commons Attribution v1.0).

## Evidence cited
- Hoppe C, Poepel A, Elger CE (2007). Epilepsy: accuracy of patient seizure counts. Archives of Neurology. https://pubmed.ncbi.nlm.nih.gov/17998441/
- WHO epilepsy fact sheet: https://www.who.int/news-room/fact-sheets/detail/epilepsy

## Open-source used
- **Python libraries:**
  - numpy: https://numpy.org
  - scipy: https://scipy.org
  - pandas: https://pandas.pydata.org
  - scikit-learn: https://scikit-learn.org
  - joblib: https://joblib.readthedocs.io
  - MNE-Python: https://mne.tools
  - Flask: https://flask.palletsprojects.com
- **LLM serving:**
  - llama.cpp: https://github.com/ggml-org/llama.cpp
  - IBM Granite 3.3 2B Instruct: https://huggingface.co/ibm-granite

## Limits (said plainly)
- CHB-MIT is children and young people with hard-to-treat epilepsy, recorded in hospital after medication was withdrawn.
- Alerts are suspected events for clinician review, not diagnoses.
- In the live demo the clinician's confirmation uses the expert annotation of that seizure.
- Replay is accelerated.
