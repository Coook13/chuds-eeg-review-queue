# Results: everything below was trained and measured on IBM LinuxONE (s390x)

**Machine:**
- IBM LinuxONE Community Cloud VM `ibmzdatathonuom`.
- s390x, big-endian, 3 CPUs, 7.8 GB RAM, Ubuntu 24.04 (`linuxone_proof_linuxone.txt`).

**Shipped model:** IBM Snap ML gradient boosting (`snapml.BoostingMachineClassifier`, 200 rounds).
- Trained on the CPU.
- Every score is computed on the IBM Integrated Accelerator for AI, the on-chip AI unit, through zDNN (`TREE_FORMAT=zdnn_tensors`).
- The earlier scikit-learn model is shown alongside for comparison.

**Headline alert threshold: 0.8.** It was declared in `src/detector.py` before any results were seen. We kept it after switching to Snap ML because it did not collapse. Full sweeps from 0.5 to 0.9 are in the JSON files.

## HEADLINE (most credible run, threshold 0.8, Snap ML on the AI accelerator)

**Data:**
- 23 patients, **321 recordings (477 hours)**, 191 seizures.
- Every seizure recording, plus **8 seizure-free recordings per patient**, up from 2. False alarms per day are therefore measured over a lot of ordinary EEG.

**Shipped models:**
- **General model (every new bed): v2 features, trained on the CHB-MIT children plus the 14 Siena adults.**
  - v2 is v1 plus each channel relative to its own last 10 minutes, plus cross-channel max/mean (520 features).
  - Set with `GENERAL_FSET=v2 GENERAL_EXTRA=siena GENERAL_TAG=_s`; the models are `runs/models/v2_s_snap-zdnn_general_chbXX.joblib`.
  - The held-out child is never in training, and no Siena patient is ever tested here.
- **Personal model (after the doctor confirms): v1 features** (234).
- v1 is a prefix of v2, so one live feature computation feeds both.

**Sources:**
- v2 general model, children + Siena adults: `exp_v1_snap_gv2s.json`, `metrics_v1_snap_gv2s_th0.8.json`.
- v2 general model, children only: `exp_v1_snap_gv2.json`, `metrics_v1_snap_gv2_th0.8.json`.
- v1 general model: `exp_v1_snap_more.json`, `metrics_v1_snap_more_th0.8.json`.
- Same data, same threshold, same personal models.

| Situation | Seizures caught (event sensitivity) | False alarms per day | Median s to alert | Window sensitivity | Window specificity | AUC |
|---|---|---|---|---|---|---|
| **Brand-new patient, general model (v2, children + Siena adults, shipped)** | **103 / 191 (54%)** | **17.2** | **13** | **0.49** | **0.98** | **0.90** |
| Brand-new patient, general model (v2, children only) | 102 / 191 (53%) | 22.8 | 12.5 | 0.45 | 0.98 | 0.86 |
| Brand-new patient, general model (v1, earlier) | 82 / 191 (43%) | 148.0 | 13.5 | 0.37 | 0.90 | 0.74 |
| Same 303 h after the doctor confirms: general model (v2, children only) | 84 / 168 (50%) | 27.1 | 13 | 0.42 | 0.98 | 0.85 |
| Same 303 h after the doctor confirms: general model (v1) | 67 / 168 (40%) | 179.7 | 13 | 0.35 | 0.90 | 0.72 |
| **Same 303 h after the doctor confirms: personalised** | **141 / 168 (84%)** | **12.1** | **11** | **0.72** | **0.99** | **0.95** |
| Personal model trained only on earlier recordings | 135 / 164 (82%) | 2.9 | 12 | 0.68 | 0.997 | 0.96 |

**What changed with the v2 general model (children only, vs v1):**
- A brand-new patient now gets 6.5x fewer false alarms (148 → 22.8 per day).
- It also catches 20 more seizures.
- After confirmation, personalising still beats it on the same hours: 141 vs 84 seizures caught, 12.1 vs 27.1 false alarms per day.

**Per patient vs the v2 general model** (`exp_v1_snap_gv2.json` → `confirm` → `0.8` → `by_patient`):
- **More false alarms after personalising (5 of 23 patients):**
  - chb03: 0 → 12.9 per day
  - **chb08: 19.7 → 108.3**
  - chb15: 0 → 1.4
  - **chb17: 20.9 → 167.3**
  - chb24: 6.1 → 20.3
- **Fewer seizures caught (4 patients):** chb07 2/2 → 1/2, chb16 1/9 → 0/9, chb17 2/2 → 1/2, chb18 4/5 → 3/5.
- **Biggest gains:** chb12 5/32 → 31/32, chb15 2/19 → 18/19, chb14 0/7 → 6/7, chb06 1/9 → 6/9 (70.7 → 0.8 per day).

**Does training on other hospitals' patients help a brand-new child?**
Every row tests the same 321 CHB-MIT recordings (191 seizures, 477 h), leaving one patient out, on Snap ML with the zDNN accelerator.

| General model trained on | 0.5 | 0.6 | 0.7 | **0.8 (declared)** | 0.9 | AUC |
|---|---|---|---|---|---|---|
| Children only (CHB-MIT) | 143 / 156.5 | 134 / 84.9 | 120 / 45.9 | 102 / 22.8 | 84 / 9.1 | 0.86 |
| **+ Siena adults (shipped)** | **157 / 132.8** | **137 / 73.7** | **120 / 37.2** | **103 / 17.2** | **90 / 6.6** | **0.90** |
| + Siena adults + Helsinki newborns | 124 / 73.5 | 111 / 40.6 | 103 / 20.8 | 91 / 9.1 | 73 / 3.0 | 0.88 |

Cells are seizures caught out of 191 / false alarms per day.
- **Adding the Siena adults** catches at least as many seizures with fewer false alarms at every threshold. At 0.8, false alarms fall 25% (22.8 → 17.2 per day) for one more seizure caught, and AUC rises from 0.86 to 0.90. The extra catch is within noise; the false-alarm cut and the AUC are the real gain. The ship rule was set before the run: more caught with no more false alarms.
- **Adding the Helsinki newborns as well** is not shipped. It adds 42,653 newborn seizure windows, about 4x the children's, so the model leans towards newborn seizure patterns and catches fewer at 0.8 (91 vs 102). It ranks windows better than children-only (AUC 0.88), but worse than adults-only.
- **Not re-run with the new general model:** the confirm-mode "general model" rows. The personalised rows are unaffected, because after a confirmation only the personal model scores.
- **Wall-clock on LinuxONE:** 893 s for the 23 adults-added models; 1,245 s with newborns too.
- **Code:** `src/experiments.py`, env `GENERAL_EXTRA` / `GENERAL_TAG`.

**Tested and not shipped:** mixing other patients' data into the personal model (`CONFIRM_MIX=20000`, laptop HGB run).
- It cut false alarms after the 1st confirmation from 53.5 to 4.6 per day.
- But it caught 128/167 seizures instead of 139/167.

**How the measures are defined:**
- **Median s to alert:** from seizure onset to the moment the alert appears on screen, i.e. the end of the window that completes 5 windows above threshold (commit d6f8506). That is about 9 s of built-in smoothing. Recomputed from cached scores on LinuxONE (`metrics_v1_snap_more_th0.8.json`, `metrics_v1_snap_gv2_th0.8.json`). Seizure counts and false alarms were unchanged by the fix.
- **Event sensitivity:** seizures with an overlapping alert, out of all seizures.
- **Window sensitivity and specificity:** each 2-second window is called seizure or not on the raw model score at 0.8.
- **Why specificity looks high:** non-seizure windows outnumber seizure windows about 1000 to 1. False alarms per day is the clinically meaningful counterpart.
- **AUC:** threshold-free, over all windows.

**Learning curve after confirmations** (personalised, same hours):

| Confirmed so far | Caught | False alarms per day | Hours |
|---|---|---|---|
| 1 | 18/23 | 28.9 | 46.5 |
| 2 to 3 | 32/40 | 22.5 | 85.5 |
| 4 or more | 91/105 | 2.4 | 171.3 |

**Per patient vs the v1 general model:** personalising raised false alarms for 6 of 23 patients (chb03, chb08, chb15, chb17, chb23, chb24).
- **chb08:** 37 → 108 per day.
- **chb17:** 0 → 167 per day, though it caught 1/2 seizures instead of 0/2.
- **chb09:** 550 → 0 per day.
- **chb06:** 680 → 0.8 per day, but 8/9 → 6/9 seizures caught.

**Stricter threshold until the 2nd confirmation** (`metrics_v1_snap_more_th0.8_fc*.json`): it only helps a little.

| Threshold until 2nd confirmation | Seizures caught | False alarms per day |
|---|---|---|
| 0.8 (default) | 141 / 168 | 12.2 |
| 0.9 | 140 / 168 | 11.3 |
| 0.95 | 139 / 168 | 10.6 |
| 0.98 | 138 / 168 | 10.1 |

**Wall-clock on LinuxONE for this run** (`timings_more.txt`):

| Step | Time |
|---|---|
| Extra download | 99 s |
| Features | 492 s |
| Experiments: 535 models | 1,219 s |

**Delays in the sections below use the old definition:** they were measured from the back-dated event start, about 5 s earlier than the alert appears. Those runs have no score cache, so they can't be recomputed.

The sections below are the first Snap ML run (2 seizure-free recordings per patient, 186 recordings). They stay for traceability.

## Earlier run: 2 seizure-free recordings per patient (threshold 0.8, Snap ML on the AI accelerator)

| Situation | Seizures caught | False alarms per day | Median seconds to alert |
|---|---|---|---|
| Brand-new patient, general model only | 78 / 191 (41%) | 122.4 | 11 |
| After the doctor confirms seizures (personalised), same later stretches | 142 / 168 (85%) | 14.0 (general model on those same hours: 127.6) | 6 (general: 10.5) |
| Personal model trained only on the patient's earlier recordings | 136 / 164 (83%) | 5.1 | 6 |

Source: `exp_v1_snap.json` → `general` / `confirm` / `past` → `0.8`.

## Snap ML (shipped) vs scikit-learn (earlier), both on LinuxONE, threshold 0.8

| | Snap ML + accelerator (`exp_v1_snap.json`) | scikit-learn HGB (`exp_v1.json`, `exp_v1_confirm.json`) |
|---|---|---|
| New patient: caught, false per day, seconds | 78/191, 122.4, 11 | 77/191, 106.4, 10 |
| Confirm mode, same hours: general | 64/168, 127.6, 10.5 | 64/168, 111.0, 10 |
| Confirm mode, same hours: personalised | 142/168, **14.0**, 6 | 142/168, 16.1, 6 |
| Past-only personal | 136/164, **5.1**, 6 | 139/164, 6.0, 6 |
| Retrain one patient (median / max) | **1.39 s / 3.14 s** (2 threads) | 2.7 s / 3.54 s (1 thread) |
| Full experiment run (586 models) | **1,308 s** | 1,554 s |

## The IBM Integrated Accelerator for AI is really used
**Call count:** an `LD_PRELOAD` shim counts calls into `libzdnn` (`accel_shim.so`, from the app lane).
- One live scoring step for 1 bed made **2** `zdnn_matmul_op` calls, all with status 0 (`zdnn_step_1.json`).
- One step for 100 beds made **7** calls, all with status 0, taking 1.0 ms inside the accelerator operations (`zdnn_step_100.json`).
- The scores carry the label "IBM Integrated Accelerator for AI (Snap ML + zDNN)" (`zdnn_step_*.log`).

**Same answers as the CPU:** the largest difference between accelerator and CPU scores over 34,084 windows is 2.2e-16 (`zdnn_vs_cpu.json`).

**Honest speed note:** for a model this small, the accelerator is slower than Snap ML on the CPU.

| | One window | Batch of 100 |
|---|---|---|
| Accelerator | 0.088 ms | 0.49 ms |
| CPU | 0.009 ms | 0.07 ms |

(`zdnn_vs_cpu.json`.) Both are far below the 1-second budget per bed. Setting `TREE_FORMAT=compress_trees` switches scoring to the CPU with identical outputs.

## (a) Data
- **Source:** CHB-MIT Scalp EEG.
- **Size:** 23 patients, 186 recordings, 191 seizures, 262.5 hours of EEG (`exp_v1_snap.json`).
- **Same patient:** chb01 and chb21 are one person and always sit in the same fold.
- **Excluded:** chb12_28 and chb12_29, because their channels can't be converted to the standard montage.

## (b) Brand-new patient (general model, leave-one-patient-out)
At 0.8:
- caught 78 of 191 seizures;
- 122.44 false alarms per day;
- median 11 s to alert.

## (c) After the doctor confirms seizures (confirm mode, past-only)
**Protocol:**
- Each patient starts on the general model.
- After each seizure ends, plus 60 s for the doctor to confirm, a personal model is retrained on all of that patient's data up to that moment, and used until the next confirmation.
- Only time after the first confirmation is compared. The general model is scored on exactly the same hours (192.4 h, 168 seizures).

**Results at 0.8:**

| Model on the same 192.4 h | Caught | False alarms per day | Median s to alert |
|---|---|---|---|
| General (before) | 64 / 168 (38%) | 127.63 | 10.5 |
| Personalised (after) | 142 / 168 (85%) | 13.97 | 6.0 |

**Learning curve,** by confirmed seizures so far:

| Confirmed so far | Caught | False alarms per day | Hours |
|---|---|---|---|
| 1 | 18 / 23 (78%) | 44.15 | 30.4 |
| 2 to 3 | 33 / 40 (83%) | 16.42 | 55.5 |
| 4 or more | 91 / 105 (87%) | 4.06 | 106.4 |

**Per patient** (`by_patient`): personalisation is not better for everyone.
- **More false alarms after personalising (6 of 23 patients):** chb03 (0 → 21.7 per day), **chb08 (3.9 → 224.8)**, chb16 (13.5 → 18.0), **chb17 (0 → 138.6)**, chb23 (0 → 5.8), chb24 (2.0 → 20.3).
- **Fewer seizures caught (1 patient):** chb06 went from 8/9 to 6/9, while its false alarms fell from 536.9 to 0 per day.
- **Biggest gains:** chb12 (0/32 → 31/32), chb15 (0/19 → 18/19), chb14 (0/7 → 6/7), chb20 (0/7 → 6/7).
- **chb09**, the bed flagged in the live audit: offline, personalising it cut false alarms from 468.3 to 7.8 per day, with 3/3 seizures caught either way.

## (d) Personal model, past-only
At 0.8:
- caught 136 of 164 seizures;
- 5.06 false alarms per day;
- median 6 s.

Recordings with no earlier seizure for that patient aren't scored here.

For reference, leave-one-recording-out personal is optimistic because it learns from later recordings: 169/191, 5.12 per day, 6 s.

## (e) Scale test on LinuxONE (`scale_test_snap.json`)
Real recordings streaming for N beds at once. Features are computed per bed, then one batched accelerator call scores all beds.

| Beds at once | ms per step (all beds) | ms per window | 95th percentile step |
|---|---|---|---|
| 50 | 15.35 | 0.307 | 20.08 ms |
| 100 | 30.83 | 0.308 | 41.58 ms |

- **Sequential scoring,** one model call per bed: 0.625 ms per window at 100 beds.
- **Feature extraction** dominates the step time; the model itself is a small part.
- **Max beds in real time:** about 3,243 by straight-line extrapolation. That is an estimate. We measured up to 100.

## (f) Retraining one patient on LinuxONE (`retrain_timing_snap.json`)
- **What was timed:** exactly what the dashboard's Confirm button does, for 8 patients.
- **Median 1.39 s, max 3.14 s.**
- Each fit used 1,708 to 5,784 training rows.

## (g) Wall-clock on LinuxONE
| Step | Time | File |
|---|---|---|
| Download | 11.51 GB, 187 recordings, 941 s (15.7 min) via PhysioNet's AWS mirror | `linuxone_download.log`, `linuxone_t_*` |
| Features for 186 recordings | 278 s (4.6 min) | `linuxone_features.log` |
| Experiments, Snap ML: 586 models, every mode | 1,308 s (21.8 min) | `timings_snap.txt`, `exp_v1_snap.log` |
| Scale test, Snap ML | 38 s | `timings_snap.txt` |

## Live demo notes
- **Demo beds follow a fixed rule** (`src/engine.py`, `choose_beds`): the first 5 patients by ID that have a recording with 2 or more seizures (their first such recording), plus one seizure-free control recording from the next patient by ID.
- **Live "seconds to alert" are in recording time,** not wall-clock time. The replay runs 10x faster.
- **The dashboard's before/after panel compares unequal time spans,** for example 103 min before vs 4.6 min after. It illustrates the switch. It is not evidence. The evidence is the confirm-mode table above, which uses identical hours.

## Things to say carefully
- **The doctor's confirmation** uses the expert annotation of that seizure, offline and in the demo. That assumes the doctor marks it correctly.
- **Gains are concentrated in patients with many seizures.** After the first confirmation: 78% caught at 44 false alarms per day.
- **Personalisation raised false alarms for 6 of 23 patients.** See (c).
- **False alarms per day are measured on recordings chosen around seizures,** plus 2 seizure-free recordings per patient.
- **The patients are children and young people** with hard-to-treat epilepsy, recorded in hospital after medication was withdrawn.
