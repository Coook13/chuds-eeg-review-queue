# Seizure detector across three populations (threshold 0.8)

A brand-new patient the model has never seen, in each age group. Event-level: a seizure counts as caught if an alert overlaps it; a false alert is an alert with no seizure nearby. Same detector code, same threshold.

| Population | Model trained on | Patients | Seizures | Caught | False alerts per 24 h | Median delay to alert | EEG hours | Features |
|---|---|---|---|---|---|---|---|---|
| Children (CHB-MIT) | other children (leave-one-patient-out) | 23 | 191 | 78 (41%) | 122.44 | 11 s | 262.5 h | v1 |
| Adults (Siena) | children + the other adults | 14 | 47 | 36 (77%) | 22.12 | 18 s | 141.0 h | v2 |
| Newborns (Helsinki) | children + adults + the other newborns | 79 | 315 | 106 (34%) | 41.18 | 32 s | 111.9 h | v2 |

Children: `runs/exp_v1_snap.json` (the newer file; `runs/exp_v1.json` is an older run of the same experiment and gives 77/191 caught, 106.35 false alerts per 24 h, 10 s). Adults and newborns use the v2 features (change from the patient's own last 10 minutes); the children number is only available with v1, so the first row is not like-for-like with the other two. With v1 for all three: adults 21/47 caught, 34.21 false alerts per 24 h; newborns 52/315 caught, 28.96 false alerts per 24 h.

## What makes each row optimistic or pessimistic

- **Children, optimistic:** training and test come from the same hospital, amplifier and bipolar montage, and the 186 recordings are the seizure-rich files plus a few seizure-free ones, not whole days. **Pessimistic:** only the v1 features were run here; v2 was not tried on this data.
- **Adults, optimistic:** the recordings are trimmed around the seizure, so the background is calm and the false-alert rate per day is probably too low for real monitoring; with 47 seizures in 14 patients one seizure moves the result 2 points. **Pessimistic:** 6 of 47 seizures last 20 s or less, which the event logic (5 smoothed windows in a row) can barely catch, and false alerts pile up in a few patients (118 of 130 alerts in four patients).
- **Newborns, optimistic:** all 79 come from one hospital and one amplifier, so the other newborns in training share the test newborn's hardware, and the truth is the consensus of 3 experts, the seizures all of them agreed on. Recordings are continuous (median 74 min, not trimmed), so false alerts per day are closer to real monitoring than Siena's, but they are still short NICU sessions, not days. **Pessimistic:** alerts on seconds that only one or two experts marked count as false against the consensus truth; on the 22 newborns no expert marked, the false-alert rate is 25.16 per 24 h (30 alerts in 28.6 h).
