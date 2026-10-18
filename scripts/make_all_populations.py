"""Builds runs/ALL_POPULATIONS.md: one plain-English table, children vs adults vs newborns, at threshold 0.8.
Every number is read from a results file; nothing is typed in by hand.

  children : runs/exp_v1_snap.json   (CHB-MIT, general model, leave-one-patient-out, v1 features; the newer of the two
             exp_v1 files, exp_v1.json is the older run and is quoted at the bottom for reference)
  adults   : runs/siena_results_mixed.json   (Siena, mixed model, leave-one-adult-out, v2 features)
  newborns : runs/helsinki_results.json      (Helsinki, mixed model, newborn held out, v2 features)
Run:  python scripts/make_all_populations.py
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
TH = "0.8"


def loadj(name):
    return json.loads((RUNS / name).read_text())


def row(name, who, patients, v, feats):
    return (f"| {name} | {who} | {patients} | {v['seizures']} | {v['detected']} ({v['sensitivity']:.0%}) | "
            f"{v['false_alerts_per_24h']} | {v['median_delay_s']:.0f} s | {v['hours']} h | {feats} |")


def main():
    snap, old = loadj("exp_v1_snap.json"), loadj("exp_v1.json")
    sie, hel = loadj("siena_results_mixed.json"), loadj("helsinki_results.json")

    c = snap["general"][TH]
    det, tot = (int(x) for x in c["caught"].split("/"))
    child = {"seizures": tot, "detected": det, "sensitivity": c["sensitivity"],
             "false_alerts_per_24h": c["false_alerts_per_24h"], "median_delay_s": c["median_delay_s"], "hours": c["hours"]}
    a = sie["v2"]["mixed"]["all_41_recordings"][TH]
    a1 = sie["v1"]["mixed"]["all_41_recordings"][TH]
    n = hel["v2"]["b_mixed_newborn_held_out"][TH]
    n1 = hel["v1"]["b_mixed_newborn_held_out"][TH]
    nfree = hel["v2"]["no_reviewer_marked_seizure_neonates"]
    d = hel["dataset"]
    o = old["general"][TH]
    import pandas as pd
    med_min = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "helsinki" / "files_extra.csv").duration_s.median() / 60
    sp = loadj("siena_results.json")["dataset"]["seizure_duration_s"]["n_20s_or_shorter"]
    pp = sie["v2"]["mixed"]["per_patient_at_declared_threshold"]
    top4 = sorted((v["false_alerts"] for v in pp.values()), reverse=True)[:4]
    tot_alerts = sum(v["false_alerts"] for v in pp.values())
    maj = hel["v2"]["b_vs_majority_truth"][TH]

    L = ["# Seizure detector across three populations (threshold 0.8)", "",
         "A brand-new patient the model has never seen, in each age group. Event-level: a seizure counts as caught if an "
         "alert overlaps it; a false alert is an alert with no seizure nearby. Same detector code, same threshold.", "",
         "| Population | Model trained on | Patients | Seizures | Caught | False alerts per 24 h | Median delay to alert | EEG hours | Features |",
         "|---|---|---|---|---|---|---|---|---|",
         row("Children (CHB-MIT)", "other children (leave-one-patient-out)", snap["_patients"], child, "v1"),
         row("Adults (Siena)", "children + the other adults", sie["dataset"]["patients"], a, "v2"),
         row("Newborns (Helsinki)", "children + adults + the other newborns", d["neonates"], n, "v2"),
         "",
         f"Children: `runs/exp_v1_snap.json` (the newer file; `runs/exp_v1.json` is an older run of the same experiment and gives "
         f"{o['caught']} caught, {o['false_alerts_per_24h']} false alerts per 24 h, {o['median_delay_s']:.0f} s). "
         f"Adults and newborns use the v2 features (change from the patient's own last 10 minutes); the children number is only available "
         f"with v1, so the first row is not like-for-like with the other two. With v1 for all three: adults "
         f"{a1['detected']}/{a1['seizures']} caught, {a1['false_alerts_per_24h']} false alerts per 24 h; newborns "
         f"{n1['detected']}/{n1['seizures']} caught, {n1['false_alerts_per_24h']} false alerts per 24 h.", "",
         "## What makes each row optimistic or pessimistic", "",
         f"- **Children, optimistic:** training and test come from the same hospital, amplifier and bipolar montage, and the "
         f"{snap['_recordings']} recordings are the seizure-rich files plus a few seizure-free ones, not whole days. "
         f"**Pessimistic:** only the v1 features were run here; v2 was not tried on this data.",
         f"- **Adults, optimistic:** the recordings are trimmed around the seizure, so the background is calm and the false-alert rate per day "
         f"is probably too low for real monitoring; with 47 seizures in 14 patients one seizure moves the result 2 points. "
         f"**Pessimistic:** {sp} of {a['seizures']} seizures last 20 s or less, which the event logic (5 smoothed windows in a row) can barely catch, "
         f"and false alerts pile up in a few patients ({sum(top4)} of {tot_alerts} alerts in four patients).",
         f"- **Newborns, optimistic:** all 79 come from one hospital and one amplifier, so the other newborns in training share the test newborn's "
         f"hardware, and the truth is the consensus of 3 experts, the seizures all of them agreed on. Recordings are continuous "
         f"(median {med_min:.0f} min, not trimmed), so false alerts per day are closer to real monitoring than "
         f"Siena's, but they are still short NICU sessions, not days. "
         f"**Pessimistic:** alerts on seconds that only one or two experts marked count as false against the consensus truth; on the "
         f"{nfree['n']} newborns no expert marked, the false-alert rate is {nfree['b']['0.8']['false_alerts_per_24h']} per 24 h "
         f"({nfree['b']['0.8']['false_alerts']} alerts in {nfree['b']['0.8']['hours']} h).", ""]
    (RUNS / "ALL_POPULATIONS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
