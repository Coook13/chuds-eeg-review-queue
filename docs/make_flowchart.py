"""Draws docs/flowchart.svg: the full EEG Review Queue architecture on IBM LinuxONE."""
from pathlib import Path

W, H = 2010, 1430
BLUE, INK, MUTED, GREEN, PALE, PURPLE = "#0f62fe", "#161616", "#525252", "#198038", "#edf5ff", "#6929c4"
BW, BH, GAP, X0 = 214, 96, 22, 62
out = []
cx = lambda c: X0 + (c - 1) * (BW + GAP)  # left x of column c (1..8)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(c, y, title, lines, color=INK, span=1, fill="#ffffff"):
    x, w = cx(c), BW * span + GAP * (span - 1)
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="9" fill="{fill}" stroke="{color}" stroke-width="2"/>')
    out.append(f'<text x="{x + w / 2}" y="{y + 23}" text-anchor="middle" font-weight="700" font-size="15.5" fill="{color}">{esc(title)}</text>')
    for i, t in enumerate(lines):
        out.append(f'<text x="{x + w / 2}" y="{y + 44 + i * 18}" text-anchor="middle" font-size="13" fill="{MUTED}">{esc(t)}</text>')


def path(pts, color=INK, dash=False):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    da = ' stroke-dasharray="6 5"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{da} marker-end="url(#m{color[1:]})"/>')


def row(y, n1, n2, color=INK):  # arrows between consecutive columns n1..n2
    for c in range(n1, n2):
        path([(cx(c) + BW, y + BH / 2), (cx(c + 1) - 3, y + BH / 2)], color)


def lane(y, text, color, c=1):
    out.append(f'<text x="{cx(c)}" y="{y - 10}" font-weight="800" font-size="19" fill="{color}">{esc(text)}</text>')


out.append(f'<text x="{W / 2}" y="44" text-anchor="middle" font-weight="800" font-size="30" fill="{INK}">EEG Review Queue: full architecture on IBM LinuxONE</text>')
out.append(f'<text x="{W / 2}" y="72" text-anchor="middle" font-size="16" fill="{MUTED}">Diagnose, Detect, Predict, Learn, plus IBM Granite handover notes. Every box inside the blue frame runs on one IBM LinuxONE mainframe.</text>')

# open data
y = 120
lane(y, "Open EEG data (PhysioNet, Zenodo), downloaded straight onto the mainframe", MUTED)
box(1, y, "CHB-MIT", ["23 children, 321 recordings", "477 h, 191 seizures"], span=2)
box(3, y, "Siena Scalp EEG", ["14 adults, 47 seizures", "mixed into general model"], span=2)
box(5, y, "Helsinki neonatal", ["79 newborns", "third population tested"], span=2)
box(7, y, "Guinea-Bissau / Nigeria", ["low-cost Emotiv headset", "epilepsy vs control, 5 min"], span=2)

# mainframe frame
FY = 262
out.insert(0, f'<rect x="30" y="{FY}" width="{W - 60}" height="1008" rx="18" fill="{PALE}" stroke="{BLUE}" stroke-width="3"/>')
out.append(f'<text x="{X0}" y="{FY + 28}" font-weight="800" font-size="18" fill="{BLUE}">IBM LinuxONE (s390x, IBM machine type 9175, on-chip IBM Integrated Accelerator for AI)</text>')

# lane 1 train
y1 = FY + 75
lane(y1, "Train (offline, on the mainframe)", INK)
box(1, y1, "Read EDF (MNE)", ["18 bipolar channels", "resample, label seizures"])
box(2, y1, "Windowing", ["2 s window", "1 s step"])
box(3, y1, "Features v1 (234)", ["band power 5 bands,", "line length, Hjorth"])
box(4, y1, "Features v2 (520)", ["vs own last 10 min,", "cross-channel"])
box(5, y1, "Leave-one-patient-out", ["model never sees the", "patient it is tested on"])
box(6, y1, "IBM Snap ML", ["BoostingMachine", "Classifier"], BLUE)
box(7, y1, "optimize_trees()", ["tree_format =", "zdnn_tensors"], BLUE)
box(8, y1, "Model store", ["23 general models,", "personal models (joblib)"])
row(y1, 1, 8)

# lane 2 detect
y2 = y1 + 165
lane(y2, "2  Detect (real time, every second, per bed)", BLUE)
box(1, y2, "Bed streams", ["6 beds replayed live,", "scale-tested to 100"])
box(2, y2, "Ring buffer", ["last 2 s of EEG", "+ 10 min baseline"])
box(3, y2, "Feature extractor", ["same code as training", "v2 general / v1 personal"])
box(4, y2, "Model router", ["general model until a", "confirm, then personal"])
box(5, y2, "zDNN inference", ["on the on-chip AI", "accelerator (NNPA)"], BLUE)
box(6, y2, "Smoothing", ["mean of last", "5 scores"])
box(7, y2, "Event logic", [">= 0.8 for 5 s opens", "an alert (fixed in advance)"])
box(8, y2, "Rules engine", ["review: any suspected", "urgent: 5+ min or 2 in 1 h"])
row(y2, 1, 8)
gy = y2 - 36
path([(cx(8) + BW / 2, y1 + BH), (cx(8) + BW / 2, gy), (cx(4) + BW / 2, gy), (cx(4) + BW / 2, y2 - 3)], BLUE)

# lane 3 learn + predict
y3 = y2 + 165
lane(y3, "4  Learn (doctor in the loop)", GREEN)
lane(y3, "3  Predict", PURPLE, 5)
box(1, y3, "Doctor's answer", ["Confirm seizure or", "Not a seizure"], GREEN)
box(2, y3, "Label store", ["confirmed = positive,", "rejected = negative"], GREEN)
box(3, y3, "Retrain personal model", ["IBM Snap ML, this patient", "1.3 to 3.1 s"], GREEN)
box(4, y3, "Hot-swap", ["bed switches model,", "other beds keep going"], GREEN)
row(y3, 1, 4, GREEN)
path([(cx(4) + BW / 2, y3 - 3), (cx(4) + BW / 2, y2 + BH + 3)], GREEN)
box(5, y3, "Seizure history", ["count so far,", "time since last"], PURPLE)
box(6, y3, "Risk model", ["logistic regression", "AUC 0.754 vs 0.716"], PURPLE)
box(7, y3, "Next-hour risk", ["who to watch closely", "(never dosing)"], PURPLE)
row(y3, 5, 7, PURPLE)
path([(cx(7) + 30, y2 + BH), (cx(7) + 30, y3 - 34), (cx(5) + BW - 20, y3 - 34), (cx(5) + BW - 20, y3 - 3)], PURPLE)

# lane 4 serve
y4 = y3 + 165
lane(y4, "Serve: Flask app on the mainframe (port 8050)", INK, 2)
box(1, y4, "/api/confirm, /reject", ["doctor's answer in,", "retrain triggered"], GREEN)
box(2, y4, "/api/ack + audit log", ["who, when, which bed,", "every action logged"])
box(3, y4, "/api/evidence", ["EEG clip, model spans,", "doctor-marked seizures"])
box(4, y4, "/api/handover", ["bed numbers to Granite,", "note back"])
box(5, y4, "/api/diagnose", ["5-min EEG in,", "likelihood + reasons out"])
box(6, y4, "/system", ["live /proc/sysinfo,", "zDNN call counters"], BLUE)
box(7, y4, "/api/metrics", ["latency per window,", "retrain timings"])
box(8, y4, "/api/beds", ["queue: urgent first,", "status + risk per bed"])
path([(cx(1) + BW / 2, y4 - 3), (cx(1) + BW / 2, y3 + BH + 3)], GREEN)
path([(cx(8) + BW / 2, y2 + BH), (cx(8) + BW / 2, y4 - 3)])
path([(cx(7) + BW, y3 + BH / 2), (cx(8) + 40, y3 + BH / 2), (cx(8) + 40, y4 - 3)], PURPLE)

# lane 5 diagnose + generate
y5 = y4 + 165
lane(y5, "1  Diagnose", BLUE)
lane(y5, "Generate (IBM Granite)", BLUE, 6)
box(1, y5, "5-min resting EEG", ["14-channel Emotiv,", "eyes open / closed"])
box(2, y5, "Diagnose features", ["band power per channel", "+ signal quality"])
box(3, y5, "Classifier", ["cross-validated AUC", "0.898 / 0.935"])
box(4, y5, "Epilepsy likely?", ["yes / no + reasons,", "doctor decides"], BLUE)
row(y5, 1, 4)
box(5, y5, "Bed numbers", ["alerts, durations,", "confirms, risk"])
box(6, y5, "Safety prompt", ["no drug, no dose,", "no diagnosis, <=110 words"])
box(7, y5, "IBM Granite 3.3 2B", ["llama.cpp built for s390x,", "big-endian GGUF, :8081"], BLUE)
box(8, y5, "Handover note", ["flagged for clinician", "review (template fallback)"])
row(y5, 5, 8)
path([(cx(5) + 30, y4 + BH), (cx(5) + 30, y5 - 40), (cx(1) + BW - 40, y5 - 40), (cx(1) + BW - 40, y5 - 3)])
path([(cx(4) + BW / 2, y4 + BH), (cx(4) + BW / 2, y5 - 30), (cx(5) + BW - 30, y5 - 30), (cx(5) + BW - 30, y5 - 3)])

# lane 6 platform
y6 = y5 + 150
lane(y6, "Platform layer", BLUE)
box(1, y6, "s390x, big-endian", ["Ubuntu 24.04", "3 CPUs, 7.8 GB RAM"], BLUE, 2)
box(3, y6, "Integrated Accelerator for AI", ["nnpa CPU flag, libzdnn,", "zDNN calls verified, 0 errors"], BLUE, 2)
box(5, y6, "Python stack", ["Snap ML 1.17.2, scikit-learn,", "NumPy, SciPy, MNE, Flask"], BLUE, 2)
box(7, y6, "Scale + speed", ["100 beds at 0.31 ms per window,", "app under 1 ms per window"], BLUE, 2)

# outside: clinician + judges
y7 = FY + 1008 + 40
box(1, y7, "Clinician view  /", ["queue, EEG evidence, risk,", "Acknowledge, Confirm, Not a seizure, Handover"], INK, 3)
box(4, y7, "Intake  /intake", ["Diagnose a new person", "from a 5-min EEG"], INK, 2)
box(6, y7, "IBM Z system view  /system", ["machine facts read live, accelerator,", "scale test, retrain times, zDNN calls"], BLUE, 3)
out.append(f'<text x="{W - 40}" y="{y7 - 12}" text-anchor="end" font-size="14" fill="{MUTED}">Browsers outside the mainframe, over an SSH tunnel</text>')

defs = "".join(f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
               for c in (INK, GREEN, MUTED, BLUE, PURPLE))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">'
       f'<defs>{defs}</defs><rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(out) + "</svg>")
Path(__file__).with_name("flowchart.svg").write_text(svg, encoding="utf-8")
print("wrote flowchart.svg")
