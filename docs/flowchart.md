# Flowchart: EEG Review Queue

Same picture as `flowchart.svg`. Everything inside the LinuxONE box runs on one IBM LinuxONE VM. The clinician's browser is outside and connects through an SSH tunnel.

```mermaid
flowchart LR
    subgraph LX["IBM LinuxONE (s390x), Ubuntu 24.04, 3 CPUs, 7.8 GB RAM"]
        direction LR
        EEG["1. EEG streams<br/>CHB-MIT recordings on the VM,<br/>replayed live, one per bed<br/>186 recordings, 23 patients"]
        FEAT["2. Features every 1 s<br/>2 s window, 18 channels<br/>234 features"]
        subgraph MODEL["3. Seizure model, one per bed"]
            GEN["General model<br/>trained only on other patients<br/>every bed starts here"]
            PER["Personal model<br/>this patient only<br/>after a confirmed seizure"]
        end
        RULES["4. Event logic + escalation rules<br/>score 0 to 1, smoothed, alert after 5 in a row<br/>any suspected seizure: review<br/>5+ min or 2+ per hour: urgent<br/>urgent never hidden, clinician can edit"]
        APP["5. Web app on the VM (Flask)<br/>Review queue<br/>Evidence viewer, flagged span shaded<br/>Acknowledge log<br/>Before/after table per bed"]
        RETRAIN["6. Retrain personal model on LinuxONE<br/>confirmed seizure + recording so far<br/>+ the patient's earlier recordings<br/>takes [RETRAIN SECONDS ON LINUXONE] s,<br/>other beds keep streaming"]
        GRANITE["7. IBM Granite 3.3 2B<br/>llama.cpp built for s390x, big-endian model<br/>writes the handover note<br/>never suggests treatment"]
    end

    BROWSER["Clinician's browser<br/>laptop, outside LinuxONE"]
    CONFIRM{{"Clinician confirms seizure"}}

    EEG --> FEAT --> GEN
    FEAT --> PER
    GEN -->|score| RULES
    PER -->|score| RULES
    RULES --> APP
    APP <-->|SSH tunnel| BROWSER
    BROWSER -->|Acknowledge| APP
    BROWSER --> CONFIRM
    CONFIRM --> RETRAIN
    EEG -. "earlier recordings + recording so far" .-> RETRAIN
    RETRAIN -->|bed switches to personal model| PER
    APP -->|bed numbers| GRANITE
    GRANITE -->|handover note| APP

    classDef personal fill:#e7f5f3,stroke:#0b7a75,color:#0b4f4b
    classDef gen fill:#f4effc,stroke:#6929c4,color:#3b1f66
    classDef outside fill:#f6f7f9,stroke:#5b6475,color:#1d2433
    class PER,RETRAIN,CONFIRM personal
    class GRANITE gen
    class BROWSER outside
    style LX fill:#f5f8ff,stroke:#0f62fe,stroke-width:2px,stroke-dasharray:6 4
```

How it maps to the IBM labs:

- Hybrid multi-model is the general model, the personal model and the escalation rules together deciding each bed's status.
- Predictive plus generative is the seizure detector flagging events and Granite writing the handover note from them.

Note: in the real app the confirm request goes from the browser through the web app to the retrain step. The diagram draws "Clinician confirms seizure" as its own step so the loop is easy to see.
