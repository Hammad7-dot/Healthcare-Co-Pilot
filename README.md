# AI Healthcare Co-Pilot 🩺
### Hackathon Theme: "Build an AI Co-Pilot for Industry" — Healthcare Assistant Challenge

An AI Co-Pilot that helps clinicians triage patients by combining **three data
modalities**, a **deep learning predictive model**, **explainable AI**, and a
**human-in-the-loop (HITL)** approval workflow — packaged as a working web app
with automated PDF report generation.

## ✅ Mandatory requirements coverage

| Requirement | How it's met |
|---|---|
| 3+ data modalities | **Image** (chest X-ray), **Tabular** (cardiac risk factors), **Text** (free-text symptom notes) |
| Predictive deep learning model | **CNN** for pneumonia detection + **ANN** for cardiac risk |
| LLM usage (non-predictive) | Optional Anthropic API call to *summarize* the text note into a clinical narrative — never used to make the diagnosis |
| Human-in-the-Loop | Reviewer tab: clinician must **Approve / Modify / Reject** before a report is finalized; every decision is logged in an audit trail |
| Explainable AI | **Grad-CAM** heatmaps for the CNN, **SHAP** feature importance for the ANN, confidence scores for every prediction |
| Working web application | `streamlit` app (`app.py`) — 4-tab workflow |
| Downloadable report | One-click **PDF** report (imaging + tabular + text + HITL decision) via `reportlab` |
| Business model | See `BUSINESS_MODEL.md` |

## 🏗️ Architecture

```
User → Streamlit UI (app.py)
         ├─ Tab 1: Image upload → CNN (cnn_pneumonia.py) → Grad-CAM (gradcam.py)
         ├─ Tab 2: Tabular form  → ANN (tabular_heart.py) → SHAP (shap_explain.py)
         ├─ Tab 3: Text note     → keyword extractor → optional LLM summary (text_triage.py)
         └─ Tab 4: HITL review  → clinician decision → PDF report (report_generator.py)
```

- **CNN**: 3-conv-block Keras model predicting NORMAL vs. PNEUMONIA from 64×64 grayscale X-rays.
- **ANN**: 2-hidden-layer Keras model predicting cardiac risk from 13 clinical features (same schema as UCI Heart Disease).
- **Text**: rule-based symptom/urgency keyword extractor, feeding an optional LLM call for narrative summarization (report generation only, never prediction — per hackathon rules).
- **HITL**: every AI output is provisional until a named reviewer logs Approve/Modify/Reject with notes; this decision is embedded in the final PDF and an in-session audit trail.

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, both models auto-train on **synthetic data** (~10-20 seconds on CPU) so the whole app works immediately with zero external downloads. Trained weights are cached to `saved_models/` for subsequent runs.

## 🔄 Swapping in real datasets (for your actual submission)

**Chest X-rays** (Kaggle "Chest X-Ray Pneumonia" — `paultimothymooney/chest-xray-pneumonia`):
1. Download & unzip into `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`
2. In `models/cnn_pneumonia.py`, call `train_from_directory("data/chest_xray")` instead of `train_on_synthetic()`

**Cardiac risk** (UCI Heart Disease — `archive.ics.uci.edu/dataset/45/heart+disease`):
1. Download the CSV to `data/heart.csv` (with the 13 standard feature columns + `target`)
2. In `models/tabular_heart.py`, call `train(csv_path="data/heart.csv")`

**LLM summarization** (optional): set the `ANTHROPIC_API_KEY` environment variable. Without it, the app falls back to a transparent rule-based summary — no functionality is lost.

## 📁 Project structure

```
healthcare_copilot/
├── app.py                     # Streamlit web app (main entry point)
├── models/
│   ├── cnn_pneumonia.py        # CNN for X-ray classification
│   └── tabular_heart.py        # ANN for cardiac risk
├── utils/
│   ├── gradcam.py               # Grad-CAM explainability
│   ├── shap_explain.py          # SHAP explainability
│   ├── text_triage.py           # Text/NLP symptom extraction + optional LLM summary
│   └── report_generator.py      # PDF report builder
├── data/
│   └── synthetic_data.py        # Offline demo data generators (swap for real data — see above)
├── saved_models/                # Trained model weights (auto-generated)
├── requirements.txt
├── README.md
└── BUSINESS_MODEL.md
```

## ⚠️ Disclaimer
This is a decision-support prototype for the hackathon. It is **not a certified medical device** and every output requires clinician review (enforced by the HITL step) before any clinical action.
