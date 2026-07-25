# AI Healthcare Co-Pilot 🩺
### Hackathon Theme: "Build an AI Co-Pilot for Industry" — Healthcare Assistant Challenge

An AI Co-Pilot that helps clinicians triage patients by combining **three data
modalities**, deep learning, **explainable AI**, and a **human-in-the-loop
(HITL)** approval workflow — with automated PDF report generation.

## ✅ Requirements coverage

| Requirement | How it's met |
|---|---|
| 3+ data modalities | Image (chest X-ray), Tabular (cardiac risk), Text (symptom notes) |
| Predictive DL models | CNN (pneumonia) + ANN (cardiac risk) |
| LLM usage (non-predictive) | Optional Anthropic API call to summarize notes — never used to diagnose |
| Human-in-the-Loop | Reviewer must Approve/Modify/Reject before a report is finalized; every decision is logged |
| Explainable AI | Grad-CAM (CNN) + SHAP (ANN) + confidence scores |
| Working web app | Streamlit, 4-tab workflow (`app.py`) |
| Downloadable report | One-click PDF via `reportlab` |
| Business model | See `BUSINESS_MODEL.md` |

## 🏗️ Architecture

```
User → Streamlit UI (app.py)
         ├─ Tab 1: Image  → CNN (cnn_pneumonia.py) → Grad-CAM
         ├─ Tab 2: Tabular → ANN (tabular_heart.py) → SHAP
         ├─ Tab 3: Text    → keyword extractor → optional LLM summary
         └─ Tab 4: HITL review → clinician decision → PDF report
```

Both models auto-detect real data if present (`data/chest_xray/`,
`data/heart_disease_raw/processed.cleveland.data`) and fall back to synthetic
data otherwise — no manual config needed either way.

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Real data is already bundled in this repo, so both models train on it
automatically on first run. Trained weights are cached to `saved_models/`.

**Optional LLM summarization:** set `ANTHROPIC_API_KEY`. Without it, the app
uses a transparent rule-based summary instead — no functionality is lost.

> ⚠️ `data/heart.csv` is a *different* dataset (Kaggle "Heart Failure
> Prediction", incompatible schema) — not used by the app. Use
> `load_uci_processed_cleveland()` for the real UCI data, as `app.py` already does.

## 📁 Project structure

```
├── app.py                    # Streamlit app (entry point)
├── models/                   # CNN + ANN
├── utils/                    # Grad-CAM, SHAP, text triage, PDF report
├── data/                     # Real + synthetic datasets
├── saved_models/             # Trained weights (auto-generated)
└── BUSINESS_MODEL.md
```

## ⚠️ Disclaimer
Decision-support prototype only — **not a certified medical device**. Every
output requires clinician review (enforced by the HITL step).