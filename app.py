"""
AI Healthcare Co-Pilot -- main Streamlit application.

Run with: streamlit run app.py

Improvements over the original version:
  - Patient ID sanitized before being used to build file paths (no path traversal)
  - Model loading wrapped in try/except with a friendly error banner instead of a crash
  - SHAP background data cached so it isn't rebuilt on every form submit
  - Generated PDF report is cleaned up after being served, and read via a fresh
    buffer instead of an unclosed file handle
  - Audit trail (HITL decisions) persisted to a local CSV so it survives app restarts,
    not just kept in st.session_state
  - Clear on-screen indicator of whether the LLM path or rule-based fallback was used
    for the text-note summary (matters for explainability/audit)
"""
import os
import re
import io
import sys
import tempfile
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_data import generate_synthetic_xray, generate_synthetic_heart_data
from models import cnn_pneumonia, tabular_heart
from utils import gradcam, text_triage, report_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthcare_copilot")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UCI_HEART_PATH = os.path.join(APP_DIR, "data", "heart_disease_raw", "processed.cleveland.data")
AUDIT_LOG_PATH = os.path.join(APP_DIR, "data", "audit_trail.csv")

st.set_page_config(page_title="AI Healthcare Co-Pilot", page_icon="🩺", layout="wide")


def sanitize_patient_id(raw_id: str) -> str:
    """Keep patient IDs filesystem-safe: alphanumerics, dash, underscore only."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", raw_id or "")
    return cleaned[:64] or "PT-UNKNOWN"


def detect_image_type(file_bytes: bytes) -> str | None:
    """
    Minimal magic-byte sniffing for PNG/JPEG, replacing the stdlib `imghdr`
    module (removed in Python 3.13). Returns 'png', 'jpeg', or None.
    """
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if file_bytes[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return None


@st.cache_resource(show_spinner="Loading / training models (first run only)...")
def load_models():
    cnn = cnn_pneumonia.load_or_train()
    if os.path.exists(UCI_HEART_PATH):
        df = tabular_heart.load_uci_processed_cleveland(UCI_HEART_PATH)
        ann, scaler = tabular_heart.train(epochs=40, df=df)
    else:
        ann, scaler = tabular_heart.load_or_train()
    return cnn, ann, scaler


@st.cache_data(show_spinner=False)
def get_shap_background():
    """Cache the SHAP background dataset so it isn't rebuilt on every submit."""
    if os.path.exists(UCI_HEART_PATH):
        return tabular_heart.load_uci_processed_cleveland(UCI_HEART_PATH)
    return generate_synthetic_heart_data(60)


def append_audit_row(row: dict) -> None:
    """Persist a HITL decision to a local CSV so the audit trail survives restarts."""
    df_row = pd.DataFrame([row])
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    header = not os.path.exists(AUDIT_LOG_PATH)
    try:
        df_row.to_csv(AUDIT_LOG_PATH, mode="a", header=header, index=False)
    except OSError as e:
        logger.warning("Could not persist audit trail: %s", e)


def load_audit_trail() -> pd.DataFrame:
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            return pd.read_csv(AUDIT_LOG_PATH)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


st.title("🩺 AI Healthcare Co-Pilot")
st.caption(
    "Multimodal AI decision-support: chest X-ray imaging (CNN), cardiac risk "
    "(tabular ANN), and patient notes (text/NLP) -- with explainability and "
    "human-in-the-loop review. Demo runs on synthetic data; see README to plug in "
    "real Kaggle/UCI datasets."
)

try:
    cnn_model, ann_model, scaler = load_models()
    models_ready = True
except Exception as e:
    logger.exception("Model loading failed")
    st.error(
        f"Models failed to load or train: {e}\n\n"
        "The app cannot run predictions until this is resolved. Check that "
        "dependencies in requirements.txt are installed and that any dataset "
        "paths referenced in the README exist."
    )
    models_ready = False
    st.stop()

if "hitl_log" not in st.session_state:
    st.session_state.hitl_log = []

raw_patient_id = st.text_input("Patient ID", value="PT-0001")
patient_id = sanitize_patient_id(raw_patient_id)
if patient_id != raw_patient_id:
    st.caption(f"Patient ID sanitized to `{patient_id}` (letters, numbers, `-`, `_` only).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Imaging (CNN)", "2. Cardiac Risk (Tabular ANN)", "3. Patient Notes (Text)", "4. Review & Report"]
)

# ---------------- TAB 1: IMAGING ----------------
with tab1:
    st.subheader("Chest X-Ray Pneumonia Screening")
    st.write("Upload a chest X-ray, or generate a synthetic demo sample.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate synthetic demo X-ray"):
            imgs, labels = generate_synthetic_xray(1)
            st.session_state.xray_img = imgs[0]
            st.session_state.xray_true_label = int(labels[0])
            st.session_state.xray_is_uploaded = False
    with col2:
        uploaded = st.file_uploader("Or upload a grayscale X-ray (png/jpg)", type=["png", "jpg", "jpeg"])
        if uploaded:
            MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
            file_bytes = uploaded.getvalue()
            if len(file_bytes) > MAX_UPLOAD_BYTES:
                st.error(f"File too large ({len(file_bytes) / 1e6:.1f} MB). Max allowed is 8 MB.")
            elif detect_image_type(file_bytes) is None:
                st.error(
                    "This doesn't look like a valid PNG/JPEG file (checked by content, "
                    "not just the file extension). Please upload a genuine image file."
                )
            else:
                try:
                    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((64, 64))
                    arr = np.array(img).astype(np.float32) / 255.0
                    st.session_state.xray_img = arr[..., np.newaxis]
                    st.session_state.xray_true_label = None
                    st.session_state.xray_is_uploaded = True
                except Exception as e:
                    st.error(f"Could not read the uploaded image: {e}")

    if st.session_state.get("xray_is_uploaded") and cnn_pneumonia.get_training_source() == "synthetic":
        st.warning(
            "⚠️ The currently loaded CNN was trained on **synthetic** demo data only "
            "(no real X-ray dataset found in `data/chest_xray/`). Predictions on this "
            "real, uploaded image are **not clinically meaningful** — the model has never "
            "seen a real chest X-ray. Train on the real Kaggle dataset (see README) before "
            "trusting predictions on real images.",
            icon="⚠️",
        )

    if "xray_img" in st.session_state:
        image = st.session_state.xray_img
        try:
            result = cnn_pneumonia.predict(cnn_model, image)
            st.session_state.xray_result = result

            heatmap = gradcam.make_gradcam_heatmap(image, cnn_model)
            overlay = gradcam.overlay_heatmap(image[:, :, 0], heatmap)

            c1, c2 = st.columns(2)
            with c1:
                st.image(image[:, :, 0], caption="Input X-ray", clamp=True, width=280)
            with c2:
                fig, ax = plt.subplots()
                ax.imshow(overlay)
                ax.axis("off")
                ax.set_title(f"Grad-CAM: regions driving the '{result['label']}' prediction")
                st.pyplot(fig)
                fig_path = os.path.join(tempfile.gettempdir(), f"{patient_id}_gradcam_overlay.png")
                fig.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)
                st.session_state.xray_image_path = fig_path

            st.metric("Prediction", result["label"], f"{result['confidence']*100:.1f}% confidence")
            if st.session_state.get("xray_true_label") is not None:
                st.caption(
                    f"(synthetic ground truth label: "
                    f"{'PNEUMONIA' if st.session_state.xray_true_label else 'NORMAL'})"
                )
        except Exception as e:
            logger.exception("Imaging prediction failed")
            st.error(f"Imaging prediction failed: {e}")

# ---------------- TAB 2: TABULAR ----------------
with tab2:
    st.subheader("Cardiac Risk Assessment")
    if os.path.exists(UCI_HEART_PATH):
        demo_df = (
            tabular_heart.load_uci_processed_cleveland(UCI_HEART_PATH)
            .sample(1, random_state=None)
            .reset_index(drop=True)
        )
    else:
        demo_df = generate_synthetic_heart_data(1)
    defaults = demo_df.iloc[0].to_dict()

    with st.form("heart_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, int(defaults["age"]))
        sex_options = [1, 0]
        sex = c1.selectbox(
            "Sex", sex_options, format_func=lambda x: "Male" if x == 1 else "Female",
            index=sex_options.index(int(defaults["sex"]))
        )
        cp_options = [1, 2, 3, 4]
        cp = c1.selectbox(
            "Chest pain type (1-4)", cp_options,
            index=cp_options.index(int(defaults["cp"])) if int(defaults["cp"]) in cp_options else 0
        )
        trestbps = c2.number_input("Resting BP (mmHg)", 80, 220, int(defaults["trestbps"]))
        chol = c2.number_input("Cholesterol (mg/dl)", 100, 600, int(defaults["chol"]))
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], index=int(defaults["fbs"]))
        restecg = c3.selectbox("Resting ECG (0-2)", [0, 1, 2], index=int(defaults["restecg"]))
        thalach = c3.number_input("Max heart rate achieved", 60, 220, int(defaults["thalach"]))
        exang = c3.selectbox("Exercise-induced angina", [0, 1], index=int(defaults["exang"]))
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, float(defaults["oldpeak"]))
        slope_options = [1, 2, 3]
        slope = st.selectbox(
            "Slope of peak exercise ST (1-3)", slope_options,
            index=slope_options.index(int(defaults["slope"])) if int(defaults["slope"]) in slope_options else 0
        )
        ca_options = [0, 1, 2, 3]
        ca = st.selectbox(
            "Number of major vessels (0-3)", ca_options,
            index=ca_options.index(int(defaults["ca"])) if int(defaults["ca"]) in ca_options else 0
        )
        thal_options = [3, 6, 7]
        thal = st.selectbox(
            "Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)", thal_options,
            index=thal_options.index(int(defaults["thal"])) if int(defaults["thal"]) in thal_options else 0
        )
        submitted = st.form_submit_button("Run cardiac risk model")

        if submitted:
            patient_dict = dict(
                age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
            )
            try:
                result = tabular_heart.predict(ann_model, scaler, patient_dict)
                st.session_state.heart_result = result
                st.session_state.patient_dict = patient_dict
                st.metric("Prediction", result["label"], f"{result['confidence']*100:.1f}% confidence")
            except Exception as e:
                logger.exception("Cardiac prediction failed")
                st.error(f"Cardiac risk prediction failed: {e}")
                patient_dict = None

            if patient_dict is not None:
                with st.spinner("Computing SHAP feature importance..."):
                    try:
                        from utils.shap_explain import get_shap_explanation

                        background_df = get_shap_background()
                        patient_row_df = pd.DataFrame([patient_dict])
                        shap_pairs = get_shap_explanation(
                            ann_model, scaler, background_df, patient_row_df,
                            tabular_heart.FEATURE_COLS
                        )
                        st.session_state.shap_pairs = shap_pairs
                        shap_df = pd.DataFrame(shap_pairs, columns=["feature", "impact"])
                        st.bar_chart(shap_df.set_index("feature"))
                    except Exception as e:
                        st.warning(f"SHAP explanation unavailable in this environment: {e}")
                        st.session_state.shap_pairs = None

# ---------------- TAB 3: TEXT ----------------
with tab3:
    st.subheader("Patient Symptom Notes")
    note = st.text_area(
        "Enter the patient's free-text symptom note",
        value="Patient reports shortness of breath and a persistent cough for 4 days, mild fever.",
        height=100,
    )
    if "note_analysis_count" not in st.session_state:
        st.session_state.note_analysis_count = 0
    MAX_ANALYSES_PER_SESSION = 20

    if st.button("Analyze note"):
        if st.session_state.note_analysis_count >= MAX_ANALYSES_PER_SESSION:
            st.error(
                f"Rate limit reached ({MAX_ANALYSES_PER_SESSION} note analyses per session). "
                "Please refresh the app to start a new session. This limit protects against "
                "runaway LLM API usage."
            )
            st.stop()
        st.session_state.note_analysis_count += 1
        try:
            extraction = text_triage.extract_symptoms(note)
            context = ""
            if "xray_result" in st.session_state:
                context += f"X-ray: {st.session_state.xray_result['label']}. "
            if "heart_result" in st.session_state:
                context += f"Cardiac risk: {st.session_state.heart_result['label']}."

            llm_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
            summary = text_triage.llm_summary(note, extraction, context)
            st.session_state.text_summary = summary

            st.write("**Extracted symptom signals:**", extraction["symptoms"] or "None detected")
            if extraction["urgent_flags"]:
                st.error(f"Urgent language detected: {extraction['urgent_flags']}")

            source_label = "LLM-generated summary" if llm_available else "Rule-based summary (no API key set)"
            st.caption(f"Summary source: {source_label}")
            st.info(summary)
        except Exception as e:
            logger.exception("Text triage failed")
            st.error(f"Note analysis failed: {e}")

# ---------------- TAB 4: HITL + REPORT ----------------
with tab4:
    st.subheader("Human-in-the-Loop Review")
    st.write("A clinician must review AI recommendations before they are finalized.")
    missing = []
    if "xray_result" not in st.session_state:
        missing.append("imaging (Tab 1)")
    if "heart_result" not in st.session_state:
        missing.append("cardiac risk (Tab 2)")
    if "text_summary" not in st.session_state:
        missing.append("patient notes (Tab 3)")
    if missing:
        st.info(
            "ℹ️ No data yet for: " + ", ".join(missing) + ". "
            "The report can still be generated, but those sections will be marked "
            "'not provided' rather than showing a full multimodal picture."
        )

    reviewer_name = st.text_input("Reviewer name", value="Dr. A. Sharma")
    decision = st.radio("Decision", ["Approve", "Modify", "Reject"], horizontal=True)
    notes = st.text_area("Reviewer notes / modifications", value="")

    if st.button("Confirm decision & generate report"):
        if not reviewer_name.strip():
            st.error("Reviewer name is required before a decision can be logged.")
        else:
            audit_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_id": patient_id,
                "reviewer": reviewer_name.strip(),
                "decision": decision,
                "notes": notes,
            }
            st.session_state.hitl_log.append(audit_row)
            append_audit_row(audit_row)

            report_path = os.path.join(tempfile.gettempdir(), f"{patient_id}_report.pdf")
            try:
                report_generator.generate_report(
                    patient_id=patient_id,
                    xray_result=st.session_state.get("xray_result"),
                    xray_image_path=st.session_state.get("xray_image_path"),
                    heart_result=st.session_state.get("heart_result"),
                    shap_pairs=st.session_state.get("shap_pairs"),
                    text_summary=st.session_state.get("text_summary"),
                    hitl_decision=decision,
                    hitl_notes=notes,
                    reviewer_name=reviewer_name,
                    output_path=report_path,
<<<<<<< HEAD
                    cnn_training_source=cnn_pneumonia.get_training_source(),
                    heart_data_source="real (UCI Cleveland)" if os.path.exists(UCI_HEART_PATH) else "synthetic",
=======
>>>>>>> 2d7e0bf0112ff57140acb7e3ac7c815ad5beba17
                )
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{patient_id}_copilot_report.pdf",
                    mime="application/pdf",
                )
                st.success("Report generated and HITL decision logged.")
            except Exception as e:
                logger.exception("Report generation failed")
                st.error(f"Report generation failed, but the HITL decision was still logged: {e}")
            finally:
                if os.path.exists(report_path):
                    try:
                        os.remove(report_path)
                    except OSError:
                        pass

    st.write("### Audit Trail (this session)")
    if st.session_state.hitl_log:
        st.dataframe(pd.DataFrame(st.session_state.hitl_log))

    with st.expander("Full persisted audit trail (all sessions)"):
        full_trail = load_audit_trail()
        if not full_trail.empty:
            st.dataframe(full_trail)
            st.download_button(
                "⬇️ Download full audit trail (CSV)",
                data=full_trail.to_csv(index=False).encode("utf-8"),
                file_name="audit_trail.csv",
                mime="text/csv",
            )
        else:
            st.caption("No persisted audit entries yet.")