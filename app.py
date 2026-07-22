"""
AI Healthcare Co-Pilot -- main Streamlit application.

Run with:  streamlit run app.py
"""

import os
import sys
import io
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.synthetic_data import generate_synthetic_xray, generate_synthetic_heart_data
from models import cnn_pneumonia, tabular_heart
from utils import gradcam, text_triage, report_generator

st.set_page_config(page_title="AI Healthcare Co-Pilot", layout="wide")


@st.cache_resource
def load_models():
    cnn = cnn_pneumonia.load_or_train()
    uci_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "heart_disease_raw", "processed.cleveland.data")
    if os.path.exists(uci_path):
        df = tabular_heart.load_uci_processed_cleveland(uci_path)
        ann, scaler = tabular_heart.train(epochs=40, df=df)
    else:
        ann, scaler = tabular_heart.load_or_train()
    return cnn, ann, scaler


st.title("🩺 AI Healthcare Co-Pilot")
st.caption(
    "Multimodal AI decision-support: chest X-ray imaging (CNN), cardiac risk "
    "(tabular ANN), and patient notes (text/NLP) -- with explainability and "
    "human-in-the-loop review. Demo runs on synthetic data; see README to plug in "
    "real Kaggle/UCI datasets."
)

cnn_model, ann_model, scaler = load_models()

if "hitl_log" not in st.session_state:
    st.session_state.hitl_log = []

patient_id = st.text_input("Patient ID", value="PT-0001")

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
    with col2:
        uploaded = st.file_uploader("Or upload a grayscale X-ray (png/jpg)", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("L").resize((64, 64))
            arr = np.array(img).astype(np.float32) / 255.0
            st.session_state.xray_img = arr[..., np.newaxis]
            st.session_state.xray_true_label = None

    if "xray_img" in st.session_state:
        image = st.session_state.xray_img
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
            ax.set_title("Grad-CAM: regions driving the prediction")
            st.pyplot(fig)
            fig_path = os.path.join(tempfile.gettempdir(), "gradcam_overlay.png")
            fig.savefig(fig_path, bbox_inches="tight")
            st.session_state.xray_image_path = fig_path

        st.metric("Prediction", result["label"], f"{result['confidence']*100:.1f}% confidence")
        if st.session_state.get("xray_true_label") is not None:
            st.caption(f"(synthetic ground truth label: {'PNEUMONIA' if st.session_state.xray_true_label else 'NORMAL'})")

# ---------------- TAB 2: TABULAR ----------------
with tab2:
    st.subheader("Cardiac Risk Assessment")
    uci_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "heart_disease_raw", "processed.cleveland.data")
    if os.path.exists(uci_path):
        demo_df = tabular_heart.load_uci_processed_cleveland(uci_path).sample(1, random_state=None).reset_index(drop=True)
    else:
        demo_df = generate_synthetic_heart_data(1)
    defaults = demo_df.iloc[0].to_dict()

    with st.form("heart_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, int(defaults["age"]))
        sex_options = [1, 0]
        sex = c1.selectbox("Sex", sex_options, format_func=lambda x: "Male" if x == 1 else "Female",
                            index=sex_options.index(int(defaults["sex"])))
        cp_options = [1, 2, 3, 4]
        cp = c1.selectbox("Chest pain type (1-4)", cp_options,
                           index=cp_options.index(int(defaults["cp"])) if int(defaults["cp"]) in cp_options else 0)
        trestbps = c2.number_input("Resting BP (mmHg)", 80, 220, int(defaults["trestbps"]))
        chol = c2.number_input("Cholesterol (mg/dl)", 100, 600, int(defaults["chol"]))
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], index=int(defaults["fbs"]))
        restecg = c3.selectbox("Resting ECG (0-2)", [0, 1, 2], index=int(defaults["restecg"]))
        thalach = c3.number_input("Max heart rate achieved", 60, 220, int(defaults["thalach"]))
        exang = c3.selectbox("Exercise-induced angina", [0, 1], index=int(defaults["exang"]))
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, float(defaults["oldpeak"]))
        slope_options = [1, 2, 3]
        slope = st.selectbox("Slope of peak exercise ST (1-3)", slope_options,
                              index=slope_options.index(int(defaults["slope"])) if int(defaults["slope"]) in slope_options else 0)
        ca_options = [0, 1, 2, 3]
        ca = st.selectbox("Number of major vessels (0-3)", ca_options,
                           index=ca_options.index(int(defaults["ca"])) if int(defaults["ca"]) in ca_options else 0)
        thal_options = [3, 6, 7]
        thal = st.selectbox("Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)", thal_options,
                             index=thal_options.index(int(defaults["thal"])) if int(defaults["thal"]) in thal_options else 0)
        submitted = st.form_submit_button("Run cardiac risk model")

    if submitted:
        patient_dict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                             fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                             oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        result = tabular_heart.predict(ann_model, scaler, patient_dict)
        st.session_state.heart_result = result
        st.session_state.patient_dict = patient_dict
        st.metric("Prediction", result["label"], f"{result['confidence']*100:.1f}% confidence")

        with st.spinner("Computing SHAP feature importance..."):
            try:
                from utils.shap_explain import get_shap_explanation
                uci_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data", "heart_disease_raw", "processed.cleveland.data")
                if os.path.exists(uci_path):
                    background_df = tabular_heart.load_uci_processed_cleveland(uci_path)
                else:
                    background_df = generate_synthetic_heart_data(60)
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
    if st.button("Analyze note"):
        extraction = text_triage.extract_symptoms(note)
        context = ""
        if "xray_result" in st.session_state:
            context += f"X-ray: {st.session_state.xray_result['label']}. "
        if "heart_result" in st.session_state:
            context += f"Cardiac risk: {st.session_state.heart_result['label']}."
        summary = text_triage.llm_summary(note, extraction, context)
        st.session_state.text_summary = summary
        st.write("**Extracted symptom signals:**", extraction["symptoms"] or "None detected")
        if extraction["urgent_flags"]:
            st.error(f"Urgent language detected: {extraction['urgent_flags']}")
        st.info(summary)

# ---------------- TAB 4: HITL + REPORT ----------------
with tab4:
    st.subheader("Human-in-the-Loop Review")
    st.write("A clinician must review AI recommendations before they are finalized.")

    reviewer_name = st.text_input("Reviewer name", value="Dr. A. Sharma")
    decision = st.radio("Decision", ["Approve", "Modify", "Reject"], horizontal=True)
    notes = st.text_area("Reviewer notes / modifications", value="")

    if st.button("Confirm decision & generate report"):
        st.session_state.hitl_log.append({
            "patient_id": patient_id, "reviewer": reviewer_name,
            "decision": decision, "notes": notes,
        })
        path = report_generator.generate_report(
            patient_id=patient_id,
            xray_result=st.session_state.get("xray_result"),
            xray_image_path=st.session_state.get("xray_image_path"),
            heart_result=st.session_state.get("heart_result"),
            shap_pairs=st.session_state.get("shap_pairs"),
            text_summary=st.session_state.get("text_summary"),
            hitl_decision=decision,
            hitl_notes=notes,
            reviewer_name=reviewer_name,
            output_path=os.path.join(tempfile.gettempdir(), f"{patient_id}_report.pdf"),
        )
        with open(path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=f"{patient_id}_copilot_report.pdf")
        st.success("Report generated and HITL decision logged.")

    if st.session_state.hitl_log:
        st.write("### Audit Trail")
        st.dataframe(pd.DataFrame(st.session_state.hitl_log))
