"""
SHAP-based explainability for the tabular heart-risk ANN.
Satisfies the "Explainable AI" mandatory requirement for the tabular modality
(feature importance / SHAP values).
"""

import numpy as np
import shap


def get_shap_explanation(model, scaler, background_df, patient_row_df, feature_cols):
    """
    background_df: DataFrame of representative training rows (for SHAP background)
    patient_row_df: single-row DataFrame with the same feature_cols, for the patient being explained
    Returns: dict of {feature_name: shap_value} sorted by absolute impact, descending
    """
    background = scaler.transform(background_df[feature_cols].values.astype(np.float32))
    patient = scaler.transform(patient_row_df[feature_cols].values.astype(np.float32))

    # KernelExplainer works model-agnostically with any predict function
    def predict_fn(x):
        return model.predict(x, verbose=0)

    background_sample = shap.sample(background, min(50, len(background)))
    explainer = shap.KernelExplainer(predict_fn, background_sample)
    shap_values = explainer.shap_values(patient, nsamples=100)

    values = np.array(shap_values).flatten()
    pairs = list(zip(feature_cols, values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs
