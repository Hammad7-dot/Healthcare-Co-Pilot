"""
ANN model for cardiac risk prediction from tabular patient data.
Modality: TABULAR (2nd of the 3+ required modalities)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_data import generate_synthetic_heart_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "heart_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "heart_scaler.joblib")

FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_data(csv_path="data/heart.csv"):
    """
    Swap csv_path to a real heart-disease CSV for the actual submission.
    NOTE: the bundled data/heart.csv is the Kaggle "Heart Failure Prediction"
    dataset, which uses a completely different (tab-separated, 11-column,
    string-categorical) schema than FEATURE_COLS below. Loading it directly
    here would raise a confusing KeyError deep inside train(). Prefer
    load_uci_processed_cleveland() with data/heart_disease_raw/
    processed.cleveland.data instead -- that file already matches
    FEATURE_COLS and is what app.py uses by default.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_COLS + ["target"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing expected columns {missing}. "
                "This file does not match the UCI schema FEATURE_COLS expects "
                "(it looks like the Kaggle Heart Failure Prediction dataset, "
                "which is a different schema). Use load_uci_processed_cleveland("
                "'data/heart_disease_raw/processed.cleveland.data') instead, or "
                "pass a CSV with columns: " + ", ".join(FEATURE_COLS + ["target"])
            )
        return df
    return generate_synthetic_heart_data()


def load_uci_processed_cleveland(path):
    """
    Loads the raw UCI 'processed.cleveland.data' file (no header, '?' for
    missing values, target column 'num' is 0-4 severity).
    Returns a DataFrame with FEATURE_COLS + binary 'target' (0 = no disease,
    1 = disease present), ready to feed straight into train().
    """
    cols = FEATURE_COLS + ["num"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    df = df.dropna().reset_index(drop=True)
    for c in df.columns:
        df[c] = df[c].astype(float)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def build_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(epochs=40, csv_path="data/heart.csv", df=None):
    if df is None:
        df = load_data(csv_path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = build_model(X.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=16, verbose=2)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return train()


def predict(model, scaler, patient_dict):
    """patient_dict: {feature_name: value, ...} for all FEATURE_COLS"""
    row = np.array([[patient_dict[c] for c in FEATURE_COLS]], dtype=np.float32)
    row_scaled = scaler.transform(row)
    prob = float(model.predict(row_scaled, verbose=0)[0][0])
    label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    confidence = prob if label == "HIGH RISK" else 1 - prob
    return {"label": label, "confidence": confidence, "raw_prob": prob}


if __name__ == "__main__":
    model, scaler = train(epochs=10)
    df = load_data()
    sample = df.iloc[0][FEATURE_COLS].to_dict()
    print(predict(model, scaler, sample), "true:", df.iloc[0]["target"])
