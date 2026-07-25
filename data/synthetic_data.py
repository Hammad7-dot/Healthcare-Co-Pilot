"""
Synthetic data generators for demo/training purposes.

IMPORTANT: These generators let the entire pipeline run end-to-end with NO
external downloads (useful for the hackathon demo and for CI). For the real
submission, swap these out for the actual datasets:

  Chest X-Ray images -> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  Heart disease table -> https://archive.ics.uci.edu/dataset/45/heart+disease

To swap in real data:
  1. Download the Kaggle chest X-ray dataset, unzip into data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
  2. Point models/cnn_pneumonia.py's `train_from_directory()` at that path instead of `train_on_synthetic()`
  3. Download the UCI heart disease CSV into data/heart.csv
  4. Point models/tabular_heart.py's `load_data()` at that CSV instead of `generate_synthetic_heart_data()`
"""

import numpy as np
import pandas as pd


def generate_synthetic_xray(n_samples=400, img_size=64, seed=42):
    """
    Generates synthetic 'chest X-ray-like' grayscale images.
    PNEUMONIA class images get added blob/haze patterns (simulating opacity),
    NORMAL class images stay cleaner. This is a stand-in for real X-ray data
    so the CNN + Grad-CAM pipeline is fully runnable offline.
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, img_size, img_size, 1), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int32)

    for i in range(n_samples):
        base = rng.normal(0.35, 0.05, (img_size, img_size))
        # simulate rib/lung field structure
        for r in range(0, img_size, 8):
            base[r:r + 2, :] += 0.05
        label = i % 2
        if label == 1:  # PNEUMONIA -> add cloudy opacity blobs
            n_blobs = rng.integers(2, 5)
            for _ in range(n_blobs):
                cx, cy = rng.integers(10, img_size - 10, size=2)
                radius = rng.integers(6, 14)
                yy, xx = np.ogrid[:img_size, :img_size]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
                base[mask] += rng.uniform(0.25, 0.45)
        base = np.clip(base, 0, 1)
        images[i, :, :, 0] = base
        labels[i] = label

    idx = rng.permutation(n_samples)
    return images[idx], labels[idx]


def generate_synthetic_heart_data(n_samples=800, seed=42):
    """
    Generates a synthetic tabular dataset with the same schema as the
    UCI Heart Disease dataset, with a realistic-ish signal so the ANN
    and SHAP explainability actually learn something meaningful.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(29, 78, n_samples)
    sex = rng.integers(0, 2, n_samples)  # 1 = male, 0 = female
    cp = rng.integers(1, 5, n_samples)   # chest pain type, UCI encoding: 1-4
    trestbps = rng.integers(94, 200, n_samples)  # resting blood pressure
    chol = rng.integers(126, 565, n_samples)     # cholesterol
    fbs = (rng.random(n_samples) < 0.15).astype(int)  # fasting blood sugar > 120
    restecg = rng.integers(0, 3, n_samples)
    thalach = rng.integers(71, 202, n_samples)   # max heart rate achieved
    exang = (rng.random(n_samples) < 0.3).astype(int)  # exercise induced angina
    oldpeak = np.round(rng.uniform(0, 6.2, n_samples), 1)
    slope = rng.integers(1, 4, n_samples)  # UCI encoding: 1-3
    ca = rng.integers(0, 4, n_samples)   # number of major vessels
    thal = rng.choice([3, 6, 7], n_samples)  # UCI encoding: 3=normal, 6=fixed, 7=reversible

    # Construct a risk score with real clinical direction, then binarize
    risk = (
        0.03 * (age - 50)
        + 0.6 * sex
        + 0.4 * cp
        + 0.02 * (trestbps - 130)
        + 0.01 * (chol - 240)
        + 0.5 * fbs
        - 0.02 * (thalach - 150)
        + 0.7 * exang
        + 0.35 * oldpeak
        + 0.4 * ca
        + rng.normal(0, 1.0, n_samples)
    )
    target = (risk > np.median(risk)).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        "target": target,
    })
    return df


if __name__ == "__main__":
    imgs, labels = generate_synthetic_xray(10)
    print("X-ray demo:", imgs.shape, labels[:10])
    df = generate_synthetic_heart_data(10)
    print(df.head())
