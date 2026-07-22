"""
CNN model for Pneumonia detection from chest X-ray images.
Modality: IMAGE (1 of the 3+ required modalities)
Predictive engine: CNN (satisfies "at least one predictive deep learning model")
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_data import generate_synthetic_xray

IMG_SIZE = 64
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "saved_models", "cnn_pneumonia.keras")


def build_model(img_size=IMG_SIZE):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(16, 3, activation="relu", padding="same", name="conv1")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same", name="conv3_gradcam")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_on_synthetic(epochs=8, n_samples=600):
    """Train on synthetic data. Swap for train_from_directory() with real Kaggle data."""
    images, labels = generate_synthetic_xray(n_samples=n_samples, img_size=IMG_SIZE)
    split = int(0.85 * n_samples)
    x_train, y_train = images[:split], labels[:split]
    x_val, y_val = images[split:], labels[split:]

    model = build_model()
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=32, verbose=2)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    return model


def train_from_directory(data_dir, epochs=15, img_size=IMG_SIZE):
    """
    Real-data training path. Point data_dir at a folder with
    train/NORMAL, train/PNEUMONIA subfolders (Kaggle chest-xray-pneumonia layout).
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"), color_mode="grayscale",
        image_size=(img_size, img_size), batch_size=32, label_mode="binary")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"), color_mode="grayscale",
        image_size=(img_size, img_size), batch_size=32, label_mode="binary")

    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # The Kaggle chest-xray-pneumonia train split is heavily imbalanced
    # (~3875 PNEUMONIA vs ~1341 NORMAL). Without class weighting the model
    # learns to predict PNEUMONIA almost always, which inflates overall
    # accuracy while tanking NORMAL recall (missed "healthy" calls aren't
    # dangerous, but it also means PNEUMONIA precision is misleadingly low
    # and the confidence scores are unreliable). Weight classes inversely
    # to their frequency so both classes contribute equally to the loss.
    counts = {"NORMAL": 0, "PNEUMONIA": 0}
    for cls in counts:
        d = os.path.join(data_dir, "train", cls)
        if os.path.isdir(d):
            counts[cls] = len(os.listdir(d))
    total = sum(counts.values())
    class_weight = None
    if counts["NORMAL"] and counts["PNEUMONIA"]:
        class_weight = {
            0: total / (2 * counts["NORMAL"]),
            1: total / (2 * counts["PNEUMONIA"]),
        }

    model = build_model(img_size)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2,
              class_weight=class_weight)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    return model


def load_or_train():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return train_from_directory("data/chest_xray")


def predict(model, image):
    """image: np.array (H, W) or (H, W, 1), values 0-1"""
    if image.ndim == 2:
        image = image[..., np.newaxis]
    batch = np.expand_dims(image, axis=0).astype(np.float32)
    prob = float(model.predict(batch, verbose=0)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob
    return {"label": label, "confidence": confidence, "raw_prob": prob}


if __name__ == "__main__":
    m = train_on_synthetic(epochs=5, n_samples=200)
    imgs, labels = generate_synthetic_xray(5)
    for i in range(5):
        print(predict(m, imgs[i]), "true:", labels[i])

