"""
Grad-CAM implementation: visual explainability for the CNN pneumonia model.
Satisfies the "Explainable AI" mandatory requirement for the image modality.
"""

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


def make_gradcam_heatmap(image, model, last_conv_layer_name="conv3_gradcam"):
    """
    image: np.array (H, W, 1), normalized 0-1
    Returns a (H, W) heatmap in range [0, 1].

    NOTE: this is a binary sigmoid model with a single output unit representing
    P(PNEUMONIA). Grad-CAM always needs to differentiate w.r.t. the *predicted*
    class's score, not a fixed class -- otherwise, for a NORMAL prediction, the
    heatmap would show what pushed the image *towards* PNEUMONIA even though
    that's not what the model actually predicted. We flip the sign of the loss
    when the predicted class is NORMAL so the heatmap always explains the
    class that was actually output.
    """
    grad_model = tf.keras.models.Model(
         model.input, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_batch = np.expand_dims(image, axis=0).astype(np.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        pneumonia_score = predictions[:, 0]
        # If the model predicts NORMAL (score < 0.5), explain "not pneumonia"
        # (1 - score) instead, so the heatmap reflects the predicted class.
        predicted_pneumonia = pneumonia_score[0] >= 0.5
        loss = pneumonia_score if predicted_pneumonia else (1.0 - pneumonia_score)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image_2d, heatmap, alpha=0.4):
    """
    image_2d: (H, W) grayscale image, values 0-1
    heatmap: (h, w) Grad-CAM heatmap, smaller resolution than image
    Returns an (H, W, 3) RGB overlay image, values 0-1, for display.
    """
    h, w = image_2d.shape
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy()[..., 0]

    try:
        jet = matplotlib.colormaps["jet"]
    except AttributeError:
        # older matplotlib (<3.5) doesn't have matplotlib.colormaps
        jet = plt.get_cmap("jet")
    jet_colors = jet(heatmap_resized)[:, :, :3]

    gray_rgb = np.stack([image_2d] * 3, axis=-1)
    overlay = jet_colors * alpha + gray_rgb * (1 - alpha)
    return np.clip(overlay, 0, 1)