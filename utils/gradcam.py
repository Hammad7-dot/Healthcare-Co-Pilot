"""
Grad-CAM implementation: visual explainability for the CNN pneumonia model.
Satisfies the "Explainable AI" mandatory requirement for the image modality.
"""

import numpy as np
import tensorflow as tf
import matplotlib.cm as cm


def make_gradcam_heatmap(image, model, last_conv_layer_name="conv3_gradcam"):
    """
    image: np.array (H, W, 1), normalized 0-1
    Returns a (H, W) heatmap in range [0, 1].
    """
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_batch = np.expand_dims(image, axis=0).astype(np.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

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

    jet = cm.get_cmap("jet")
    jet_colors = jet(heatmap_resized)[:, :, :3]

    gray_rgb = np.stack([image_2d] * 3, axis=-1)
    overlay = jet_colors * alpha + gray_rgb * (1 - alpha)
    return np.clip(overlay, 0, 1)
