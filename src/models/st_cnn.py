"""
ST-CNN-5 (Spatio-Temporal CNN) for ECG multi-label classification.
Uses Conv2D: input as 2D grid (time_steps × 12 leads × 1 channel).
- Five temporal Conv2D blocks with kernel (5, 1) — process along the time axis only.
- One spatial Conv2D block with kernel (1, 12) — span across all 12 leads.
- GlobalAveragePooling2D, then two FC layers with dropout, sigmoid output.
Uses standard Dropout for MC dropout at inference.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def _temporal_block_2d(x, filters, kernel_time=5, l2_reg=1e-4, name_prefix=""):
    """One temporal Conv2D block: kernel (kernel_time, 1) along time axis. Skip before ReLU."""
    shortcut = x
    # Conv2D kernel (height, width) = (kernel_time, 1) for "along time only"
    x = layers.Conv2D(
        filters, (kernel_time, 1), padding="same", kernel_regularizer=l2(l2_reg), name=f"{name_prefix}conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}bn")(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same", name=f"{name_prefix}skip_conv")(shortcut)
    x = layers.Add(name=f"{name_prefix}add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}relu")(x)
    return x


def build_st_cnn(
    input_shape=(5000, 12),
    n_classes=10,
    dropout_rate=0.3,
    l2_reg=1e-4,
    temporal_filters=(32, 32, 64, 64, 128),
    fc_units=(128, 64),
):
    """
    ST-CNN-5 with Conv2D. Accepts input_shape=(time_steps, leads) e.g. (5000, 12);
    reshapes to (time_steps, leads, 1) for Conv2D.
    - Temporal blocks: Conv2D kernel (5, 1) along time.
    - Spatial block: Conv2D kernel (1, 12) across all leads.
    - GAP2D, then two FC layers with dropout, sigmoid.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    # (batch, time, leads) -> (batch, time, leads, 1) for Conv2D
    x = layers.Reshape((*input_shape, 1), name="reshape_2d")(inputs)

    # Five temporal Conv2D blocks: kernel (5, 1) along time axis; dropout after later blocks for MC dropout / DR sweep
    for i, f in enumerate(temporal_filters):
        x = _temporal_block_2d(x, f, kernel_time=5, l2_reg=l2_reg, name_prefix=f"t{i}_")
        if i >= 2:  # Dropout after last 3 temporal blocks
            x = layers.Dropout(dropout_rate, name=f"t{i}_dropout")(x)

    # One spatial Conv2D block: kernel (1, 12) spanning all 12 leads (valid: width 12 -> 1)
    x = layers.Conv2D(
        128, (1, 12), padding="valid", kernel_regularizer=l2(l2_reg), name="spatial_conv"
    )(x)
    x = layers.BatchNormalization(name="spatial_bn")(x)
    x = layers.Activation("relu", name="spatial_relu")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    for i, units in enumerate(fc_units):
        x = layers.Dense(units, activation="relu", kernel_regularizer=l2(l2_reg), name=f"fc{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout{i}")(x)

    outputs = layers.Dense(n_classes, activation="sigmoid", name="output")(x)
    return models.Model(inputs, outputs, name="st_cnn")
