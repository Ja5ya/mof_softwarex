"""
ResNet1D for ECG multi-label classification.
Uses tf.keras.layers.Dropout (not SpatialDropout1D) for MC dropout at inference.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)
    return x


def build_resnet1d(
    input_shape=(5000, 12),
    n_classes=10,
    initial_filters=64,
    num_blocks=(2, 2, 2),
    dropout_rate=0.3,
):
    """
    ResNet1D for ECG. Dropout is standard Dropout so MC dropout works at inference.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(initial_filters, 7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    filters = initial_filters
    for i, blocks in enumerate(num_blocks):
        for b in range(blocks):
            stride = 2 if b == 0 and i > 0 else 1
            x = resnet_block(x, filters, stride=stride)
        x = layers.Dropout(dropout_rate)(x)  # After each residual stage for meaningful MC dropout variance
        filters *= 2
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="resnet1d")
