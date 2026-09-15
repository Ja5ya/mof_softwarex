"""
CNN-LSTM for ECG multi-label classification: CNN front-end + LSTM + dense.
Uses standard Dropout for MC dropout at inference.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm(
    input_shape=(5000, 12),
    n_classes=10,
    dropout_rate=0.3,
    cnn_filters=(64, 128, 128),
    lstm_units=128,
    bidirectional=True,
    fc_units=128,
):
    """
    CNN front-end (Conv1D blocks) + LSTM (optionally bidirectional) + FC + sigmoid.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs

    for i, f in enumerate(cnn_filters):
        x = layers.Conv1D(f, 5, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.MaxPooling1D(2, name=f"pool{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_cnn{i}")(x)

    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, name="lstm"), name="bilstm")(x)
    else:
        x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)

    x = layers.Dropout(dropout_rate, name="drop_lstm")(x)
    x = layers.Dense(fc_units, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc")(x)
    outputs = layers.Dense(n_classes, activation="sigmoid", name="output")(x)
    return models.Model(inputs, outputs, name="cnn_lstm")
