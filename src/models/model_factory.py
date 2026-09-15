"""
Model factory: create_model(architecture, ...) dispatches to resnet1d, st_cnn, cnn_lstm.
All models share the same interface: input_shape, num_classes, dropout_rate; output is sigmoid multi-label.
"""

import tensorflow as tf
from tensorflow.keras import Model

from .resnet1d import build_resnet1d


def create_model(
    architecture: str,
    input_shape: tuple,
    num_classes: int,
    dropout_rate: float = 0.3,
    **kwargs,
) -> Model:
    """
    Build a Keras model for ECG multi-label classification.
    All architectures use standard Dropout (for MC dropout at inference).
    """
    if architecture == "resnet1d":
        return build_resnet1d(
            input_shape=input_shape,
            n_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    if architecture == "st_cnn":
        from .st_cnn import build_st_cnn
        return build_st_cnn(
            input_shape=input_shape,
            n_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    if architecture == "cnn_lstm":
        from .cnn_lstm import build_cnn_lstm
        return build_cnn_lstm(
            input_shape=input_shape,
            n_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    raise ValueError(f"Unknown architecture: {architecture}. Use one of: resnet1d, st_cnn, cnn_lstm")
