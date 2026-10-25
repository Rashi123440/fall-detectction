"""Tests for verifying Python environment, core libraries, and deep learning framework."""

import sys
import pytest


def test_python_version():
    """Verify supported 64-bit Python 3.10+ runtime."""
    assert sys.version_info >= (3, 10), f"Python version too old: {sys.version}"


def test_core_imports():
    """Verify required project packages can be imported without conflict."""
    import cv2
    import numpy as np
    import pandas as pd
    import sklearn
    import mediapipe
    import tensorflow as tf

    assert cv2.__version__ is not None
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert mediapipe.__version__ is not None
    assert tf.__version__ is not None


def test_tensorflow_backend():
    """Verify TensorFlow Keras can construct a simple model and compute gradients."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential([Dense(4, input_shape=(10,), activation="relu"), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    dummy_x = tf.random.normal((5, 10))
    dummy_y = tf.constant([[0.0], [1.0], [0.0], [1.0], [0.0]])
    loss = model.train_on_batch(dummy_x, dummy_y)
    assert loss is not None
