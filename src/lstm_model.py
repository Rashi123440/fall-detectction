"""LSTM Classifier for Fall Detection.
Defines model architecture, compilation, training loop with callbacks, and serialization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.config import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    LEARNING_RATE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    PATIENCE,
    DEFAULT_MODEL_PATH,
    METADATA_PATH,
)


def build_lstm_model(
    input_shape: Tuple[int, int] = (SEQUENCE_LENGTH, TOTAL_FEATURE_DIM),
    lstm_units: list = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = LEARNING_RATE,
):
    """Builds and compiles a 2-layer stacked LSTM neural network with Dropout regularization.

    Args:
        input_shape: (sequence_length, feature_dimension), e.g. (30, 135)
        lstm_units: Number of hidden units in LSTM layers.
        dropout_rate: Dropout fraction to prevent overfitting.
        dense_units: Units in post-LSTM dense classification head.
        learning_rate: Initial Adam optimizer learning rate.

    Returns:
        Compiled tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

    model = Sequential([
        Input(shape=input_shape, name="pose_sequence_input"),
        LSTM(lstm_units[0], return_sequences=True, name="lstm_layer_1"),
        Dropout(dropout_rate, name="dropout_1"),
        BatchNormalization(name="batch_norm_1"),
        LSTM(lstm_units[1], return_sequences=False, name="lstm_layer_2"),
        Dropout(dropout_rate, name="dropout_2"),
        Dense(dense_units, activation="relu", name="dense_features"),
        Dense(1, activation="sigmoid", name="fall_prediction"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_fall_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    is_synthetic_training: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Trains LSTM classifier on sequence keypoints and saves the best model checkpoint.

    Returns:
        (trained_model, history_dict)
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n[LSTM Trainer] Starting training for {epochs} epochs (Batch size: {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "input_shape": list(input_shape),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "is_synthetic_training": is_synthetic_training,
        "disclaimer": "Trained on synthetic verification data for pipeline testing. Do not claim real-world accuracy." if is_synthetic_training else "Trained on verified benchmark dataset.",
        "history": {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        },
        "final_metrics": {
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "val_precision": float(history.history.get("val_precision", [0.0])[-1]),
            "val_recall": float(history.history.get("val_recall", [0.0])[-1]),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[LSTM Trainer] Successfully trained model saved to: {model_path}")
    print(f"[LSTM Trainer] Metadata saved to: {metadata_path}")
    return model, meta


def load_model_for_inference(model_path: Path = DEFAULT_MODEL_PATH):
    """Loads a saved Keras LSTM model with explicit error handling for missing models."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by running the training pipeline."
        )

    try:
        model = tf.keras.models.load_model(str(model_path))
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")
