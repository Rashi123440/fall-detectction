"""Automated unit and integration tests for the Experimental Fall Detection pipeline."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM, MODELS_DIR
from src.train_experimental import (
    build_experimental_lstm_model,
    ExperimentalTrainer,
    EXPERIMENTAL_MODEL_PATH,
    EXPERIMENTAL_SCALER_PATH,
)


def test_build_experimental_lstm_model_architecture():
    """Verifies that experimental LSTM matches the expected (30, 135) input."""
    model = build_experimental_lstm_model(
        input_shape=(SEQUENCE_LENGTH, TOTAL_FEATURE_DIM),
    )
    assert model.input_shape == (None, 30, 135)
    assert model.output_shape == (None, 1)

    # Test forward pass on mock batch
    mock_batch = np.random.randn(4, 30, 135).astype(np.float32)
    preds = model.predict(mock_batch, verbose=0)
    assert preds.shape == (4, 1)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_validation_threshold_tuning_logic():
    """Verifies that threshold tuning selects a valid threshold between 0.30 and 0.85."""
    trainer = ExperimentalTrainer()

    # Mock predictions on validation set
    y_val = np.array([0] * 30 + [1] * 70, dtype=np.int32)
    # Simulate slightly separated probabilities
    np.random.seed(42)
    mock_val_probs = np.concatenate([
        np.random.uniform(0.20, 0.60, size=30),  # No Fall
        np.random.uniform(0.40, 0.90, size=70),  # Fall
    ]).reshape(-1, 1)

    class MockModel:
        def predict(self, X, verbose=0):
            return mock_val_probs

    mock_x_val = np.zeros((100, 30, 135), dtype=np.float32)
    th, summary = trainer.tune_threshold_on_validation(
        model=MockModel(),
        X_val_scaled=mock_x_val,
        y_val=y_val,
        min_fall_recall=0.75,
    )

    assert 0.30 <= th <= 0.85
    assert "val_balanced_acc" in summary
    assert len(summary["tuning_curve"]) > 0


def test_experimental_filenames_do_not_overwrite_baseline():
    """Guarantees that experimental artifact paths are completely distinct from baseline paths."""
    from src.config import DEFAULT_MODEL_PATH, SCALER_PATH, METADATA_PATH
    from src.train_experimental import (
        EXPERIMENTAL_MODEL_PATH,
        EXPERIMENTAL_SCALER_PATH,
        EXPERIMENTAL_METADATA_PATH,
    )

    assert EXPERIMENTAL_MODEL_PATH != DEFAULT_MODEL_PATH
    assert EXPERIMENTAL_SCALER_PATH != SCALER_PATH
    assert EXPERIMENTAL_METADATA_PATH != METADATA_PATH
    assert "experimental" in EXPERIMENTAL_MODEL_PATH.name
    assert "experimental" in EXPERIMENTAL_SCALER_PATH.name
