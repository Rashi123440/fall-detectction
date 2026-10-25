"""Tests for preprocessing, sliding windows, missing landmark interpolation, and scaling."""

import numpy as np
import pytest
from src.preprocessor import KeypointPreprocessor
from src.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM, STRIDE_STEP


def test_sliding_window_shapes():
    preprocessor = KeypointPreprocessor(sequence_length=SEQUENCE_LENGTH, stride=STRIDE_STEP)

    # 100 frames of dummy keypoints
    dummy_frames = np.ones((100, TOTAL_FEATURE_DIM), dtype=np.float32)
    X, y = preprocessor.create_sliding_windows(dummy_frames, label=1)

    expected_windows = (100 - SEQUENCE_LENGTH) // STRIDE_STEP + 1
    assert X.shape == (expected_windows, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)
    assert y.shape == (expected_windows,)
    assert np.all(y == 1)


def test_short_frame_sequence():
    """Sequences shorter than window size should return empty arrays gracefully."""
    preprocessor = KeypointPreprocessor(sequence_length=30)
    short_frames = np.ones((15, TOTAL_FEATURE_DIM), dtype=np.float32)
    X, y = preprocessor.create_sliding_windows(short_frames, label=0)
    assert len(X) == 0
    assert len(y) == 0


def test_missing_landmark_interpolation():
    preprocessor = KeypointPreprocessor()
    seq = np.ones((10, TOTAL_FEATURE_DIM), dtype=np.float32)
    # Simulate lost tracking on frames 4 and 5
    seq[4] = 0.0
    seq[5] = 0.0

    clean = preprocessor.interpolate_missing_frames(seq)
    assert not np.any(np.all(clean == 0.0, axis=1))
    assert np.all(clean[4] == 1.0)
    assert np.all(clean[5] == 1.0)


def test_dataset_splitting():
    preprocessor = KeypointPreprocessor()
    X = np.random.randn(100, 30, TOTAL_FEATURE_DIM).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    assert (y_train == 1).sum() == 35  # Stratified
