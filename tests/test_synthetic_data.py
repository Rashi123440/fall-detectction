"""Tests for synthetic kinematics generator and test dataset generation."""

import numpy as np
import pytest
from src.synthetic_data import SyntheticKinematicsGenerator
from src.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM, LABEL_FALL, LABEL_NO_FALL


def test_synthetic_sequence_shape():
    generator = SyntheticKinematicsGenerator(seed=42)

    # Test ADL sequence
    adl_seq = generator.generate_sequence(is_fall=False, seq_length=SEQUENCE_LENGTH)
    assert adl_seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)
    assert not np.isnan(adl_seq).any()

    # Test Fall sequence
    fall_seq = generator.generate_sequence(is_fall=True, seq_length=SEQUENCE_LENGTH)
    assert fall_seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)
    assert not np.isnan(fall_seq).any()


def test_synthetic_dataset_generation():
    generator = SyntheticKinematicsGenerator(seed=100)
    X, y, meta = generator.generate_dataset(num_samples=20, fall_ratio=0.5, seq_length=SEQUENCE_LENGTH)

    assert X.shape == (20, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)
    assert y.shape == (20,)
    assert meta["is_synthetic"] is True
    assert (y == LABEL_FALL).sum() == 10
    assert (y == LABEL_NO_FALL).sum() == 10
