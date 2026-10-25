"""Tests for MediaPipe pose extraction and feature normalization."""

import numpy as np
import pytest
from src.pose_extractor import PoseExtractor
from src.config import TOTAL_FEATURE_DIM


def test_pose_extractor_initialization():
    extractor = PoseExtractor()
    assert extractor is not None
    extractor.close()


def test_empty_and_blank_frames():
    """Verify robust handling of empty frames without throwing exceptions."""
    extractor = PoseExtractor()

    # None frame
    success, feats, lms = extractor.extract_keypoints(None)
    assert not success
    assert feats.shape == (TOTAL_FEATURE_DIM,)
    assert np.all(feats == 0)

    # Empty frame
    success, feats, lms = extractor.extract_keypoints(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not success
    assert feats.shape == (TOTAL_FEATURE_DIM,)

    # Solid black frame (no human pose)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    success, feats, lms = extractor.extract_keypoints(black_frame)
    assert not success
    assert feats.shape == (TOTAL_FEATURE_DIM,)
    assert lms is None
    extractor.close()


def test_privacy_skeleton_canvas():
    """Verify that skeleton rendering returns valid BGR canvas with correct dimensions."""
    extractor = PoseExtractor()
    canvas = extractor.render_privacy_skeleton(
        frame_shape=(480, 640, 3),
        pose_landmarks=None,
        bg_black=True,
    )
    assert canvas.shape == (480, 640, 3)
    assert np.all(canvas == 0)  # Blank black canvas when no landmarks
    extractor.close()
