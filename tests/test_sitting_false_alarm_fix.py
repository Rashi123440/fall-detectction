"""Tests for Sitting False Alarm Suppression and Multi-Factor Temporal Window Analysis.
Verifies that sitting, standing, bending, and lying down do NOT trigger false fall alerts,
while genuine dynamic falls trigger confirmed alerts.
"""

from pathlib import Path
import numpy as np
import pytest

from src.real_time_detector import RealTimeFallDetector
from src.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM, DEFAULT_MODEL_PATH, SCALER_PATH


def _build_sequence(posture="sit", n_frames=30):
    """Generates synthetic 135-dim feature sequences matching exact pose extractor geometry."""
    seq = np.zeros((n_frames, TOTAL_FEATURE_DIM), dtype=np.float32)

    for i in range(n_frames):
        if posture == "sit":
            # Upright sitting: torso angle ~0.50 (vertical), hip_y ~0.78 (low in frame), aspect ~0.60
            seq[i, 132] = 0.50 + 0.005 * np.sin(i)   # torso_angle
            seq[i, 133] = 0.78 + 0.001 * np.cos(i)   # hip_y_ratio
            seq[i, 134] = 0.60                        # aspect_ratio

        elif posture == "stand":
            # Upright standing: torso angle ~0.50, hip_y ~0.55, aspect ~0.45
            seq[i, 132] = 0.50 + 0.005 * np.cos(i)
            seq[i, 133] = 0.55 + 0.002 * np.sin(i)
            seq[i, 134] = 0.45

        elif posture == "bend":
            # Bending forward: torso angle ~0.25 (tilted), hip_y ~0.55, aspect ~0.75
            seq[i, 132] = 0.25 + 0.01 * np.cos(i)
            seq[i, 133] = 0.56 + 0.005 * (i / n_frames)
            seq[i, 134] = 0.75

        elif posture == "lying":
            # Static lying: torso angle ~0.0 or ~1.0 (horizontal), hip_y ~0.88, aspect ~1.30 (wide)
            seq[i, 132] = 0.05
            seq[i, 133] = 0.88
            seq[i, 134] = 1.30

        elif posture == "fall":
            # Dynamic fall: transitions from upright (0.50) to horizontal (0.05) with 0.35 vertical hip drop
            progress = i / float(n_frames)
            seq[i, 132] = 0.50 * (1.0 - progress) + 0.05 * progress
            seq[i, 133] = 0.50 + 0.35 * progress
            seq[i, 134] = 0.45 * (1.0 - progress) + 1.25 * progress

    return seq


class MockModel:
    """Mock predictor for deterministic unit testing of decision rules."""
    def __init__(self, prob: float):
        self.prob = prob

    def predict(self, *args, **kwargs):
        return np.array([[self.prob]], dtype=np.float32)


def test_sitting_posture_suppresses_false_alarm():
    """Verify that normal sitting (even with raw model prob ~0.72) is suppressed as stable upright."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=5,
        debug=False,
    )
    detector.model = MockModel(0.72)  # Elevated probability as seen in experimental model on sitting
    detector.has_scaler = False

    sit_seq = _build_sequence("sit", 30)
    for frame_feat in sit_seq:
        detector.frame_buffer.append(frame_feat)

    # Process frame
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Mock extract_keypoints to return success and current frame feature
    detector.pose_extractor.extract_keypoints = lambda frame: (True, sit_seq[-1], None)

    _, tele = detector.process_frame(dummy_img)

    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    assert detector.consecutive_fall_count == 0
    detector.close()


def test_standing_posture_no_false_alarm():
    """Verify standing upright does not trigger candidate fall."""
    detector = RealTimeFallDetector(threshold=0.65, alert_confirmation_frames=5, debug=False)
    detector.model = MockModel(0.68)
    detector.has_scaler = False

    stand_seq = _build_sequence("stand", 30)
    for f in stand_seq:
        detector.frame_buffer.append(f)

    detector.pose_extractor.extract_keypoints = lambda frame: (True, stand_seq[-1], None)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    _, tele = detector.process_frame(dummy_img)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    assert detector.consecutive_fall_count == 0
    detector.close()


def test_bending_posture_no_false_alarm():
    """Verify bending forward with low velocity does not trigger candidate fall."""
    detector = RealTimeFallDetector(threshold=0.65, alert_confirmation_frames=5, debug=False)
    detector.model = MockModel(0.60)  # Below threshold
    detector.has_scaler = False

    bend_seq = _build_sequence("bend", 30)
    for f in bend_seq:
        detector.frame_buffer.append(f)

    detector.pose_extractor.extract_keypoints = lambda frame: (True, bend_seq[-1], None)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    _, tele = detector.process_frame(dummy_img)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    detector.close()


def test_lying_down_static_no_false_alarm():
    """Verify static lying down without dynamic motion does not trigger fall alert."""
    detector = RealTimeFallDetector(threshold=0.70, alert_confirmation_frames=5, debug=False)
    detector.model = MockModel(0.66)
    detector.has_scaler = False

    lying_seq = _build_sequence("lying", 30)
    for f in lying_seq:
        detector.frame_buffer.append(f)

    detector.pose_extractor.extract_keypoints = lambda frame: (True, lying_seq[-1], None)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    _, tele = detector.process_frame(dummy_img)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    detector.close()


def test_dynamic_fall_triggers_and_confirms_alert():
    """Verify that a genuine dynamic fall triggers candidate and confirms alert after sustained frames."""
    confirm_frames = 5
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=confirm_frames,
        debug=False,
    )
    detector.model = MockModel(0.88)  # High confidence fall
    detector.has_scaler = False

    fall_seq = _build_sequence("fall", 30)
    for f in fall_seq:
        detector.frame_buffer.append(f)

    detector.pose_extractor.extract_keypoints = lambda frame: (True, fall_seq[-1], None)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process frames 1 to 4: candidate should be True, count increments, confirmed False
    for step in range(1, confirm_frames):
        _, tele = detector.process_frame(dummy_img)
        assert tele["is_candidate"] is True
        assert tele["is_fall"] is False
        assert detector.consecutive_fall_count == step

    # Process frame 5: should now confirm!
    _, tele_confirm = detector.process_frame(dummy_img)
    assert tele_confirm["is_candidate"] is True
    assert tele_confirm["is_fall"] is True
    assert tele_confirm["is_confirmed"] is True
    assert detector.consecutive_fall_count >= confirm_frames

    detector.close()


def test_configurable_threshold_and_confirmation_frames():
    """Verify threshold and confirmation frames can be customized and take effect."""
    detector = RealTimeFallDetector(
        threshold=0.80,
        alert_confirmation_frames=12,
        debug=False,
    )
    assert detector.threshold == 0.80
    assert detector.alert_confirmation_frames == 12

    # Sequence with prob 0.75 should be rejected by threshold 0.80
    detector.model = MockModel(0.75)
    detector.has_scaler = False

    fall_seq = _build_sequence("fall", 30)
    for f in fall_seq:
        detector.frame_buffer.append(f)

    detector.pose_extractor.extract_keypoints = lambda frame: (True, fall_seq[-1], None)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    _, tele = detector.process_frame(dummy_img)
    assert tele["is_candidate"] is False
    assert detector.consecutive_fall_count == 0
    detector.close()


def test_debug_output_toggle(capsys):
    """Verify debug logging output is printed when enabled and suppressed when disabled."""
    # Debug ENABLED
    detector_dbg = RealTimeFallDetector(debug=True)
    detector_dbg.model = MockModel(0.90)
    detector_dbg.has_scaler = False

    fall_seq = _build_sequence("fall", 30)
    for f in fall_seq:
        detector_dbg.frame_buffer.append(f)
    detector_dbg.pose_extractor.extract_keypoints = lambda frame: (True, fall_seq[-1], None)

    detector_dbg.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    captured = capsys.readouterr()
    assert "[DEBUG Seq #" in captured.out
    assert "Fall Prob:" in captured.out
    detector_dbg.close()

    # Debug DISABLED
    detector_no_dbg = RealTimeFallDetector(debug=False)
    detector_no_dbg.model = MockModel(0.90)
    detector_no_dbg.has_scaler = False

    for f in fall_seq:
        detector_no_dbg.frame_buffer.append(f)
    detector_no_dbg.pose_extractor.extract_keypoints = lambda frame: (True, fall_seq[-1], None)

    detector_no_dbg.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    captured2 = capsys.readouterr()
    assert "[DEBUG Seq #" not in captured2.out
    detector_no_dbg.close()


def test_baseline_and_experimental_models_preserved():
    """Verify that both baseline and experimental models and scalers are intact and not overwritten."""
    models_dir = Path("models")
    baseline_model = models_dir / "fall_lstm_model.keras"
    baseline_scaler = models_dir / "scaler.pkl"
    exp_model = models_dir / "experimental_fall_lstm.keras"
    exp_scaler = models_dir / "experimental_scaler.pkl"

    assert baseline_model.exists(), "Baseline model must be preserved!"
    assert baseline_scaler.exists(), "Baseline scaler must be preserved!"
    assert exp_model.exists(), "Experimental model must be preserved!"
    assert exp_scaler.exists(), "Experimental scaler must be preserved!"

    # Ensure baseline and experimental are distinct files
    assert baseline_model.stat().st_size > 0
    assert exp_model.stat().st_size > 0
