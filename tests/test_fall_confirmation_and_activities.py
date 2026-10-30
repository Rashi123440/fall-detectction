"""Regression tests for sitting, walking, standing, genuine fall sequences,
confirmation-counter reset, and experimental model/scaler compatibility.
"""

from pathlib import Path
import numpy as np
import pytest
import joblib

from src.real_time_detector import RealTimeFallDetector
from src.config import TOTAL_FEATURE_DIM, SEQUENCE_LENGTH, MODELS_DIR
from src.preprocessor import KeypointPreprocessor


class MockExtractor:
    """Mock PoseExtractor returning predetermined feature vector and landmarks."""
    def __init__(self, feature_vector: np.ndarray):
        self.feat = feature_vector

    def extract_keypoints(self, frame_bgr):
        return True, self.feat.copy(), []

    def render_privacy_skeleton(self, frame_shape, pose_landmarks, bg_black=True, base_frame=None):
        return np.zeros(frame_shape, dtype=np.uint8)

    def close(self):
        pass


class MockModel:
    """Mock Keras model returning a fixed probability."""
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict(self, *args, **kwargs):
        return np.array([[self.prob]], dtype=np.float32)


def _create_sequence(
    hip_start: float,
    hip_end: float,
    torso_angle: float = 0.50,
    aspect_ratio: float = 0.45,
    noise: float = 0.001,
) -> np.ndarray:
    """Creates a synthetic 30-frame sequence with controlled kinematics.
    
    Feature indices:
      - 132: torso_angle (normalized by pi, ~0.50 is upright)
      - 133: mid_hip_y (ratio in [0, 1], lower in image is larger value)
      - 134: aspect_ratio (width / height)
    """
    seq = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURE_DIM), dtype=np.float32)
    hip_trajectory = np.linspace(hip_start, hip_end, SEQUENCE_LENGTH)
    for i in range(SEQUENCE_LENGTH):
        noisy_hip = float(hip_trajectory[i] + np.random.normal(0, noise))
        seq[i, 132] = float(torso_angle)
        seq[i, 133] = noisy_hip
        seq[i, 134] = float(aspect_ratio)
    return seq


# ---------------------------------------------------------------------------
# 1. Sitting Normally (Stationary and Minor Posture Shift)
# ---------------------------------------------------------------------------

def test_sitting_normally_does_not_increment_counter():
    """Verify sitting normally with prob ~0.528 or even above threshold never increments counter."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=10,
        debug=False,
    )
    detector.has_scaler = False
    detector.model = MockModel(0.528)  # user's observed probability

    # 30 frames of stable seated posture: hip at 0.72, upright angle 0.50, aspect 0.50
    sit_seq = _create_sequence(0.72, 0.72, torso_angle=0.50, aspect_ratio=0.50)
    for feat in sit_seq:
        detector.frame_buffer.append(feat)

    detector.pose_extractor = MockExtractor(sit_seq[-1])
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process 15 frames
    for _ in range(15):
        _, tele = detector.process_frame(dummy_img)
        assert detector.is_fall_candidate is False
        assert detector.consecutive_fall_count == 0
        assert detector.is_fall_confirmed is False
        assert tele["is_fall"] is False

    # Even if model probability spikes to 0.72 while sitting still
    detector.model = MockModel(0.72)
    for _ in range(5):
        _, tele = detector.process_frame(dummy_img)
        # Stable upright posture suppresses the candidate
        assert detector.is_fall_candidate is False
        assert detector.consecutive_fall_count == 0
        assert detector.is_fall_confirmed is False

    detector.close()


# ---------------------------------------------------------------------------
# 2. Walking Normally
# ---------------------------------------------------------------------------

def test_walking_normally_does_not_trigger_fall():
    """Verify walking across camera view (upright with motion) does not trigger fall alerts."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=10,
        debug=False,
    )
    detector.has_scaler = False
    # Model probability elevated on walking (e.g. 0.70)
    detector.model = MockModel(0.70)

    # Walking: hip bobs slightly and moves across frame; vertical hip fluctuates 0.45 to 0.52
    # but torso remains upright (0.50) and aspect ratio is vertical (0.42)
    walk_seq = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURE_DIM), dtype=np.float32)
    for i in range(SEQUENCE_LENGTH):
        # Gait oscillation
        hip_y = 0.48 + 0.04 * np.sin(i * 0.8)
        walk_seq[i, 132] = 0.50  # upright
        walk_seq[i, 133] = hip_y
        walk_seq[i, 134] = 0.42  # slim upright aspect

    for feat in walk_seq:
        detector.frame_buffer.append(feat)

    detector.pose_extractor = MockExtractor(walk_seq[-1])
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        _, tele = detector.process_frame(dummy_img)
        # Upright posture without horizontal collapse is NOT genuine fall evidence
        assert detector.is_fall_candidate is False
        assert detector.consecutive_fall_count == 0
        assert detector.is_fall_confirmed is False

    detector.close()


# ---------------------------------------------------------------------------
# 3. Genuine Fall Sequence
# ---------------------------------------------------------------------------

def test_genuine_fall_sequence_confirms_fall():
    """Verify rapid downward drop followed by horizontal collapse triggers and confirms fall alert."""
    confirm_frames = 5
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=confirm_frames,
        debug=False,
    )
    detector.has_scaler = False
    detector.model = MockModel(0.92)  # High fall probability

    # Genuine fall: hips drop from standing (0.35) to floor (0.85), disp = 0.50
    # Torso rotates from upright (0.50) to horizontal floor posture (0.85)
    # Aspect ratio widens to horizontal (> 1.2)
    fall_seq = _create_sequence(0.35, 0.85, torso_angle=0.85, aspect_ratio=1.4)
    for feat in fall_seq:
        detector.frame_buffer.append(feat)

    detector.pose_extractor = MockExtractor(fall_seq[-1])
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Feed consecutive fall frames
    for i in range(1, confirm_frames):
        _, tele = detector.process_frame(dummy_img)
        assert detector.is_fall_candidate is True
        assert detector.consecutive_fall_count == i
        assert detector.is_fall_confirmed is False

    # Final frame reaches confirmation threshold
    _, tele_final = detector.process_frame(dummy_img)
    assert detector.consecutive_fall_count >= confirm_frames
    assert detector.is_fall_confirmed is True
    assert tele_final["is_fall"] is True

    detector.close()


# ---------------------------------------------------------------------------
# 4. Confirmation-Counter Immediate Reset on Non-Fall Frame
# ---------------------------------------------------------------------------

def test_confirmation_counter_resets_immediately_when_condition_is_false():
    """Verify counter does NOT decay by 1, but resets strictly to 0 when interrupted."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=10,
        debug=False,
    )
    detector.has_scaler = False
    detector.model = MockModel(0.90)

    # 1. Inject 4 frames of fall candidate to bring counter to 4
    fall_seq = _create_sequence(0.35, 0.85, torso_angle=0.85, aspect_ratio=1.4)
    for feat in fall_seq:
        detector.frame_buffer.append(feat)
    detector.pose_extractor = MockExtractor(fall_seq[-1])
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(4):
        detector.process_frame(dummy_img)

    assert detector.consecutive_fall_count == 4
    assert detector.is_fall_confirmed is False

    # 2. Next frame: user is sitting normally with prob 0.528 < 0.65
    sit_seq = _create_sequence(0.72, 0.72, torso_angle=0.50, aspect_ratio=0.50)
    for feat in sit_seq:
        detector.frame_buffer.append(feat)
    detector.pose_extractor = MockExtractor(sit_seq[-1])
    detector.model = MockModel(0.528)

    _, tele = detector.process_frame(dummy_img)

    # Crucial check: counter MUST reset to 0, not stay at 3 or 4
    assert detector.is_fall_candidate is False
    assert detector.consecutive_fall_count == 0
    assert tele["consecutive_fall_count"] == 0
    assert detector.is_fall_confirmed is False

    detector.close()


# ---------------------------------------------------------------------------
# 5. Experimental Model and Scaler Verification
# ---------------------------------------------------------------------------

def test_experimental_model_and_scaler_compatibility():
    """Verify experimental model and scaler exist, match 135 features, and preprocessor scales correctly."""
    exp_model_path = MODELS_DIR / "experimental_fall_lstm.keras"
    exp_scaler_path = MODELS_DIR / "experimental_scaler.pkl"

    assert exp_model_path.exists(), f"Experimental model not found at {exp_model_path}"
    assert exp_scaler_path.exists(), f"Experimental scaler not found at {exp_scaler_path}"

    scaler = joblib.load(exp_scaler_path)
    assert scaler.n_features_in_ == TOTAL_FEATURE_DIM  # 135

    preprocessor = KeypointPreprocessor()
    loaded = preprocessor.load_scaler(exp_scaler_path)
    assert loaded is True

    # Test transformation of dummy sequence
    dummy_seq = np.random.randn(2, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM).astype(np.float32)
    transformed = preprocessor.transform(dummy_seq)
    assert transformed.shape == (2, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)
    assert not np.isnan(transformed).any()
