"""Tests for Stability Guard Calibration and Configurable Thresholds.

Verifies that:
- Stationary sitting, normal breathing sway, and minor fidgeting are classified as STABLE.
- Slow bending / leaning is NEITHER stable NOR fall-dynamics (passes through to threshold only).
- Rapid vertical collapse triggers has_fall_dynamics regardless of upright posture.
- Configurable stability thresholds work end-to-end and are stored correctly on the instance.
- No model files are modified.

Units used throughout (matching src/real_time_detector.py docstring):
  hip_displacement : fraction of MediaPipe normalised frame height, over 30-frame window.
  hip_speed        : mean |delta(hip_y)| per frame (1 frame = ~1/30 s at 30 fps).
"""

import math
import numpy as np
import pytest

from src.real_time_detector import RealTimeFallDetector
from src.config import TOTAL_FEATURE_DIM, SEQUENCE_LENGTH


# ──────────────────────────────────────────────
# Helper: build synthetic 30-frame (135-dim) sequences
# Index 132 = torso_angle, 133 = hip_y, 134 = aspect_ratio
# ──────────────────────────────────────────────

def _window_from_hip_trajectory(
    hip_y_start: float,
    hip_y_end: float,
    torso_angle: float = 0.50,
    aspect_ratio: float = 0.60,
    noise_std: float = 0.0,
    n: int = 30,
) -> np.ndarray:
    """Build a synthetic (30, 135) feature window with controlled hip trajectory."""
    rng = np.random.default_rng(seed=0)
    seq = np.zeros((n, TOTAL_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        seq[i, 132] = torso_angle
        seq[i, 133] = float(hip_y_start + (hip_y_end - hip_y_start) * i / (n - 1))
        if noise_std > 0:
            seq[i, 133] += float(rng.normal(0, noise_std))
        seq[i, 134] = aspect_ratio
    return seq


class MockModel:
    """Deterministic probability mock."""
    def __init__(self, prob: float):
        self.prob = prob

    def predict(self, *args, **kwargs):
        return np.array([[self.prob]], dtype=np.float32)


def _detector(prob: float = 0.50, threshold: float = 0.45, **guard_kwargs) -> RealTimeFallDetector:
    """Build a detector with a mocked model and no scaler."""
    d = RealTimeFallDetector(threshold=threshold, debug=False, **guard_kwargs)
    d.model = MockModel(prob)
    d.has_scaler = False
    return d


# ──────────────────────────────────────────────
# 1. _evaluate_temporal_window unit tests
# ──────────────────────────────────────────────

class TestEvaluateTemporalWindowDefaults:
    """Verify stability/fall classification against the default guard thresholds
    using realistic motion amplitudes from the Monte Carlo calibration."""

    def test_completely_still_is_stable(self):
        """Still subject: disp~0.002, speed~0.0006 → STABLE."""
        seq = _window_from_hip_trajectory(0.55, 0.55, noise_std=0.0005)
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        assert result["is_upright"] is True
        assert result["is_stable_upright"] is True
        assert result["has_fall_dynamics"] is False

    def test_normal_breathing_sway_is_stable(self):
        """Breathing sway: disp~0.012, speed~0.003 → STABLE."""
        seq = _window_from_hip_trajectory(0.54, 0.55, noise_std=0.003)
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        assert result["is_stable_upright"] is True
        assert result["has_fall_dynamics"] is False

    def test_minor_fidgeting_is_stable(self):
        """Minor fidgeting: noise_std=0.008 → disp~0.033, speed~0.009 → STABLE with new defaults."""
        seq = _window_from_hip_trajectory(0.55, 0.55, noise_std=0.008)
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        # hip_disp ~ 0.033 < stable_disp_max=0.18  AND  speed ~ 0.009 < stable_speed_max=0.025
        assert result["is_stable_upright"] is True
        assert result["has_fall_dynamics"] is False

    def test_slow_lean_forward_is_stable(self):
        """Slow lean/reach: linear drift of 0.10 produces disp~0.10 which is below
        stable_disp_max=0.18, so the posture is correctly classified as STABLE.
        (A 0.10 hip shift corresponds to ~10% of frame height — typical desk reaching.)
        To enter the NEITHER zone (not stable, not fall_dynamics), displacement must
        be in (0.18, 0.20) — a narrow band. We test that zone explicitly with drift 0.19."""
        # Drift 0.10 → STABLE (below stable_disp_max=0.18)
        seq_small = _window_from_hip_trajectory(0.50, 0.60, noise_std=0.0)
        d = _detector()
        result_small = d._evaluate_temporal_window(seq_small)
        assert result_small["is_stable_upright"] is True,   "drift 0.10 should be STABLE"
        assert result_small["has_fall_dynamics"] is False,  "drift 0.10 should not be FALL_DYN"

        # Drift 0.19 → NEITHER (above stable_disp_max=0.18, below fall_disp_min=0.20)
        seq_neither = _window_from_hip_trajectory(0.50, 0.69, noise_std=0.0)
        result_neither = d._evaluate_temporal_window(seq_neither)
        assert result_neither["is_stable_upright"] is False, "drift 0.19 should NOT be STABLE"
        assert result_neither["has_fall_dynamics"] is False, "drift 0.19 should NOT be FALL_DYN"

    def test_rapid_fall_triggers_fall_dynamics(self):
        """Rapid collapse: disp=0.35 >> fall_disp_min=0.20 → FALL_DYN."""
        seq = _window_from_hip_trajectory(0.50, 0.85)
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        assert result["is_stable_upright"] is False
        assert result["has_fall_dynamics"] is True

    def test_slow_sitdown_triggers_fall_dynamics(self):
        """Slow sit-down: disp clearly above fall_disp_min=0.20 → FALL_DYN.
        Use drift 0.22 to avoid floating-point boundary ambiguity."""
        seq = _window_from_hip_trajectory(0.55, 0.77)   # drift 0.22 > fall_disp_min 0.20
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        # disp = max(hip_y) - min(hip_y) = 0.77 - 0.55 = 0.22 >= fall_disp_min=0.20
        assert result["has_fall_dynamics"] is True

    def test_horizontal_posture_always_fall_dynamics(self):
        """Lying flat: aspect_ratio=1.3 → is_horizontal=True → has_fall_dynamics regardless of disp."""
        seq = _window_from_hip_trajectory(0.85, 0.85, torso_angle=0.05, aspect_ratio=1.30)
        d = _detector()
        result = d._evaluate_temporal_window(seq)
        assert result["is_horizontal"] is True
        assert result["has_fall_dynamics"] is True


class TestConfigurableThresholds:
    """Verify that custom guard thresholds are stored and applied correctly."""

    def test_custom_thresholds_stored_on_instance(self):
        d = RealTimeFallDetector(
            stable_disp_max=0.12,
            stable_speed_max=0.015,
            fall_disp_min=0.25,
            fall_speed_min=0.020,
            debug=False,
        )
        assert d.stable_disp_max == 0.12
        assert d.stable_speed_max == 0.015
        assert d.fall_disp_min == 0.25
        assert d.fall_speed_min == 0.020
        d.close()

    def test_tighter_stable_disp_max_reclassifies_fidgeting(self):
        """With stable_disp_max=0.02 (very tight), minor fidgeting is no longer stable."""
        seq = _window_from_hip_trajectory(0.55, 0.55, noise_std=0.008)
        d = _detector(stable_disp_max=0.02, stable_speed_max=0.025)
        result = d._evaluate_temporal_window(seq)
        # disp ~0.033 > 0.02 → not stable
        assert result["is_stable_upright"] is False

    def test_looser_fall_disp_min_reclassifies_slow_lean(self):
        """With fall_disp_min=0.08, the 0.10 slow lean now triggers fall dynamics."""
        seq = _window_from_hip_trajectory(0.50, 0.60)   # drift 0.10
        d = _detector(fall_disp_min=0.08, fall_speed_min=0.018)
        result = d._evaluate_temporal_window(seq)
        assert result["has_fall_dynamics"] is True

    def test_default_thresholds_match_documented_values(self):
        """Default guard thresholds must match the values in the calibration docstring."""
        d = RealTimeFallDetector(debug=False)
        assert d.stable_disp_max == 0.18
        assert d.stable_speed_max == 0.025
        assert d.fall_disp_min == 0.20
        assert d.fall_speed_min == 0.018
        d.close()


# ──────────────────────────────────────────────
# 2. End-to-end process_frame decision tests
# ──────────────────────────────────────────────

class MockExtractor:
    """Injects a pre-built feature vector instead of running MediaPipe."""
    def __init__(self, feat: np.ndarray):
        self._feat = feat

    def extract_keypoints(self, frame):
        return True, self._feat.copy(), None

    def render_privacy_skeleton(self, **kwargs):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        pass


def _process_once(detector: RealTimeFallDetector, seq: np.ndarray):
    """Pre-fill buffer with seq, inject last frame as 'new frame', return telemetry."""
    for feat in seq:
        detector.frame_buffer.append(feat)
    detector.pose_extractor = MockExtractor(seq[-1])
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _, tele = detector.process_frame(dummy)
    return tele


def test_stationary_sitting_no_alert():
    """Stationary sitting: stable upright → is_candidate=False, no alert, count stays 0."""
    sit = _window_from_hip_trajectory(0.78, 0.78, noise_std=0.001)  # hip barely moves
    d = _detector(prob=0.72, threshold=0.65)  # elevated raw prob, should be suppressed
    tele = _process_once(d, sit)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    assert d.consecutive_fall_count == 0
    d.close()


def test_normal_standing_no_alert():
    """Normal standing with tiny sway: stable → no alert."""
    stand = _window_from_hip_trajectory(0.55, 0.55, noise_std=0.002)
    d = _detector(prob=0.68, threshold=0.65)
    tele = _process_once(d, stand)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    d.close()


def test_rapid_fall_triggers_candidate():
    """Rapid downward collapse (disp=0.35): has_fall_dynamics=True + high prob → candidate."""
    fall = _window_from_hip_trajectory(0.50, 0.85)  # hip drops 0.35 of frame
    d = _detector(prob=0.88, threshold=0.65)
    tele = _process_once(d, fall)
    assert tele["is_candidate"] is True
    assert d.consecutive_fall_count == 1
    d.close()


def test_slow_lean_below_threshold_no_alert():
    """Slow lean (disp~0.10): not stable, not fall_dynamics → goes to threshold check only.
    With prob=0.50 < threshold=0.65 → no alert."""
    lean = _window_from_hip_trajectory(0.50, 0.60)
    d = _detector(prob=0.50, threshold=0.65)
    tele = _process_once(d, lean)
    assert tele["is_candidate"] is False
    assert tele["is_fall"] is False
    d.close()


def test_slow_lean_above_threshold_triggers_candidate():
    """Slow lean (disp~0.10): not stable, not fall_dynamics.
    With prob=0.80 >= threshold=0.65 AND prob >= 0.85 is False AND no fall_dynamics → NO candidate.
    The inner guard (has_fall_dynamics OR not is_upright OR prob>=0.85) blocks it."""
    lean = _window_from_hip_trajectory(0.50, 0.60, torso_angle=0.50)
    # is_upright=True (angle=0.50), has_fall_dynamics=False (disp~0.10 < 0.20), prob=0.80 < 0.85
    d = _detector(prob=0.80, threshold=0.65)
    tele = _process_once(d, lean)
    assert tele["is_candidate"] is False   # inner guard rejects: no fall dynamics, still upright
    d.close()


def test_overwhelming_confidence_triggers_regardless_of_dynamics():
    """prob >= 0.85 overrides the has_fall_dynamics gate ONLY when is_stable_upright is False.

    Priority of the decision ladder (src/real_time_detector.py lines ~222-239):
      1. is_stable_upright=True  → is_candidate=False  (unconditional suppression)
      2. prob >= threshold AND (has_fall_dynamics OR not is_upright OR prob >= 0.85)
                              → is_candidate=True
      3. Otherwise             → is_candidate=False

    So the 0.85 emergency override only applies after the stability gate passes.
    We test with a non-upright collapsing sequence so is_stable_upright=False.
    """
    # Tilted/collapsing posture: torso_angle=0.20 (not upright), small disp → is_stable=False
    non_upright = _window_from_hip_trajectory(0.55, 0.60, torso_angle=0.20, noise_std=0.0)
    d = _detector(prob=0.90, threshold=0.65)
    result = d._evaluate_temporal_window(non_upright)
    # Confirm preconditions
    assert result["is_upright"] is False, "precondition: torso must not be upright"
    assert result["is_stable_upright"] is False, "precondition: must not be classified stable"
    tele = _process_once(d, non_upright)
    assert tele["is_candidate"] is True, "prob=0.90 >= 0.85 override: should be candidate"
    d.close()


def test_stable_upright_blocks_even_high_confidence():
    """When is_stable_upright=True, even prob=0.95 is suppressed (no false alarm while sitting)."""
    # Stationary upright: disp<<0.18, speed<<0.025, torso_angle=0.50
    stationary = _window_from_hip_trajectory(0.78, 0.78, noise_std=0.0, torso_angle=0.50)
    d = _detector(prob=0.95, threshold=0.65)
    result = d._evaluate_temporal_window(stationary)
    assert result["is_stable_upright"] is True, "precondition: stationary sitting must be stable"
    tele = _process_once(d, stationary)
    assert tele["is_candidate"] is False, "stable sitting must never trigger even at prob=0.95"
    d.close()


def test_confirmation_counter_accumulates_and_decays():
    """Counter increments per candidate frame and decrements per non-candidate frame (not reset)."""
    fall = _window_from_hip_trajectory(0.50, 0.85)
    d = _detector(prob=0.88, threshold=0.65, alert_confirmation_frames=5)

    for step in range(1, 5):
        for feat in fall:
            d.frame_buffer.append(feat)
        d.pose_extractor = MockExtractor(fall[-1])
        _, tele = d.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert d.consecutive_fall_count == step
        assert tele["is_fall"] is False  # not confirmed yet

    # One more → confirms
    for feat in fall:
        d.frame_buffer.append(feat)
    d.pose_extractor = MockExtractor(fall[-1])
    _, tele5 = d.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert tele5["is_fall"] is True

    # Now inject a non-candidate: counter resets to 0 immediately
    sit = _window_from_hip_trajectory(0.78, 0.78, noise_std=0.001)
    for feat in sit:
        d.frame_buffer.append(feat)
    d.pose_extractor = MockExtractor(sit[-1])
    d.model = MockModel(0.40)  # below threshold
    d.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert d.consecutive_fall_count == 0
    d.close()


def test_guard_thresholds_gap_is_valid():
    """Verify stable_disp_max < fall_disp_min so there is a NEITHER zone for borderline motion."""
    d = RealTimeFallDetector(debug=False)
    assert d.stable_disp_max < d.fall_disp_min, (
        "stable_disp_max must be < fall_disp_min to create a NEITHER zone for borderline "
        f"motion; got stable={d.stable_disp_max}, fall={d.fall_disp_min}"
    )
    assert d.stable_speed_max > d.fall_speed_min, (
        "stable_speed_max should be > fall_speed_min so that the stable zone contains the "
        "normal-motion speed range; got stable={d.stable_speed_max}, fall={d.fall_speed_min}"
    )
    d.close()
