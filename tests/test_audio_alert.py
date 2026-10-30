"""Unit tests for automatic local fall-alert, repeating alarm, acknowledgement,
and duplicate-prevention mechanisms in the Privacy-Preserving Fall Detection system.
"""

import time
import threading
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.audio_alert import AudioAlertManager, alert_manager
from src.real_time_detector import RealTimeFallDetector
from src.config import TOTAL_FEATURE_DIM


class DummyModel:
    """Mock LSTM model returning a preset probability."""
    def __init__(self, prob: float):
        self.prob = prob

    def predict(self, *args, **kwargs):
        return np.array([[self.prob]], dtype=np.float32)


# ---------------------------------------------------------------------------
# AudioAlertManager Unit Tests
# ---------------------------------------------------------------------------

def test_audio_alert_manager_single_trigger_cooldown():
    """Verify single-shot trigger_alert respects cooldown."""
    mgr = AudioAlertManager(cooldown=2.0)
    with patch.object(mgr, "_play_sound_worker") as mock_play:
        assert mgr.trigger_alert() is True
        # Immediate second call should be suppressed by cooldown
        assert mgr.trigger_alert() is False
        # Forced call should bypass cooldown
        assert mgr.trigger_alert(force=True) is True


def test_audio_alert_manager_start_and_stop_alarm():
    """Verify start_alarm begins repeating alarm loop and stop_alarm halts it cleanly."""
    mgr = AudioAlertManager(repeat_interval=0.1)
    play_count = 0

    def mock_beep():
        nonlocal play_count
        play_count += 1

    with patch.object(mgr, "_play_beep_once", side_effect=mock_beep):
        assert mgr.is_alarm_active is False
        started = mgr.start_alarm()
        assert started is True
        assert mgr.is_alarm_active is True

        # Second start_alarm call should be idempotent (return False)
        assert mgr.start_alarm() is False

        # Allow alarm loop to run for a short duration
        time.sleep(0.25)
        assert play_count >= 1

        # Stop alarm
        stopped = mgr.stop_alarm()
        assert stopped is True
        assert mgr.is_alarm_active is False

        # Repeated stop_alarm call should be idempotent
        assert mgr.stop_alarm() is False


def test_audio_alert_manager_error_resilience():
    """Verify that an audio playback exception is caught and logged without raising unhandled errors."""
    mgr = AudioAlertManager()
    with patch("winsound.Beep", side_effect=RuntimeError("Audio hardware error")):
        # Should not raise exception
        mgr._play_beep_once()


# ---------------------------------------------------------------------------
# Detector Alert Triggering & Confirmation Logic Tests
# ---------------------------------------------------------------------------

def test_alert_not_triggered_before_confirmation():
    """Verify alarm is NOT started on positive predictions before confirmation count is reached."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=5,
    )
    detector.has_scaler = False
    detector.model = DummyModel(0.95)

    # Populate 30-frame buffer with falling-like dynamic posture features
    dummy_frame = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    # Set non-upright angle (e.g. 0.85 horizontal) to pass candidate gate
    dummy_frame[132] = 0.85
    dummy_frame[134] = 1.3
    for _ in range(30):
        detector.frame_buffer.append(dummy_frame.copy())

    mock_mgr = MagicMock()
    with patch("src.real_time_detector.alert_manager", mock_mgr):
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed 4 frames (confirmation threshold is 5)
        for _ in range(4):
            # Manually inject dummy pose landmarks and bypass extraction for unit test speed
            with patch.object(detector.pose_extractor, "extract_keypoints", return_value=(True, dummy_frame, [])):
                detector.process_frame(raw_frame)

            assert detector.is_fall_candidate is True
            assert detector.is_fall_confirmed is False
            # Alarm must NOT have been started yet
            mock_mgr.start_alarm.assert_not_called()

        detector.close()


def test_alert_triggered_when_fall_is_confirmed():
    """Verify repeating alarm starts exactly when confirmation frames threshold is reached."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=3,
    )
    detector.has_scaler = False
    detector.model = DummyModel(0.95)

    dummy_frame = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    dummy_frame[132] = 0.85  # horizontal/fallen
    dummy_frame[134] = 1.3
    for _ in range(30):
        detector.frame_buffer.append(dummy_frame.copy())

    mock_mgr = MagicMock()
    with patch("src.real_time_detector.alert_manager", mock_mgr):
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(detector.pose_extractor, "extract_keypoints", return_value=(True, dummy_frame, [])):
            # Frame 1: count=1
            detector.process_frame(raw_frame)
            assert detector.consecutive_fall_count == 1
            mock_mgr.start_alarm.assert_not_called()

            # Frame 2: count=2
            detector.process_frame(raw_frame)
            assert detector.consecutive_fall_count == 2
            mock_mgr.start_alarm.assert_not_called()

            # Frame 3: count=3 -> CONFIRMED!
            _, tele3 = detector.process_frame(raw_frame)
            assert detector.consecutive_fall_count >= 3
            assert detector.is_fall_confirmed is True
            assert tele3["is_fall"] is True
            assert tele3["alarm_active"] is True
            mock_mgr.start_alarm.assert_called_once()

        detector.close()


# ---------------------------------------------------------------------------
# Alert Acknowledgement Tests
# ---------------------------------------------------------------------------

def test_alert_acknowledgement_stops_alarm_and_resets_state():
    """Verify acknowledge_alert stops alarm, resets confirmation state, and sets alert_acknowledged."""
    detector = RealTimeFallDetector()
    detector._alert_was_firing = True
    detector.is_fall_confirmed = True
    detector.is_fall_detected = True
    detector.consecutive_fall_count = 10

    mock_mgr = MagicMock()
    with patch("src.real_time_detector.alert_manager", mock_mgr):
        result = detector.acknowledge_alert()
        assert result is True
        mock_mgr.stop_alarm.assert_called_once()
        assert detector._alert_was_firing is False
        assert detector.is_fall_confirmed is False
        assert detector.is_fall_detected is False
        assert detector.consecutive_fall_count == 0
        assert detector.alert_acknowledged is True
        assert detector.alert_acknowledged_at > 0.0

    detector.close()


def test_alert_acknowledgement_when_no_alert_active():
    """Verify acknowledge_alert returns False gracefully if no alert is currently active."""
    detector = RealTimeFallDetector()
    detector._alert_was_firing = False
    detector.is_fall_confirmed = False

    result = detector.acknowledge_alert()
    assert result is False
    detector.close()


# ---------------------------------------------------------------------------
# Duplicate Alert Prevention & Cooldown Tests
# ---------------------------------------------------------------------------

def test_duplicate_prevention_while_alarm_is_already_firing():
    """Verify alarm is not re-started on every subsequent frame of an ongoing fall event."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=2,
    )
    detector.has_scaler = False
    detector.model = DummyModel(0.95)

    dummy_frame = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    dummy_frame[132] = 0.85
    dummy_frame[134] = 1.3
    for _ in range(30):
        detector.frame_buffer.append(dummy_frame.copy())

    mock_mgr = MagicMock()
    with patch("src.real_time_detector.alert_manager", mock_mgr):
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(detector.pose_extractor, "extract_keypoints", return_value=(True, dummy_frame, [])):
            # Frame 1 & 2: reach confirmation
            detector.process_frame(raw_frame)
            detector.process_frame(raw_frame)
            assert mock_mgr.start_alarm.call_count == 1

            # Frames 3, 4, 5: still in confirmed fall state
            detector.process_frame(raw_frame)
            detector.process_frame(raw_frame)
            detector.process_frame(raw_frame)

            # Crucial: start_alarm should STILL only have been called ONCE
            assert mock_mgr.start_alarm.call_count == 1

        detector.close()


def test_cooldown_prevents_duplicate_alarm_after_acknowledgement():
    """Verify cooldown period prevents immediate re-triggering of the alarm after dismissal."""
    detector = RealTimeFallDetector(
        threshold=0.65,
        alert_confirmation_frames=1,
    )
    detector.has_scaler = False
    detector.model = DummyModel(0.95)
    detector.alert_ack_cooldown = 10.0  # 10s cooldown

    dummy_frame = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    dummy_frame[132] = 0.85
    dummy_frame[134] = 1.3
    for _ in range(30):
        detector.frame_buffer.append(dummy_frame.copy())

    mock_mgr = MagicMock()
    with patch("src.real_time_detector.alert_manager", mock_mgr):
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(detector.pose_extractor, "extract_keypoints", return_value=(True, dummy_frame, [])):
            # 1. Trigger first alert
            detector.process_frame(raw_frame)
            assert mock_mgr.start_alarm.call_count == 1

            # 2. User acknowledges alert
            detector.acknowledge_alert()
            assert detector.alert_acknowledged is True

            # 3. Next frame immediately following acknowledgment while still on floor:
            # Reached confirmation count 1, but ack cooldown is active
            detector.process_frame(raw_frame)

            # Alarm must NOT start again during cooldown
            assert mock_mgr.start_alarm.call_count == 1

            # 4. Fast-forward past cooldown
            detector.alert_acknowledged_at = time.monotonic() - 15.0  # > 10.0s ago
            detector._alert_was_firing = False
            detector.process_frame(raw_frame)

            # Now alarm is allowed to start for a new sustained event
            assert mock_mgr.start_alarm.call_count == 2

        detector.close()
