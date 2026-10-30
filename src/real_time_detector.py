"""Real-Time Fall Detection Engine.
Performs in-memory frame extraction, sliding window buffer management,
LSTM prediction, configurable alert confirmation rules, local audio alerts,
and privacy-preserving skeleton rendering.
"""

import sys
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import cv2

# Bootstrap project root for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    FALL_PROBABILITY_THRESHOLD,
    DEFAULT_MODEL_PATH,
    SCALER_PATH,
    MODELS_DIR,
)
from src.pose_extractor import PoseExtractor
from src.preprocessor import KeypointPreprocessor
from src.lstm_model import load_model_for_inference
from src.audio_alert import alert_manager


class CameraAccessError(IOError):
    """Raised when webcam hardware is inaccessible, permissions denied, or stream disconnected."""
    pass


class ModelLoadingError(RuntimeError):
    """Raised when model file is missing or weights cannot be loaded into TensorFlow."""
    pass


def open_camera_safe(camera_id: int = 0) -> cv2.VideoCapture:
    """Attempts to open webcam with explicit validation of hardware and frame capture.

    Raises:
        CameraAccessError: If device cannot be opened or fails to yield frames.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise CameraAccessError(
            f"Cannot access camera with ID {camera_id}. "
            "Please check camera connections and Windows privacy/camera permissions."
        )

    # Validate actual frame read
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        cap.release()
        raise CameraAccessError(
            f"Camera {camera_id} is recognized but returned empty or invalid frames."
        )

    return cap


class RealTimeFallDetector:
    """Processes live video frames in-memory without saving raw frames to disk."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold: float = 0.65,
        alert_confirmation_frames: int = 10,
        debug: bool = True,
        stable_disp_max: float = 0.18,
        stable_speed_max: float = 0.025,
        fall_disp_min: float = 0.20,
        fall_speed_min: float = 0.018,
    ):
        """
        Args:
            model_path: Path to trained Keras LSTM model.
            scaler_path: Path to fitted StandardScaler.
            threshold: Probability decision boundary for candidate fall (tuned from validation set).
            alert_confirmation_frames: Number of consecutive positive windows required
                                       before confirming an alert (prevents noise spikes).
            debug: Whether to print per-sequence diagnosis info to console.

            Stability Guard Thresholds (all in units of MediaPipe normalised frame-height fraction,
            i.e. 0.0 = top of frame, 1.0 = bottom of frame):

            stable_disp_max: Maximum vertical hip displacement over the 30-frame window that
                             still counts as a stable (non-falling) posture.
                             [units: fraction of frame height per 30-frame window]
                             Default 0.18 (~18% of frame height, ~100 px in 540p frame).
                             Calibrated from: normal breathing sway ~0.012, minor fidgeting ~0.033,
                             reaching forward ~0.043.  Slow leaning (~0.10) and slow sit-down (~0.20)
                             are classified as FALL_DYN by fall_disp_min, not suppressed here.

            stable_speed_max: Maximum mean per-frame vertical hip speed that still counts as stable.
                             [units: fraction of frame height per frame (1 frame = 1/30 s)]
                             Default 0.025.  Normal fidgeting peaks at ~0.013 (99th-percentile).

            fall_disp_min: Minimum vertical hip displacement required (together with prob >= threshold
                           and not is_stable_upright) to trigger has_fall_dynamics.
                           [units: fraction of frame height per 30-frame window]
                           Default 0.20. Slow sit-down is ~0.20; rapid falls are 0.35+.

            fall_speed_min: Minimum mean per-frame hip speed to trigger has_fall_dynamics.
                           [units: fraction of frame height per frame]
                           Default 0.018. Standing-up-from-chair peaks at ~0.009 (99th-pct),
                           keeping deliberate body movements from false-triggering fall dynamics.
        """
        self.threshold = threshold
        self.alert_confirmation_frames = max(1, alert_confirmation_frames)
        self.debug = debug

        # Stability guard thresholds (configurable, units documented above)
        self.stable_disp_max  = stable_disp_max
        self.stable_speed_max = stable_speed_max
        self.fall_disp_min    = fall_disp_min
        self.fall_speed_min   = fall_speed_min

        self.pose_extractor = PoseExtractor()
        self.preprocessor = KeypointPreprocessor()

        # Load scaler
        self.has_scaler = self.preprocessor.load_scaler(scaler_path)
        if not self.has_scaler:
            print("[RealTimeFallDetector] Notice: Feature scaler not found. Raw normalized features will be used.")

        # Load model safely
        self.model = None
        self.model_error = None
        try:
            self.model = load_model_for_inference(model_path)
            print(f"[RealTimeFallDetector] Verified LSTM model loaded successfully from {model_path}.")
        except Exception as e:
            self.model_error = str(e)
            print(f"[RealTimeFallDetector] Model loading notice: {e}. Running in pass-through pose mode.")

        # In-memory temporal window buffer (30 frames)
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions: List[Dict[str, Any]] = []

        # Alert confirmation state
        self.last_probability = 0.0
        self.consecutive_fall_count = 0
        self.is_fall_candidate = False
        self.is_fall_confirmed = False
        self.is_fall_detected = False
        self.last_pose_detected = False
        self.frames_processed = 0

        # Acknowledgement / duplicate-event guard
        # After the user acknowledges an alert, no new alarm fires for alert_ack_cooldown seconds.
        self._alert_was_firing = False        # True while alarm is running for the current event
        self.alert_acknowledged = False       # Set to True when user presses SPACE
        self.alert_acknowledged_at = 0.0     # monotonic timestamp of last acknowledgement
        self.alert_ack_cooldown = 15.0       # seconds before the next alarm can fire after ack

    def _evaluate_temporal_window(self, seq_array: np.ndarray) -> Dict[str, Any]:
        """Analyzes 30-frame temporal window for multi-factor posture and motion dynamics.

        Combines biomechanical posture (torso angle, hip height ratio, aspect ratio)
        and normalized temporal displacement to distinguish stable upright postures
        (sitting, standing) from true falls without relying solely on a downward-velocity gate.
        """
        # Feature columns:
        # index 132: torso_angle (normalized by pi, ~0.50 is vertical)
        # index 133: hip_y_ratio (mid-hip y coordinate in [0, 1])
        # index 134: aspect_ratio (bbox width / height)
        torso_angles = seq_array[:, 132]
        hip_ys = seq_array[:, 133]
        aspect_ratios = seq_array[:, 134]

        # Recent posture across the last 5 frames (~160 ms at 30 fps)
        recent_angle = float(np.mean(torso_angles[-5:]))
        recent_aspect = float(np.mean(aspect_ratios[-5:]))

        # Full-window motion metrics
        hip_displacement = float(np.max(hip_ys) - np.min(hip_ys))
        hip_speed = float(np.mean(np.abs(np.diff(hip_ys))))

        # Posture classification
        is_upright = (0.35 <= recent_angle <= 0.65)
        is_horizontal = (recent_aspect > 1.1) or (recent_angle < 0.25 or recent_angle > 0.75)

        # Stable upright: subject is currently upright AND motion over the full 30-frame window is
        # below the configurable thresholds.  Uses self.stable_disp_max / self.stable_speed_max.
        # Calibrated against: breathing sway (disp~0.012), minor fidgeting (disp~0.033, speed~0.009),
        # desk reaching (disp~0.043).  All well below the 0.18 / 0.025 defaults.
        is_stable_upright = (
            is_upright
            and (hip_displacement < self.stable_disp_max)
            and (hip_speed < self.stable_speed_max)
        )

        # Significant fall-like dynamics: large vertical displacement OR rapid hip motion OR
        # body has collapsed to horizontal.  Uses self.fall_disp_min / self.fall_speed_min.
        # Calibrated against: slow sit-down (disp~0.20), rapid fall (disp~0.35+, speed~0.012+).
        # Deliberate standing-up-from-chair (speed~0.009) does NOT cross fall_speed_min=0.018.
        has_fall_dynamics = (
            (hip_displacement >= self.fall_disp_min)
            or (hip_speed >= self.fall_speed_min)
            or is_horizontal
        )

        return {
            "recent_angle": recent_angle,
            "recent_aspect": recent_aspect,
            "hip_displacement": hip_displacement,
            "hip_speed": hip_speed,
            "is_upright": is_upright,
            "is_horizontal": is_horizontal,
            "is_stable_upright": is_stable_upright,
            "has_fall_dynamics": has_fall_dynamics,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        skeleton_only_mode: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes a single BGR frame in-memory.

        Args:
            frame_bgr: Raw camera frame in RAM.
            skeleton_only_mode: If True, returns a completely black background with only
                               skeleton lines for 100% visual privacy.

        Returns:
            Tuple of (annotated_frame, telemetry_dict)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8), {
                "pose_detected": False,
                "error": "Empty frame provided",
            }

        self.frames_processed += 1
        h, w, c = frame_bgr.shape

        # 1. Extract body keypoints in RAM
        success, feature_vector, landmarks = self.pose_extractor.extract_keypoints(frame_bgr)
        self.last_pose_detected = success

        # 2. Append to sliding window buffer (fill zeros if no person visible)
        self.frame_buffer.append(feature_vector)

        # 3. Model inference when buffer is full
        if len(self.frame_buffer) == SEQUENCE_LENGTH and self.model is not None and success:
            seq_array = np.array(self.frame_buffer, dtype=np.float32)  # (30, 135)

            # Apply scaler
            if self.has_scaler:
                seq_array_scaled = self.preprocessor.transform(seq_array[np.newaxis, ...])
            else:
                seq_array_scaled = seq_array[np.newaxis, ...]

            # Model prediction
            try:
                preds = self.model.predict(seq_array_scaled, verbose=0)
                prob = float(preds[0][0])
                self.last_probability = prob

                # Multi-Factor Temporal Window Analysis
                window_eval = self._evaluate_temporal_window(seq_array)
                is_stable_upright = window_eval["is_stable_upright"]
                has_fall_dynamics = window_eval["has_fall_dynamics"]
                is_upright = window_eval["is_upright"]
                is_horizontal = window_eval["is_horizontal"]
                hip_disp = window_eval["hip_displacement"]

                # Candidate Fall Decision:
                # 1. Suppress false positive if user is in a stable upright sitting or standing posture
                if is_stable_upright:
                    is_candidate = False
                    rejection_reason = "stable_upright"
                elif prob < self.threshold:
                    # Model probability is below configured threshold
                    is_candidate = False
                    rejection_reason = "below_threshold"
                else:
                    # prob >= self.threshold AND not stable_upright.
                    # Genuine evidence of a fall requires:
                    # - Subject collapsed into a horizontal/fallen posture (is_horizontal), OR
                    # - Significant fall dynamics (has_fall_dynamics) with either loss of upright posture
                    #   (not is_upright) or a major vertical downward drop (hip_disp >= fall_disp_min).
                    # Upright walking and ordinary movements with low vertical displacement (disp < fall_disp_min)
                    # remain upright and are NOT falls.
                    if is_horizontal:
                        is_candidate = True
                        rejection_reason = None
                    elif has_fall_dynamics and (not is_upright or hip_disp >= self.fall_disp_min):
                        is_candidate = True
                        rejection_reason = None
                    else:
                        is_candidate = False
                        rejection_reason = "insufficient_fall_evidence"

                self.is_fall_candidate = is_candidate

                # Strictly consecutive confirmation:
                # Counter increases ONLY when the complete fall condition is true,
                # and RESETS TO ZERO when that condition is false.
                if is_candidate:
                    self.consecutive_fall_count += 1
                else:
                    self.consecutive_fall_count = 0

                # Confirm alert only after sustained threshold breach
                self.is_fall_confirmed = (self.consecutive_fall_count >= self.alert_confirmation_frames)
                self.is_fall_detected = self.is_fall_confirmed

                if self.is_fall_confirmed:
                    # Start the repeating alarm the FIRST time this event fires.
                    # _alert_was_firing guards against re-starting on every subsequent frame.
                    now_mono = time.monotonic()
                    ack_cooldown_ok = (
                        (now_mono - self.alert_acknowledged_at) >= self.alert_ack_cooldown
                    )
                    if not self._alert_was_firing and ack_cooldown_ok:
                        try:
                            alert_manager.start_alarm()
                        except Exception as e:
                            print(f"[RealTimeFallDetector] Error starting alarm: {e}")
                        self._alert_was_firing = True
                        self.alert_acknowledged = False
                        print("[FALL ALERT] FALL DETECTED — PLEASE CHECK IMMEDIATELY")
                else:
                    # Fall no longer confirmed — stop alarm and reset firing flag
                    if self._alert_was_firing:
                        try:
                            alert_manager.stop_alarm()
                        except Exception as e:
                            print(f"[RealTimeFallDetector] Error stopping alarm: {e}")
                        self._alert_was_firing = False

                # Temporary debug output showing predicted class and class probabilities for completed sequence
                if self.debug:
                    pred_class = "FALL" if is_candidate else "NO FALL"
                    posture_desc = "Upright" if is_upright else ("Horizontal" if window_eval["is_horizontal"] else "Tilted")
                    status_note = "Alert Confirmed!" if self.is_fall_confirmed else f"Count: {self.consecutive_fall_count}/{self.alert_confirmation_frames}"
                    print(
                        f"[DEBUG Seq #{self.frames_processed:04d}] "
                        f"Fall Prob: {prob:.4f} (Thresh: {self.threshold:.2f}) | "
                        f"Pred: {pred_class:7s} | "
                        f"Posture: {posture_desc:10s} | "
                        f"Stable: {str(is_stable_upright):5s} | "
                        f"Disp: {hip_disp:.3f} | "
                        f"{status_note}"
                    )

                # In-memory prediction telemetry log (bounded to last 100 entries)
                if len(self.recent_predictions) > 100:
                    self.recent_predictions.pop(0)

                self.recent_predictions.append({
                    "timestamp": time.time(),
                    "probability": round(prob, 4),
                    "candidate": self.is_fall_candidate,
                    "confirmed": self.is_fall_confirmed,
                    "consecutive_count": self.consecutive_fall_count,
                    "is_stable_upright": is_stable_upright,
                })
            except Exception as e:
                print(f"[RealTimeFallDetector] Inference error: {e}")
        elif not success:
            # Decay probability and reset counter when no subject is visible
            self.last_probability = max(0.0, self.last_probability * 0.8)
            self.consecutive_fall_count = 0
            self.is_fall_candidate = False
            self.is_fall_confirmed = False
            self.is_fall_detected = False
            if self._alert_was_firing:
                try:
                    alert_manager.stop_alarm()
                except Exception as e:
                    print(f"[RealTimeFallDetector] Error stopping alarm on pose loss: {e}")
                self._alert_was_firing = False

        # 4. Render privacy-preserving visualization
        if skeleton_only_mode:
            display_frame = self.pose_extractor.render_privacy_skeleton(
                frame_shape=(h, w, c),
                pose_landmarks=landmarks,
                bg_black=True,
            )
        else:
            display_frame = self.pose_extractor.render_privacy_skeleton(
                frame_shape=(h, w, c),
                pose_landmarks=landmarks,
                bg_black=False,
                base_frame=frame_bgr,
            )

        # 5. Overlay visual indicators and telemetry
        display_frame = self._draw_ui_overlays(display_frame, w, h, skeleton_only_mode)

        telemetry = {
            "pose_detected": success,
            "buffer_fill": len(self.frame_buffer),
            "buffer_target": SEQUENCE_LENGTH,
            "fall_probability": self.last_probability,
            "is_candidate": self.is_fall_candidate,
            "is_fall": self.is_fall_confirmed,
            "is_confirmed": self.is_fall_confirmed,
            "consecutive_fall_count": self.consecutive_fall_count,
            "alert_confirmation_frames": self.alert_confirmation_frames,
            "alert_acknowledged": self.alert_acknowledged,
            "alarm_active": self._alert_was_firing,
            "frames_processed": self.frames_processed,
            "model_ready": self.model is not None,
            "model_error": self.model_error,
        }
        return display_frame, telemetry

    def _draw_ui_overlays(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        skeleton_only_mode: bool,
    ) -> np.ndarray:
        """Draws informational badges, fall warnings, confirmation counters, and privacy tags."""
        # Top banner background
        cv2.rectangle(frame, (0, 0), (width, 70), (20, 20, 25), -1)

        # Privacy Mode Indicator
        privacy_text = "PRIVACY: SKELETON ONLY (NO RAW VIDEO STORED)" if skeleton_only_mode else "PRIVACY: IN-MEMORY STREAM"
        cv2.putText(frame, privacy_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Active configuration indicator
        cfg_text = f"Thresh: {self.threshold:.2f} | Confirm: {self.alert_confirmation_frames}f"
        cv2.putText(frame, cfg_text, (max(15, width - 260), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # Status text & color
        if self.model is None:
            status_text = "MODEL NOT LOADED (PASS-THROUGH POSE MODE)"
            status_color = (0, 165, 255)
        elif not self.last_pose_detected:
            status_text = "NO PERSON DETECTED"
            status_color = (180, 180, 180)
        elif self.is_fall_confirmed:
            status_text = f"*** CONFIRMED FALL DETECTED! *** ({self.last_probability * 100:.1f}%)"
            status_color = (0, 0, 255)  # Bright Red
        elif self.consecutive_fall_count > 0:
            status_text = f"CONFIRMING FALL ({self.consecutive_fall_count}/{self.alert_confirmation_frames}) ({self.last_probability * 100:.1f}%)"
            status_color = (0, 180, 255)  # Orange
        else:
            status_text = f"NORMAL / NO FALL ({self.last_probability * 100:.1f}%)"
            status_color = (0, 255, 0)  # Green

        cv2.putText(frame, status_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2, cv2.LINE_AA)

        # Flashing red perimeter box and prominent alert message when Fall Confirmed
        if self.is_fall_confirmed:
            # Animated border: flickers by using frame count modulo
            border_color = (0, 0, 255) if (self.frames_processed % 20 < 10) else (0, 0, 180)
            cv2.rectangle(frame, (4, 4), (width - 4, height - 4), border_color, 8)

            # Primary alert banner
            banner_y = height // 2 - 30
            cv2.rectangle(frame, (30, banner_y - 55), (width - 30, banner_y + 85), (0, 0, 180), -1)
            cv2.rectangle(frame, (30, banner_y - 55), (width - 30, banner_y + 85), (0, 0, 255), 3)

            cv2.putText(
                frame,
                "FALL DETECTED",
                (width // 2 - 155, banner_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "PLEASE CHECK IMMEDIATELY",
                (width // 2 - 200, banner_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2, cv2.LINE_AA,
            )
            # Dismiss hint
            cv2.putText(
                frame,
                "Press SPACE = I am OK / Dismiss Alert",
                (width // 2 - 210, banner_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # Probability meter bar at bottom
        bar_w = int(width * min(1.0, max(0.0, self.last_probability)))
        bar_color = (0, 0, 255) if self.is_fall_confirmed else ((0, 180, 255) if self.consecutive_fall_count > 0 else (0, 200, 0))
        cv2.rectangle(frame, (0, height - 12), (bar_w, height), bar_color, -1)

        # Show acknowledged state briefly
        if self.alert_acknowledged and (time.monotonic() - self.alert_acknowledged_at) < 5.0:
            cv2.putText(
                frame, "Alert dismissed. Monitoring continues.",
                (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 1, cv2.LINE_AA,
            )

        return frame

    def acknowledge_alert(self) -> bool:
        """User-triggered acknowledgement: stops the alarm and resets fall-confirmed state.

        Should be called when the user presses SPACE (or another dismiss control).
        After acknowledgement, no new alarm fires for alert_ack_cooldown seconds.

        Returns:
            bool: True if an active alarm was acknowledged, False if no alarm was running.
        """
        if not self._alert_was_firing and not self.is_fall_confirmed:
            return False

        try:
            alert_manager.stop_alarm()
        except Exception as e:
            print(f"[RealTimeFallDetector] Error stopping alarm on acknowledgement: {e}")

        self._alert_was_firing = False
        self.alert_acknowledged = True
        self.alert_acknowledged_at = time.monotonic()
        # Reset confirmation state so the window must re-accumulate for a new alert
        self.consecutive_fall_count = 0
        self.is_fall_confirmed = False
        self.is_fall_detected = False
        self.is_fall_candidate = False
        print("[FALL ALERT] Alert acknowledged by user. Monitoring continues.")
        return True

    def close(self):
        """Releases all detector resources."""
        try:
            alert_manager.stop_alarm()
        except Exception:
            pass
        self.pose_extractor.close()


def run_webcam_demo(
    camera_id: int = 0,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    scaler_path: Union[str, Path] = SCALER_PATH,
    privacy_mode: bool = True,
    confirmation_frames: int = 10,
    threshold: float = 0.65,
    debug: bool = True,
    stable_disp_max: float = 0.18,
    stable_speed_max: float = 0.025,
    fall_disp_min: float = 0.20,
    fall_speed_min: float = 0.018,
):
    """Executes standalone real-time webcam demo with OpenCV display window."""
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    # Validate and auto-pair: ensure experimental model uses experimental scaler
    if "experimental" in model_path.name.lower() and scaler_path == SCALER_PATH:
        exp_scaler = MODELS_DIR / "experimental_scaler.pkl"
        if exp_scaler.exists():
            scaler_path = exp_scaler

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print("=" * 65)
    print("  PRIVACY-PRESERVING FALL DETECTION - WEBCAM DEMO")
    print("=" * 65)
    print(f"Opening camera ID:        {camera_id}")
    print(f"Model file:              {model_path}")
    print(f"Scaler file:             {scaler_path}")
    print(f"Fall threshold:          {threshold:.2f} (validation tuned)")
    print(f"Confirmation rule:       {confirmation_frames} consecutive frames (~{confirmation_frames/30.0:.2f}s)")
    print(f"Stability guard:         stable_disp<{stable_disp_max} | stable_speed<{stable_speed_max}")
    print(f"Fall dynamics gate:      fall_disp>={fall_disp_min} | fall_speed>={fall_speed_min}")
    print(f"Debug logging:           {'ENABLED' if debug else 'DISABLED'}")
    print(f"Initial Privacy Mode:    {'SKELETON ONLY (Black Canvas)' if privacy_mode else 'Webcam Overlay'}")
    print("Keyboard Controls:")
    print("  'q' or 'ESC' : Stop camera and quit")
    print("  'p'          : Toggle privacy mode on/off")
    print("  SPACE        : I am OK / Dismiss fall alert & stop alarm")
    print("=" * 65)

    try:
        cap = open_camera_safe(camera_id)
    except CameraAccessError as e:
        print(f"\n[Camera Error] {e}")
        return False

    detector = RealTimeFallDetector(
        model_path=model_path,
        scaler_path=scaler_path,
        threshold=threshold,
        alert_confirmation_frames=confirmation_frames,
        debug=debug,
        stable_disp_max=stable_disp_max,
        stable_speed_max=stable_speed_max,
        fall_disp_min=fall_disp_min,
        fall_speed_min=fall_speed_min,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[Camera Error] Camera disconnected or frame read failed.")
                break

            display_frame, telemetry = detector.process_frame(
                frame,
                skeleton_only_mode=privacy_mode,
            )

            cv2.imshow("Privacy-Preserving Fall Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # 'q' or ESC
                print("[Webcam Demo] Stop requested by user.")
                break
            elif key == ord("p"):
                privacy_mode = not privacy_mode
                print(f"[Webcam Demo] Privacy mode toggled: {'SKELETON ONLY' if privacy_mode else 'OVERLAY'}")
            elif key == ord(" "):  # SPACE = dismiss / I am OK
                acked = detector.acknowledge_alert()
                if acked:
                    print("[Webcam Demo] Alert dismissed by user (SPACE pressed).")
                else:
                    print("[Webcam Demo] SPACE pressed but no active alert to dismiss.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[Webcam Demo] Camera stopped and resources released safely.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Fall Detection Webcam Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device ID")
    parser.add_argument("--overlay", action="store_true",
                        help="Display skeleton over camera frame rather than black background")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Fall confidence threshold (sigmoid output, validation-tuned)")
    parser.add_argument("--confirm-frames", type=int, default=10,
                        help="Consecutive candidate windows required to confirm a fall alert")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable per-sequence debug output to console")
    parser.add_argument("--no-debug", action="store_false", dest="debug",
                        help="Disable per-sequence debug output")
    parser.add_argument(
        "--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to trained Keras model file",
    )
    parser.add_argument(
        "--scaler-path", type=str, default=str(SCALER_PATH),
        help="Path to fitted StandardScaler pickle file",
    )
    # Stability guard thresholds (configurable for different camera heights / body sizes)
    # Units: fraction of MediaPipe normalised frame height [0.0, 1.0] per 30-frame window (disp)
    #        or per frame (speed, i.e. per 1/30 s).
    parser.add_argument("--stable-disp-max", type=float, default=0.18,
                        help="Max vertical hip displacement (fraction of frame height / 30 frames) "
                             "to classify posture as stable upright. Default 0.18.")
    parser.add_argument("--stable-speed-max", type=float, default=0.025,
                        help="Max mean per-frame vertical hip speed (fraction of frame height / frame) "
                             "to classify posture as stable upright. Default 0.025.")
    parser.add_argument("--fall-disp-min", type=float, default=0.20,
                        help="Min vertical hip displacement over 30 frames to flag fall dynamics. "
                             "Default 0.20 (~20%% of frame height).")
    parser.add_argument("--fall-speed-min", type=float, default=0.018,
                        help="Min mean per-frame hip speed to flag fall dynamics. Default 0.018.")
    args = parser.parse_args()

    run_webcam_demo(
        camera_id=args.camera,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        privacy_mode=not args.overlay,
        confirmation_frames=args.confirm_frames,
        threshold=args.threshold,
        debug=args.debug,
        stable_disp_max=args.stable_disp_max,
        stable_speed_max=args.stable_speed_max,
        fall_disp_min=args.fall_disp_min,
        fall_speed_min=args.fall_speed_min,
    )
