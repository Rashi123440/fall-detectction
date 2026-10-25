"""Real-Time Fall Detection Engine.
Performs in-memory frame extraction, sliding window buffer management,
LSTM prediction, configurable alert confirmation rules, local audio alerts,
and privacy-preserving skeleton rendering.
"""

import time
import argparse
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

from src.config import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    FALL_PROBABILITY_THRESHOLD,
    DEFAULT_MODEL_PATH,
    SCALER_PATH,
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
        threshold: float = FALL_PROBABILITY_THRESHOLD,
        alert_confirmation_frames: int = 3,
    ):
        """
        Args:
            model_path: Path to trained Keras LSTM model.
            scaler_path: Path to fitted StandardScaler.
            threshold: Probability decision boundary for candidate fall.
            alert_confirmation_frames: Number of consecutive positive windows required
                                       before confirming an alert (prevents noise spikes).
        """
        self.threshold = threshold
        self.alert_confirmation_frames = max(1, alert_confirmation_frames)
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
            print("[RealTimeFallDetector] Verified LSTM fall detection model loaded successfully.")
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

                # Alert Confirmation Rule
                is_candidate = prob >= self.threshold
                self.is_fall_candidate = is_candidate

                if is_candidate:
                    self.consecutive_fall_count += 1
                else:
                    self.consecutive_fall_count = max(0, self.consecutive_fall_count - 1)

                # Confirm alert only after sustained threshold breach
                self.is_fall_confirmed = (self.consecutive_fall_count >= self.alert_confirmation_frames)
                self.is_fall_detected = self.is_fall_confirmed

                if self.is_fall_confirmed:
                    # Trigger non-blocking local audio alarm
                    alert_manager.trigger_alert()

                # In-memory prediction telemetry log (bounded to last 100 entries)
                if len(self.recent_predictions) > 100:
                    self.recent_predictions.pop(0)

                self.recent_predictions.append({
                    "timestamp": time.time(),
                    "probability": round(prob, 4),
                    "candidate": self.is_fall_candidate,
                    "confirmed": self.is_fall_confirmed,
                    "consecutive_count": self.consecutive_fall_count,
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

        # Flashing red perimeter box if Fall Confirmed
        if self.is_fall_confirmed:
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 6)
            banner_y = height // 2
            cv2.rectangle(frame, (50, banner_y - 45), (width - 50, banner_y + 45), (0, 0, 200), -1)
            cv2.putText(
                frame,
                "ALERT: FALL CONFIRMED!",
                (width // 2 - 210, banner_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # Probability meter bar at bottom
        bar_w = int(width * min(1.0, max(0.0, self.last_probability)))
        bar_color = (0, 0, 255) if self.is_fall_confirmed else ((0, 180, 255) if self.consecutive_fall_count > 0 else (0, 200, 0))
        cv2.rectangle(frame, (0, height - 12), (bar_w, height), bar_color, -1)

        return frame

    def close(self):
        """Releases all detector resources."""
        self.pose_extractor.close()


def run_webcam_demo(
    camera_id: int = 0,
    model_path=DEFAULT_MODEL_PATH,
    privacy_mode: bool = True,
    confirmation_frames: int = 3,
):
    """Executes standalone real-time webcam demo with OpenCV display window."""
    print("=" * 65)
    print("  PRIVACY-PRESERVING FALL DETECTION - WEBCAM DEMO")
    print("=" * 65)
    print(f"Opening camera ID:        {camera_id}")
    print(f"Confirmation rule:       {confirmation_frames} consecutive frames")
    print(f"Initial Privacy Mode:    {'SKELETON ONLY (Black Canvas)' if privacy_mode else 'Webcam Overlay'}")
    print("Keyboard Controls:")
    print("  'q' or 'ESC' : Stop camera and quit")
    print("  'p'          : Toggle privacy mode on/off")
    print("=" * 65)

    try:
        cap = open_camera_safe(camera_id)
    except CameraAccessError as e:
        print(f"\n[Camera Error] {e}")
        return False

    detector = RealTimeFallDetector(
        model_path=model_path,
        alert_confirmation_frames=confirmation_frames,
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[Webcam Demo] Camera stopped and resources released safely.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--overlay", action="store_true", help="Display skeleton over camera frame rather than black background")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Confirmation frames rule (default: 3)")
    args = parser.parse_args()

    run_webcam_demo(
        camera_id=args.camera,
        privacy_mode=not args.overlay,
        confirmation_frames=args.confirm_frames,
    )
