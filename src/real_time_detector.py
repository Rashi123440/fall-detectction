"""Real-Time Fall Detection Engine.
Performs in-memory frame extraction, sliding window buffer management,
LSTM prediction, local audio alerts, and privacy-preserving skeleton rendering.
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


class RealTimeFallDetector:
    """Processes live video frames in-memory without saving raw frames to disk."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold: float = FALL_PROBABILITY_THRESHOLD,
    ):
        self.threshold = threshold
        self.pose_extractor = PoseExtractor()
        self.preprocessor = KeypointPreprocessor()

        # Load scaler
        self.has_scaler = self.preprocessor.load_scaler(scaler_path)
        if not self.has_scaler:
            print("[RealTimeFallDetector] Warning: Feature scaler not found. Raw normalized features will be used.")

        # Load model
        self.model = None
        self.model_error = None
        try:
            self.model = load_model_for_inference(model_path)
            print("[RealTimeFallDetector] LSTM fall detection model loaded successfully.")
        except Exception as e:
            self.model_error = str(e)
            print(f"[RealTimeFallDetector] Notice: Model not loaded ({e}). Detection will run in pass-through mode.")

        # Sliding window buffer in RAM
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions: List[Dict[str, Any]] = []

        # Current state
        self.last_probability = 0.0
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
        self.frames_processed += 1
        h, w, c = frame_bgr.shape

        # 1. Extract body keypoints in RAM
        success, feature_vector, landmarks = self.pose_extractor.extract_keypoints(frame_bgr)
        self.last_pose_detected = success

        # 2. Append to sliding window buffer (fill zeros if no person visible)
        self.frame_buffer.append(feature_vector)

        # 3. Predict if buffer is full and model is loaded
        if len(self.frame_buffer) == SEQUENCE_LENGTH and self.model is not None and success:
            seq_array = np.array(self.frame_buffer, dtype=np.float32)  # (30, 135)

            # Apply scaler
            if self.has_scaler:
                seq_array_scaled = self.preprocessor.transform(seq_array[np.newaxis, ...])
            else:
                seq_array_scaled = seq_array[np.newaxis, ...]

            # Model inference
            try:
                preds = self.model.predict(seq_array_scaled, verbose=0)
                prob = float(preds[0][0])
                self.last_probability = prob
                self.is_fall_detected = prob >= self.threshold

                if self.is_fall_detected:
                    # Fire local non-blocking audio alarm
                    alert_manager.trigger_alert()

                # Keep log of recent events in memory
                if len(self.recent_predictions) > 100:
                    self.recent_predictions.pop(0)

                self.recent_predictions.append({
                    "timestamp": time.time(),
                    "probability": round(prob, 4),
                    "is_fall": self.is_fall_detected,
                })
            except Exception as e:
                print(f"[RealTimeFallDetector] Inference error: {e}")
        elif not success:
            # If no person visible, probability decays
            self.last_probability = max(0.0, self.last_probability * 0.8)
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
            "is_fall": self.is_fall_detected,
            "frames_processed": self.frames_processed,
            "model_ready": self.model is not None,
        }
        return display_frame, telemetry

    def _draw_ui_overlays(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        skeleton_only_mode: bool,
    ) -> np.ndarray:
        """Draws informational badges, fall warnings, and privacy tags."""
        # Top banner background
        cv2.rectangle(frame, (0, 0), (width, 70), (20, 20, 25), -1)

        # Privacy Mode Indicator
        privacy_text = "PRIVACY: SKELETON ONLY (NO RAW VIDEO STORED)" if skeleton_only_mode else "PRIVACY: IN-MEMORY STREAM"
        cv2.putText(frame, privacy_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Status & Probability text
        if self.model is None:
            status_text = "MODEL NOT LOADED (PASS-THROUGH POSE MODE)"
            status_color = (0, 165, 255)
        elif not self.last_pose_detected:
            status_text = "NO PERSON DETECTED"
            status_color = (180, 180, 180)
        elif self.is_fall_detected:
            status_text = f"*** FALL DETECTED! *** ({self.last_probability * 100:.1f}%)"
            status_color = (0, 0, 255)  # Bright Red
        else:
            status_text = f"NORMAL MOVEMENT ({self.last_probability * 100:.1f}%)"
            status_color = (0, 255, 0)  # Green

        cv2.putText(frame, status_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

        # Large flashing warning box if Fall Detected
        if self.is_fall_detected:
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 6)
            # Centered alert banner
            banner_y = height // 2
            cv2.rectangle(frame, (50, banner_y - 40), (width - 50, banner_y + 40), (0, 0, 200), -1)
            cv2.putText(
                frame,
                "ALERT: FALL DETECTED!",
                (width // 2 - 200, banner_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # Probability meter bar at bottom
        bar_w = int(width * self.last_probability)
        bar_color = (0, 0, 255) if self.is_fall_detected else (0, 200, 0)
        cv2.rectangle(frame, (0, height - 12), (bar_w, height), bar_color, -1)

        return frame

    def close(self):
        """Releases all resources."""
        self.pose_extractor.close()


def run_webcam_demo(
    camera_id: int = 0,
    model_path=DEFAULT_MODEL_PATH,
    privacy_mode: bool = True,
):
    """Executes standalone real-time webcam demo with OpenCV display window."""
    print("=" * 60)
    print("  PRIVACY-PRESERVING FALL DETECTION - WEBCAM DEMO")
    print("=" * 60)
    print(f"Opening camera index {camera_id}...")
    print("Privacy Mode:", "SKELETON ONLY (Black canvas)" if privacy_mode else "Skeleton Overlay")
    print("Press 'q' or 'ESC' in the video window to quit.")
    print("Press 'p' to toggle Privacy Mode on/off.")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"\n[Camera Error] Could not access camera ID {camera_id}.")
        print("Please check camera connections and Windows privacy/camera permissions.")
        return

    detector = RealTimeFallDetector(model_path=model_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera Error] Failed to read frame from webcam.")
                break

            display_frame, telemetry = detector.process_frame(
                frame,
                skeleton_only_mode=privacy_mode,
            )

            cv2.imshow("Privacy-Preserving Fall Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # 'q' or ESC
                print("[Webcam Demo] Quit requested by user.")
                break
            elif key == ord("p"):
                privacy_mode = not privacy_mode
                print(f"[Webcam Demo] Privacy mode toggled: {privacy_mode}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[Webcam Demo] Camera stopped and resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--overlay", action="store_true", help="Display skeleton over camera frame rather than black background")
    args = parser.parse_args()

    run_webcam_demo(camera_id=args.camera, privacy_mode=not args.overlay)
