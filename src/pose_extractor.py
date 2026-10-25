"""Pose Landmark Extraction and Biomechanical Feature Engineering.
Processes video frames strictly in-memory using MediaPipe Pose.
Computes privacy-preserving kinematic features (normalized coordinates, torso inclination, aspect ratio).
Supports MediaPipe Tasks PoseLandmarker and legacy Solutions API.
"""

import math
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

from src.config import (
    NUM_LANDMARKS,
    RAW_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
    MODELS_DIR,
)

# MediaPipe landmark indices of key anatomical joints
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Standard 33-landmark anatomical connections
SKELETON_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso sides
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31), (27, 31),  # Left foot
    (28, 30), (30, 32), (28, 32),  # Right foot
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face left
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face right
    (9, 10),  # Mouth
]

MODEL_ASSET_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_ASSET_PATH = MODELS_DIR / "pose_landmarker_lite.task"


class PoseExtractor:
    """Extracts 33 body keypoints from image frames and computes normalized biomechanical features.

    Privacy Guarantee:
    - Analyzes frames strictly in RAM.
    - Never writes raw RGB images to persistent disk or network sockets.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker = None
        self._legacy_pose = None
        self._mode = None

    def _ensure_model_asset(self) -> Path:
        """Downloads pose_landmarker_lite.task if not already cached locally."""
        if not MODEL_ASSET_PATH.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[PoseExtractor] Downloading pose landmark model to {MODEL_ASSET_PATH}...")
            urllib.request.urlretrieve(MODEL_ASSET_URL, MODEL_ASSET_PATH)
            print("[PoseExtractor] Model asset downloaded successfully.")
        return MODEL_ASSET_PATH

    def _init_detector(self):
        """Initializes MediaPipe detector supporting both Tasks API and Legacy Solutions."""
        if self._mode is not None:
            return

        import mediapipe as mp

        # Try MediaPipe Tasks API
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            asset_path = self._ensure_model_asset()
            base_options = mp_python.BaseOptions(model_asset_path=str(asset_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._mode = "tasks"
            return
        except Exception as e:
            pass

        # Try Legacy Solutions API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._legacy_pose = mp.solutions.pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._mode = "legacy"
            return

        raise RuntimeError("Could not initialize MediaPipe Pose detector (neither Tasks nor Solutions available).")

    def extract_keypoints(self, frame_bgr: np.ndarray) -> Tuple[bool, np.ndarray, Optional[List[Dict[str, float]]]]:
        """Processes a single BGR frame in-memory and extracts normalized feature vector.

        Args:
            frame_bgr: OpenCV BGR image matrix in RAM.

        Returns:
            Tuple of:
            - success (bool): True if human pose was detected.
            - feature_vector (np.ndarray): Shape (TOTAL_FEATURE_DIM,), normalized features.
            - raw_landmarks: Landmark list for drawing stick figure.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32), None

        self._init_detector()
        h, w = frame_bgr.shape[:2]

        # Extract landmarks using active mode
        raw_coords = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        landmarks_list = []

        if self._mode == "tasks":
            import mediapipe as mp
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self._landmarker.detect(mp_image)

            if not res.pose_landmarks or len(res.pose_landmarks) == 0:
                return False, np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32), None

            pose_lm = res.pose_landmarks[0]
            for i, lm in enumerate(pose_lm):
                vis = getattr(lm, "visibility", 0.90) or 0.90
                raw_coords[i] = [lm.x, lm.y, lm.z, vis]
                landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": vis})

        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self._legacy_pose.process(frame_rgb)

            if not results.pose_landmarks:
                return False, np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32), None

            for i, lm in enumerate(results.pose_landmarks.landmark):
                raw_coords[i] = [lm.x, lm.y, lm.z, lm.visibility]
                landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})

        # 2. Compute anchor points (Mid-Hip and Mid-Shoulder)
        mid_hip_x = (raw_coords[LEFT_HIP, 0] + raw_coords[RIGHT_HIP, 0]) / 2.0
        mid_hip_y = (raw_coords[LEFT_HIP, 1] + raw_coords[RIGHT_HIP, 1]) / 2.0
        mid_hip_z = (raw_coords[LEFT_HIP, 2] + raw_coords[RIGHT_HIP, 2]) / 2.0

        mid_shoulder_x = (raw_coords[LEFT_SHOULDER, 0] + raw_coords[RIGHT_SHOULDER, 0]) / 2.0
        mid_shoulder_y = (raw_coords[LEFT_SHOULDER, 1] + raw_coords[RIGHT_SHOULDER, 1]) / 2.0

        # Torso length scale
        torso_dist = math.sqrt(
            (mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2
        )
        scale_factor = max(torso_dist, 0.05)

        # 3. Position and scale-invariant normalization (hip-centered)
        norm_coords = np.zeros_like(raw_coords)
        norm_coords[:, 0] = (raw_coords[:, 0] - mid_hip_x) / scale_factor
        norm_coords[:, 1] = (raw_coords[:, 1] - mid_hip_y) / scale_factor
        norm_coords[:, 2] = (raw_coords[:, 2] - mid_hip_z) / scale_factor
        norm_coords[:, 3] = raw_coords[:, 3]

        flattened_coords = norm_coords.flatten()

        # 4. Biomechanical derived indicators
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y
        torso_angle = math.atan2(dy, dx) / math.pi
        hip_y_ratio = float(mid_hip_y)

        xs = raw_coords[:, 0]
        ys = raw_coords[:, 1]
        bbox_w = max(np.max(xs) - np.min(xs), 0.01)
        bbox_h = max(np.max(ys) - np.min(ys), 0.01)
        aspect_ratio = float(min(bbox_w / bbox_h, 5.0))

        derived_features = np.array([torso_angle, hip_y_ratio, aspect_ratio], dtype=np.float32)
        full_features = np.concatenate([flattened_coords, derived_features])

        return True, full_features, landmarks_list

    def render_privacy_skeleton(
        self,
        frame_shape: Tuple[int, int, int],
        pose_landmarks: Optional[List[Dict[str, float]]],
        bg_black: bool = True,
        base_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Renders body landmarks without identifiable facial or environmental textures.

        Args:
            frame_shape: (height, width, channels)
            pose_landmarks: List of landmarks with x, y coordinates [0, 1].
            bg_black: If True, draws skeleton onto a 100% black canvas for total privacy.
            base_frame: Optional base image if bg_black is False.

        Returns:
            np.ndarray: BGR image containing drawn stick-figure skeleton.
        """
        h, w, c = frame_shape

        if bg_black or base_frame is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = base_frame.copy()

        if not pose_landmarks or len(pose_landmarks) < NUM_LANDMARKS:
            return canvas

        # Convert normalized coordinates to pixel coordinates
        pts = {}
        for idx, lm in enumerate(pose_landmarks):
            px = int(lm["x"] * w)
            py = int(lm["y"] * h)
            pts[idx] = (px, py)

        # Draw bone connections
        for idx1, idx2 in SKELETON_CONNECTIONS:
            if idx1 in pts and idx2 in pts:
                cv2.line(canvas, pts[idx1], pts[idx2], (0, 255, 128), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, pt in pts.items():
            cv2.circle(canvas, pt, 3, (0, 180, 255), -1, cv2.LINE_AA)

        return canvas

    def close(self):
        """Releases detector resources."""
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        if self._legacy_pose:
            try:
                self._legacy_pose.close()
            except Exception:
                pass
            self._legacy_pose = None
        self._mode = None
