"""Biomechanical Synthetic Kinematic Data Generator.
Used exclusively for pipeline verification, unit testing, and architectural validation.
Simulates temporal 3D trajectory patterns for 33 MediaPipe landmarks across ADLs and Falls.

DISCLAIMER:
Synthetic kinematic data is intended for development and automated testing only.
Do not use synthetic data to claim real-world fall detection accuracy.
"""

import math
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np

from src.config import (
    NUM_LANDMARKS,
    RAW_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
    SEQUENCE_LENGTH,
    LABEL_NO_FALL,
    LABEL_FALL,
    SYNTHETIC_DATA_DIR,
)

# Reference landmark base offsets in standard upright T/A-pose relative to mid-hip
# Format: [dx, dy, dz] where dy is negative going up towards head, positive down towards feet
BASE_SKELETON_OFFSETS = {
    # Torso & Head
    0: [0.0, -1.05, 0.0],    # nose
    11: [-0.35, -0.75, 0.0], # left_shoulder
    12: [0.35, -0.75, 0.0],  # right_shoulder
    23: [-0.20, 0.0, 0.0],   # left_hip
    24: [0.20, 0.0, 0.0],    # right_hip
    # Left Arm
    13: [-0.45, -0.40, 0.0], # left_elbow
    15: [-0.48, -0.05, 0.0], # left_wrist
    # Right Arm
    14: [0.45, -0.40, 0.0],  # right_elbow
    16: [0.48, -0.05, 0.0],  # right_wrist
    # Left Leg
    25: [-0.22, 0.50, 0.0],  # left_knee
    27: [-0.22, 1.00, 0.0],  # left_ankle
    # Right Leg
    26: [0.22, 0.50, 0.0],   # right_knee
    28: [0.22, 1.00, 0.0],   # right_ankle
}


class SyntheticKinematicsGenerator:
    """Generates synthetic temporal keypoint sequences simulating human movement kinematics."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _generate_upright_frame(
        self,
        center_x: float,
        center_y: float,
        torso_angle_deg: float = 90.0,
        scale: float = 0.35,
        noise_std: float = 0.01,
    ) -> np.ndarray:
        """Constructs a 33-landmark coordinate array for a given posture."""
        frame = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)

        # Rotation matrix around Z-axis (2D image plane)
        angle_rad = math.radians(torso_angle_deg - 90.0)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for lm_id in range(NUM_LANDMARKS):
            if lm_id in BASE_SKELETON_OFFSETS:
                dx, dy, dz = BASE_SKELETON_OFFSETS[lm_id]
            else:
                # Approximate other landmarks (ears, hands, feet) near parent joints
                if lm_id < 11:  # Face
                    dx, dy, dz = (self.rng.uniform(-0.1, 0.1), -1.0 + self.rng.uniform(-0.1, 0.1), 0.0)
                elif lm_id in [17, 19, 21]:  # Left hand
                    dx, dy, dz = (-0.50, 0.0, 0.0)
                elif lm_id in [18, 20, 22]:  # Right hand
                    dx, dy, dz = (0.50, 0.0, 0.0)
                elif lm_id in [29, 31]:  # Left foot
                    dx, dy, dz = (-0.22, 1.05, 0.05)
                else:  # Right foot
                    dx, dy, dz = (0.22, 1.05, 0.05)

            # Apply rotation & scale
            rx = (dx * cos_a - dy * sin_a) * scale
            ry = (dx * sin_a + dy * cos_a) * scale
            rz = dz * scale

            # Add sensor noise & anchor at center
            x = center_x + rx + self.rng.normal(0, noise_std)
            y = center_y + ry + self.rng.normal(0, noise_std)
            z = rz + self.rng.normal(0, noise_std)
            visibility = max(0.6, min(0.99, self.rng.normal(0.90, 0.05)))

            frame[lm_id] = [x, y, z, visibility]

        return frame

    def _convert_raw_frame_to_feature_vector(self, raw_frame: np.ndarray) -> np.ndarray:
        """Converts (33, 4) landmark array to (TOTAL_FEATURE_DIM,) feature vector."""
        mid_hip_x = (raw_frame[23, 0] + raw_frame[24, 0]) / 2.0
        mid_hip_y = (raw_frame[23, 1] + raw_frame[24, 1]) / 2.0
        mid_hip_z = (raw_frame[23, 2] + raw_frame[24, 2]) / 2.0

        mid_shoulder_x = (raw_frame[11, 0] + raw_frame[12, 0]) / 2.0
        mid_shoulder_y = (raw_frame[11, 1] + raw_frame[12, 1]) / 2.0

        torso_dist = math.sqrt(
            (mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2
        )
        scale_factor = max(torso_dist, 0.05)

        norm_coords = np.zeros_like(raw_frame)
        norm_coords[:, 0] = (raw_frame[:, 0] - mid_hip_x) / scale_factor
        norm_coords[:, 1] = (raw_frame[:, 1] - mid_hip_y) / scale_factor
        norm_coords[:, 2] = (raw_frame[:, 2] - mid_hip_z) / scale_factor
        norm_coords[:, 3] = raw_frame[:, 3]

        flattened = norm_coords.flatten()

        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y
        torso_angle = math.atan2(dy, dx) / math.pi
        hip_y_ratio = float(mid_hip_y)

        xs = raw_frame[:, 0]
        ys = raw_frame[:, 1]
        bbox_w = max(np.max(xs) - np.min(xs), 0.01)
        bbox_h = max(np.max(ys) - np.min(ys), 0.01)
        aspect_ratio = float(min(bbox_w / bbox_h, 5.0))

        derived = np.array([torso_angle, hip_y_ratio, aspect_ratio], dtype=np.float32)
        return np.concatenate([flattened, derived])

    def generate_sequence(self, is_fall: bool, seq_length: int = SEQUENCE_LENGTH) -> np.ndarray:
        """Generates a temporal sequence array of shape (seq_length, TOTAL_FEATURE_DIM)."""
        sequence_features = np.zeros((seq_length, TOTAL_FEATURE_DIM), dtype=np.float32)

        base_cx = self.rng.uniform(0.40, 0.60)
        base_cy = self.rng.uniform(0.45, 0.55)
        base_scale = self.rng.uniform(0.30, 0.38)

        if not is_fall:
            # ADL: Walking / Normal movement
            activity_type = self.rng.choice(["walk", "sit", "bend", "stand"])
            for t in range(seq_length):
                phase = (t / seq_length) * 2 * math.pi
                if activity_type == "walk":
                    cx = base_cx + 0.05 * math.sin(phase)
                    cy = base_cy + 0.015 * math.cos(2 * phase)  # Vertical oscillation
                    angle = 90.0 + self.rng.uniform(-4.0, 4.0)
                elif activity_type == "sit":
                    # Gradual descent then stable
                    descent = min(1.0, t / (seq_length * 0.7))
                    cx = base_cx
                    cy = base_cy + 0.15 * descent
                    angle = 90.0 - 15.0 * descent
                elif activity_type == "bend":
                    # Bending forward and returning
                    bend_amp = math.sin((t / seq_length) * math.pi)
                    cx = base_cx
                    cy = base_cy + 0.08 * bend_amp
                    angle = 90.0 - 45.0 * bend_amp
                else:  # standing
                    cx = base_cx + self.rng.normal(0, 0.005)
                    cy = base_cy + self.rng.normal(0, 0.003)
                    angle = 90.0 + self.rng.normal(0, 1.5)

                raw_frame = self._generate_upright_frame(cx, cy, angle, base_scale)
                sequence_features[t] = self._convert_raw_frame_to_feature_vector(raw_frame)

        else:
            # Fall event: Normal upright for 1st third, sudden fall collapse, horizontal posture on floor
            fall_start = int(seq_length * self.rng.uniform(0.25, 0.40))
            fall_duration = int(seq_length * self.rng.uniform(0.20, 0.30))
            fall_end = min(seq_length, fall_start + fall_duration)
            fall_direction = self.rng.choice([0.0, 180.0])  # Forward or backward collapse

            for t in range(seq_length):
                if t < fall_start:
                    # Upright before fall
                    cx = base_cx + self.rng.normal(0, 0.005)
                    cy = base_cy + self.rng.normal(0, 0.005)
                    angle = 90.0 + self.rng.normal(0, 2.0)
                elif t < fall_end:
                    # In mid-fall: rapid acceleration downwards and rotation
                    progress = (t - fall_start) / max(1, fall_duration)
                    # Non-linear quadratic acceleration: d = 0.5 * g * t^2
                    accel_curve = progress ** 2
                    cx = base_cx + (0.15 if fall_direction == 0.0 else -0.15) * progress
                    cy = base_cy + (0.35 * accel_curve)  # Rapid drop towards ground (0.80+)
                    angle = 90.0 + (fall_direction - 90.0) * progress
                else:
                    # Fallen on ground (horizontal resting state)
                    cx = base_cx + (0.15 if fall_direction == 0.0 else -0.15) + self.rng.normal(0, 0.002)
                    cy = min(0.88, base_cy + 0.35 + self.rng.normal(0, 0.003))
                    angle = fall_direction + self.rng.normal(0, 1.0)

                raw_frame = self._generate_upright_frame(cx, cy, angle, base_scale)
                sequence_features[t] = self._convert_raw_frame_to_feature_vector(raw_frame)

        return sequence_features

    def generate_dataset(
        self,
        num_samples: int = 400,
        fall_ratio: float = 0.5,
        seq_length: int = SEQUENCE_LENGTH,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generates a complete balanced synthetic dataset with metadata.

        Returns:
            X (np.ndarray): Shape (num_samples, seq_length, TOTAL_FEATURE_DIM)
            y (np.ndarray): Shape (num_samples,), 0 for ADL, 1 for Fall
            metadata (dict): Information detailing synthetic nature of data
        """
        num_falls = int(num_samples * fall_ratio)
        num_adls = num_samples - num_falls

        X = np.zeros((num_samples, seq_length, TOTAL_FEATURE_DIM), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int32)

        idx = 0
        for _ in range(num_adls):
            X[idx] = self.generate_sequence(is_fall=False, seq_length=seq_length)
            y[idx] = LABEL_NO_FALL
            idx += 1

        for _ in range(num_falls):
            X[idx] = self.generate_sequence(is_fall=True, seq_length=seq_length)
            y[idx] = LABEL_FALL
            idx += 1

        # Shuffle indices
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)
        X = X[indices]
        y = y[indices]

        metadata = {
            "is_synthetic": True,
            "disclaimer": "Generated for pipeline testing and verification. Do not report as real-world clinical performance.",
            "num_samples": num_samples,
            "num_falls": num_falls,
            "num_adls": num_adls,
            "sequence_length": seq_length,
            "feature_dim": TOTAL_FEATURE_DIM,
        }
        return X, y, metadata


def create_and_save_synthetic_dataset(
    output_dir: Path = SYNTHETIC_DATA_DIR,
    num_samples: int = 500,
) -> Tuple[Path, Path, Path]:
    """Utility function to create and save synthetic benchmark arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticKinematicsGenerator(seed=123)
    X, y, meta = generator.generate_dataset(num_samples=num_samples)

    x_path = output_dir / "synthetic_X.npy"
    y_path = output_dir / "synthetic_y.npy"
    meta_path = output_dir / "synthetic_metadata.json"

    np.save(x_path, X)
    np.save(y_path, y)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SyntheticGenerator] Created synthetic dataset: {X.shape} at {output_dir}")
    return x_path, y_path, meta_path
