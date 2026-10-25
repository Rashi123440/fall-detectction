"""Reproducible UP-Fall Keypoint Dataset Preparation Module.
Parses downloaded UP-Fall 3D keypoint CSV files (Joint1_X..Joint33_Z, LABEL),
applies scale-invariant hip-centered normalization and derived biomechanical features,
and generates temporal sequence sliding windows (30 frames, 135 features) without raw video frames.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    NUM_LANDMARKS,
    TOTAL_FEATURE_DIM,
    SEQUENCE_LENGTH,
    STRIDE_STEP,
    LABEL_FALL,
    LABEL_NO_FALL,
)

# UP-Fall CSV joint column mapping:
# 1-indexed Joint names in CSV:
# Joint12 = Left Shoulder, Joint13 = Right Shoulder
# Joint24 = Left Hip, Joint25 = Right Hip
LEFT_SHOULDER_IDX = 11  # 0-indexed (Joint 12)
RIGHT_SHOULDER_IDX = 12 # 0-indexed (Joint 13)
LEFT_HIP_IDX = 22       # 0-indexed (Joint 23 in some or Joint 24)
RIGHT_HIP_IDX = 23      # 0-indexed (Joint 24 in some or Joint 25)


class UPFallDataPreparer:
    """Processes downloaded UP-Fall keypoint CSVs into privacy-preserving training sequences."""

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR / "up_fall",
        output_dir: Path = PROCESSED_DATA_DIR,
        sequence_length: int = SEQUENCE_LENGTH,
        stride: int = STRIDE_STEP,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.stride = stride

        # Verify raw directory
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"UP-Fall directory '{self.raw_dir}' not found. "
                "Please place downloaded SUBJECT folders in data/raw/up_fall/."
            )

    def discover_files(self) -> List[Path]:
        """Finds all UP-Fall CSV files recursively."""
        files = sorted(list(self.raw_dir.rglob("*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")
        return files

    def inspect_dataset(self) -> Dict[str, Any]:
        """Performs initial inspection of actual files, headers, and labels without guessing."""
        files = self.discover_files()

        # Group by subject folders
        subjects = sorted(list({f.parent.name for f in files}))

        # Read first file to inspect schema
        sample_df = pd.read_csv(files[0])
        columns = list(sample_df.columns)

        has_label_col = "LABEL" in columns or "Label" in columns or "label" in columns
        label_col = "LABEL" if "LABEL" in columns else ("Label" if "Label" in columns else "label")

        joint_cols = [c for c in columns if c.startswith("Joint") or c.startswith("joint")]

        # Count total frames and label distribution
        total_frames = 0
        label_counts = {0: 0, 1: 0}

        for f in files:
            df = pd.read_csv(f)
            total_frames += len(df)
            labels = df.iloc[:, 99].astype(int)
            vc = labels.value_counts().to_dict()
            for k, v in vc.items():
                label_counts[int(k)] = label_counts.get(int(k), 0) + int(v)

        inspection_report = {
            "total_files": len(files),
            "subjects_found": subjects,
            "total_raw_frames": total_frames,
            "total_columns": len(columns),
            "joint_columns_count": len(joint_cols),
            "label_column_name": label_col,
            "frame_label_distribution": label_counts,
            "sample_file": str(files[0].name),
        }
        return inspection_report

    def _normalize_frame_keypoints(self, raw_99_coords: np.ndarray) -> np.ndarray:
        """Converts raw (99,) [x1,y1,z1...x33,y33,z33] into hip-centered normalized (135,) vector."""
        # Reshape to (33, 3)
        coords = raw_99_coords.reshape((NUM_LANDMARKS, 3))

        # Add default high visibility = 1.0 to match (33, 4) standard
        coords_4d = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        coords_4d[:, :3] = coords
        coords_4d[:, 3] = 1.0  # Visibility

        # Locate Left Hip (Joint 24 -> index 23) and Right Hip (Joint 25 -> index 24)
        # Note: 0-indexed: Joint 24 is index 23, Joint 25 is index 24
        mid_hip_x = (coords_4d[23, 0] + coords_4d[24, 0]) / 2.0
        mid_hip_y = (coords_4d[23, 1] + coords_4d[24, 1]) / 2.0
        mid_hip_z = (coords_4d[23, 2] + coords_4d[24, 2]) / 2.0

        # Locate Left Shoulder (Joint 12 -> index 11) and Right Shoulder (Joint 13 -> index 12)
        mid_shoulder_x = (coords_4d[11, 0] + coords_4d[12, 0]) / 2.0
        mid_shoulder_y = (coords_4d[11, 1] + coords_4d[12, 1]) / 2.0

        # Torso scale factor
        torso_dist = math.sqrt(
            (mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2
        )
        scale_factor = max(torso_dist, 0.05)

        # Scale-invariant normalization
        norm_coords = np.zeros_like(coords_4d)
        norm_coords[:, 0] = (coords_4d[:, 0] - mid_hip_x) / scale_factor
        norm_coords[:, 1] = (coords_4d[:, 1] - mid_hip_y) / scale_factor
        norm_coords[:, 2] = (coords_4d[:, 2] - mid_hip_z) / scale_factor
        norm_coords[:, 3] = coords_4d[:, 3]

        flattened = norm_coords.flatten()  # 132 elements

        # Biomechanical derived features
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y
        torso_angle = math.atan2(dy, dx) / math.pi
        hip_y_ratio = float(mid_hip_y)

        xs = coords_4d[:, 0]
        ys = coords_4d[:, 1]
        bbox_w = max(np.max(xs) - np.min(xs), 0.01)
        bbox_h = max(np.max(ys) - np.min(ys), 0.01)
        aspect_ratio = float(min(bbox_w / bbox_h, 5.0))

        derived = np.array([torso_angle, hip_y_ratio, aspect_ratio], dtype=np.float32)
        return np.concatenate([flattened, derived])  # 135 features

    def process_and_save(self) -> Tuple[Path, Path, Path]:
        """Preprocesses all CSV files into temporal sequence arrays and saves to data/processed/."""
        files = self.discover_files()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_windows: List[np.ndarray] = []
        all_labels: List[int] = []
        all_subjects: List[str] = []

        total_files_processed = 0
        total_raw_frames = 0

        for idx, fpath in enumerate(files):
            df = pd.read_csv(fpath)
            subject_name = fpath.parent.name

            coords_data = df.iloc[:, :99].values.astype(np.float32)
            frame_labels = df.iloc[:, 99].values.astype(np.int32)
            num_frames = len(coords_data)
            total_raw_frames += num_frames

            # Handle missing landmarks (NaNs, Infs, or all-zero frames)
            if np.isnan(coords_data).any() or np.isinf(coords_data).any():
                coords_df = pd.DataFrame(coords_data)
                coords_df = coords_df.ffill().bfill().fillna(0.0)
                coords_data = coords_df.values.astype(np.float32)

            # Impute intermittent all-zero missing frames
            is_zero_frame = np.all(coords_data == 0.0, axis=1)
            if np.any(is_zero_frame):
                last_valid = None
                for i in range(num_frames):
                    if not is_zero_frame[i]:
                        last_valid = coords_data[i]
                    elif last_valid is not None:
                        coords_data[i] = last_valid

            # Extract normalized feature vector for each frame
            normalized_frames = np.zeros((num_frames, TOTAL_FEATURE_DIM), dtype=np.float32)
            for i in range(num_frames):
                normalized_frames[i] = self._normalize_frame_keypoints(coords_data[i])

            # Generate sliding windows
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                window = normalized_frames[start_idx:end_idx]
                win_labels = frame_labels[start_idx:end_idx]

                # If >= 30% of frames in the window represent a fall, label window as Fall (1)
                win_label = LABEL_FALL if (np.sum(win_labels == 1) >= (self.sequence_length * 0.3)) else LABEL_NO_FALL

                all_windows.append(window)
                all_labels.append(win_label)
                all_subjects.append(subject_name)

            total_files_processed += 1

        X_array = np.array(all_windows, dtype=np.float32)
        y_array = np.array(all_labels, dtype=np.int32)
        subjects_array = np.array(all_subjects, dtype="<U16")

        # Output paths
        x_path = self.output_dir / "up_fall_X.npy"
        y_path = self.output_dir / "up_fall_y.npy"
        subjects_path = self.output_dir / "up_fall_subjects.npy"
        meta_path = self.output_dir / "up_fall_metadata.json"

        np.save(x_path, X_array)
        np.save(y_path, y_array)
        np.save(subjects_path, subjects_array)

        num_falls = int(np.sum(y_array == LABEL_FALL))
        num_no_falls = int(np.sum(y_array == LABEL_NO_FALL))

        metadata = {
            "dataset_name": "UP-Fall Detection Dataset",
            "source_directory": str(self.raw_dir),
            "files_processed": total_files_processed,
            "total_raw_frames": total_raw_frames,
            "total_windows_generated": len(y_array),
            "fall_windows": num_falls,
            "no_fall_windows": num_no_falls,
            "sequence_length": self.sequence_length,
            "feature_dim": TOTAL_FEATURE_DIM,
            "subjects_breakdown": {
                s: int(np.sum(subjects_array == s)) for s in sorted(list(set(all_subjects)))
            },
            "output_files": {
                "X": str(x_path),
                "y": str(y_path),
                "subjects": str(subjects_path),
            },
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return x_path, y_path, subjects_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="UP-Fall Keypoint Dataset Preparation")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect files without generating output arrays")
    args = parser.parse_args()

    preparer = UPFallDataPreparer()
    report = preparer.inspect_dataset()

    print("=" * 70)
    print(" [UP-FALL DATASET INSPECTION]")
    print("=" * 70)
    print(f"Total CSV Files:       {report['total_files']}")
    print(f"Subjects Found:        {', '.join(report['subjects_found'])}")
    print(f"Total Raw Frames:      {report['total_raw_frames']}")
    print(f"Columns per File:      {report['total_columns']} ({report['joint_columns_count']} joint coords + '{report['label_column_name']}')")
    print("Frame Label Counts:")
    print(f"  * Class 0 (No Fall / ADL): {report['frame_label_distribution'].get(0, 0)} frames")
    print(f"  * Class 1 (Fall event):    {report['frame_label_distribution'].get(1, 0)} frames")
    print("=" * 70)

    if args.inspect_only:
        print("Inspection finished (--inspect-only specified).")
        return

    print("\nStarting reproducible keypoint normalization and sliding window generation...")
    x_p, y_p, subj_p, meta_p = preparer.process_and_save()

    with open(meta_p) as f:
        meta = json.load(f)

    print("\n" + "=" * 70)
    print(" [UP-FALL PREPROCESSING COMPLETE]")
    print("=" * 70)
    print(f"Generated X Array:     {x_p} (Shape: {meta['total_windows_generated']}, {meta['sequence_length']}, {meta['feature_dim']})")
    print(f"Generated Y Array:     {y_p} (Shape: {meta['total_windows_generated']},)")
    print(f"Window Distribution:   {meta['no_fall_windows']} No Fall windows vs {meta['fall_windows']} Fall windows")
    print(f"Metadata Saved:        {meta_p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
