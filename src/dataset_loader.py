"""Dataset Preparation and Inspection Module.
Supports documented fall-detection datasets (UP-Fall, Le2i, and custom categorized video clips).
Extracts keypoints in-memory and saves only privacy-preserving features.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    LABEL_FALL,
    LABEL_NO_FALL,
    TOTAL_FEATURE_DIM,
)
from src.pose_extractor import PoseExtractor


class DatasetInspector:
    """Inspects filesystem directories and provides diagnostic summaries of available datasets."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR, synthetic_dir: Path = SYNTHETIC_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.synthetic_dir = Path(synthetic_dir)

    def inspect_all(self) -> Dict[str, Any]:
        """Provides a comprehensive health check and inventory of data directories."""
        summary = {
            "raw_dir_exists": self.raw_dir.exists(),
            "processed_dir_exists": self.processed_dir.exists(),
            "synthetic_dir_exists": self.synthetic_dir.exists(),
            "benchmark_datasets": {},
            "processed_datasets": {},
            "synthetic_datasets": {},
            "status": "NO_DATA",
            "recommendation": "",
        }

        # 1. Inspect UP-Fall and Le2i in raw_dir
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        raw_videos = [f for f in self.raw_dir.rglob("*") if f.suffix.lower() in video_exts]

        up_fall_videos = [f for f in raw_videos if "up_fall" in str(f).lower() or "activity" in str(f).lower()]
        le2i_videos = [f for f in raw_videos if "le2i" in str(f).lower() or "home" in str(f).lower() or "office" in str(f).lower()]

        summary["benchmark_datasets"] = {
            "total_raw_videos": len(raw_videos),
            "up_fall_candidate_videos": len(up_fall_videos),
            "le2i_candidate_videos": len(le2i_videos),
        }

        # 2. Inspect processed keypoint files
        proc_x_files = list(self.processed_dir.glob("*_X.npy"))
        summary["processed_datasets"] = {
            "feature_files": [f.name for f in proc_x_files],
            "total_processed_sets": len(proc_x_files),
        }

        # 3. Inspect synthetic data
        synth_x = self.synthetic_dir / "synthetic_X.npy"
        synth_y = self.synthetic_dir / "synthetic_y.npy"
        synth_meta = self.synthetic_dir / "synthetic_metadata.json"
        if synth_x.exists() and synth_y.exists():
            try:
                x_arr = np.load(synth_x, mmap_mode="r")
                summary["synthetic_datasets"] = {
                    "available": True,
                    "samples": int(x_arr.shape[0]),
                    "sequence_length": int(x_arr.shape[1]),
                    "feature_dim": int(x_arr.shape[2]),
                }
            except Exception as e:
                summary["synthetic_datasets"] = {"available": False, "error": str(e)}
        else:
            summary["synthetic_datasets"] = {"available": False}

        # Determine overall diagnostic status
        if len(proc_x_files) > 0:
            summary["status"] = "PROCESSED_READY"
            summary["recommendation"] = "Processed keypoints ready for preprocessor & LSTM training."
        elif len(raw_videos) > 0:
            summary["status"] = "RAW_PRESENT_UNPROCESSED"
            summary["recommendation"] = f"Found {len(raw_videos)} raw video files. Run feature extraction to generate privacy-preserving keypoints."
        elif summary["synthetic_datasets"].get("available", False):
            summary["status"] = "SYNTHETIC_READY_FOR_TESTING"
            summary["recommendation"] = "Synthetic verification data is ready for pipeline testing and architecture verification. (Do not claim as real-world accuracy)."
        else:
            summary["status"] = "DATASET_MISSING"
            summary["recommendation"] = "No datasets found. Download UP-Fall / Le2i using docs/dataset_guide.md or generate synthetic test data using src/synthetic_data.py."

        return summary


class DatasetProcessor:
    """Processes video datasets into privacy-preserving keypoints without storing raw video frames."""

    def __init__(self, output_dir: Path = PROCESSED_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pose_extractor = PoseExtractor()

    def process_video_stream(self, video_path: Path) -> Tuple[np.ndarray, int]:
        """Extracts keypoint features frame-by-frame from video strictly in RAM.

        Returns:
            features (np.ndarray): Shape (num_frames, TOTAL_FEATURE_DIM)
            valid_detections (int): Number of frames where a human pose was successfully detected.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        frames_features = []
        valid_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract landmarks in RAM
                success, feat_vec, _ = self.pose_extractor.extract_keypoints(frame)
                frames_features.append(feat_vec)
                if success:
                    valid_count += 1
        finally:
            cap.release()

        if len(frames_features) == 0:
            return np.zeros((0, TOTAL_FEATURE_DIM), dtype=np.float32), 0

        return np.array(frames_features, dtype=np.float32), valid_count

    def parse_up_fall_label(self, filepath: Path) -> int:
        """Determines Fall (1) or No Fall (0) label for UP-Fall dataset based on directory or filename.

        UP-Fall convention:
        - Activities 1 through 6: Falls (Forward, backward, sideways, sitting down empty, etc.) -> Class 1
        - Activities 7 through 11: ADLs (Walking, standing, sitting, picking object, jumping) -> Class 0
        """
        path_str = str(filepath).lower()
        match = re.search(r"activity\s*(\d+)", path_str)
        if match:
            act_num = int(match.group(1))
            return LABEL_FALL if act_num <= 6 else LABEL_NO_FALL

        # Fallback check for keywords
        if any(w in path_str for w in ["fall", "falling"]):
            return LABEL_FALL
        return LABEL_NO_FALL

    def extract_from_folder(
        self,
        input_dir: Path,
        dataset_name: str = "custom",
    ) -> Tuple[Path, Path]:
        """Batch processes video files in an input folder into privacy-preserving numpy arrays.

        Args:
            input_dir: Path containing raw videos.
            dataset_name: Name identifier for saved output files.

        Returns:
            Tuple of paths to saved (features_path, labels_path).
        """
        input_dir = Path(input_dir)
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in video_exts]

        if not video_files:
            raise FileNotFoundError(f"No video files found matching {video_exts} in {input_dir}")

        all_video_features = []
        all_labels = []

        print(f"[DatasetProcessor] Found {len(video_files)} video clips to process in {input_dir}...")
        for idx, vid_file in enumerate(video_files):
            try:
                feats, valids = self.process_video_stream(vid_file)
                if len(feats) == 0:
                    continue

                label = self.parse_up_fall_label(vid_file)
                all_video_features.append(feats)
                all_labels.append(label)

                print(f"[{idx+1}/{len(video_files)}] {vid_file.name}: {len(feats)} frames, {valids} poses detected -> Label: {label}")
            except Exception as e:
                print(f"[DatasetProcessor] Error processing {vid_file}: {e}")

        # Save extracted features
        out_x_path = self.output_dir / f"{dataset_name}_raw_features.npy"
        out_y_path = self.output_dir / f"{dataset_name}_labels.npy"

        # Note: Videos have variable lengths; saved as an object array or list
        np.save(out_x_path, np.array(all_video_features, dtype=object), allow_pickle=True)
        np.save(out_y_path, np.array(all_labels, dtype=np.int32))

        print(f"[DatasetProcessor] Finished feature extraction. Saved to:\n  {out_x_path}\n  {out_y_path}")
        return out_x_path, out_y_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-Preserving Fall Detection: Dataset Inspection & Health Check")
    parser.add_argument("--json", action="store_true", help="Print output in raw JSON format")
    parser.add_argument("--extract", type=str, default=None, help="Extract keypoints from a raw dataset folder (e.g., data/raw/up_fall)")
    parser.add_argument("--name", type=str, default="custom", help="Dataset name identifier for extracted files")
    args = parser.parse_args()

    if args.extract:
        processor = DatasetProcessor()
        processor.extract_from_folder(Path(args.extract), dataset_name=args.name)
    else:
        inspector = DatasetInspector()
        report = inspector.inspect_all()

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print(" [PRIVACY-PRESERVING FALL DETECTION] DATASET INSPECTION REPORT")
            print("=" * 70)
            print("Directory Status:")
            print(f"  * Raw Video Dir       (`data/raw/`):       {'[OK] Present' if report['raw_dir_exists'] else '[X] Missing'}")
            print(f"  * Processed Data Dir  (`data/processed/`): {'[OK] Present' if report['processed_dir_exists'] else '[X] Missing'}")
            print(f"  * Synthetic Data Dir  (`data/synthetic/`): {'[OK] Present' if report['synthetic_dir_exists'] else '[X] Missing'}")
            print("-" * 70)
            print("Data Inventory:")
            print(f"  * Raw Benchmark Videos Found:   {report['benchmark_datasets']['total_raw_videos']} clips")
            print(f"  * Processed Keypoint Datasets:  {report['processed_datasets']['total_processed_sets']} sets")
            synth = report["synthetic_datasets"]
            if synth.get("available", False):
                print(f"  * Synthetic Verification Data:  {synth['samples']} samples (Seq: {synth['sequence_length']}, Dim: {synth['feature_dim']})")
            else:
                print("  * Synthetic Verification Data:  None generated")
            print("-" * 70)
            print(f"System State:    [{report['status']}]")
            print(f"Recommendation:  {report['recommendation']}")
            print("=" * 70)
            print("Privacy Note: Raw videos in 'data/raw/' are strictly ignored by git.")
            print("Frames are processed in RAM; only keypoints are written to disk.")
            print("=" * 70)

