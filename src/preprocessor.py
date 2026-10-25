"""Preprocessing Pipeline for Pose Keypoints.
Handles temporal sliding window generation, missing landmark interpolation,
feature standardization, and train/val/test dataset splitting.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    SEQUENCE_LENGTH,
    STRIDE_STEP,
    TOTAL_FEATURE_DIM,
    SCALER_PATH,
    LABEL_FALL,
    LABEL_NO_FALL,
)


class KeypointPreprocessor:
    """Prepares and normalizes keypoint sequence data for LSTM training and inference."""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, stride: int = STRIDE_STEP):
        self.sequence_length = sequence_length
        self.stride = stride
        self.scaler = None

    def interpolate_missing_frames(self, sequence: np.ndarray) -> np.ndarray:
        """Interpolates missing landmarks (all-zero frames) using nearest valid keypoint frames.

        Args:
            sequence: Array of shape (num_frames, feature_dim)

        Returns:
            clean_sequence: Array with missing frames imputed.
        """
        if sequence.shape[0] == 0:
            return sequence

        clean = sequence.copy()
        is_missing = np.all(clean == 0.0, axis=1)

        if not np.any(is_missing):
            return clean

        # Forward fill and backward fill
        last_valid = None
        for i in range(len(clean)):
            if not is_missing[i]:
                last_valid = clean[i].copy()
            elif last_valid is not None:
                clean[i] = last_valid.copy()

        # Backward fill if leading frames were missing
        next_valid = None
        for i in range(len(clean) - 1, -1, -1):
            if not is_missing[i]:
                next_valid = clean[i].copy()
            elif next_valid is not None and np.all(clean[i] == 0.0):
                clean[i] = next_valid.copy()

        return clean

    def create_sliding_windows(
        self,
        frames_array: np.ndarray,
        label: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts continuous video frame features into fixed-length temporal sequence windows.

        Args:
            frames_array: Array of shape (N_frames, feature_dim)
            label: Integer label (0 = No Fall, 1 = Fall)

        Returns:
            X_windows: Shape (num_windows, sequence_length, feature_dim)
            y_windows: Shape (num_windows,)
        """
        num_frames = len(frames_array)
        if num_frames < self.sequence_length:
            return np.zeros((0, self.sequence_length, TOTAL_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        clean_frames = self.interpolate_missing_frames(frames_array)

        windows = []
        labels = []

        for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            window = clean_frames[start_idx:end_idx]
            windows.append(window)
            labels.append(label)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)

    def process_multiple_videos(
        self,
        video_features_list: List[np.ndarray],
        video_labels_list: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts windows across an entire collection of recorded video sequences."""
        all_X = []
        all_y = []

        for features, lbl in zip(video_features_list, video_labels_list):
            X_win, y_win = self.create_sliding_windows(features, lbl)
            if len(X_win) > 0:
                all_X.append(X_win)
                all_y.append(y_win)

        if not all_X:
            return np.zeros((0, self.sequence_length, TOTAL_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def fit_scaler(self, X_train: np.ndarray, save_path: Optional[Path] = SCALER_PATH) -> StandardScaler:
        """Fits StandardScaler across flattened feature vectors of the training set.

        Args:
            X_train: Shape (N, sequence_length, feature_dim)
            save_path: Optional path to save the fitted scaler.
        """
        n_samples, seq_len, feat_dim = X_train.shape
        flat_x = X_train.reshape(-1, feat_dim)

        self.scaler = StandardScaler()
        self.scaler.fit(flat_x)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, save_path)
            print(f"[KeypointPreprocessor] Fitted and saved feature scaler to {save_path}")

        return self.scaler

    def load_scaler(self, load_path: Path = SCALER_PATH) -> bool:
        """Loads a previously fitted scaler from disk."""
        load_path = Path(load_path)
        if not load_path.exists():
            return False
        try:
            self.scaler = joblib.load(load_path)
            return True
        except Exception as e:
            print(f"[KeypointPreprocessor] Error loading scaler: {e}")
            return False

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales 3D temporal sequence array using the fitted/loaded scaler.

        Args:
            X: Shape (N, sequence_length, feature_dim)
        """
        if self.scaler is None:
            # Fallback to loading scaler or returning raw if none exists
            if not self.load_scaler():
                return X.astype(np.float32)

        n_samples, seq_len, feat_dim = X.shape
        flat_x = X.reshape(-1, feat_dim)
        scaled_flat = self.scaler.transform(flat_x)
        return scaled_flat.reshape(n_samples, seq_len, feat_dim).astype(np.float32)

    def transform_single_frame(self, frame_feature: np.ndarray) -> np.ndarray:
        """Transforms a single 1D feature vector for real-time inference."""
        if self.scaler is None:
            if not self.load_scaler():
                return frame_feature
        reshaped = frame_feature.reshape(1, -1)
        return self.scaler.transform(reshaped).flatten()

    def split_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified split into Train, Validation, and Test partitions.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        total = train_ratio + val_ratio + test_ratio
        train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

        # First split into train and temp (val + test)
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=temp_ratio,
            stratify=y,
            random_state=random_state,
        )

        # Split temp into val and test
        val_proportion_of_temp = val_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - val_proportion_of_temp),
            stratify=y_temp,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
