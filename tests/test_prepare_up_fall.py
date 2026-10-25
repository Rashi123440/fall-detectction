"""Tests for UP-Fall keypoint dataset preparation and preprocessing."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.prepare_up_fall import UPFallDataPreparer
from src.config import TOTAL_FEATURE_DIM, SEQUENCE_LENGTH, LABEL_FALL, LABEL_NO_FALL


def test_up_fall_data_preparer_on_mock_csv():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        subj_dir = tmp_path / "SUBJECT1"
        subj_dir.mkdir(parents=True)

        # Create a mock 100-frame, 100-column CSV
        num_frames = 100
        cols = [f"Joint{i}_{coord}" for i in range(1, 34) for coord in ["X", "Y", "Z"]] + ["LABEL"]
        data = np.zeros((num_frames, 100), dtype=np.float32)

        # Set mid hip and shoulder
        # Joint 12 (Left Shoulder) -> cols 33..35
        # Joint 13 (Right Shoulder) -> cols 36..38
        # Joint 24 (Left Hip) -> cols 69..71
        # Joint 25 (Right Hip) -> cols 72..74
        data[:, 33:36] = [0.4, 0.3, 0.0]
        data[:, 36:39] = [0.6, 0.3, 0.0]
        data[:, 69:72] = [0.45, 0.6, 0.0]
        data[:, 72:75] = [0.55, 0.6, 0.0]

        # Frames 0..29 are ADL (0), 30..99 are Fall (1)
        data[30:, 99] = 1.0

        df = pd.DataFrame(data, columns=cols)
        csv_file = subj_dir / "C1S1A1T1.csv"
        df.to_csv(csv_file, index=False)

        # Run preparer
        out_dir = tmp_path / "processed"
        preparer = UPFallDataPreparer(raw_dir=tmp_path, output_dir=out_dir)

        # Inspect
        report = preparer.inspect_dataset()
        assert report["total_files"] == 1
        assert report["total_raw_frames"] == 100
        assert report["frame_label_distribution"][0] == 30
        assert report["frame_label_distribution"][1] == 70

        # Process and save
        x_path, y_path, subj_path, meta_path = preparer.process_and_save()
        assert x_path.exists()
        assert y_path.exists()
        assert subj_path.exists()
        assert meta_path.exists()

        X = np.load(x_path)
        y = np.load(y_path)

        assert X.shape[1] == SEQUENCE_LENGTH
        assert X.shape[2] == TOTAL_FEATURE_DIM
        assert len(X) == len(y)
        assert LABEL_FALL in y
        assert LABEL_NO_FALL in y
