"""End-to-End Test for the full Fall Detection Pipeline.
Generates small synthetic verification data, fits preprocessor, trains an LSTM for 2 epochs,
evaluates metrics, and verifies model saving and loading.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.synthetic_data import SyntheticKinematicsGenerator
from src.preprocessor import KeypointPreprocessor
from src.lstm_model import build_lstm_model, train_fall_detector, load_model_for_inference
from src.evaluate import evaluate_model
from src.config import SEQUENCE_LENGTH, TOTAL_FEATURE_DIM


def test_pipeline_e2e():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Generate small synthetic dataset for pipeline test
        generator = SyntheticKinematicsGenerator(seed=777)
        X, y, meta = generator.generate_dataset(num_samples=30, fall_ratio=0.5, seq_length=SEQUENCE_LENGTH)
        assert X.shape == (30, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)

        # 2. Preprocessing & Splitting
        preprocessor = KeypointPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
        )

        scaler_path = tmp_path / "scaler.pkl"
        preprocessor.fit_scaler(X_train, save_path=scaler_path)
        assert scaler_path.exists()

        X_train_scaled = preprocessor.transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        X_test_scaled = preprocessor.transform(X_test)

        # 3. Train LSTM model for 2 epochs
        model_path = tmp_path / "test_model.keras"
        meta_path = tmp_path / "metadata.json"

        model, train_meta = train_fall_detector(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            epochs=2,
            batch_size=8,
            model_path=model_path,
            metadata_path=meta_path,
            is_synthetic_training=True,
        )

        assert model_path.exists()
        assert meta_path.exists()

        # 4. Load trained model
        loaded_model = load_model_for_inference(model_path)
        assert loaded_model is not None

        # 5. Evaluate metrics
        eval_results = evaluate_model(
            model=loaded_model,
            X_test=X_test_scaled,
            y_test=y_test,
            threshold=0.5,
            save_dir=tmp_path,
            plot_figures=True,
        )

        assert "accuracy" in eval_results
        assert "precision" in eval_results
        assert "recall" in eval_results
        assert "f1_score" in eval_results
        assert "confusion_matrix" in eval_results
        assert (tmp_path / "evaluation_results.json").exists()
        assert (tmp_path / "confusion_matrix.png").exists()
