"""LSTM Classifier Training & Evaluation Pipeline for UP-Fall Keypoints.
Enforces strict subject-level isolation to prevent data leakage (Leave-One-Subject-Out).
Trains stacked LSTM with class balancing, saves checkpoints, and computes test diagnostics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    DEFAULT_MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    FALL_PROBABILITY_THRESHOLD,
    DEFAULT_EPOCHS,
    BATCH_SIZE,
    CLASS_NAMES,
)
from src.lstm_model import build_lstm_model
from src.evaluate import evaluate_model


class SubjectLeakageFreeTrainer:
    """Manages training with Leave-One-Subject-Out partitioning to prevent subject data leakage."""

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
        test_subject: str = "SUBJECT3",
    ):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.test_subject = test_subject

        # Load verified processed UP-Fall dataset
        x_path = self.processed_dir / "up_fall_X.npy"
        y_path = self.processed_dir / "up_fall_y.npy"
        subj_path = self.processed_dir / "up_fall_subjects.npy"

        if not x_path.exists() or not y_path.exists() or not subj_path.exists():
            raise FileNotFoundError(
                "Processed UP-Fall dataset not found. Please run 'python -m src.prepare_up_fall' first."
            )

        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.subjects = np.load(subj_path)

    def partition_by_subject(
        self,
        val_subject: str = "SUBJECT2",
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partitions dataset strictly by subject identifier.

        Guarantee:
        - Test set contains ONLY self.test_subject (unseen during all training and tuning).
        - Validation set contains 50% of val_subject.
        - Training set contains all remaining subjects + other 50% of val_subject.
        Zero frame or subject overlap across splits.
        """
        # 1. Held-Out Test Set: Exclusively test_subject
        test_mask = (self.subjects == self.test_subject)
        X_test = self.X[test_mask]
        y_test = self.y[test_mask]

        # 2. Candidate pool for Train & Validation: All subjects except test_subject
        train_pool_mask = ~test_mask
        X_pool = self.X[train_pool_mask]
        y_pool = self.y[train_pool_mask]
        subjs_pool = self.subjects[train_pool_mask]

        # Split val_subject into train and val to keep both in training pool
        val_sub_mask = (subjs_pool == val_subject)
        X_val_sub = X_pool[val_sub_mask]
        y_val_sub = y_pool[val_sub_mask]

        # Stratified 50/50 split of the validation subject
        X_val_sub_train, X_val, y_val_sub_train, y_val = train_test_split(
            X_val_sub, y_val_sub,
            test_size=0.5,
            stratify=y_val_sub,
            random_state=random_state,
        )

        # Remaining subjects go purely to train (e.g. SUBJECT1)
        other_sub_mask = (subjs_pool != val_subject)
        X_other = X_pool[other_sub_mask]
        y_other = y_pool[other_sub_mask]

        X_train = np.concatenate([X_other, X_val_sub_train], axis=0)
        y_train = np.concatenate([y_other, y_val_sub_train], axis=0)

        # Shuffle training set
        rng = np.random.RandomState(random_state)
        train_idx = np.arange(len(y_train))
        rng.shuffle(train_idx)
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(
        self,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Executes leak-free subject training, model checkpoint saving, and test evaluation."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        # 1. Subject-level split
        X_train, X_val, X_test, y_train, y_val, y_test = self.partition_by_subject()

        print("=" * 70)
        print(" [SUBJECT-LEVEL LEAK-FREE PARTITIONING]")
        print("=" * 70)
        print(f"Training Set:   {len(y_train)} windows (Fall: {(y_train == 1).sum()}, No Fall: {(y_train == 0).sum()})")
        print(f"Validation Set: {len(y_val)} windows (Fall: {(y_val == 1).sum()}, No Fall: {(y_val == 0).sum()})")
        print(f"Held-Out Test:  {len(y_test)} windows (Fall: {(y_test == 1).sum()}, No Fall: {(y_test == 0).sum()})")
        print(f"Held-Out Person: '{self.test_subject}' (100% unseen by training/validation)")
        print("=" * 70)

        # 2. Fit StandardScaler strictly on Training set
        scaler = StandardScaler()
        flat_x_train = X_train.reshape(-1, TOTAL_FEATURE_DIM)
        scaler.fit(flat_x_train)

        # Transform all partitions
        X_train_scaled = scaler.transform(flat_x_train).reshape(X_train.shape).astype(np.float32)
        X_val_scaled = scaler.transform(X_val.reshape(-1, TOTAL_FEATURE_DIM)).reshape(X_val.shape).astype(np.float32)
        X_test_scaled = scaler.transform(X_test.reshape(-1, TOTAL_FEATURE_DIM)).reshape(X_test.shape).astype(np.float32)

        # Save scaler for real-time webcam & dashboard
        joblib.dump(scaler, SCALER_PATH)
        print(f"[Trainer] Scaler fitted on training data and saved to: {SCALER_PATH}")

        # Save test partition for standalone evaluation & auditing
        test_x_path = self.models_dir / "test_X.npy"
        test_y_path = self.models_dir / "test_y.npy"
        np.save(test_x_path, X_test_scaled)
        np.save(test_y_path, y_test)

        # 3. Compute class weights to address natural activity class imbalance
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}
        print(f"[Trainer] Computed balanced class weights: {class_weight_dict}")

        # 4. Build Stacked LSTM classifier
        model = build_lstm_model(
            input_shape=(SEQUENCE_LENGTH, TOTAL_FEATURE_DIM),
            learning_rate=learning_rate,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(DEFAULT_MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        print(f"\n[Trainer] Training LSTM on {len(X_train_scaled)} sequences for up to {epochs} epochs...")
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        # Load best restored weights
        model = tf.keras.models.load_model(str(DEFAULT_MODEL_PATH))

        # 5. Evaluate on actual held-out test data (SUBJECT3)
        print("\n" + "=" * 70)
        print(f" [HELD-OUT TEST EVALUATION: UNSEEN {self.test_subject}]")
        print("=" * 70)
        eval_metrics = evaluate_model(
            model=model,
            X_test=X_test_scaled,
            y_test=y_test,
            threshold=FALL_PROBABILITY_THRESHOLD,
            save_dir=self.models_dir,
            plot_figures=True,
        )

        # Save comprehensive model metadata
        meta = {
            "model_type": "Stacked LSTM Classifier",
            "dataset": "UP-Fall Detection Dataset",
            "held_out_test_subject": self.test_subject,
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "epochs_trained": len(history.history["loss"]),
            "class_weights": class_weight_dict,
            "test_metrics": {
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "specificity": eval_metrics["specificity"],
                "f1_score": eval_metrics["f1_score"],
                "roc_auc": eval_metrics["roc_auc"],
            },
            "confusion_matrix": eval_metrics["confusion_matrix"],
        }

        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Classifier on Verified UP-Fall Sequences")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs to train (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--test-subject", type=str, default="SUBJECT3", help="Held-out subject (default: SUBJECT3)")
    args = parser.parse_args()

    trainer = SubjectLeakageFreeTrainer(test_subject=args.test_subject)
    results = trainer.train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\n" + "=" * 70)
    print(" [FINAL HELD-OUT TEST RESULTS]")
    print("=" * 70)
    print(f"Accuracy:            {results['accuracy'] * 100:.2f}%")
    print(f"Precision:           {results['precision'] * 100:.2f}%")
    print(f"Recall (Sensitivity):{results['recall'] * 100:.2f}%")
    print(f"Specificity:         {results['specificity'] * 100:.2f}%")
    print(f"F1-Score:            {results['f1_score']:.4f}")
    print(f"ROC-AUC:             {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"  True Negatives (ADL correctly identified):  {cm['true_negatives']}")
    print(f"  False Positives (ADL false alarms):        {cm['false_positives']}")
    print(f"  False Negatives (Missed falls):             {cm['false_negatives']}")
    print(f"  True Positives (Falls correctly detected):  {cm['true_positives']}")
    print("-" * 70)
    print("Classification Report:")
    print(results["classification_report"])
    print("=" * 70)


if __name__ == "__main__":
    main()
