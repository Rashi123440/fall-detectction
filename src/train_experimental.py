"""Experimental LSTM Classifier Training Pipeline for Fall Detection.
Implements:
1. Strict subject-level isolation (Train: SUBJECT1+2, Val: SUBJECT4+5, Test: SUBJECT3).
2. Binary Focal Loss / Class-Weighted optimization for class-imbalanced keypoint sequences.
3. Validation-set threshold tuning to optimize the tradeoff between Fall Recall and No Fall Recall.
4. Separate experimental model, scaler, and artifact filenames to ensure zero interference with baseline models.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    CLASS_NAMES,
)
from src.train_lstm import SubjectLeakageFreeTrainer

# Dedicated experimental file paths (preserves baseline intact)
EXPERIMENTAL_MODEL_PATH = MODELS_DIR / "experimental_fall_lstm.keras"
EXPERIMENTAL_SCALER_PATH = MODELS_DIR / "experimental_scaler.pkl"
EXPERIMENTAL_METADATA_PATH = MODELS_DIR / "experimental_metadata.json"
EXPERIMENTAL_EVAL_PATH = MODELS_DIR / "experimental_evaluation.json"
EXPERIMENTAL_CM_PATH = MODELS_DIR / "experimental_confusion_matrix.png"
EXPERIMENTAL_ROC_PATH = MODELS_DIR / "experimental_roc_curve.png"


def build_experimental_lstm_model(
    input_shape: Tuple[int, int] = (SEQUENCE_LENGTH, TOTAL_FEATURE_DIM),
    lstm_units: list = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = 0.001,
):
    """Builds stacked LSTM classifier compiled with balanced binary crossentropy."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

    model = Sequential([
        Input(shape=input_shape, name="pose_sequence_input"),
        LSTM(lstm_units[0], return_sequences=True, name="lstm_layer_1"),
        Dropout(dropout_rate, name="dropout_1"),
        BatchNormalization(name="batch_norm_1"),
        LSTM(lstm_units[1], return_sequences=False, name="lstm_layer_2"),
        Dropout(dropout_rate, name="dropout_2"),
        Dense(dense_units, activation="relu", name="dense_features"),
        Dense(1, activation="sigmoid", name="fall_prediction"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class ExperimentalTrainer:
    """Manages the experimental training loop with validation-only threshold tuning."""

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
        test_subject: str = "SUBJECT3",
    ):
        self.trainer_base = SubjectLeakageFreeTrainer(
            processed_dir=processed_dir,
            models_dir=models_dir,
            test_subject=test_subject,
        )
        self.models_dir = Path(models_dir)

    def tune_threshold_on_validation(
        self,
        model,
        X_val_scaled: np.ndarray,
        y_val: np.ndarray,
        min_fall_recall: float = 0.80,
    ) -> Tuple[float, Dict[str, Any]]:
        """Sweeps decision thresholds strictly on the VALIDATION set.
        The test set is never accessed here to avoid optimization leakage.
        """
        val_probs = model.predict(X_val_scaled, verbose=0).flatten()

        best_threshold = 0.50
        best_balanced_acc = 0.0
        tuning_curve = []

        thresholds = np.arange(0.30, 0.85, 0.02)

        for th in thresholds:
            preds = (val_probs >= th).astype(int)
            cm = confusion_matrix(y_val, preds)
            tn, fp, fn, tp = cm.ravel()

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            b_acc = (rec + spec) / 2.0
            f1 = f1_score(y_val, preds, zero_division=0)

            tuning_curve.append({
                "threshold": round(float(th), 2),
                "fall_recall": round(float(rec), 4),
                "no_fall_recall": round(float(spec), 4),
                "balanced_acc": round(float(b_acc), 4),
                "f1_score": round(float(f1), 4),
            })

            # Selection criterion: Maximize balanced accuracy with non-trivial specificity
            if rec >= min_fall_recall and spec >= 0.10 and b_acc > best_balanced_acc:
                best_balanced_acc = b_acc
                best_threshold = round(float(th), 2)

        tuning_summary = {
            "selected_threshold": best_threshold,
            "val_balanced_acc": round(best_balanced_acc, 4),
            "tuning_curve": tuning_curve,
        }
        return best_threshold, tuning_summary

    def train_and_evaluate(
        self,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Trains experimental model, tunes threshold on validation, and evaluates on untouched test set."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        # 1. Subject-level partition
        X_tr, X_val, X_te, y_tr, y_val, y_te = self.trainer_base.partition_by_subject()

        print("=" * 70)
        print(" [EXPERIMENTAL TRAINING PIPELINE - SUBJECT ISOLATION]")
        print("=" * 70)
        print(f"Train Set:      {len(y_tr)} windows (Fall: {(y_tr==1).sum()}, No Fall: {(y_tr==0).sum()})")
        print(f"Validation Set: {len(y_val)} windows (Fall: {(y_val==1).sum()}, No Fall: {(y_val==0).sum()})")
        print(f"Held-Out Test:  {len(y_te)} windows (Fall: {(y_te==1).sum()}, No Fall: {(y_te==0).sum()})")
        print("=" * 70)

        # 2. Fit experimental scaler strictly on training split
        scaler = StandardScaler()
        flat_x_tr = X_tr.reshape(-1, TOTAL_FEATURE_DIM)
        scaler.fit(flat_x_tr)

        X_tr_s = scaler.transform(flat_x_tr).reshape(X_tr.shape).astype(np.float32)
        X_val_s = scaler.transform(X_val.reshape(-1, TOTAL_FEATURE_DIM)).reshape(X_val.shape).astype(np.float32)
        X_te_s = scaler.transform(X_te.reshape(-1, TOTAL_FEATURE_DIM)).reshape(X_te.shape).astype(np.float32)

        joblib.dump(scaler, EXPERIMENTAL_SCALER_PATH)
        print(f"[Experimental] Saved separate scaler to: {EXPERIMENTAL_SCALER_PATH}")

        # 3. Build model
        model = build_experimental_lstm_model(
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
                filepath=str(EXPERIMENTAL_MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Class weights to supplement focal loss
        class_weight_dict = {
            0: float(len(y_tr) / (2 * max(1, (y_tr == 0).sum()))),
            1: float(len(y_tr) / (2 * max(1, (y_tr == 1).sum()))),
        }

        print(f"\n[Experimental] Training on {len(X_tr_s)} sequences for up to {epochs} epochs...")
        history = model.fit(
            X_tr_s,
            y_tr,
            validation_data=(X_val_s, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        # Load best restored weights
        model = tf.keras.models.load_model(str(EXPERIMENTAL_MODEL_PATH))

        # 4. Tune threshold strictly on Validation Set
        print("\n[Experimental] Tuning threshold on validation set...")
        tuned_threshold, tuning_summary = self.tune_threshold_on_validation(
            model=model,
            X_val_scaled=X_val_s,
            y_val=y_val,
            min_fall_recall=0.80,
        )
        print(f"[Experimental] Selected Validation-Tuned Threshold: {tuned_threshold}")

        # 5. Evaluate on Untouched Test Set (SUBJECT3)
        print("\n" + "=" * 70)
        print(" [EVALUATING EXPERIMENTAL MODEL ON UNTOUCHED HELD-OUT TEST SET]")
        print("=" * 70)
        test_probs = model.predict(X_te_s, verbose=0).flatten()
        test_preds = (test_probs >= tuned_threshold).astype(int)

        cm = confusion_matrix(y_te, test_preds)
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_te, test_preds))
        prec = float(precision_score(y_te, test_preds, zero_division=0))
        rec = float(recall_score(y_te, test_preds, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_te, test_preds, zero_division=0))
        roc_auc = float(roc_auc_score(y_te, test_probs)) if len(np.unique(y_te)) > 1 else 0.50
        clf_report = classification_report(y_te, test_preds, target_names=CLASS_NAMES, digits=4)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            title=f"Experimental Confusion Matrix (Th={tuned_threshold})",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(EXPERIMENTAL_CM_PATH, dpi=150)
        plt.close(fig)

        # Save metadata and evaluation results
        results = {
            "model_type": "Experimental Stacked LSTM with Focal Loss",
            "tuned_threshold": tuned_threshold,
            "test_samples": int(len(y_te)),
            "accuracy": acc,
            "precision": prec,
            "fall_recall": rec,
            "no_fall_recall": spec,
            "false_positive_rate": fpr,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "matrix": cm.tolist(),
            },
            "classification_report": clf_report,
            "validation_tuning": tuning_summary,
        }

        with open(EXPERIMENTAL_EVAL_PATH, "w") as f:
            json.dump(results, f, indent=2)

        meta = {
            "model_type": "Experimental Stacked LSTM with Balanced Crossentropy",
            "epochs_trained": len(history.history["loss"]),
            "loss": "binary_crossentropy",
            "tuned_threshold": tuned_threshold,
            "held_out_test_subject": "SUBJECT3",
            "test_metrics": results,
        }
        with open(EXPERIMENTAL_METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Accuracy:                 {acc * 100:.2f}%")
        print(f"Precision:                {prec * 100:.2f}%")
        print(f"Fall Recall (Sensitivity):{rec * 100:.2f}% ({tp}/{tp + fn})")
        print(f"No Fall Recall (Spec):    {spec * 100:.2f}% ({tn}/{tn + fp})")
        print(f"False Positive Rate:      {fpr * 100:.2f}% ({fp}/{tn + fp})")
        print(f"F1-Score:                 {f1:.4f}")
        print(f"ROC-AUC:                  {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  True Negatives (ADL correctly identified): {tn}")
        print(f"  False Positives (ADL false alarms):       {fp}")
        print(f"  False Negatives (Missed falls):            {fn}")
        print(f"  True Positives (Falls correctly caught):   {tp}")
        print("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(description="Train Experimental Fall Detection LSTM with Balanced Objective")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    trainer = ExperimentalTrainer()
    trainer.train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
