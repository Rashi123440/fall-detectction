"""Model Evaluation Module.
Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
Generates evaluation plots and saves structured metric reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.config import CLASS_NAMES, MODELS_DIR, FALL_PROBABILITY_THRESHOLD


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = FALL_PROBABILITY_THRESHOLD,
    save_dir: Optional[Path] = MODELS_DIR,
    plot_figures: bool = True,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics on held-out test data.

    Args:
        model: Trained Keras/TensorFlow model or callable predictor.
        X_test: Test sequences of shape (N, sequence_length, feature_dim).
        y_test: Ground truth labels of shape (N,).
        threshold: Decision boundary for classifying as Fall (default 0.50).
        save_dir: Directory where metrics JSON and confusion matrix PNG will be saved.
        plot_figures: Whether to generate and save PNG plots.

    Returns:
        Dict containing all computed metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
    )

    # 1. Generate model predictions
    raw_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (raw_probs >= threshold).astype(int)

    # 2. Compute core metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, raw_probs))
    except Exception:
        auc = 0.5

    # 3. Compute Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics_result = {
        "threshold": float(threshold),
        "test_samples": int(len(y_test)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist(),
        },
        "classification_report": report_str,
    }

    # 4. Save metrics and plots if directory requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(metrics_result, f, indent=2)
        print(f"[Evaluator] Saved metrics JSON to: {json_path}")

        if plot_figures:
            try:
                import matplotlib
                matplotlib.use("Agg")  # Non-interactive backend
                import matplotlib.pyplot as plt

                # Plot Confusion Matrix
                plt.figure(figsize=(6, 5))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Fall Detection Confusion Matrix", fontsize=14)
                plt.colorbar()
                tick_marks = np.arange(len(CLASS_NAMES))
                plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
                plt.yticks(tick_marks, CLASS_NAMES)

                thresh_cm = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh_cm else "black",
                            fontsize=12,
                        )

                plt.ylabel("True Label", fontsize=12)
                plt.xlabel("Predicted Label", fontsize=12)
                plt.tight_layout()
                cm_plot_path = save_dir / "confusion_matrix.png"
                plt.savefig(cm_plot_path, dpi=200)
                plt.close()
                metrics_result["confusion_matrix_plot"] = str(cm_plot_path)

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(y_test, raw_probs)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
                plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate (Recall)", fontsize=12)
                plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
                plt.legend(loc="lower right")
                plt.tight_layout()
                roc_plot_path = save_dir / "roc_curve.png"
                plt.savefig(roc_plot_path, dpi=200)
                plt.close()
                metrics_result["roc_curve_plot"] = str(roc_plot_path)
            except Exception as e:
                print(f"[Evaluator] Warning: Could not generate plots: {e}")

    return metrics_result
