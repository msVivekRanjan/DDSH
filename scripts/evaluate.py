import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
evaluate.py — Comprehensive Model Evaluation & Metrics Pipeline

After training, this script:
1. Loads the trained model from disk
2. Evaluates on test dataset
3. Generates confusion matrix
4. Calculates: Accuracy, Precision, Recall, F1-Score
5. Generates ROC/AUC curve
6. Compares results with paper's expected values for sanity check
7. Saves all plots to outputs/

Reference: Bhanja et al., ROBOMECH Journal (2025)
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import config
from preprocess import prepare_train_test_split


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────

def load_model(model_path: str = None) -> tf.keras.Model:
    """
    Load pre-trained model from disk.

    Args:
        model_path: Path to model file. Defaults to config.MODEL_PATH.

    Returns:
        Loaded model ready for inference.
    """
    if model_path is None:
        model_path = config.MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"\n📦 Loading model from: {model_path}")
    # compile=False avoids legacy Adam optimizer version mismatch on TF 2.13 + macOS
    model = tf.keras.models.load_model(model_path, compile=False)
    # Recompile fresh with current optimizer
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=["accuracy"],
    )
    print(f"   ✓ Model loaded and recompiled successfully")
    return model


# ──────────────────────────────────────────────
# Predict on tf.data.Dataset
# ──────────────────────────────────────────────

def predict_on_dataset(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions by iterating over a tf.data.Dataset batch by batch.

    Args:
        model     : Trained Keras model.
        test_ds   : Batched tf.data.Dataset yielding (images, labels).
        threshold : Classification threshold. Default 0.5.

    Returns:
        scores      : Raw drowsiness scores (N,)
        predictions : Binary predictions (N,) ∈ {0, 1}
        y_true      : Ground-truth labels (N,)
    """
    print(f"\n🔮 Generating predictions (threshold={threshold})...")

    all_scores  = []
    all_labels  = []

    for batch_images, batch_labels in test_ds:
        batch_scores = model.predict(batch_images, verbose=0).flatten()
        all_scores.extend(batch_scores.tolist())
        all_labels.extend(batch_labels.numpy().tolist())

    scores      = np.array(all_scores,  dtype=np.float32)
    y_true      = np.array(all_labels,  dtype=np.int32)
    predictions = (scores >= threshold).astype(int)

    print(f"   ✓ Predictions complete")
    print(f"   Score range         : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   Predicted 0 (open)  : {np.sum(predictions == 0)} samples")
    print(f"   Predicted 1 (closed): {np.sum(predictions == 1)} samples")

    return scores, predictions, y_true


# ──────────────────────────────────────────────
# Compute metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).

    Returns:
        dict with accuracy, precision, recall, f1_score, tp, tn, fp, fn, confusion_matrix.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_score":         f1,
        "tp":               tp,
        "tn":               tn,
        "fp":               fp,
        "fn":               fn,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────

def print_evaluation_report(
    metrics: dict, y_true: np.ndarray, y_scores: np.ndarray
) -> None:
    """Print comprehensive evaluation report with paper comparison."""

    print("\n" + "=" * 70)
    print("📊 MODEL EVALUATION — DDSH (Bhanja et al., ROBOMECH 2025)")
    print("=" * 70)

    print("\n🎯 Classification Metrics:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  | Paper: {config.PAPER_ACCURACY:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  | Paper: {config.PAPER_PRECISION:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  | Paper: {config.PAPER_RECALL:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}  | Paper: {config.PAPER_F1_SCORE:.4f}")

    print("\n📋 Confusion Matrix:")
    print(f"  True Positives (TP)  : {metrics['tp']}  | Paper: {config.PAPER_TP}")
    print(f"  True Negatives (TN)  : {metrics['tn']}  | Paper: {config.PAPER_TN}")
    print(f"  False Positives (FP) : {metrics['fp']}  | Paper: {config.PAPER_FP}")
    print(f"  False Negatives (FN) : {metrics['fn']}  | Paper: {config.PAPER_FN}")

    specificity = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
    sensitivity = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0

    print(f"\n📈 Additional Metrics:")
    print(f"  Sensitivity (Recall) : {sensitivity:.4f}")
    print(f"  Specificity          : {specificity:.4f}")

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"  ROC-AUC              : {roc_auc:.4f}")
    except Exception:
        pass

    print("=" * 70 + "\n")


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, output_dir: str = None) -> str:
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Open Eyes", "Closed Eyes"],
        yticklabels=["Open Eyes", "Closed Eyes"],
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title("Confusion Matrix — DDSH Model", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label",      fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Confusion matrix saved: {plot_path}")
    plt.close()
    return plot_path


def plot_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, output_dir: str = None
) -> str:
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate",  fontsize=12, fontweight="bold")
    plt.title("ROC Curve — DDSH Model", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ ROC curve saved: {plot_path}")
    plt.close()
    return plot_path


def plot_metric_comparison(metrics: dict, output_dir: str = None) -> str:
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    our_values   = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    paper_values = [config.PAPER_ACCURACY, config.PAPER_PRECISION, config.PAPER_RECALL, config.PAPER_F1_SCORE]

    x     = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, our_values,   width, label="Our Model",              color="skyblue")
    bars2 = ax.bar(x + width / 2, paper_values, width, label="Paper (Bhanja et al.)",  color="lightcoral")

    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score",   fontsize=12, fontweight="bold")
    ax.set_title("Metric Comparison: Our Model vs Paper", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.02,
            f"{height:.3f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Metrics comparison saved: {plot_path}")
    plt.close()
    return plot_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    """
    Main evaluation pipeline: Load Model → Predict → Metrics → Plots
    """
    print("\n" + "=" * 70)
    print("DDSH — Model Evaluation Pipeline")
    print("=" * 70)

    try:
        # Step 1: Load dataset as tf.data generators
        print("\n📥 Loading dataset...")
        train_ds, test_ds, train_info, test_info = prepare_train_test_split(
            config.TRAIN_DIR, config.TEST_DIR, batch_size=config.BATCH_SIZE
        )

        # Step 2: Load trained model
        model = load_model()

        # Step 3: Generate predictions + collect ground-truth labels
        scores, predictions, y_true = predict_on_dataset(
            model, test_ds, threshold=config.DROWSINESS_THRESHOLD
        )

        # Step 4: Compute metrics
        metrics = compute_metrics(y_true, predictions)

        # Step 5: Print report
        print_evaluation_report(metrics, y_true, scores)

        # Step 6: Generate plots
        print("\n📊 Generating evaluation plots...")
        plot_confusion_matrix(metrics)
        plot_roc_curve(y_true, scores)
        plot_metric_comparison(metrics)

        print("\n✅ Evaluation complete!")
        print(f"   All plots saved to: {config.OUTPUT_DIR}/")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("   Make sure you've completed training (run train.py first)")

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()