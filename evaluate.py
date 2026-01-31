"""
Comprehensive Evaluation Script for Scene Classification Models.

This script provides detailed evaluation metrics including:
- Per-class accuracy
- Confusion matrix generation and visualization
- Precision, Recall, F1-Score (per-class and macro/weighted averages)
- Top-k accuracy (k=1, 3, 5)
- ROC curves and AUC scores (one-vs-rest)
- Export all results to JSON for reproducibility

Usage:
    python evaluate.py [--model_dir ARTIFACTS_DIR] [--data_dir DATASET_DIR]
"""

from __future__ import annotations

import argparse
import json
import ssl
import certifi
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import Preprocess as pre
from models1 import ANNClassifier, CNNRegularized, ResNetTransfer

# Configuration
IMAGE_SIZE = 224
DEFAULT_DATA_DIR = Path("DATASET") / "images"
DEFAULT_ARTIFACT_DIR = Path("artifacts")


def load_model_checkpoint(checkpoint_path: Path, num_classes: int, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", "CNNRegularized")
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    model_classes = {
        "ANNClassifier": ANNClassifier,
        "CNNRegularized": CNNRegularized,
        "ResNetTransfer": ResNetTransfer,
    }

    ModelClass = model_classes.get(arch, CNNRegularized)
    model = ModelClass(num_classes=num_classes, input_size=image_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, arch, image_size


def compute_predictions(model, dataloader, device, num_classes: int):
    """
    Run inference on entire dataset and collect predictions.
    
    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted class indices
        all_probs: Prediction probabilities for all classes
    """
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def calculate_topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(label in top_k for label, top_k in zip(labels, top_k_preds))
    return (correct / len(labels)) * 100


def calculate_per_class_accuracy(labels: np.ndarray, preds: np.ndarray, classes: list) -> dict:
    """Calculate accuracy for each class."""
    per_class = {}
    for idx, class_name in enumerate(classes):
        mask = labels == idx
        if mask.sum() > 0:
            correct = (preds[mask] == idx).sum()
            per_class[class_name] = {
                "accuracy": float(correct / mask.sum() * 100),
                "total_samples": int(mask.sum()),
                "correct": int(correct),
            }
    return per_class


def generate_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    classes: list,
    output_path: Path,
    normalize: bool = True,
):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(labels, preds)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero

    # Create figure with appropriate size
    fig_size = max(12, len(classes) * 0.3)
    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=len(classes) <= 20,  # Only show annotations for smaller matrices
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        square=True,
    )

    plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cm


def generate_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    classes: list,
    output_path: Path,
    max_classes_to_plot: int = 15,
):
    """Generate ROC curves for multi-class classification (one-vs-rest)."""
    n_classes = len(classes)
    
    # Binarize labels
    labels_bin = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        labels_bin[i, label] = 1

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot individual class ROC curves (limited for readability)
    sorted_indices = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)
    colors = plt.cm.tab20(np.linspace(0, 1, min(max_classes_to_plot, n_classes)))

    for idx, i in enumerate(sorted_indices[:max_classes_to_plot]):
        plt.plot(
            fpr[i], tpr[i],
            color=colors[idx],
            lw=1.5,
            alpha=0.7,
            label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Plot macro-average
    plt.plot(
        all_fpr, mean_tpr,
        color="navy",
        linestyle="--",
        lw=2,
        label=f"Macro-average (AUC = {macro_auc:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {classes[i]: float(roc_auc[i]) for i in range(n_classes)}, float(macro_auc)


def evaluate_model(
    model_dir: Path,
    data_dir: Path,
    device: torch.device,
    output_dir: Path,
):
    """Run comprehensive evaluation on a trained model."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Load classes
    classes_path = model_dir / "classes.json"
    with open(classes_path, "r") as f:
        classes = json.load(f)["classes"]
    num_classes = len(classes)
    print(f"[INFO] Number of classes: {num_classes}")

    # Load model
    model_path = model_dir / "best_model.pt"
    model, arch, image_size = load_model_checkpoint(model_path, num_classes, device)
    print(f"[INFO] Loaded model: {arch} (image_size={image_size})")

    # Load validation data
    _, val_loader, _ = pre.load_data(data_dir, batch_size=32, augment=False, num_workers=4)
    print(f"[INFO] Validation samples: {len(val_loader.dataset)}")

    # Get predictions
    print("[INFO] Running inference...")
    labels, preds, probs = compute_predictions(model, val_loader, device, num_classes)

    # Calculate metrics
    results = {
        "model_architecture": arch,
        "image_size": image_size,
        "num_classes": num_classes,
        "total_samples": int(len(labels)),
        "evaluation_date": datetime.now().isoformat(),
    }

    # Overall accuracy
    overall_acc = accuracy_score(labels, preds) * 100
    results["overall_accuracy"] = float(overall_acc)
    print(f"\n✔ Overall Accuracy: {overall_acc:.2f}%")

    # Top-k accuracy
    for k in [1, 3, 5]:
        if k <= num_classes:
            topk_acc = calculate_topk_accuracy(probs, labels, k)
            results[f"top_{k}_accuracy"] = float(topk_acc)
            print(f"✔ Top-{k} Accuracy: {topk_acc:.2f}%")

    # Per-class accuracy
    per_class_acc = calculate_per_class_accuracy(labels, preds, classes)
    results["per_class_accuracy"] = per_class_acc

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    results["macro_metrics"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1_score": float(macro_f1),
    }
    results["weighted_metrics"] = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1_score": float(weighted_f1),
    }

    print(f"\n✔ Macro F1-Score: {macro_f1:.4f}")
    print(f"✔ Weighted F1-Score: {weighted_f1:.4f}")

    # Per-class precision/recall/F1
    per_class_metrics = {}
    for idx, class_name in enumerate(classes):
        per_class_metrics[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
        }
    results["per_class_metrics"] = per_class_metrics

    # Generate confusion matrix
    print("\n[INFO] Generating confusion matrix...")
    cm_path = output_dir / "confusion_matrix.png"
    cm = generate_confusion_matrix(labels, preds, classes, cm_path)
    results["confusion_matrix_path"] = str(cm_path)

    # Generate ROC curves
    print("[INFO] Generating ROC curves...")
    roc_path = output_dir / "roc_curves.png"
    auc_scores, macro_auc = generate_roc_curves(labels, probs, classes, roc_path)
    results["auc_scores"] = auc_scores
    results["macro_auc"] = macro_auc
    print(f"✔ Macro AUC: {macro_auc:.4f}")

    # Save classification report
    report = classification_report(labels, preds, target_names=classes, output_dict=True)
    results["classification_report"] = report

    # Save results to JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✔ Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


def evaluate_all_models(
    artifact_dir: Path,
    data_dir: Path,
    device: torch.device,
):
    """Evaluate all trained models and create comparison."""
    model_dirs = [d for d in artifact_dir.iterdir() if d.is_dir() and (d / "best_model.pt").exists()]
    
    if not model_dirs:
        # Check if there's a single model in artifacts root
        if (artifact_dir / "best_model.pt").exists():
            model_dirs = [artifact_dir]

    all_results = {}
    
    for model_dir in model_dirs:
        model_name = model_dir.name if model_dir != artifact_dir else "default"
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        results = evaluate_model(model_dir, data_dir, device, model_dir)
        all_results[model_name] = results

    # Create comparison summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':>10} {'Top-3':>10} {'Macro F1':>10} {'AUC':>10}")
        print("-" * 60)
        
        for name, res in all_results.items():
            acc = res.get("overall_accuracy", 0)
            top3 = res.get("top_3_accuracy", 0)
            f1 = res.get("macro_metrics", {}).get("f1_score", 0)
            auc_val = res.get("macro_auc", 0)
            print(f"{name:<20} {acc:>9.2f}% {top3:>9.2f}% {f1:>10.4f} {auc_val:>10.4f}")

    # Save comparison
    comparison_path = artifact_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔ Comparison saved to: {comparison_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scene Classification Models")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained model(s)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dataset",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    evaluate_all_models(args.model_dir, args.data_dir, device)


if __name__ == "__main__":
    main()
