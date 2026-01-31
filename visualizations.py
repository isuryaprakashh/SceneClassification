"""
Visualization Utilities for Scene Classification.

This module provides functions to generate publication-quality visualizations:
- Training/validation loss and accuracy curves
- Model comparison bar charts
- Confusion matrix heatmaps
- t-SNE feature visualization
- Sample prediction displays

Usage:
    python visualizations.py --results_dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DEFAULT_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed', '#0891b2']


def plot_training_curves(
    training_history: Dict[str, Dict],
    output_path: Path,
    title: str = "Training Progress",
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        training_history: Dict with model names as keys, each containing
                         'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        output_path: Path to save the figure
        title: Plot title
    """
    num_models = len(training_history)
    fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 8))
    
    if num_models == 1:
        axes = axes.reshape(-1, 1)

    for idx, (model_name, history) in enumerate(training_history.items()):
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss plot
        ax_loss = axes[0, idx]
        if 'train_loss' in history:
            ax_loss.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
        if 'val_loss' in history:
            ax_loss.plot(epochs, history['val_loss'], 'r--', label='Validation', linewidth=2)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{model_name} - Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy plot
        ax_acc = axes[1, idx]
        if 'train_acc' in history:
            ax_acc.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
        if 'val_acc' in history:
            ax_acc.plot(epochs, history['val_acc'], 'r--', label='Validation', linewidth=2)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title(f'{model_name} - Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Training curves saved to: {output_path}")


def plot_model_comparison(
    results: Dict[str, Dict],
    output_path: Path,
    metrics: List[str] = ['overall_accuracy', 'macro_f1', 'macro_auc'],
):
    """
    Create bar chart comparing models across metrics.
    
    Args:
        results: Dict with model names as keys, metrics as nested values
        output_path: Path to save the figure
        metrics: List of metrics to compare
    """
    model_names = list(results.keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    metric_labels = {
        'overall_accuracy': ('Accuracy (%)', lambda r: r.get('overall_accuracy', 0)),
        'macro_f1': ('Macro F1-Score', lambda r: r.get('macro_metrics', {}).get('f1_score', 0)),
        'macro_auc': ('Macro AUC', lambda r: r.get('macro_auc', 0)),
        'top_3_accuracy': ('Top-3 Accuracy (%)', lambda r: r.get('top_3_accuracy', 0)),
        'top_5_accuracy': ('Top-5 Accuracy (%)', lambda r: r.get('top_5_accuracy', 0)),
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        label, extractor = metric_labels.get(metric, (metric, lambda r: 0))
        
        values = [extractor(results[m]) for m in model_names]
        bars = ax.bar(model_names, values, color=DEFAULT_COLORS[:len(model_names)], edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold',
            )
        
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticklabels(model_names, rotation=45, ha='right')

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Model comparison saved to: {output_path}")


def plot_per_class_metrics(
    per_class_data: Dict[str, Dict],
    output_path: Path,
    metric: str = 'f1_score',
    top_n: int = 20,
):
    """
    Plot per-class metrics as horizontal bar chart.
    
    Args:
        per_class_data: Dict with class names as keys, metrics as values
        output_path: Path to save the figure
        metric: Metric to plot ('precision', 'recall', 'f1_score')
        top_n: Number of classes to show (sorted by metric value)
    """
    # Sort by metric value
    sorted_classes = sorted(
        per_class_data.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True
    )[:top_n]
    
    class_names = [c[0] for c in sorted_classes]
    values = [c[1].get(metric, 0) for c in sorted_classes]

    plt.figure(figsize=(10, max(6, len(class_names) * 0.3)))
    
    colors = ['#22c55e' if v >= 0.7 else '#f59e0b' if v >= 0.5 else '#ef4444' for v in values]
    
    bars = plt.barh(class_names[::-1], values[::-1], color=colors[::-1], edgecolor='black', alpha=0.8)
    
    plt.xlabel(f'{metric.replace("_", " ").title()}')
    plt.title(f'Per-Class {metric.replace("_", " ").title()} (Top {top_n})')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Per-class metrics saved to: {output_path}")


def plot_accuracy_distribution(
    per_class_accuracy: Dict[str, Dict],
    output_path: Path,
):
    """
    Plot histogram of per-class accuracy distribution.
    """
    accuracies = [v.get('accuracy', 0) for v in per_class_accuracy.values()]
    
    plt.figure(figsize=(10, 6))
    
    n, bins, patches = plt.hist(accuracies, bins=10, edgecolor='black', alpha=0.7, color='#3b82f6')
    
    # Color bins by accuracy range
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center >= 80:
            patch.set_facecolor('#22c55e')
        elif bin_center >= 60:
            patch.set_facecolor('#f59e0b')
        else:
            patch.set_facecolor('#ef4444')

    plt.axvline(np.mean(accuracies), color='navy', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.1f}%')
    plt.axvline(np.median(accuracies), color='darkgreen', linestyle=':', linewidth=2, label=f'Median: {np.median(accuracies):.1f}%')
    
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Per-Class Accuracy')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Accuracy distribution saved to: {output_path}")


def create_summary_table(
    results: Dict[str, Dict],
    output_path: Path,
):
    """
    Create a summary table image with key metrics.
    """
    fig, ax = plt.subplots(figsize=(12, max(3, len(results) + 1)))
    ax.axis('off')

    headers = ['Model', 'Accuracy', 'Top-3 Acc', 'Top-5 Acc', 'Macro F1', 'AUC', 'Samples']
    data = []
    
    for name, res in results.items():
        row = [
            name,
            f"{res.get('overall_accuracy', 0):.2f}%",
            f"{res.get('top_3_accuracy', 0):.2f}%",
            f"{res.get('top_5_accuracy', 0):.2f}%",
            f"{res.get('macro_metrics', {}).get('f1_score', 0):.4f}",
            f"{res.get('macro_auc', 0):.4f}",
            str(res.get('total_samples', 0)),
        ]
        data.append(row)

    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i, header in enumerate(headers):
        table[(0, i)].set_facecolor('#1e40af')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#f1f5f9')

    plt.title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Summary table saved to: {output_path}")


def generate_all_visualizations(results_dir: Path):
    """
    Generate all visualizations from evaluation results.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Try to load comparison results
    comparison_path = results_dir / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, "r") as f:
            all_results = json.load(f)
    else:
        # Load individual evaluation results
        eval_path = results_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                all_results = {"default": json.load(f)}
        else:
            print("[ERROR] No evaluation results found. Run evaluate.py first.")
            return

    # Generate model comparison
    if len(all_results) >= 1:
        plot_model_comparison(
            all_results,
            results_dir / "model_comparison.png",
            metrics=['overall_accuracy', 'macro_f1', 'macro_auc'],
        )

    # Generate summary table
    create_summary_table(all_results, results_dir / "summary_table.png")

    # Generate per-model visualizations
    for model_name, results in all_results.items():
        model_dir = results_dir / model_name if model_name != "default" else results_dir
        
        # Per-class metrics
        if 'per_class_metrics' in results:
            plot_per_class_metrics(
                results['per_class_metrics'],
                model_dir / "per_class_f1.png",
                metric='f1_score',
            )

        # Accuracy distribution
        if 'per_class_accuracy' in results:
            plot_accuracy_distribution(
                results['per_class_accuracy'],
                model_dir / "accuracy_distribution.png",
            )

    # Load training history if available
    history_path = results_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history = json.load(f)
        plot_training_curves(
            training_history,
            results_dir / "training_curves.png",
        )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for Scene Classification")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing evaluation results",
    )
    args = parser.parse_args()

    generate_all_visualizations(args.results_dir)


if __name__ == "__main__":
    main()
