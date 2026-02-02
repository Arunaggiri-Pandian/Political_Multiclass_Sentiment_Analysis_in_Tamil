"""
Model Analysis and Visualization for Tamil Political Sentiment Analysis
Generates comprehensive reports to understand model behavior
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import (
    load_data,
    TamilSentimentDataset,
    clean_text,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_model(model_path: str, device: torch.device):
    """Load trained model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def get_predictions_with_confidence(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """Get predictions, true labels, and confidence scores."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []

    for batch in tqdm(dataloader, desc="Getting predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'confidences': np.array(all_confidences)
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    normalize: bool = True
):
    """Plot and save confusion matrix."""
    labels = list(LABEL2ID.keys())
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_history(history_path: str, output_path: str):
    """Plot training curves from history CSV."""
    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train', marker='o')
    axes[0].plot(df['epoch'], df['dev_loss'], 'r-', label='Dev', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(df['epoch'], df['train_accuracy'], 'b-', label='Train', marker='o')
    axes[1].plot(df['epoch'], df['dev_accuracy'], 'r-', label='Dev', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True)

    # F1 curve
    axes[2].plot(df['epoch'], df['train_macro_f1'], 'b-', label='Train', marker='o')
    axes[2].plot(df['epoch'], df['dev_macro_f1'], 'r-', label='Dev', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Macro F1 Curve')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_metrics(report: Dict, output_path: str):
    """Plot per-class precision, recall, F1."""
    labels = list(LABEL2ID.keys())
    metrics = ['precision', 'recall', 'f1-score']

    data = []
    for label in labels:
        if label in report:
            for metric in metrics:
                data.append({
                    'Class': label,
                    'Metric': metric.capitalize(),
                    'Value': report[label][metric]
                })

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=df, x='Class', y='Value', hue='Metric')

    plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(title='Metric', loc='lower right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_distribution(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """Plot confidence distribution for correct vs incorrect predictions."""
    correct_mask = predictions == labels
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall distribution
    axes[0].hist(correct_conf, bins=20, alpha=0.7, label=f'Correct (n={len(correct_conf)})', color='green')
    axes[0].hist(incorrect_conf, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})', color='red')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()

    # Per-class confidence for incorrect predictions
    class_names = list(LABEL2ID.keys())
    incorrect_by_class = []
    for i, name in enumerate(class_names):
        mask = (labels == i) & (~correct_mask)
        if mask.sum() > 0:
            incorrect_by_class.append({
                'Class': name,
                'Mean Confidence': confidences[mask].mean(),
                'Count': mask.sum()
            })

    if incorrect_by_class:
        df = pd.DataFrame(incorrect_by_class)
        bars = axes[1].barh(df['Class'], df['Mean Confidence'], color='coral')
        axes[1].set_xlabel('Mean Confidence')
        axes[1].set_title('Avg Confidence on Misclassified Samples')
        axes[1].set_xlim(0, 1)

        # Add count labels
        for bar, count in zip(bars, df['Count']):
            axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'n={count}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_error_analysis(
    texts: List[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    output_path: str,
    max_samples: int = 50
):
    """Generate detailed error analysis CSV."""
    incorrect_mask = predictions != labels
    incorrect_indices = np.where(incorrect_mask)[0]

    errors = []
    for idx in incorrect_indices[:max_samples]:
        errors.append({
            'text': texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx],
            'true_label': ID2LABEL[labels[idx]],
            'predicted_label': ID2LABEL[predictions[idx]],
            'confidence': f"{confidences[idx]:.3f}",
            'text_length': len(texts[idx])
        })

    df = pd.DataFrame(errors)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Summary statistics
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)

    # Most common confusion pairs
    confusion_pairs = {}
    for idx in incorrect_indices:
        pair = (ID2LABEL[labels[idx]], ID2LABEL[predictions[idx]])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Confusion Pairs:")
    for (true_label, pred_label), count in sorted_pairs[:10]:
        print(f"  {true_label:20s} -> {pred_label:20s}: {count:3d}")

    return df


def plot_class_distribution_comparison(
    train_labels: np.ndarray,
    dev_labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str
):
    """Compare class distributions across train, dev, and predictions."""
    class_names = list(LABEL2ID.keys())

    train_dist = np.bincount(train_labels, minlength=NUM_LABELS) / len(train_labels)
    dev_dist = np.bincount(dev_labels, minlength=NUM_LABELS) / len(dev_labels)
    pred_dist = np.bincount(predictions, minlength=NUM_LABELS) / len(predictions)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, train_dist, width, label='Train', color='steelblue')
    bars2 = ax.bar(x, dev_dist, width, label='Dev (True)', color='forestgreen')
    bars3 = ax.bar(x + width, pred_dist, width, label='Dev (Predicted)', color='coral')

    ax.set_xlabel('Class')
    ax.set_ylabel('Proportion')
    ax.set_title('Class Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(train_dist.max(), dev_dist.max(), pred_dist.max()) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_full_report(model_dir: str, data_dir: str = "data"):
    """Generate comprehensive analysis report for a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create analysis output directory
    analysis_dir = os.path.join(model_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    print("="*60)
    print(f"GENERATING ANALYSIS REPORT")
    print(f"Model: {model_dir}")
    print("="*60)

    # Load model
    best_model_path = os.path.join(model_dir, 'best_model')
    if not os.path.exists(best_model_path):
        best_model_path = model_dir

    print("\nLoading model...")
    model, tokenizer = load_model(best_model_path, device)

    # Load data
    print("Loading data...")
    train_df, dev_df, _ = load_data(data_dir)

    # Create dev dataloader
    dev_dataset = TamilSentimentDataset(
        texts=dev_df['content'].tolist(),
        labels=dev_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=256
    )
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # Get predictions
    print("Running inference...")
    results = get_predictions_with_confidence(model, dev_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(results['labels'], results['predictions'])
    macro_f1 = f1_score(results['labels'], results['predictions'], average='macro')
    report = classification_report(
        results['labels'],
        results['predictions'],
        target_names=list(LABEL2ID.keys()),
        output_dict=True
    )

    print(f"\nDev Accuracy: {accuracy:.4f}")
    print(f"Dev Macro F1: {macro_f1:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion Matrix
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        os.path.join(analysis_dir, 'confusion_matrix.png')
    )

    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        os.path.join(analysis_dir, 'confusion_matrix_counts.png'),
        normalize=False
    )

    # 2. Training History
    history_path = os.path.join(model_dir, 'training_history.csv')
    if os.path.exists(history_path):
        plot_training_history(
            history_path,
            os.path.join(analysis_dir, 'training_curves.png')
        )

    # 3. Per-class metrics
    plot_per_class_metrics(
        report,
        os.path.join(analysis_dir, 'per_class_metrics.png')
    )

    # 4. Confidence distribution
    plot_confidence_distribution(
        results['confidences'],
        results['predictions'],
        results['labels'],
        os.path.join(analysis_dir, 'confidence_distribution.png')
    )

    # 5. Class distribution comparison
    plot_class_distribution_comparison(
        train_df['label_id'].values,
        results['labels'],
        results['predictions'],
        os.path.join(analysis_dir, 'class_distribution.png')
    )

    # 6. Error analysis
    error_df = generate_error_analysis(
        dev_df['content'].tolist(),
        results['predictions'],
        results['labels'],
        results['confidences'],
        os.path.join(analysis_dir, 'error_analysis.csv')
    )

    # 7. Save summary report
    summary = {
        'model_path': model_dir,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'total_samples': len(results['labels']),
        'correct_predictions': int((results['predictions'] == results['labels']).sum()),
        'incorrect_predictions': int((results['predictions'] != results['labels']).sum()),
        'mean_confidence_correct': float(results['confidences'][results['predictions'] == results['labels']].mean()),
        'mean_confidence_incorrect': float(results['confidences'][results['predictions'] != results['labels']].mean()) if (results['predictions'] != results['labels']).sum() > 0 else 0,
        'per_class_metrics': {k: v for k, v in report.items() if k in LABEL2ID}
    }

    with open(os.path.join(analysis_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {analysis_dir}")
    print("="*60)
    print("\nGenerated files:")
    for f in os.listdir(analysis_dir):
        print(f"  - {f}")

    return summary


def compare_models(model_dirs: List[str], output_dir: str, data_dir: str = "data"):
    """Compare multiple trained models."""
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    for model_dir in model_dirs:
        summary_path = os.path.join(model_dir, 'analysis', 'summary_report.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
                summary['model_name'] = os.path.basename(model_dir)
                summaries.append(summary)

    if not summaries:
        print("No model summaries found. Run analysis on individual models first.")
        return

    # Create comparison DataFrame
    comparison_data = []
    for s in summaries:
        row = {
            'Model': s['model_name'],
            'Accuracy': s['accuracy'],
            'Macro F1': s['macro_f1'],
            'Avg Conf (Correct)': s['mean_confidence_correct'],
            'Avg Conf (Incorrect)': s['mean_confidence_incorrect']
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy and F1 comparison
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width/2, df['Accuracy'], width, label='Accuracy')
    axes[0].bar(x + width/2, df['Macro F1'], width, label='Macro F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Confidence comparison
    axes[1].bar(x - width/2, df['Avg Conf (Correct)'], width, label='Correct', color='green')
    axes[1].bar(x + width/2, df['Avg Conf (Incorrect)'], width, label='Incorrect', color='red')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Model Confidence Comparison')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()

    print(f"Comparison saved to: {output_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trained models")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--compare", type=str, nargs='+',
                        help="Compare multiple models")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                        help="Output directory for comparison")

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.output_dir, args.data_dir)
    else:
        generate_full_report(args.model_dir, args.data_dir)
