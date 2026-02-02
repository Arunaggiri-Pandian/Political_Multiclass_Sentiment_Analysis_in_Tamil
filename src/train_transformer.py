"""
Transformer-based training script for Tamil Political Sentiment Analysis
Supports MuRIL, IndicBERT, XLM-RoBERTa fine-tuning with Focal Loss
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import (
    load_data,
    TamilSentimentDataset,
    get_class_weights,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regularization.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = targets_one_hot * self.confidence + self.smoothing / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=-1)
        return loss.mean()


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                criterion: nn.Module,
                device: torch.device,
                gradient_accumulation_steps: int = 1) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss = loss / gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }


@torch.no_grad()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=list(LABEL2ID.keys()),
        output_dict=True
    )

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'classification_report': report
    }


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = load_data(args.data_dir, use_augmented=args.use_augmented)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = TamilSentimentDataset(
        texts=train_df['content'].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    dev_dataset = TamilSentimentDataset(
        texts=dev_df['content'].tolist(),
        labels=dev_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Create dataloaders
    # Create weighted sampler for class imbalance
    train_labels = train_df['label_id'].tolist()
    class_counts = np.bincount(train_labels, minlength=NUM_LABELS)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = [class_weights_sampler[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    # Setup loss function
    if args.loss_type == 'focal':
        class_weights = get_class_weights(train_df['label_id'].tolist())
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
    elif args.loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(NUM_LABELS, smoothing=args.label_smoothing)
        print(f"Using Label Smoothing Loss with smoothing={args.label_smoothing}")
    else:
        class_weights = get_class_weights(train_df['label_id'].tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using Weighted Cross Entropy Loss")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Setup scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.early_stopping_patience} epochs")
    best_macro_f1 = 0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*50)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro F1: {train_metrics['macro_f1']:.4f}")

        # Evaluate
        dev_metrics = evaluate(model, dev_loader, criterion, device)
        print(f"Dev - Loss: {dev_metrics['loss']:.4f}, "
              f"Acc: {dev_metrics['accuracy']:.4f}, "
              f"Macro F1: {dev_metrics['macro_f1']:.4f}, "
              f"Weighted F1: {dev_metrics['weighted_f1']:.4f}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_macro_f1': train_metrics['macro_f1'],
            'dev_loss': dev_metrics['loss'],
            'dev_accuracy': dev_metrics['accuracy'],
            'dev_macro_f1': dev_metrics['macro_f1'],
            'dev_weighted_f1': dev_metrics['weighted_f1']
        })

        # Save best model
        if dev_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = dev_metrics['macro_f1']
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            print(f"New best model! Macro F1: {best_macro_f1:.4f}")

            model.save_pretrained(os.path.join(output_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(output_dir, 'best_model'))

            # Save classification report
            with open(os.path.join(output_dir, 'best_classification_report.json'), 'w') as f:
                json.dump(dev_metrics['classification_report'], f, indent=2)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Generate and save training curves
    print("\nGenerating training curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs_range = history_df['epoch']

    # Loss curve
    axes[0].plot(epochs_range, history_df['train_loss'], 'b-', label='Train', marker='o', markersize=3)
    axes[0].plot(epochs_range, history_df['dev_loss'], 'r-', label='Dev', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs_range, history_df['train_accuracy'], 'b-', label='Train', marker='o', markersize=3)
    axes[1].plot(epochs_range, history_df['dev_accuracy'], 'r-', label='Dev', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Macro F1 curve
    axes[2].plot(epochs_range, history_df['train_macro_f1'], 'b-', label='Train', marker='o', markersize=3)
    axes[2].plot(epochs_range, history_df['dev_macro_f1'], 'r-', label='Dev', marker='s', markersize=3)
    axes[2].axhline(y=best_macro_f1, color='g', linestyle='--', label=f'Best ({best_macro_f1:.3f})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Macro F1 Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Training Progress - {args.model_name.split("/")[-1]}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'training_curves.png')}")

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {output_dir}")
    print('='*50)

    # Print final classification report
    print("\nBest Model Classification Report:")
    with open(os.path.join(output_dir, 'best_classification_report.json'), 'r') as f:
        report = json.load(f)
        for label in LABEL2ID.keys():
            if label in report:
                print(f"{label:20s}: P={report[label]['precision']:.3f}, "
                      f"R={report[label]['recall']:.3f}, "
                      f"F1={report[label]['f1-score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer for Tamil sentiment analysis")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased",
                        help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                        help="Output directory")
    parser.add_argument("--use_augmented", action="store_true",
                        help="Use augmented training data")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")

    # Loss arguments
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["focal", "weighted_ce", "label_smoothing"],
                        help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop training if no improvement for N epochs")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    main(args)
