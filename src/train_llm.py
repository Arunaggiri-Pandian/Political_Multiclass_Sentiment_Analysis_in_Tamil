"""
LLM Fine-tuning script for Tamil Political Sentiment Analysis
Supports Gemma, Qwen, LLaMA with LoRA/QLoRA
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import (
    load_data,
    clean_text,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)


# Instruction template for classification
SYSTEM_PROMPT = """You are an expert in Tamil political sentiment analysis. Classify the given Tamil tweet into exactly one of these categories:
- Substantiated: Contains evidence, references, statistics, or factual claims
- Sarcastic: Uses sarcasm, mockery, irony, or humor
- Opinionated: Expresses strong personal views or biased beliefs
- Positive: Shows approval, optimism, or appreciation
- Negative: Expresses criticism or dissatisfaction
- Neutral: Factual political information without expressing opinions
- None of the above: Not related to politics

Respond with ONLY the category name, nothing else."""

PROMPT_TEMPLATE = """Tweet: {tweet}

Category:"""


class LLMSentimentDataset(Dataset):
    """Dataset for LLM fine-tuning with instruction format."""

    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 tokenizer=None,
                 max_length: int = 512,
                 is_train: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = clean_text(self.texts[idx])
        prompt = PROMPT_TEMPLATE.format(tweet=text)

        if self.is_train and self.labels is not None:
            label_text = ID2LABEL[self.labels[idx]]
            # Format: <s>[INST] {system}\n{prompt} [/INST] {response}</s>
            # Adjust based on model's chat template
            full_text = f"{SYSTEM_PROMPT}\n\n{prompt} {label_text}"

            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            # Create labels - mask the prompt part
            labels = input_ids.clone()

            # Find where the response starts (after "Category: ")
            prompt_with_system = f"{SYSTEM_PROMPT}\n\n{prompt}"
            prompt_encoding = self.tokenizer(
                prompt_with_system,
                return_tensors='pt'
            )
            prompt_len = prompt_encoding['input_ids'].shape[1]

            # Mask prompt tokens
            labels[:prompt_len] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            # For inference
            full_text = f"{SYSTEM_PROMPT}\n\n{prompt}"

            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable length sequences."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        padding_len = max_len - seq_len

        # Left padding for causal LM
        input_ids.append(
            torch.cat([
                torch.full((padding_len,), pad_token_id),
                item['input_ids']
            ])
        )
        attention_mask.append(
            torch.cat([
                torch.zeros(padding_len),
                item['attention_mask']
            ])
        )
        if 'labels' in item:
            labels.append(
                torch.cat([
                    torch.full((padding_len,), -100),
                    item['labels']
                ])
            )

    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask).long()
    }

    if labels:
        result['labels'] = torch.stack(labels)

    return result


def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device):
    """Evaluate model by generating predictions."""
    model.eval()
    all_preds = []
    all_labels = []

    # Create label mapping for fuzzy matching
    label_keywords = {
        'substantiated': 0, 'evidence': 0, 'factual': 0,
        'sarcastic': 1, 'sarcasm': 1, 'irony': 1, 'mockery': 1,
        'opinionated': 2, 'opinion': 2, 'personal': 2, 'bias': 2,
        'positive': 3, 'approval': 3, 'appreciation': 3,
        'negative': 4, 'criticism': 4, 'dissatisfaction': 4,
        'neutral': 5, 'factual': 5,
        'none': 6, 'not related': 6, 'off-topic': 6
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode only the generated part
        for i, output in enumerate(outputs):
            generated_ids = output[input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

            # Match to label
            pred_label = 6  # Default to "None of the above"
            for keyword, label_id in label_keywords.items():
                if keyword in generated_text:
                    pred_label = label_id
                    break

            all_preds.append(pred_label)

        if 'labels' in batch:
            # Extract true labels from the batch
            labels = batch['labels']
            for label_seq in labels:
                # Find the non-masked label
                valid_labels = label_seq[label_seq != -100]
                if len(valid_labels) > 0:
                    label_text = tokenizer.decode(valid_labels, skip_special_tokens=True).strip().lower()
                    true_label = 6
                    for keyword, label_id in label_keywords.items():
                        if keyword in label_text:
                            true_label = label_id
                            break
                    all_labels.append(true_label)

    results = {'predictions': all_preds}

    if all_labels:
        results['accuracy'] = accuracy_score(all_labels, all_preds)
        results['macro_f1'] = f1_score(all_labels, all_preds, average='macro')
        results['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted')
        results['labels'] = all_labels

    return results


def main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    output_dir = os.path.join(args.output_dir, f"llm_{model_short_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = load_data(args.data_dir)

    # Quantization config for QLoRA
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        print("Using 4-bit quantization (QLoRA)")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None
    )

    # Prepare for k-bit training if using quantization
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    train_dataset = LLMSentimentDataset(
        texts=train_df['content'].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True
    )

    dev_dataset = LLMSentimentDataset(
        texts=dev_df['content'].tolist(),
        labels=dev_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=4
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=4
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Setup scheduler
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_macro_f1 = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*50)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate (every eval_epochs)
        if (epoch + 1) % args.eval_epochs == 0:
            print("Evaluating...")
            dev_metrics = evaluate(model, dev_loader, tokenizer, device)

            if 'macro_f1' in dev_metrics:
                print(f"Dev - Acc: {dev_metrics['accuracy']:.4f}, "
                      f"Macro F1: {dev_metrics['macro_f1']:.4f}")

                history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'dev_accuracy': dev_metrics['accuracy'],
                    'dev_macro_f1': dev_metrics['macro_f1']
                })

                # Save best model
                if dev_metrics['macro_f1'] > best_macro_f1:
                    best_macro_f1 = dev_metrics['macro_f1']
                    print(f"New best model! Macro F1: {best_macro_f1:.4f}")

                    model.save_pretrained(os.path.join(output_dir, 'best_model'))
                    tokenizer.save_pretrained(os.path.join(output_dir, 'best_model'))

    # Save final model
    model.save_pretrained(os.path.join(output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'final_model'))

    # Save history
    if history:
        pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Tamil sentiment analysis")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                        help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                        help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_epochs", type=int, default=1,
                        help="Evaluate every N epochs")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Quantization arguments
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA)")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    main(args)
