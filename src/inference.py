"""
Inference script for Tamil Political Sentiment Analysis
Supports transformer models, LLMs, and ensemble predictions
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import (
    load_data,
    TamilSentimentDataset,
    clean_text,
    create_submission,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)


def load_transformer_model(model_path: str, device: torch.device):
    """Load fine-tuned transformer model."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_llm_model(model_path: str, base_model: str, device: torch.device):
    """Load fine-tuned LLM with LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def predict_transformer(model,
                        tokenizer,
                        texts: List[str],
                        device: torch.device,
                        batch_size: int = 32,
                        max_length: int = 256) -> Dict:
    """Generate predictions using transformer model."""
    dataset = TamilSentimentDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    all_preds = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().float().numpy())

    return {
        'predictions': all_preds,
        'probabilities': np.array(all_probs)
    }


@torch.no_grad()
def predict_llm(model,
                tokenizer,
                texts: List[str],
                device: torch.device,
                batch_size: int = 8,
                max_length: int = 512) -> Dict:
    """Generate predictions using LLM."""
    from src.train_llm import SYSTEM_PROMPT, PROMPT_TEMPLATE

    # Label mapping for fuzzy matching
    label_keywords = {
        'substantiated': 0, 'evidence': 0,
        'sarcastic': 1, 'sarcasm': 1, 'irony': 1,
        'opinionated': 2, 'opinion': 2,
        'positive': 3, 'approval': 3,
        'negative': 4, 'criticism': 4,
        'neutral': 5,
        'none': 6, 'not related': 6
    }

    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]

        # Prepare prompts
        prompts = []
        for text in batch_texts:
            cleaned = clean_text(text)
            prompt = f"{SYSTEM_PROMPT}\n\n{PROMPT_TEMPLATE.format(tweet=cleaned)}"
            prompts.append(prompt)

        # Tokenize
        encodings = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode predictions
        for j, output in enumerate(outputs):
            generated_ids = output[input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

            # Match to label
            pred_label = 6  # Default
            for keyword, label_id in label_keywords.items():
                if keyword in generated_text:
                    pred_label = label_id
                    break

            all_preds.append(pred_label)

    return {'predictions': all_preds}


def ensemble_predictions(predictions_list: List[np.ndarray],
                        probabilities_list: Optional[List[np.ndarray]] = None,
                        weights: Optional[List[float]] = None,
                        method: str = 'voting') -> np.ndarray:
    """
    Ensemble predictions from multiple models.

    Args:
        predictions_list: List of prediction arrays
        probabilities_list: List of probability arrays (for soft voting)
        weights: Model weights
        method: 'voting' (hard), 'soft' (weighted average of probs), or 'weighted_voting'

    Returns:
        Final predictions
    """
    n_samples = len(predictions_list[0])
    n_models = len(predictions_list)

    if weights is None:
        weights = [1.0 / n_models] * n_models

    if method == 'voting':
        # Hard voting
        predictions_array = np.array(predictions_list)
        final_preds = []
        for i in range(n_samples):
            votes = predictions_array[:, i]
            counts = np.bincount(votes, minlength=NUM_LABELS)
            final_preds.append(np.argmax(counts))
        return np.array(final_preds)

    elif method == 'weighted_voting':
        # Weighted hard voting
        predictions_array = np.array(predictions_list)
        final_preds = []
        for i in range(n_samples):
            weighted_counts = np.zeros(NUM_LABELS)
            for j, w in enumerate(weights):
                weighted_counts[predictions_array[j, i]] += w
            final_preds.append(np.argmax(weighted_counts))
        return np.array(final_preds)

    elif method == 'soft' and probabilities_list is not None:
        # Soft voting (weighted average of probabilities)
        weighted_probs = np.zeros((n_samples, NUM_LABELS))
        for probs, w in zip(probabilities_list, weights):
            weighted_probs += w * probs
        return np.argmax(weighted_probs, axis=1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    _, _, test_df = load_data(args.data_dir)
    test_texts = test_df['content'].tolist()
    print(f"Test samples: {len(test_texts)}")

    predictions_list = []
    probabilities_list = []
    model_names = []

    # Load and predict with each model
    for model_path in args.model_paths:
        print(f"\nLoading model: {model_path}")

        if args.model_type == 'transformer':
            model, tokenizer = load_transformer_model(model_path, device)
            results = predict_transformer(
                model, tokenizer, test_texts, device,
                batch_size=args.batch_size,
                max_length=args.max_length
            )
            predictions_list.append(np.array(results['predictions']))
            probabilities_list.append(results['probabilities'])

        elif args.model_type == 'llm':
            model, tokenizer = load_llm_model(model_path, args.base_model, device)
            results = predict_llm(
                model, tokenizer, test_texts, device,
                batch_size=args.batch_size,
                max_length=args.max_length
            )
            predictions_list.append(np.array(results['predictions']))

        model_names.append(os.path.basename(model_path))

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Ensemble if multiple models
    if len(predictions_list) > 1:
        print(f"\nEnsembling {len(predictions_list)} models using {args.ensemble_method}...")

        if args.ensemble_method == 'soft' and probabilities_list:
            final_predictions = ensemble_predictions(
                predictions_list,
                probabilities_list=probabilities_list,
                method='soft'
            )
        else:
            final_predictions = ensemble_predictions(
                predictions_list,
                method=args.ensemble_method
            )
    else:
        final_predictions = predictions_list[0]

    # Create submission
    output_path = os.path.join(args.output_dir, args.output_name)
    submission_df = create_submission(
        final_predictions.tolist(),
        test_df,
        output_path
    )

    print(f"\nSubmission saved to: {output_path}")

    # Save individual model predictions for analysis
    if len(predictions_list) > 1:
        analysis_df = test_df.copy()
        for i, (preds, name) in enumerate(zip(predictions_list, model_names)):
            analysis_df[f'pred_{name}'] = [ID2LABEL[p] for p in preds]
        analysis_df['final_pred'] = [ID2LABEL[p] for p in final_predictions]
        analysis_path = os.path.join(args.output_dir, 'ensemble_analysis.csv')
        analysis_df.to_csv(analysis_path, index=False)
        print(f"Ensemble analysis saved to: {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for Tamil sentiment analysis")

    # Model arguments
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="Paths to trained models")
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "llm"],
                        help="Type of model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for LLM (required for LoRA models)")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions",
                        help="Output directory")
    parser.add_argument("--output_name", type=str, default="submission.csv",
                        help="Output filename")

    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")

    # Ensemble arguments
    parser.add_argument("--ensemble_method", type=str, default="soft",
                        choices=["voting", "weighted_voting", "soft"],
                        help="Ensemble method")

    args = parser.parse_args()
    main(args)
