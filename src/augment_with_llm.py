"""
LLM-based Data Augmentation for Tamil Political Sentiment Analysis
Uses Claude via AWS Bedrock to create meaning-preserving paraphrases
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import time
import boto3
from botocore.config import Config


# AWS Bedrock model IDs (Haiku is cheapest)
BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",  # Cheapest, fast
    "sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",   # Mid-range
}


def get_bedrock_client(region: str = "us-east-1"):
    """Get AWS Bedrock client using SSO credentials."""
    config = Config(
        region_name=region,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
    return boto3.client('bedrock-runtime', config=config)


def augment_with_claude(
    text: str,
    label: str,
    client,
    model_id: str,
    num_variations: int = 2
) -> List[str]:
    """
    Use Claude via Bedrock to create meaning-preserving variations of Tamil text.
    """

    prompt = f"""You are helping with Tamil political sentiment analysis data augmentation.

Given this Tamil tweet with sentiment label "{label}", create {num_variations} meaningful variations.

Original Tamil tweet:
{text}

Instructions:
1. First, understand the meaning of the Tamil tweet
2. Create {num_variations} different Tamil paraphrases that:
   - Preserve the EXACT same sentiment ({label})
   - Keep the same political meaning/opinion
   - Use different words or sentence structure
   - Are grammatically correct Tamil
   - Keep any hashtags or mentions

Return ONLY a JSON array with the {num_variations} Tamil variations, nothing else.
Example format: ["variation 1 in Tamil", "variation 2 in Tamil"]

Important: The variations must sound natural to a Tamil speaker and preserve the original sentiment."""

    try:
        # Bedrock request format
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        })

        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        result_text = response_body['content'][0]['text'].strip()

        # Try to parse as JSON
        if result_text.startswith('['):
            variations = json.loads(result_text)
            return [v for v in variations if v and v != text]
        else:
            # Try to find JSON array in response
            import re
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                variations = json.loads(match.group())
                return [v for v in variations if v and v != text]

        return []

    except Exception as e:
        print(f"Error: {e}")
        return []


def augment_dataset_with_llm(
    input_path: str = "data/PS_train.csv",
    output_path: str = "data/PS_train_llm_augmented.csv",
    minority_threshold: float = 0.15,
    num_variations: int = 2,
    max_samples_per_class: Optional[int] = None,
    model: str = "haiku",
    region: str = "us-east-1"
):
    """
    Augment minority classes using Claude LLM via AWS Bedrock.
    """

    # Initialize Bedrock client (uses AWS SSO credentials)
    print(f"Connecting to AWS Bedrock ({region})...")
    client = get_bedrock_client(region)
    model_id = BEDROCK_MODELS.get(model, BEDROCK_MODELS["haiku"])
    print(f"Using model: {model} ({model_id})")

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples")

    # Find minority classes
    class_counts = df['labels'].value_counts(normalize=True)
    minority_classes = class_counts[class_counts < minority_threshold].index.tolist()

    print(f"\nMinority classes to augment: {minority_classes}")
    print(f"Variations per sample: {num_variations}")

    # Augment
    new_rows = []

    for label in minority_classes:
        class_df = df[df['labels'] == label]

        if max_samples_per_class:
            class_df = class_df.head(max_samples_per_class)

        print(f"\nAugmenting {label} ({len(class_df)} samples)...")

        for idx, row in tqdm(class_df.iterrows(), total=len(class_df)):
            text = row['content']

            # Get variations from Claude via Bedrock
            variations = augment_with_claude(text, label, client, model_id, num_variations)

            # Add new rows
            for var_text in variations:
                new_row = row.copy()
                new_row['content'] = var_text
                new_rows.append(new_row)

            # Small delay to avoid throttling
            time.sleep(0.1)

    # Combine
    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"\n{'='*50}")
    print(f"Original samples: {len(df)}")
    print(f"New samples: {len(new_rows)}")
    print(f"Total samples: {len(augmented_df)}")

    print(f"\nNew class distribution:")
    print(augmented_df['labels'].value_counts())

    # Save
    augmented_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return augmented_df


def test_augmentation(num_samples: int = 3, model: str = "haiku", region: str = "us-east-1"):
    """Test augmentation on a few samples using AWS Bedrock."""

    print(f"Connecting to AWS Bedrock ({region})...")
    client = get_bedrock_client(region)
    model_id = BEDROCK_MODELS.get(model, BEDROCK_MODELS["haiku"])
    print(f"Using model: {model} ({model_id})")

    # Load a few samples
    df = pd.read_csv("data/PS_train.csv")

    # Test on different classes
    test_samples = [
        ("Neutral", df[df['labels'] == 'Neutral'].iloc[0]['content']),
        ("Negative", df[df['labels'] == 'Negative'].iloc[0]['content']),
        ("None of the above", df[df['labels'] == 'None of the above'].iloc[0]['content']),
    ]

    print("="*80)
    print("LLM AUGMENTATION TEST (AWS Bedrock)")
    print("="*80)

    for label, text in test_samples[:num_samples]:
        print(f"\n### {label} ###")
        print(f"Original: {text[:300]}")

        variations = augment_with_claude(text, label, client, model_id, num_variations=2)

        for i, var in enumerate(variations, 1):
            print(f"Variation {i}: {var[:300]}")

        print("-"*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based Tamil text augmentation via AWS Bedrock")
    parser.add_argument("--test", action="store_true", help="Test on a few samples")
    parser.add_argument("--full", action="store_true", help="Run full augmentation")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max samples per class")
    parser.add_argument("--model", type=str, default="haiku", choices=["haiku", "sonnet"],
                        help="Claude model to use (haiku=cheapest, sonnet=better)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
    parser.add_argument("--variations", type=int, default=2, help="Number of variations per sample")

    args = parser.parse_args()

    if args.test:
        test_augmentation(model=args.model, region=args.region)
    elif args.full:
        augment_dataset_with_llm(
            max_samples_per_class=args.max_per_class,
            model=args.model,
            region=args.region,
            num_variations=args.variations
        )
    else:
        print("Usage:")
        print("  Test mode:  python src/augment_with_llm.py --test")
        print("  Full run:   python src/augment_with_llm.py --full")
        print("  Limited:    python src/augment_with_llm.py --full --max_per_class 50")
        print("")
        print("Options:")
        print("  --model haiku|sonnet  Choose model (haiku=cheap, sonnet=better)")
        print("  --region us-east-1    AWS region for Bedrock")
        print("  --variations 2        Number of variations per sample")
        print("")
        print("Make sure you're logged in via AWS SSO: aws sso login")
