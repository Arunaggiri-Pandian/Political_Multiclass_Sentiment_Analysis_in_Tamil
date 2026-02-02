"""
LLM-based Data Augmentation using Google Gemini via Vertex AI
Creates meaning-preserving Tamil paraphrases for sentiment analysis
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import time
import re

# Vertex AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


def init_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialize Vertex AI with project settings."""
    vertexai.init(project=project_id, location=location)
    print(f"Initialized Vertex AI: project={project_id}, location={location}")


def augment_with_gemini(
    text: str,
    label: str,
    model: GenerativeModel,
    num_variations: int = 2
) -> List[str]:
    """
    Use Gemini to create meaning-preserving variations of Tamil text.
    """

    prompt = f"""TASK: Tamil Political Sentiment Data Augmentation

I have a Tamil political tweet. Create {num_variations} paraphrased variations.

RULES:
1. Preserve the EXACT same sentiment: {label}
2. Keep the same political meaning/opinion
3. Use different Tamil words or sentence structure
4. Must be grammatically correct and natural Tamil
5. Keep any hashtags (#) or mentions (@)

SENTIMENT MEANINGS:
- Substantiated: Contains evidence, facts, references
- Sarcastic: Uses sarcasm, mockery, irony
- Opinionated: Strong personal views or bias
- Positive: Shows approval or appreciation
- Negative: Expresses criticism or dissatisfaction
- Neutral: Factual political info without bias
- None of the above: Not political content

OUTPUT: Return ONLY a JSON array of {num_variations} Tamil strings. No explanation.

INPUT TWEET ({label}):
{text}

OUTPUT:"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.8,
                max_output_tokens=1024,
            )
        )

        result_text = response.text.strip()

        # Parse JSON array from response
        if result_text.startswith('['):
            variations = json.loads(result_text)
            return [v for v in variations if v and v != text]
        else:
            # Try to find JSON array in response
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                variations = json.loads(match.group())
                return [v for v in variations if v and v != text]

        return []

    except Exception as e:
        print(f"Error: {e}")
        return []


def augment_dataset(
    project_id: str,
    input_path: str = "data/PS_train.csv",
    output_path: str = "data/PS_train_gemini_augmented.csv",
    minority_threshold: float = 0.15,
    num_variations: int = 2,
    max_samples_per_class: Optional[int] = None,
    model_name: str = "gemini-1.5-flash",
    location: str = "us-central1"
):
    """
    Augment minority classes using Gemini via Vertex AI.
    """

    # Initialize Vertex AI
    init_vertex_ai(project_id, location)

    # Load model
    print(f"Loading model: {model_name}")
    model = GenerativeModel(model_name)

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
    errors = 0

    for label in minority_classes:
        class_df = df[df['labels'] == label]

        if max_samples_per_class:
            class_df = class_df.head(max_samples_per_class)

        print(f"\nAugmenting {label} ({len(class_df)} samples)...")

        for idx, row in tqdm(class_df.iterrows(), total=len(class_df)):
            text = row['content']

            # Get variations from Gemini
            variations = augment_with_gemini(text, label, model, num_variations)

            if not variations:
                errors += 1

            # Add new rows
            for var_text in variations:
                new_row = row.copy()
                new_row['content'] = var_text
                new_rows.append(new_row)

            # Small delay to avoid rate limiting
            time.sleep(0.2)

    # Combine
    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"\n{'='*50}")
    print(f"Original samples: {len(df)}")
    print(f"New samples: {len(new_rows)}")
    print(f"Errors: {errors}")
    print(f"Total samples: {len(augmented_df)}")

    print(f"\nNew class distribution:")
    print(augmented_df['labels'].value_counts())

    # Save
    augmented_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return augmented_df


def test_augmentation(
    project_id: str,
    model_name: str = "gemini-1.5-flash",
    location: str = "us-central1",
    num_samples: int = 3
):
    """Test augmentation on a few samples and save to HTML for Tamil viewing."""

    # Initialize
    init_vertex_ai(project_id, location)
    model = GenerativeModel(model_name)

    # Load samples
    df = pd.read_csv("data/PS_train.csv")

    test_samples = [
        ("Neutral", df[df['labels'] == 'Neutral'].iloc[0]['content']),
        ("Negative", df[df['labels'] == 'Negative'].iloc[0]['content']),
        ("None of the above", df[df['labels'] == 'None of the above'].iloc[0]['content']),
    ]

    print("="*80)
    print("GEMINI AUGMENTATION TEST")
    print("="*80)

    # Build HTML for Tamil display
    html = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Gemini Augmentation Test</title>
<style>
body { font-family: 'Noto Sans Tamil', sans-serif; padding: 20px; font-size: 18px; }
.label { color: #0066cc; font-weight: bold; }
.original { background: #e8f5e9; padding: 15px; margin: 10px 0; border-left: 4px solid #4caf50; }
.variation { background: #fff3e0; padding: 10px; margin: 5px 0 5px 20px; border-left: 4px solid #ff9800; }
</style></head><body>
<h1>Gemini Tamil Augmentation Test</h1>
'''

    for label, text in test_samples[:num_samples]:
        print(f"\n### {label} ###")
        print(f"Original: {text[:100]}...")

        html += f'<h2 class="label">{label}</h2>'
        html += f'<div class="original"><b>Original:</b><br>{text}</div>'

        variations = augment_with_gemini(text, label, model, num_variations=2)

        for i, var in enumerate(variations, 1):
            print(f"Variation {i}: {var[:100]}...")
            html += f'<div class="variation"><b>Variation {i}:</b><br>{var}</div>'

        print("-"*40)

    html += '</body></html>'

    # Save HTML
    output_path = "outputs/gemini_augmentation_test.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nSaved HTML to: {output_path}")
    print("Open this file in a browser to see Tamil text properly.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tamil augmentation using Gemini via Vertex AI")
    parser.add_argument("--project", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--test", action="store_true", help="Test on a few samples")
    parser.add_argument("--full", action="store_true", help="Run full augmentation")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max samples per class")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash",
                        choices=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                        help="Gemini model (flash=cheap, pro=better)")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--variations", type=int, default=2, help="Variations per sample")

    args = parser.parse_args()

    if args.test:
        test_augmentation(
            project_id=args.project,
            model_name=args.model,
            location=args.location
        )
    elif args.full:
        augment_dataset(
            project_id=args.project,
            model_name=args.model,
            location=args.location,
            max_samples_per_class=args.max_per_class,
            num_variations=args.variations
        )
    else:
        print("Usage:")
        print("  Test:  python src/augment_with_gemini.py --project YOUR_PROJECT_ID --test")
        print("  Full:  python src/augment_with_gemini.py --project YOUR_PROJECT_ID --full")
        print("")
        print("Models: gemini-1.5-flash (cheap), gemini-1.5-pro (better), gemini-2.0-flash (newest)")
        print("")
        print("Make sure you're authenticated: gcloud auth application-default login")
