"""
Standalone Tamil Political Sentiment Data Augmentation using Gemini
Run this in GCP Vertex AI Workbench or Colab Enterprise

Usage:
1. Upload PS_train.csv to the same directory
2. Run this script
3. Download PS_train_augmented.csv when done
"""

import json
import re
import time
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION - Edit these values
# ============================================================
GOOGLE_CLOUD_PROJECT = "gdw-team-tpgds-nandrelmodeling"
GEMINI_LOCATION = "global"
MODEL_NAME = "gemini-1.5-flash"  # Options: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash

INPUT_FILE = "PS_train.csv"
OUTPUT_FILE = "PS_train_augmented.csv"

NUM_VARIATIONS = 2          # Paraphrases per tweet
MINORITY_THRESHOLD = 0.15   # Augment classes below this %
MAX_PER_CLASS = None        # Set to e.g. 50 for testing, None for all

# ============================================================
# SETUP - Initialize Vertex AI
# ============================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

print(f"Initializing Vertex AI...")
print(f"  Project: {GOOGLE_CLOUD_PROJECT}")
print(f"  Location: {GEMINI_LOCATION}")
print(f"  Model: {MODEL_NAME}")

vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GEMINI_LOCATION)
model = GenerativeModel(MODEL_NAME)

print("✓ Vertex AI initialized successfully!\n")

# ============================================================
# AUGMENTATION FUNCTION
# ============================================================
def augment_tamil_tweet(text: str, label: str, num_variations: int = 2) -> list:
    """
    Use Gemini to create meaning-preserving Tamil paraphrases.
    """

    prompt = f"""TASK: Tamil Political Sentiment Data Augmentation

Create {num_variations} paraphrased variations of this Tamil political tweet.

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

OUTPUT FORMAT: Return ONLY a JSON array of {num_variations} Tamil strings.
Example: ["முதல் மாற்று வாக்கியம்", "இரண்டாவது மாற்று வாக்கியம்"]

INPUT TWEET (Sentiment: {label}):
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

        # Parse JSON array
        if result_text.startswith('['):
            variations = json.loads(result_text)
        else:
            # Find JSON array in response
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                variations = json.loads(match.group())
            else:
                return []

        # Filter out empty or identical results
        return [v for v in variations if v and v.strip() and v != text]

    except Exception as e:
        print(f"  Error: {e}")
        return []

# ============================================================
# TEST ON 3 SAMPLES FIRST
# ============================================================
print("=" * 60)
print("STEP 1: Testing on 3 samples")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} samples from {INPUT_FILE}\n")

# Test samples from different classes
test_cases = [
    ("Neutral", df[df['labels'] == 'Neutral'].iloc[0]['content']),
    ("Negative", df[df['labels'] == 'Negative'].iloc[0]['content']),
    ("None of the above", df[df['labels'] == 'None of the above'].iloc[0]['content']),
]

print("Testing augmentation quality:\n")
for label, text in test_cases:
    print(f"[{label}]")
    print(f"  Original: {text[:80]}...")

    variations = augment_tamil_tweet(text, label, num_variations=2)

    for i, var in enumerate(variations, 1):
        print(f"  Variation {i}: {var[:80]}...")

    print()
    time.sleep(1)  # Rate limiting

# ============================================================
# ASK FOR CONFIRMATION
# ============================================================
print("=" * 60)
print("STEP 2: Review the test results above")
print("=" * 60)
print("\nDo the variations look good? Do they preserve sentiment?")
print("If yes, the script will continue with full augmentation.\n")

# Uncomment the line below to require manual confirmation:
# input("Press Enter to continue with full augmentation, or Ctrl+C to stop...")

# ============================================================
# FULL AUGMENTATION
# ============================================================
print("=" * 60)
print("STEP 3: Running full augmentation")
print("=" * 60)

# Find minority classes
class_dist = df['labels'].value_counts(normalize=True)
print("\nClass distribution:")
print(class_dist)
print()

minority_classes = class_dist[class_dist < MINORITY_THRESHOLD].index.tolist()
print(f"Minority classes to augment (< {MINORITY_THRESHOLD*100}%): {minority_classes}")
print(f"Variations per sample: {NUM_VARIATIONS}\n")

# Count total samples to process
total_to_process = sum(
    min(len(df[df['labels'] == label]), MAX_PER_CLASS or float('inf'))
    for label in minority_classes
)
print(f"Total samples to augment: {total_to_process}")
print(f"Expected new samples: ~{total_to_process * NUM_VARIATIONS}\n")

# Augment
new_rows = []
errors = 0

for label in minority_classes:
    class_df = df[df['labels'] == label]

    if MAX_PER_CLASS:
        class_df = class_df.head(MAX_PER_CLASS)

    print(f"\nAugmenting '{label}' ({len(class_df)} samples)...")

    for idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=label):
        text = row['content']

        variations = augment_tamil_tweet(text, label, NUM_VARIATIONS)

        if not variations:
            errors += 1

        for var_text in variations:
            new_row = row.copy()
            new_row['content'] = var_text
            new_rows.append(new_row)

        # Rate limiting
        time.sleep(0.2)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Saving results")
print("=" * 60)

# Combine original + augmented
augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

print(f"\nOriginal samples: {len(df)}")
print(f"New augmented samples: {len(new_rows)}")
print(f"Errors: {errors}")
print(f"Total samples: {len(augmented_df)}")

print(f"\nNew class distribution:")
print(augmented_df['labels'].value_counts())

# Save
augmented_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nDownload '{OUTPUT_FILE}' and use it for training.")
print("Command: ./run.sh train-transformer indicbert_augmented")
