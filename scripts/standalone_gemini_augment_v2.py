"""
Tamil Political Sentiment Data Augmentation using Gemini 2.5 Flash
Version 2 - Improved prompt for better quality paraphrases

Run in GCP Vertex AI Workbench or Colab Enterprise
"""

import json
import re
import time
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
GOOGLE_CLOUD_PROJECT = "gdw-team-tpgds-nandrelmodeling"
GEMINI_LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"  # Gemini 2.5 Flash

INPUT_FILE = "PS_train.csv"
OUTPUT_FILE = "PS_train_gemini_augmented.csv"

# Variations per class - targeting ~2,000 samples each
VARIATIONS_PER_CLASS = {
    "Opinionated": 1,        # 1,361 → ~2,722
    "Sarcastic": 2,          # 790 → ~2,370
    "Neutral": 2,            # 637 → ~1,911
    "Positive": 3,           # 575 → ~2,300
    "Substantiated": 4,      # 412 → ~2,060
    "Negative": 4,           # 406 → ~2,030
    "None of the above": 0,  # Skip (too short/random to paraphrase)
}

MAX_PER_CLASS = None  # Set to e.g. 10 for testing, None for full run

# ============================================================
# SETUP
# ============================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

print("=" * 60)
print("Tamil Political Sentiment Augmentation v2")
print("=" * 60)
print(f"Project: {GOOGLE_CLOUD_PROJECT}")
print(f"Location: {GEMINI_LOCATION}")
print(f"Model: {MODEL_NAME}")
print()

vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GEMINI_LOCATION)
model = GenerativeModel(MODEL_NAME)
print("✓ Vertex AI initialized\n")

# ============================================================
# IMPROVED AUGMENTATION FUNCTION
# ============================================================
def augment_tamil_tweet(text: str, label: str, num_variations: int) -> list:
    """
    Use Gemini to create meaning-preserving Tamil paraphrases.
    Improved prompt to avoid content loss and meaning changes.
    """

    # Extract hashtags from original to enforce preservation
    original_hashtags = re.findall(r'#\w+', text)
    hashtag_instruction = ""
    if original_hashtags:
        hashtag_instruction = f"\n\n⚠️ CRITICAL: The following hashtags MUST appear exactly in each variation: {' '.join(original_hashtags)}"

    prompt = f"""நீங்கள் தமிழ் அரசியல் உணர்வு பகுப்பாய்வுக்கான தரவு விரிவாக்க நிபுணர்.

TASK: Create {num_variations} Tamil paraphrases of the given tweet.

STRICT RULES:
1. SAME LENGTH: Output must be similar length to input (±20% words)
2. SAME SENTIMENT: Must preserve exact sentiment "{label}"
3. SAME MEANING: Keep the political opinion/message intact
4. PARAPHRASE ONLY: Rewrite using different words/structure, do NOT summarize
5. NATURAL TAMIL: Must sound natural to a native Tamil speaker
6. KEEP HASHTAGS: ⚠️ COPY-PASTE all #hashtags EXACTLY as they appear - do not translate or modify them!
7. KEEP EMOJIS: Preserve any emojis from original
8. KEEP MENTIONS: Preserve any @mentions exactly{hashtag_instruction}

SENTIMENT DEFINITION for "{label}":
- Substantiated: Contains evidence, statistics, facts, or references
- Sarcastic: Uses sarcasm, mockery, irony, or satire
- Opinionated: Strong personal views, bias, or subjective beliefs
- Positive: Shows approval, appreciation, support, or praise
- Negative: Expresses criticism, dissatisfaction, complaint, or opposition
- Neutral: Factual political information without emotional bias

BAD EXAMPLES (DO NOT DO):
❌ Making output much shorter than input
❌ Changing the political stance or opinion
❌ Translating to English
❌ Adding new information not in original
❌ Removing hashtags or emojis

GOOD EXAMPLE:
Input: "முதல்வர் ஸ்டாலின் சிறப்பான பணி செய்கிறார் #DMK"
Output: ["முதலமைச்சர் ஸ்டாலின் அருமையான வேலை செய்து வருகிறார் #DMK", "ஸ்டாலின் அவர்கள் சிறந்த முறையில் பணியாற்றுகிறார் #DMK"]

INPUT TWEET:
Sentiment: {label}
Text: {text}

OUTPUT: Return ONLY a JSON array of {num_variations} Tamil strings. No explanation.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.9,
                max_output_tokens=2048,
            )
        )

        result_text = response.text.strip()

        # Parse JSON array
        # Remove markdown code blocks if present
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)

        if result_text.startswith('['):
            variations = json.loads(result_text)
        else:
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                variations = json.loads(match.group())
            else:
                print(f"  Warning: Could not parse response")
                return []

        # Filter: not empty, not same as original, reasonable length, hashtags preserved
        original_len = len(text.split())
        valid_variations = []

        for v in variations:
            if not v or not v.strip():
                continue
            if v == text:
                continue

            # Check length (allow 50% to 200% of original)
            v_len = len(v.split())
            if v_len < original_len * 0.5 or v_len > original_len * 2:
                print(f"  Warning: Skipped variation - bad length ({v_len} vs {original_len})")
                continue

            # Check hashtags are preserved
            if original_hashtags:
                variation_hashtags = set(re.findall(r'#\w+', v))
                missing_hashtags = set(original_hashtags) - variation_hashtags
                if missing_hashtags:
                    print(f"  Warning: Skipped variation - missing hashtags: {missing_hashtags}")
                    continue

            valid_variations.append(v)

        return valid_variations

    except Exception as e:
        print(f"  Error: {e}")
        return []

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} samples\n")

print("Class distribution:")
print(df['labels'].value_counts())
print()

print("Augmentation plan:")
for label, n_var in VARIATIONS_PER_CLASS.items():
    count = len(df[df['labels'] == label])
    if n_var > 0:
        new_samples = count * n_var
        print(f"  {label}: {count} × {n_var} variations = +{new_samples} new samples")
    else:
        print(f"  {label}: {count} (skipping)")
print()

# ============================================================
# TEST ON 3 SAMPLES
# ============================================================
print("=" * 60)
print("STEP 1: Testing on sample tweets")
print("=" * 60)

test_labels = ["Neutral", "Negative", "Substantiated"]
for label in test_labels:
    if VARIATIONS_PER_CLASS.get(label, 0) == 0:
        continue

    sample = df[df['labels'] == label].iloc[0]['content']
    n_var = VARIATIONS_PER_CLASS[label]

    print(f"\n[{label}] - {n_var} variations")
    print(f"Original ({len(sample.split())} words): {sample[:100]}...")

    variations = augment_tamil_tweet(sample, label, n_var)

    for i, var in enumerate(variations, 1):
        print(f"Var {i} ({len(var.split())} words): {var[:100]}...")

    time.sleep(1)

print("\n" + "=" * 60)
print("Review the test results above.")
print("=" * 60)

# Uncomment to require confirmation:
# input("\nPress Enter to continue with full augmentation, or Ctrl+C to stop...")

# ============================================================
# FULL AUGMENTATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Running full augmentation")
print("=" * 60)

new_rows = []
errors = 0
skipped = 0

for label, num_variations in VARIATIONS_PER_CLASS.items():
    if num_variations == 0:
        print(f"\nSkipping {label} (configured to skip)")
        continue

    class_df = df[df['labels'] == label]

    if MAX_PER_CLASS:
        class_df = class_df.head(MAX_PER_CLASS)

    print(f"\nAugmenting '{label}' ({len(class_df)} samples × {num_variations} variations)...")

    for idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=label):
        text = row['content']

        # Skip very short tweets
        if len(text.split()) < 3:
            skipped += 1
            continue

        variations = augment_tamil_tweet(text, label, num_variations)

        if not variations:
            errors += 1

        for var_text in variations:
            new_row = row.copy()
            new_row['content'] = var_text
            new_rows.append(new_row)

        # Rate limiting
        time.sleep(0.3)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Saving results")
print("=" * 60)

augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

print(f"\nOriginal samples: {len(df)}")
print(f"New augmented samples: {len(new_rows)}")
print(f"Errors: {errors}")
print(f"Skipped (too short): {skipped}")
print(f"Total samples: {len(augmented_df)}")

print(f"\nFinal class distribution:")
print(augmented_df['labels'].value_counts())

augmented_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("DONE! Download the augmented file and use for training.")
print("=" * 60)
