"""
Tamil Political Sentiment Data Augmentation using Gemini 2.5 Flash
Version 3 - BATCH PROCESSING for faster execution

Run in GCP Vertex AI Workbench or Colab Enterprise
"""

import json
import re
import time
import asyncio
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# CONFIGURATION
# ============================================================
GOOGLE_CLOUD_PROJECT = "gdw-team-tpgds-nandrelmodeling"
GEMINI_LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"

INPUT_FILE = "Political_Sentiment_Analysis/data/PS_train.csv"
OUTPUT_FILE = "Political_Sentiment_Analysis/data/PS_train_gemini_augmented.csv"

# Batch processing settings
BATCH_SIZE = 5  # Tweets per API call (5-10 recommended)

# Parallel processing settings
MAX_CONCURRENT = 5  # Max concurrent API calls (avoid rate limits)
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

# Variations per class - targeting ~2,000 samples each
VARIATIONS_PER_CLASS = {
    "Opinionated": 1,        # 1,361 → ~2,722
    "Sarcastic": 2,          # 790 → ~2,370
    "Neutral": 2,            # 637 → ~1,911
    "Positive": 3,           # 575 → ~2,300
    "Substantiated": 4,      # 412 → ~2,060
    "Negative": 4,           # 406 → ~2,030
    "None of the above": 0,  # SKIP (too short/random)
}

MAX_PER_CLASS = None  # Set to e.g. 20 for testing, None for full run

# ============================================================
# SETUP
# ============================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

print("=" * 60)
print("Tamil Political Sentiment Augmentation v3 (BATCH)")
print("=" * 60)
print(f"Project: {GOOGLE_CLOUD_PROJECT}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE} tweets per API call")
print()

vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GEMINI_LOCATION)
model = GenerativeModel(MODEL_NAME)
print("✓ Vertex AI initialized\n")

# ============================================================
# BATCH AUGMENTATION FUNCTION
# ============================================================
def clean_json_response(text: str) -> str:
    """Clean up malformed JSON from LLM response."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Remove control characters (keep basic whitespace)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    return text.strip()


def parse_json_safely(text: str) -> dict:
    """Try multiple strategies to parse JSON."""
    text = clean_json_response(text)

    # Strategy 1: Direct parse
    try:
        if text.startswith('{'):
            return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object in text
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Strategy 3: Aggressive cleaning - collapse all whitespace
    try:
        cleaned = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    return {}


def augment_batch(tweets_batch: list, label: str, num_variations: int, max_retries: int = 2) -> dict:
    """
    Augment multiple tweets in a single API call with retry logic.

    Args:
        tweets_batch: List of (index, text, hashtags) tuples
        label: Sentiment label
        num_variations: Number of variations per tweet
        max_retries: Number of retry attempts on failure

    Returns:
        Dict mapping index to list of variations
    """

    # Build batch prompt
    batch_items = []
    for idx, text, hashtags in tweets_batch:
        hashtag_note = f" [MUST INCLUDE: {' '.join(hashtags)}]" if hashtags else ""
        batch_items.append(f"TWEET_{idx}: {text}{hashtag_note}")

    tweets_text = "\n\n".join(batch_items)

    prompt = f"""நீங்கள் தமிழ் அரசியல் உணர்வு பகுப்பாய்வுக்கான தரவு விரிவாக்க நிபுணர்.

TASK: Create {num_variations} Tamil paraphrases for EACH tweet below.

STRICT RULES:
1. SAME LENGTH: Each output must be similar length to its input (±30% words)
2. SAME SENTIMENT: Must preserve exact sentiment "{label}"
3. SAME MEANING: Keep the political opinion/message intact
4. PARAPHRASE ONLY: Rewrite using different words/structure, do NOT summarize
5. NATURAL TAMIL: Must sound natural to a native Tamil speaker
6. KEEP HASHTAGS: ⚠️ COPY-PASTE all #hashtags EXACTLY as shown in [MUST INCLUDE]
7. KEEP EMOJIS: Preserve any emojis from original

SENTIMENT "{label}" means:
- Substantiated: Contains evidence, statistics, facts
- Sarcastic: Uses sarcasm, mockery, irony
- Opinionated: Strong personal views or bias
- Positive: Shows approval, appreciation, support
- Negative: Expresses criticism, dissatisfaction
- Neutral: Factual political info without bias

INPUT TWEETS:
{tweets_text}

OUTPUT FORMAT: Return a JSON object where each key is the tweet ID and value is an array of {num_variations} variations.
Example:
{{
  "TWEET_0": ["variation 1", "variation 2"],
  "TWEET_1": ["variation 1", "variation 2"]
}}

Return ONLY the JSON object, no explanation."""

    # Retry loop
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.9,
                    max_output_tokens=4096,
                )
            )

            result_text = response.text.strip()

            # Use safe JSON parser with multiple strategies
            results = parse_json_safely(result_text)

            if not results:
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} - JSON parse failed")
                    time.sleep(2)
                    continue
                else:
                    print(f"  Warning: Could not parse after {max_retries + 1} attempts")
                    return {}

            # If we got results, break out of retry loop
            break

        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt + 1}/{max_retries} - API error: {e}")
                time.sleep(2)
                continue
            else:
                print(f"  Error after {max_retries + 1} attempts: {e}")
                return {}

    # Validate each tweet's variations
    validated_results = {}

    for idx, text, original_hashtags in tweets_batch:
        key = f"TWEET_{idx}"
        if key not in results:
            continue

        variations = results[key]
        if not isinstance(variations, list):
            continue

        original_len = len(text.split())
        valid_variations = []

        for v in variations:
            if not v or not v.strip() or v == text:
                continue

            # Length check (allow 50% to 200%)
            v_len = len(v.split())
            if v_len < original_len * 0.5 or v_len > original_len * 2:
                continue

            # Hashtag check
            if original_hashtags:
                v_hashtags = set(re.findall(r'#\w+', v))
                if not set(original_hashtags).issubset(v_hashtags):
                    continue

            valid_variations.append(v)

        if valid_variations:
            validated_results[idx] = valid_variations

    return validated_results


# ============================================================
# ASYNC PARALLEL PROCESSING
# ============================================================
async def process_batch_async(batch_data: tuple, label: str, num_variations: int, semaphore: asyncio.Semaphore) -> tuple:
    """
    Process a single batch asynchronously with concurrency limiting.

    Args:
        batch_data: Tuple of (batch_index, batch_tweets, original_rows)
        label: Sentiment label
        num_variations: Number of variations per tweet
        semaphore: Asyncio semaphore to limit concurrency

    Returns:
        Tuple of (batch_index, results_dict, original_rows)
    """
    batch_idx, batch_tweets, original_rows = batch_data

    async with semaphore:  # Limit concurrent API calls
        loop = asyncio.get_event_loop()
        # Offload blocking Gemini API call to thread pool
        results = await loop.run_in_executor(
            executor,
            augment_batch,
            batch_tweets,
            label,
            num_variations
        )
        return (batch_idx, results, original_rows)


async def process_all_batches_parallel(batches: list, label: str, num_variations: int) -> list:
    """
    Process all batches in parallel with concurrency limiting.

    Args:
        batches: List of (batch_index, batch_tweets, original_rows) tuples
        label: Sentiment label
        num_variations: Number of variations per tweet

    Returns:
        List of (batch_index, results_dict, original_rows) tuples
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        process_batch_async(batch_data, label, num_variations, semaphore)
        for batch_data in batches
    ]

    # Run all batches concurrently (limited by semaphore)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  Batch failed with error: {r}")
        else:
            valid_results.append(r)

    return valid_results


# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} samples\n")

print("Class distribution:")
print(df['labels'].value_counts())
print()

# Calculate expected API calls
total_tweets = sum(
    min(len(df[df['labels'] == label]), MAX_PER_CLASS or float('inf'))
    for label, n in VARIATIONS_PER_CLASS.items() if n > 0
)
total_batches = (total_tweets + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Augmentation plan:")
print(f"  Total tweets to process: {total_tweets}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total API calls: ~{total_batches}")
print(f"  Estimated time: ~{total_batches * 8 / 60:.0f} minutes")
print()

# ============================================================
# TEST ON ONE BATCH
# ============================================================
print("=" * 60)
print("STEP 1: Testing on one batch")
print("=" * 60)

test_label = "Neutral"
test_df = df[df['labels'] == test_label].head(BATCH_SIZE)
test_batch = []

for i, (idx, row) in enumerate(test_df.iterrows()):
    text = row['content']
    hashtags = re.findall(r'#\w+', text)
    test_batch.append((i, text, hashtags))

print(f"\nTesting batch of {len(test_batch)} '{test_label}' tweets...")
test_results = augment_batch(test_batch, test_label, VARIATIONS_PER_CLASS[test_label])

print(f"\nResults:")
for i, text, hashtags in test_batch:
    print(f"\n[Original {i}]: {text[:80]}...")
    if i in test_results:
        for j, var in enumerate(test_results[i], 1):
            print(f"  [Var {j}]: {var[:80]}...")
    else:
        print(f"  (no valid variations)")

print("\n" + "=" * 60)
print("Review the test results above.")
print("=" * 60)

# Uncomment to require confirmation:
# input("\nPress Enter to continue, or Ctrl+C to stop...")

# ============================================================
# FULL AUGMENTATION WITH PARALLEL BATCHING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Running full augmentation (PARALLEL BATCH MODE)")
print(f"  Max concurrent API calls: {MAX_CONCURRENT}")
print("=" * 60)

new_rows = []
total_processed = 0
total_variations = 0
total_errors = 0

# Checkpoint file for incremental saves
CHECKPOINT_FILE = "Political_Sentiment_Analysis/data/PS_train_augmented_checkpoint.csv"


async def process_class(label: str, num_variations: int, class_df: pd.DataFrame):
    """Process all tweets in a class using parallel batching."""
    global total_processed, total_variations

    # Prepare all tweets with metadata
    all_tweets = []
    for idx, row in class_df.iterrows():
        text = row['content']
        if len(text.split()) < 3:  # Skip very short
            continue
        hashtags = re.findall(r'#\w+', text)
        all_tweets.append((idx, text, hashtags, row))

    num_batches = (len(all_tweets) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nAugmenting '{label}': {len(all_tweets)} tweets in {num_batches} batches (parallel)")

    # Prepare batches for parallel processing
    batches_to_process = []
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_tweets))

        batch_tweets = [(i, t[1], t[2]) for i, t in enumerate(all_tweets[start_idx:end_idx])]
        original_rows = all_tweets[start_idx:end_idx]

        batches_to_process.append((batch_num, batch_tweets, original_rows))

    # Process all batches in parallel
    print(f"  Launching {len(batches_to_process)} batches with {MAX_CONCURRENT} concurrent...")
    start_time = time.time()

    results = await process_all_batches_parallel(batches_to_process, label, num_variations)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({elapsed/len(batches_to_process):.2f}s per batch effective)")

    # Collect results
    class_new_rows = []
    for batch_idx, batch_results, original_rows in results:
        for local_idx, variations in batch_results.items():
            original_row = original_rows[local_idx][3]
            for var_text in variations:
                new_row = original_row.copy()
                new_row['content'] = var_text
                class_new_rows.append(new_row)

    total_processed += len(all_tweets)
    total_variations += len(class_new_rows)

    return class_new_rows


async def main():
    """Main async function to process all classes."""
    global new_rows

    for label, num_variations in VARIATIONS_PER_CLASS.items():
        if num_variations == 0:
            print(f"\nSkipping {label}")
            continue

        class_df = df[df['labels'] == label].copy()

        if MAX_PER_CLASS:
            class_df = class_df.head(MAX_PER_CLASS)

        # Process this class
        class_new_rows = await process_class(label, num_variations, class_df)
        new_rows.extend(class_new_rows)

        # ========== CHECKPOINT: Save after each class ==========
        print(f"  Saving checkpoint after '{label}'...")
        checkpoint_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
        print(f"  ✓ Checkpoint saved: {len(new_rows)} new rows so far")


# Run the async main function
asyncio.run(main())

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Saving results")
print("=" * 60)

augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

print(f"\nOriginal samples: {len(df)}")
print(f"Tweets processed: {total_processed}")
print(f"New variations created: {total_variations}")
print(f"Total samples: {len(augmented_df)}")

print(f"\nFinal class distribution:")
print(augmented_df['labels'].value_counts())

augmented_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
