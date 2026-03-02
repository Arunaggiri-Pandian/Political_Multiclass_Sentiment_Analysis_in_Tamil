import os
import re
import json
import time
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv(override=True)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

if not PROJECT_ID or not LOCATION:
    raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configuration
INPUT_FILE = "data/PS_train.csv"
OUTPUT_FILE = "data/PS_train_augmented.json"
TEMPLATE_FILE = "prompt_template.txt"
BATCH_SIZE = 10

# Balancing Strategy
VARIATION_MAP = {
    "Negative": 5,
    "Substantiated": 5,
    "Positive": 3,
    "Neutral": 3,
    "None of the above": 3,
    "Sarcastic": 2,
    "Opinionated": 1
}

MODEL_NAME = "gemini-2.5-flash" 
MAX_WORKERS = 10  # Number of parallel threads

def load_template(filepath):
    with open(filepath, "r") as f:
        return f.read()

def extract_hashtags(text):
    return re.findall(r"#\S+", text)

def process_single_batch(model, batch_df, label, template):
    """Processes a single batch of tweets for a specific label."""
    input_tweets_section = ""
    batch_ids = []
    
    # Determine variations for this label
    num_vars = VARIATION_MAP.get(label, 1)
    
    for idx, row in batch_df.iterrows():
        tweet_id = f"TWEET_{idx}"
        text = row['content']
        text = text.replace("\n", " ") 
        hashtags = extract_hashtags(text)
        hashtag_str = " ".join(hashtags)
        
        input_tweets_section += f"{tweet_id}: {text} [MUST INCLUDE: {hashtag_str}]\n"
        batch_ids.append(tweet_id)

    # Prepare prompt
    prompt = template.replace("{label}", label)
    prompt = prompt.replace("{num_variations}", str(num_vars))
    
    prompt_parts = prompt.split("INPUT TWEETS:")
    header = prompt_parts[0] + "INPUT TWEETS:\n"
    footer = "\nOUTPUT FORMAT:" + prompt_parts[1].split("OUTPUT FORMAT:")[1]
    
    full_prompt = header + input_tweets_section + footer
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Generate
            response = model.generate_content(full_prompt)
            
            # Parse
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            data = json.loads(response_text)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                print(f"Error processing batch {batch_ids[0]}... after {max_retries} attempts: {e}")
                return {}
    return {}

def main():
    model = GenerativeModel(MODEL_NAME)
    template = load_template(TEMPLATE_FILE)
    
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Starting full processing on {len(df)} tweets with model {MODEL_NAME}...")
    
    unique_labels = df['labels'].unique()
    print(f"Found labels: {unique_labels}")
    
    # Prepare all tasks
    tasks = []
    for label in unique_labels:
        label_df = df[df['labels'] == label]
        for i in range(0, len(label_df), BATCH_SIZE):
            batch = label_df.iloc[i:i+BATCH_SIZE]
            tasks.append((batch, label))
            
    print(f"Total batches to process: {len(tasks)}")
    
    # Load existing results if file exists to support resuming
    all_results = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Resuming... Loaded {len(all_results)} existing tweet variations.")
        except json.JSONDecodeError:
            print("Warning: Output file exists but is corrupted. Starting fresh.")

    # Filter out tasks that are already completed
    # We check if any tweet_id from the batch is already in all_results
    # Actually, a batch produces multiple IDs. If the first ID of a batch is present, we skip.
    remaining_tasks = []
    for batch, label in tasks:
        first_id = f"TWEET_{batch.index[0]}"
        if first_id not in all_results:
            remaining_tasks.append((batch, label))
            
    print(f"Remaining batches to process: {len(remaining_tasks)}")
    tasks = remaining_tasks

    completed_count = 0
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_single_batch, model, batch, label, template): (batch, label) 
            for batch, label in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            result = future.result()
            if result:
                all_results.update(result)
            
            completed_count += 1
            
            # SAVE INCREMENTALLY every 10 batches
            if completed_count % 10 == 0:
                print(f"Progress: {completed_count}/{len(tasks)} batches completed. Saving checkpoint...")
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    duration = end_time - start_time
        
    print(f"\nProcessing complete in {duration:.2f} seconds.")
    if len(df) > 0:
        print(f"Average time per tweet: {duration/len(df):.2f} seconds.")

    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_results)} tweet variations to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
