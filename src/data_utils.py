"""
Data utilities for Political Multiclass Sentiment Analysis
Tamil Twitter Comments - DravidianLangTech @ ACL 2026
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import torch


# Label mapping
LABEL2ID = {
    "Substantiated": 0,
    "Sarcastic": 1,
    "Opinionated": 2,
    "Positive": 3,
    "Negative": 4,
    "Neutral": 5,
    "None of the above": 6
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Tamil label names for reference
TAMIL_LABELS = {
    "Substantiated": "ஆதாரப்பூர்வமானது",
    "Sarcastic": "கிண்டல்",
    "Opinionated": "தனிப்பட்டக்கருத்து",
    "Positive": "நேர்மறை",
    "Negative": "எதிர்மறை",
    "Neutral": "நடுநிலை",
    "None of the above": "எதுவும்இல்லை"
}

NUM_LABELS = len(LABEL2ID)


def clean_text(text: str,
               remove_hashtags: bool = False,
               remove_mentions: bool = True,
               remove_urls: bool = True,
               normalize_whitespace: bool = True) -> str:
    """
    Clean and preprocess Tamil tweet text.

    Args:
        text: Raw tweet text
        remove_hashtags: Whether to remove hashtags (default: False, as they carry political signal)
        remove_mentions: Whether to remove @mentions
        remove_urls: Whether to remove URLs
        normalize_whitespace: Whether to normalize whitespace

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol but keep the text (political signal)
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    else:
        # Keep hashtag content, just remove the # symbol
        text = re.sub(r'#(\w+)', r'\1', text)

    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_features(text: str) -> Dict[str, float]:
    """
    Extract handcrafted features from text.

    Args:
        text: Tweet text

    Returns:
        Dictionary of features
    """
    features = {}

    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())

    # Hashtag count
    features['hashtag_count'] = len(re.findall(r'#\w+', text))

    # Number presence
    features['has_numbers'] = 1.0 if re.search(r'\d+', text) else 0.0
    features['number_count'] = len(re.findall(r'\d+', text))

    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')

    # Emoji detection (basic)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    features['has_emoji'] = 1.0 if emoji_pattern.search(text) else 0.0

    # English word presence (code-mixing indicator)
    english_words = re.findall(r'[a-zA-Z]{3,}', text)
    features['english_word_count'] = len(english_words)
    features['english_ratio'] = len(english_words) / max(features['word_count'], 1)

    return features


class TamilSentimentDataset(Dataset):
    """
    PyTorch Dataset for Tamil Political Sentiment Analysis.
    """

    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 tokenizer = None,
                 max_length: int = 256,
                 clean: bool = True):
        """
        Initialize dataset.

        Args:
            texts: List of tweet texts
            labels: List of label ids (None for test set)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            clean: Whether to clean texts
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean = clean

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.clean:
            text = clean_text(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class TamilSentimentDatasetForLLM(Dataset):
    """
    PyTorch Dataset for LLM fine-tuning with instruction format.
    """

    INSTRUCTION_TEMPLATE = """Classify the following Tamil political tweet into one of these categories:
1. Substantiated (ஆதாரப்பூர்வமானது) - Contains evidence, references, or factual claims
2. Sarcastic (கிண்டல்) - Uses sarcasm, mockery, or irony
3. Opinionated (தனிப்பட்டக்கருத்து) - Strong personal views or biased beliefs
4. Positive (நேர்மறை) - Shows approval or appreciation
5. Negative (எதிர்மறை) - Expresses criticism or dissatisfaction
6. Neutral (நடுநிலை) - Factual political information without bias
7. None of the above (எதுவும்இல்லை) - Not related to politics

Tweet: {tweet}

Category:"""

    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 tokenizer = None,
                 max_length: int = 512,
                 clean: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean = clean

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.clean:
            text = clean_text(text)

        # Format as instruction
        prompt = self.INSTRUCTION_TEMPLATE.format(tweet=text)

        if self.labels is not None:
            label_text = ID2LABEL[self.labels[idx]]
            full_text = prompt + " " + label_text
        else:
            full_text = prompt

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if self.labels is not None:
            # For causal LM, labels are same as input_ids (shifted internally)
            item['labels'] = encoding['input_ids'].squeeze(0).clone()
            # Mask prompt tokens (only compute loss on answer)
            prompt_encoding = self.tokenizer(prompt, return_tensors='pt')
            prompt_len = prompt_encoding['input_ids'].shape[1]
            item['labels'][:prompt_len] = -100  # Ignore index

        return item


def load_data(data_dir: str = "data", use_augmented: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, dev, and test datasets.

    Args:
        data_dir: Directory containing CSV files
        use_augmented: Whether to use augmented training data

    Returns:
        Tuple of (train_df, dev_df, test_df)
    """
    train_file = "PS_train_final.csv" if use_augmented else "PS_train.csv"
    train_df = pd.read_csv(f"{data_dir}/{train_file}")
    dev_df = pd.read_csv(f"{data_dir}/PS_dev.csv")
    test_df = pd.read_csv(f"{data_dir}/PS_test_without_labels.csv")

    if use_augmented:
        print(f"Using augmented training data: {train_file} ({len(train_df)} samples)")

    # Encode labels
    train_df['label_id'] = train_df['labels'].map(LABEL2ID)
    dev_df['label_id'] = dev_df['labels'].map(LABEL2ID)

    return train_df, dev_df, test_df


def get_class_weights(labels: List[int],
                      smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute class weights for handling imbalance.

    Args:
        labels: List of label ids
        smoothing: Smoothing factor

    Returns:
        Tensor of class weights
    """
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(NUM_LABELS):
        count = counts.get(i, 1)
        weight = total / (NUM_LABELS * count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    # Apply smoothing
    weights = (1 - smoothing) * weights + smoothing * weights.mean()

    return weights


def create_submission(predictions: List[int],
                      test_df: pd.DataFrame,
                      output_path: str) -> pd.DataFrame:
    """
    Create submission file.

    Args:
        predictions: List of predicted label ids
        test_df: Test dataframe
        output_path: Path to save submission

    Returns:
        Submission dataframe
    """
    submission_df = test_df.copy()
    submission_df['labels'] = [ID2LABEL[p] for p in predictions]
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Prediction distribution:")
    print(submission_df['labels'].value_counts())

    return submission_df


if __name__ == "__main__":
    # Test the module
    train_df, dev_df, test_df = load_data()

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"\nLabel distribution in train:")
    print(train_df['labels'].value_counts())

    # Test cleaning
    sample_text = train_df['content'].iloc[0]
    print(f"\nOriginal: {sample_text[:100]}...")
    print(f"Cleaned: {clean_text(sample_text)[:100]}...")

    # Test feature extraction
    features = extract_features(sample_text)
    print(f"\nExtracted features: {features}")
