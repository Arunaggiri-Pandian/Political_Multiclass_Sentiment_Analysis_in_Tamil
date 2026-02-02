"""
Data Augmentation for Tamil Political Sentiment Analysis
Techniques to increase training data diversity and reduce overfitting
"""

import random
import re
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


class TamilTextAugmenter:
    """
    Data augmentation techniques for Tamil text.
    Adapted from EDA (Easy Data Augmentation) for Tamil.
    """

    def __init__(self,
                 aug_probability: float = 0.3,
                 max_augmentations_per_sample: int = 2):
        """
        Initialize augmenter.

        Args:
            aug_probability: Probability of applying each augmentation
            max_augmentations_per_sample: Max augmented versions per original
        """
        self.aug_probability = aug_probability
        self.max_augmentations = max_augmentations_per_sample

        # Tamil stopwords (common words that can be safely manipulated)
        self.tamil_stopwords = {
            'இது', 'அது', 'என்ன', 'ஒரு', 'இந்த', 'அந்த', 'என்று', 'என்',
            'உள்ள', 'உள்ளது', 'என்பது', 'இருக்கிறது', 'செய்து', 'வந்து',
            'போல', 'மற்றும்', 'ஆனால்', 'எனவே', 'அதனால்', 'இருந்து',
            'வரை', 'பற்றி', 'மேலும்', 'கூட', 'தான்', 'மட்டும்'
        }

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.
        Preserves hashtags and mentions.
        """
        words = text.split()
        if len(words) <= 3:
            return text

        new_words = []
        for word in words:
            # Preserve hashtags and mentions
            if word.startswith('#') or word.startswith('@'):
                new_words.append(word)
            elif random.random() > p:
                new_words.append(word)

        # Ensure at least half the words remain
        if len(new_words) < len(words) // 2:
            return text

        return ' '.join(new_words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words.
        """
        words = text.split()
        if len(words) < 4:
            return text

        new_words = words.copy()
        for _ in range(n):
            # Find swappable indices (not hashtags/mentions)
            swappable = [i for i, w in enumerate(new_words)
                        if not w.startswith('#') and not w.startswith('@')]

            if len(swappable) < 2:
                break

            idx1, idx2 = random.sample(swappable, 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return ' '.join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert copies of random words.
        """
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()
        for _ in range(n):
            # Pick a random word (not hashtag/mention)
            candidates = [w for w in new_words
                         if not w.startswith('#') and not w.startswith('@')
                         and w not in self.tamil_stopwords]

            if not candidates:
                break

            word_to_insert = random.choice(candidates)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, word_to_insert)

        return ' '.join(new_words)

    def shuffle_middle(self, text: str) -> str:
        """
        Keep first and last words, shuffle middle.
        Useful for Tamil where word order is flexible.
        """
        words = text.split()
        if len(words) <= 4:
            return text

        first = words[0]
        last = words[-1]
        middle = words[1:-1]
        random.shuffle(middle)

        return ' '.join([first] + middle + [last])

    def duplicate_important_words(self, text: str) -> str:
        """
        Duplicate words that might be sentiment-bearing.
        """
        # Tamil sentiment words (simplified)
        sentiment_words = {
            'நல்ல', 'மோசம்', 'சிறந்த', 'மிக', 'அருமை', 'கொடுமை',
            'வெற்றி', 'தோல்வி', 'ஊழல்', 'நேர்மை', 'பொய்', 'உண்மை'
        }

        words = text.split()
        new_words = []

        for word in words:
            new_words.append(word)
            # Duplicate sentiment words occasionally
            if word in sentiment_words and random.random() < 0.3:
                new_words.append(word)

        return ' '.join(new_words)

    def augment_text(self, text: str) -> List[str]:
        """
        Apply multiple augmentation techniques to a single text.

        Returns:
            List of augmented texts (may include original)
        """
        augmented = []

        # Apply each technique with some probability
        techniques = [
            (self.random_deletion, {'p': 0.1}),
            (self.random_swap, {'n': 1}),
            (self.random_insertion, {'n': 1}),
            (self.shuffle_middle, {}),
            (self.duplicate_important_words, {}),
        ]

        for technique, kwargs in techniques:
            if random.random() < self.aug_probability:
                try:
                    aug_text = technique(text, **kwargs)
                    if aug_text != text and aug_text.strip():
                        augmented.append(aug_text)
                except Exception:
                    pass

            if len(augmented) >= self.max_augmentations:
                break

        return augmented


class BackTranslationAugmenter:
    """
    Back-translation augmentation using translation models.
    Tamil -> English -> Tamil
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.translator = None
        self.back_translator = None

    def load_models(self):
        """Load translation models."""
        try:
            from transformers import MarianMTModel, MarianTokenizer

            print("Loading Tamil->English model...")
            self.ta_en_tokenizer = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-ta-en'
            )
            self.ta_en_model = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-ta-en'
            ).to(self.device)

            print("Loading English->Tamil model...")
            self.en_ta_tokenizer = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-en-ta'
            )
            self.en_ta_model = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-en-ta'
            ).to(self.device)

            print("Translation models loaded!")
            return True

        except Exception as e:
            print(f"Could not load translation models: {e}")
            print("Back-translation will not be available.")
            return False

    def translate(self, text: str, model, tokenizer, max_length: int = 256) -> str:
        """Translate text using the given model."""
        inputs = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=max_length).to(self.device)
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def back_translate(self, text: str) -> Optional[str]:
        """
        Perform back-translation: Tamil -> English -> Tamil
        """
        if self.ta_en_model is None:
            return None

        try:
            # Tamil -> English
            english = self.translate(text, self.ta_en_model, self.ta_en_tokenizer)

            # English -> Tamil
            back_tamil = self.translate(english, self.en_ta_model, self.en_ta_tokenizer)

            # Only return if different from original
            if back_tamil != text and len(back_tamil) > 10:
                return back_tamil

        except Exception:
            pass

        return None


def augment_dataset(
    df: pd.DataFrame,
    text_column: str = 'content',
    label_column: str = 'labels',
    augment_minority_only: bool = True,
    minority_threshold: float = 0.15,
    aug_multiplier: int = 2,
    use_back_translation: bool = False
) -> pd.DataFrame:
    """
    Augment a dataset to increase size and balance classes.

    Args:
        df: Input dataframe
        text_column: Column containing text
        label_column: Column containing labels
        augment_minority_only: Only augment minority classes
        minority_threshold: Classes below this % are considered minority
        aug_multiplier: How many augmented samples per original
        use_back_translation: Whether to use back-translation

    Returns:
        Augmented dataframe
    """
    augmenter = TamilTextAugmenter(
        aug_probability=0.5,
        max_augmentations_per_sample=aug_multiplier
    )

    # Determine which classes to augment
    class_counts = df[label_column].value_counts(normalize=True)
    total_samples = len(df)

    if augment_minority_only:
        classes_to_augment = class_counts[class_counts < minority_threshold].index.tolist()
        print(f"Augmenting minority classes: {classes_to_augment}")
    else:
        classes_to_augment = df[label_column].unique().tolist()
        print(f"Augmenting all classes")

    # Back-translation setup
    back_translator = None
    if use_back_translation:
        back_translator = BackTranslationAugmenter()
        if not back_translator.load_models():
            back_translator = None

    # Augment
    new_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        if row[label_column] not in classes_to_augment:
            continue

        text = row[text_column]

        # EDA augmentations
        augmented_texts = augmenter.augment_text(text)

        # Back-translation
        if back_translator:
            bt_text = back_translator.back_translate(text)
            if bt_text:
                augmented_texts.append(bt_text)

        # Create new rows
        for aug_text in augmented_texts[:aug_multiplier]:
            new_row = row.copy()
            new_row[text_column] = aug_text
            new_rows.append(new_row)

    # Combine original and augmented
    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"\nAugmentation complete!")
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(new_rows)}")
    print(f"Total samples: {len(augmented_df)}")

    print(f"\nNew class distribution:")
    print(augmented_df[label_column].value_counts())

    return augmented_df


if __name__ == "__main__":
    # Test augmentation
    import sys
    sys.path.append('.')
    from src.data_utils import load_data

    print("Loading data...")
    train_df, dev_df, test_df = load_data()

    print(f"\nOriginal train size: {len(train_df)}")
    print(f"Original class distribution:")
    print(train_df['labels'].value_counts())

    print("\n" + "="*50)
    print("Running augmentation...")
    print("="*50)

    augmented_df = augment_dataset(
        train_df,
        text_column='content',
        label_column='labels',
        augment_minority_only=True,
        minority_threshold=0.15,
        aug_multiplier=2,
        use_back_translation=False  # Set True if translation models available
    )

    # Save augmented data
    output_path = 'data/PS_train_augmented.csv'
    augmented_df.to_csv(output_path, index=False)
    print(f"\nAugmented data saved to: {output_path}")
