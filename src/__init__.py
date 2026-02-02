"""
Tamil Political Sentiment Analysis
DravidianLangTech @ ACL 2026
"""

from .data_utils import (
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    TAMIL_LABELS,
    clean_text,
    extract_features,
    load_data,
    get_class_weights,
    create_submission,
    TamilSentimentDataset,
    TamilSentimentDatasetForLLM
)

__version__ = "1.0.0"
__all__ = [
    "LABEL2ID",
    "ID2LABEL",
    "NUM_LABELS",
    "TAMIL_LABELS",
    "clean_text",
    "extract_features",
    "load_data",
    "get_class_weights",
    "create_submission",
    "TamilSentimentDataset",
    "TamilSentimentDatasetForLLM"
]
