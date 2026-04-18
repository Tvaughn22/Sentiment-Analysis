from datasets import load_dataset
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


def load_sentiment_dataset(cache_dir: Optional[str] = None):
    """
    Load the cardiffnlp/tweet_eval dataset with the 'sentiment' configuration.

    Returns:
        DatasetDict containing 'train', 'validation', and 'test' splits.
    """
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", cache_dir=cache_dir)
    for split_name, split in ds.items():
        if len(split) == 0:
            raise ValueError(f"Split '{split_name}' loaded empty — check your connection or cache.")
    logger.info("Loaded tweet_eval sentiment split sizes: %s", {k: len(ds[k]) for k in ds})
    return ds


def get_splits(cache_dir: Optional[str] = None):
    """
    Return the train, validation, and test splits.
    """
    ds = load_sentiment_dataset(cache_dir)
    return ds["train"], ds["validation"], ds["test"]


def get_label_mapping() -> Dict[int, str]:
    """
    Return the numeric-to-text sentiment label mapping.
    """
    return LABEL_MAP


def sample_examples(n: int = 5, cache_dir: Optional[str] = None):
    """
    Return up to n example rows from the training split with readable labels.
    """
    train, _, _ = get_splits(cache_dir)
    return [
        {"text": train[i]["text"], "label": LABEL_MAP[train[i]["label"]]}
        for i in range(min(n, len(train)))
    ]