"""
Train a TF-IDF + LogisticRegression baseline.
Saves vectorizer and model under --out-dir.
"""

import argparse
import os
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import get_splits
from src.preprocess import batch_clean


def prepare_data(split, preprocess_args=None, subset_size=None):
    if subset_size is not None:
        split = split.select(range(min(subset_size, len(split))))

    texts = split["text"]
    labels = split["label"]

    cleaned = batch_clean(texts, **(preprocess_args or {}))
    return cleaned, np.array(labels)


def main(out_dir: str, subset_size: int = None, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    train, val, _ = get_splits()  # test split reserved for final evaluation in evaluate.py

    preprocess_args = {
        "lowercase": True,
        "replace_urls": True,
        "replace_users": True,
        "keep_punct": False,
        "handle_negation": False,
    }

    X_train, y_train = prepare_data(train, preprocess_args, subset_size)
    X_val, y_val = prepare_data(val, preprocess_args)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )

    clf.fit(X_train_vec, y_train)

    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))

    val_pred = clf.predict(X_val_vec)

    print("Validation accuracy:", accuracy_score(y_val, val_pred))
    print("Validation macro F1:", f1_score(y_val, val_pred, average="macro"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="models/baseline")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit training subset for quick experiments",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.out_dir, args.subset_size, args.seed)