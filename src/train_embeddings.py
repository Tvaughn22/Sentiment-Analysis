import argparse
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import get_splits
from src.embeddings import train_word2vec, vectorize_texts
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

    train, val, _ = get_splits()

    # Keep preprocessing aligned with baseline/evaluation so comparisons stay fair.
    preprocess_args = {
        "lowercase": True,
        "replace_urls": True,
        "replace_users": True,
        "keep_punct": False,
        "handle_negation": False,
    }

    X_train, y_train = prepare_data(train, preprocess_args, subset_size)
    X_val, y_val = prepare_data(val, preprocess_args)

    # Learn token embeddings on the cleaned training texts, then average them per example.
    w2v_model = train_word2vec(X_train)
    X_train_vec = vectorize_texts(X_train, w2v_model)
    X_val_vec = vectorize_texts(X_val, w2v_model)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train_vec, y_train)

    # Save both pieces because evaluation needs the embedding model and classifier together.
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    joblib.dump(w2v_model, os.path.join(out_dir, "w2v.pkl"))

    val_pred = clf.predict(X_val_vec)

    print("Validation accuracy:", accuracy_score(y_val, val_pred))
    print("Validation macro F1:", f1_score(y_val, val_pred, average="macro"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="models/embeddings")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit training subset for quick experiments",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.out_dir, args.subset_size, args.seed)
