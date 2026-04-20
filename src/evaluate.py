import os
import json
from typing import Sequence
import logging
from src.data_loader import LABEL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate_preds(y_true: Sequence[int], y_pred: Sequence[int], label_map: dict, out_dir: str):
    """
    Evaluate predictions and save metrics, reports, and plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    ordered_labels = sorted(label_map.keys())
    ordered_names = [label_map[i] for i in ordered_labels]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=ordered_labels,
        target_names=ordered_names,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
    }
    logger.info("Accuracy: %.4f | Macro F1: %.4f", acc, f1)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(out_dir, "classification_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ordered_names,
        yticklabels=ordered_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    df = pd.DataFrame({
        "true": list(y_true),
        "pred": list(y_pred),
        "true_label": [label_map[y] for y in y_true],
        "pred_label": [label_map[y] for y in y_pred],
    })
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    logger.info("Saved evaluation artifacts to %s", out_dir)

    return metrics


def load_preds_and_eval(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_dir: str = "reports/eval",
) -> dict:
    """
    Convenience wrapper using the standard sentiment label mapping.
    """
    return evaluate_preds(y_true, y_pred, LABEL_MAP, out_dir)