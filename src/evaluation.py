"""Shared three-class evaluation helpers."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_NAMES = ("negative", "neutral", "positive")


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0
    )
    _, _, class_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        **{f"{name}_f1": float(value) for name, value in zip(LABEL_NAMES, class_f1)},
    }

