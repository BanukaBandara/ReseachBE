from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class MetricsResult:
    accuracy: float
    f1_macro: float
    conf_mat: list[list[int]]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> MetricsResult:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred).astype(int)
    return MetricsResult(accuracy=acc, f1_macro=f1, conf_mat=cm.tolist())


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)
