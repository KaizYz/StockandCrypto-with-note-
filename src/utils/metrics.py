from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_proba),
    }


def multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        classes = np.arange(y_proba.shape[1])
    else:
        classes = np.unique(y_true)
    try:
        top2 = float(
            top_k_accuracy_score(y_true, y_proba, k=2, labels=classes)
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2
            else accuracy_score(y_true, y_pred)
        )
    except Exception:
        top2 = float(accuracy_score(y_true, y_pred))
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top2_accuracy": top2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sign_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def interval_metrics(
    y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray
) -> Dict[str, float]:
    covered = (y_true >= q_low) & (y_true <= q_high)
    width = q_high - q_low
    return {
        "coverage": float(np.mean(covered)),
        "interval_width": float(np.mean(width)),
    }


def rows_from_metric_dict(
    metric_dict: Dict[str, float], fixed_fields: Dict[str, str]
) -> Iterable[Dict[str, str | float]]:
    for key, val in metric_dict.items():
        row = dict(fixed_fields)
        row["metric"] = key
        row["value"] = val
        yield row
