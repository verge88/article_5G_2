"""Shared anomaly calibration and evaluation metrics."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("metrics require the ML extra") from exc


def calibrate_threshold(scores: np.ndarray, target_fpr: float) -> float:
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be in (0, 1)")
    if scores.size == 0:
        raise ValueError("calibration split is empty")
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def _safe_auc(metric, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(metric(y_true, y_score))


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    duration_ns: int | None = None,
) -> dict[str, float | int | None]:
    pred = (y_score > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    duration_hours = (
        duration_ns / 1e9 / 3600.0
        if duration_ns is not None and duration_ns > 0
        else 0.0
    )

    return {
        "pr_auc": _safe_auc(average_precision_score, y_true, y_score),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_score),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_alarms_per_hour": (
            float(fp / duration_hours) if duration_hours > 0 else None
        ),
    }


def temporal_duration_ns(data) -> int | None:
    if int(data.num_events) <= 1:
        return None
    duration_units = int(data.t.max()) - int(data.t.min())
    return duration_units * int(data.time_unit_ns)
