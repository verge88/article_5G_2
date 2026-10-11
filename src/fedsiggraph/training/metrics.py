"""Shared anomaly calibration and evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

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


def attack_type_breakdown(
    attack_types: Sequence[str | None],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    """Report detection recall for each attack family at one global threshold.

    The sidecar labels are evaluation-only and must be aligned one-to-one with
    the score stream. Benign rows are summarized as ``normal`` with their
    false-positive rate.
    """
    labels = np.asarray(
        ["normal" if value in (None, "", "normal") else str(value)
         for value in attack_types],
        dtype=object,
    )
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if not (labels.size == y_true.size == y_score.size):
        raise ValueError(
            "attack_types, y_true, and y_score must have identical length"
        )

    pred = (y_score > threshold).astype(np.int64)
    result: dict[str, dict[str, float | int]] = {}

    for attack_type in sorted(set(labels.tolist())):
        mask = labels == attack_type
        scores = y_score[mask]
        predictions = pred[mask]
        truths = y_true[mask]

        item: dict[str, float | int] = {
            "count": int(mask.sum()),
            "mean_score": float(scores.mean()) if scores.size else 0.0,
            "p95_score": (
                float(np.quantile(scores, 0.95)) if scores.size else 0.0
            ),
        }

        if attack_type == "normal":
            normal_mask = truths == 0
            fp = int((predictions[normal_mask] == 1).sum())
            count = int(normal_mask.sum())
            item.update(
                {
                    "false_positives": fp,
                    "false_positive_rate": float(fp / max(count, 1)),
                }
            )
        else:
            malicious_mask = truths == 1
            detected = int((predictions[malicious_mask] == 1).sum())
            count = int(malicious_mask.sum())
            item.update(
                {
                    "detected": detected,
                    "recall": float(detected / max(count, 1)),
                }
            )

        result[str(attack_type)] = item

    return result


def temporal_duration_ns(
    data,
    *,
    time_unit_ns: int,
) -> int | None:
    """Return stream duration using dataset-level time-unit metadata.

    The time unit belongs to the prepared dataset manifest rather than
    TemporalData, because TemporalDataLoader slices every custom attribute as
    event-aligned data.
    """
    if time_unit_ns <= 0:
        raise ValueError("time_unit_ns must be positive")
    if int(data.num_events) <= 1:
        return None
    duration_units = int(data.t.max()) - int(data.t.min())
    return duration_units * int(time_unit_ns)
