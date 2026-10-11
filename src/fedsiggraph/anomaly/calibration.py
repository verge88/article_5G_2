"""Benign-only robust calibration for heterogeneous anomaly components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RobustScoreStats:
    median: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustScoreStats":
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("cannot fit score calibration without finite values")

        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = 1.4826 * mad
        if scale < 1e-8:
            # Preserve ordering when benign calibration is nearly constant
            # without amplifying floating-point noise.
            std = float(np.std(finite))
            scale = std if std >= 1e-8 else 1.0
        return cls(median=median, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values - self.median) / self.scale


@dataclass(frozen=True, slots=True)
class CompositeScoreCalibrator:
    """Combine incomparable anomaly components after benign robust scaling.

    ``base_score`` is available for every event. ``continuation_score``
    is defined only where a previous explicit context exists. Missing
    continuation values contribute zero rather than being imputed.
    """

    base: RobustScoreStats
    continuation: RobustScoreStats | None
    continuation_weight: float = 1.0

    @classmethod
    def fit(
        cls,
        base_scores: np.ndarray,
        continuation_scores: np.ndarray,
        context_mask: np.ndarray,
        *,
        continuation_weight: float = 1.0,
    ) -> "CompositeScoreCalibrator":
        if continuation_weight < 0:
            raise ValueError("continuation_weight must be non-negative")

        base_scores = np.asarray(base_scores, dtype=np.float64)
        continuation_scores = np.asarray(
            continuation_scores, dtype=np.float64
        )
        context_mask = np.asarray(context_mask, dtype=bool)

        if not (
            base_scores.shape
            == continuation_scores.shape
            == context_mask.shape
        ):
            raise ValueError("score arrays and context_mask must share shape")

        base_stats = RobustScoreStats.fit(base_scores)
        continuation_stats = None
        available = context_mask & np.isfinite(continuation_scores)
        if available.any():
            continuation_stats = RobustScoreStats.fit(
                continuation_scores[available]
            )

        return cls(
            base=base_stats,
            continuation=continuation_stats,
            continuation_weight=continuation_weight,
        )

    def transform(
        self,
        base_scores: np.ndarray,
        continuation_scores: np.ndarray,
        context_mask: np.ndarray,
    ) -> np.ndarray:
        base_scores = np.asarray(base_scores, dtype=np.float64)
        continuation_scores = np.asarray(
            continuation_scores, dtype=np.float64
        )
        context_mask = np.asarray(context_mask, dtype=bool)

        if not (
            base_scores.shape
            == continuation_scores.shape
            == context_mask.shape
        ):
            raise ValueError("score arrays and context_mask must share shape")

        composite = self.base.transform(base_scores)
        if self.continuation is None or self.continuation_weight == 0:
            return composite

        available = context_mask & np.isfinite(continuation_scores)
        if available.any():
            contribution = self.continuation.transform(
                continuation_scores[available]
            )
            composite = composite.copy()
            composite[available] += self.continuation_weight * contribution
        return composite
