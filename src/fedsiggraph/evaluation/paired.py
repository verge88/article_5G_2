"""Paired statistics for continuation-score ablations.

The same seeds/splits produce base and composite metrics, so continuation gains
must be analyzed as paired differences rather than as independent samples.
"""

from __future__ import annotations

from itertools import product
import math
import statistics
from typing import Iterable


def exact_sign_flip_pvalue(
    differences: Iterable[float],
    *,
    max_exact_n: int = 20,
) -> float | None:
    """Two-sided exact randomization p-value for a paired mean difference.

    Under the symmetric zero-effect null, every sign assignment to the observed
    absolute paired differences is equally likely. Zero differences are removed.
    For more than max_exact_n non-zero pairs we deliberately return None rather
    than silently switching to an approximate significance test.
    """
    values = [
        float(value)
        for value in differences
        if math.isfinite(float(value)) and abs(float(value)) > 1e-15
    ]
    if not values:
        return 1.0
    if len(values) > max_exact_n:
        return None

    observed = abs(statistics.fmean(values))
    magnitudes = [abs(value) for value in values]

    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in product((-1.0, 1.0), repeat=len(magnitudes)):
        statistic = abs(
            statistics.fmean(
                sign * magnitude
                for sign, magnitude in zip(signs, magnitudes)
            )
        )
        total += 1
        if statistic + tolerance >= observed:
            extreme += 1

    return extreme / total


def paired_gain_summary(
    differences: Iterable[float],
) -> dict[str, float | int | None]:
    """Summarize paired metric improvements across matched seeds."""
    values = [
        float(value)
        for value in differences
        if math.isfinite(float(value))
    ]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "positive_fraction": None,
            "exact_sign_flip_pvalue": None,
        }

    positive = sum(value > 0 for value in values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "positive_fraction": positive / len(values),
        "exact_sign_flip_pvalue": exact_sign_flip_pvalue(values),
    }
