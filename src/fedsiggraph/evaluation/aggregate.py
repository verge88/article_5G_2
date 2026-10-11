"""Aggregate JSON experiment outputs without hiding individual seeds."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Iterable


_METRIC_KEYS = (
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "false_positive_rate",
    "false_alarms_per_hour",
)


def _summary(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def aggregate_results(
    results: Iterable[dict],
) -> dict[str, dict]:
    """Group run outputs by model and summarize metrics across seeds."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[str(result["model"])].append(result)

    output: dict[str, dict] = {}
    for model, runs in sorted(grouped.items()):
        metrics: dict[str, dict] = {}
        for key in _METRIC_KEYS:
            values = [
                run.get("metrics", {}).get(key)
                for run in runs
                if run.get("metrics", {}).get(key) is not None
            ]
            if values:
                metrics[key] = _summary(values)

        attack_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for run in runs:
            for attack_type, item in run.get("per_attack", {}).items():
                for key in ("recall", "false_positive_rate", "mean_score"):
                    value = item.get(key)
                    if value is not None:
                        attack_values[attack_type][key].append(float(value))

        per_attack = {
            attack_type: {
                key: _summary(values)
                for key, values in sorted(items.items())
            }
            for attack_type, items in sorted(attack_values.items())
        }

        output[model] = {
            "runs": len(runs),
            "seeds": [run.get("seed") for run in runs],
            "metrics": metrics,
            "per_attack": per_attack,
        }

    return output
