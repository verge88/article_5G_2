"""Aggregate JSON experiment outputs without hiding individual seeds."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Iterable

from .paired import paired_gain_summary


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



def _metric_summaries(items: Iterable[dict]) -> dict[str, dict]:
    materialized = list(items)
    metrics: dict[str, dict] = {}
    for key in _METRIC_KEYS:
        values = [
            item.get(key)
            for item in materialized
            if item.get(key) is not None
        ]
        if values:
            metrics[key] = _summary(values)
    return metrics


def aggregate_results(
    results: Iterable[dict],
) -> dict[str, dict]:
    """Group run outputs by model and summarize metrics across seeds."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[str(result["model"])].append(result)

    output: dict[str, dict] = {}
    for model, runs in sorted(grouped.items()):
        metrics = _metric_summaries(
            run.get("metrics", {}) for run in runs
        )

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

        scope_names = sorted(
            {
                scope
                for run in runs
                for scope in run.get("scope_metrics", {})
            }
        )
        scope_metrics: dict[str, dict] = {}
        for scope in scope_names:
            entries = [
                run.get("scope_metrics", {}).get(scope, {})
                for run in runs
            ]
            eligible = [entry for entry in entries if entry.get("eligible")]
            metric_dicts = [
                entry.get("metrics", {})
                for entry in eligible
                if entry.get("metrics") is not None
            ]
            scope_metrics[scope] = {
                "eligible_runs": len(eligible),
                "counts": _summary(
                    [float(entry.get("count", 0)) for entry in eligible]
                ) if eligible else _summary([]),
                "metrics": _metric_summaries(metric_dicts),
            }

        ablation_names = sorted(
            {
                name
                for run in runs
                for name, item in run.get("score_ablation", {}).items()
                if isinstance(item, dict) and "metrics" in item
            }
        )
        score_ablation: dict[str, dict] = {}
        for name in ablation_names:
            entries = [
                run.get("score_ablation", {}).get(name, {})
                for run in runs
            ]
            score_ablation[name] = {
                "metrics": _metric_summaries(
                    entry.get("metrics", {}) for entry in entries
                ),
                "cross_protocol": _metric_summaries(
                    entry.get("cross_protocol", {}).get("metrics", {})
                    for entry in entries
                    if entry.get("cross_protocol", {}).get("eligible")
                ),
            }

        continuation_gain: dict[str, dict] = {}
        for scope_name, path in (
            ("overall", ("metrics",)),
            ("cross_protocol", ("cross_protocol", "metrics")),
        ):
            gains: dict[str, list[float]] = defaultdict(list)
            for run in runs:
                ablation = run.get("score_ablation", {})
                base = ablation.get("base", {})
                composite = ablation.get("composite", {})

                if scope_name == "overall":
                    base_metrics = base.get("metrics", {})
                    composite_metrics = composite.get("metrics", {})
                else:
                    base_scope = base.get("cross_protocol", {})
                    composite_scope = composite.get("cross_protocol", {})
                    if not (
                        base_scope.get("eligible")
                        and composite_scope.get("eligible")
                    ):
                        continue
                    base_metrics = base_scope.get("metrics", {}) or {}
                    composite_metrics = composite_scope.get("metrics", {}) or {}

                for key in _METRIC_KEYS:
                    base_value = base_metrics.get(key)
                    composite_value = composite_metrics.get(key)
                    if base_value is None or composite_value is None:
                        continue
                    gains[key].append(
                        float(composite_value) - float(base_value)
                    )

            continuation_gain[scope_name] = {
                key: paired_gain_summary(values)
                for key, values in sorted(gains.items())
            }

        output[model] = {
            "runs": len(runs),
            "seeds": [run.get("seed") for run in runs],
            "metrics": metrics,
            "scope_metrics": scope_metrics,
            "score_ablation": score_ablation,
            "continuation_gain": continuation_gain,
            "per_attack": per_attack,
        }

    return output
