from fedsiggraph.evaluation.aggregate import aggregate_results


def test_aggregate_results_preserves_model_and_attack_statistics():
    results = [
        {
            "model": "m",
            "seed": 1,
            "metrics": {"f1": 0.5, "recall": 0.4},
            "per_attack": {
                "attack-a": {"recall": 0.25, "mean_score": 1.0},
                "normal": {"false_positive_rate": 0.01},
            },
        },
        {
            "model": "m",
            "seed": 2,
            "metrics": {"f1": 0.7, "recall": 0.8},
            "per_attack": {
                "attack-a": {"recall": 0.75, "mean_score": 2.0},
                "normal": {"false_positive_rate": 0.03},
            },
        },
    ]

    summary = aggregate_results(results)["m"]

    assert summary["runs"] == 2
    assert summary["metrics"]["f1"]["mean"] == 0.6
    assert summary["per_attack"]["attack-a"]["recall"]["mean"] == 0.5
    assert summary["per_attack"]["normal"]["false_positive_rate"]["mean"] == 0.02


def test_aggregate_results_includes_cross_protocol_scope_and_score_ablation():
    results = [
        {
            "model": "temporal_heat_context_continuation",
            "seed": 1,
            "metrics": {"pr_auc": 0.70},
            "scope_metrics": {
                "cross_protocol": {
                    "eligible": True,
                    "count": 10,
                    "metrics": {"pr_auc": 0.80, "recall": 0.60},
                }
            },
            "score_ablation": {
                "memory_trajectory": "independent_causal_replay",
                "base": {
                    "metrics": {"pr_auc": 0.60},
                    "cross_protocol": {
                        "eligible": True,
                        "metrics": {"pr_auc": 0.65, "recall": 0.40},
                    },
                },
                "composite": {
                    "metrics": {"pr_auc": 0.70},
                    "cross_protocol": {
                        "eligible": True,
                        "metrics": {"pr_auc": 0.80, "recall": 0.60},
                    },
                },
            },
            "per_attack": {},
        },
        {
            "model": "temporal_heat_context_continuation",
            "seed": 2,
            "metrics": {"pr_auc": 0.74},
            "scope_metrics": {
                "cross_protocol": {
                    "eligible": True,
                    "count": 12,
                    "metrics": {"pr_auc": 0.84, "recall": 0.70},
                }
            },
            "score_ablation": {
                "memory_trajectory": "independent_causal_replay",
                "base": {
                    "metrics": {"pr_auc": 0.62},
                    "cross_protocol": {
                        "eligible": True,
                        "metrics": {"pr_auc": 0.67, "recall": 0.45},
                    },
                },
                "composite": {
                    "metrics": {"pr_auc": 0.74},
                    "cross_protocol": {
                        "eligible": True,
                        "metrics": {"pr_auc": 0.84, "recall": 0.70},
                    },
                },
            },
            "per_attack": {},
        },
    ]

    summary = aggregate_results(results)["temporal_heat_context_continuation"]

    assert summary["scope_metrics"]["cross_protocol"]["eligible_runs"] == 2
    assert summary["scope_metrics"]["cross_protocol"]["counts"]["mean"] == 11.0
    assert (
        summary["scope_metrics"]["cross_protocol"]["metrics"]["pr_auc"]["mean"]
        == 0.82
    )
    assert (
        summary["score_ablation"]["base"]["cross_protocol"]["recall"]["mean"]
        == 0.425
    )
    assert (
        summary["score_ablation"]["composite"]["cross_protocol"]["recall"]["mean"]
        == 0.65
    )

    assert (
        summary["continuation_gain"]["overall"]["pr_auc"]["mean"]
        == 0.11
    )
    assert (
        summary["continuation_gain"]["cross_protocol"]["recall"]["mean"]
        == 0.2125
    )
