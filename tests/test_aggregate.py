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
