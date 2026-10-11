import pytest

pytest.importorskip("sklearn")

import numpy as np

from fedsiggraph.training.metrics import evaluate_masked_scores


def test_masked_scores_keep_global_threshold_and_report_class_counts():
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.8, 0.7, 0.9], dtype=np.float64)
    mask = np.asarray([False, True, True, True], dtype=bool)

    result = evaluate_masked_scores(
        y_true,
        scores,
        threshold=0.75,
        mask=mask,
    )

    assert result["eligible"] is True
    assert result["count"] == 3
    assert result["benign_count"] == 1
    assert result["attack_count"] == 2
    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["fp"] == 1
    assert result["metrics"]["fn"] == 1


def test_masked_scores_marks_empty_subset_ineligible():
    result = evaluate_masked_scores(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0.2, 0.8], dtype=np.float64),
        threshold=0.5,
        mask=np.asarray([False, False], dtype=bool),
    )

    assert result == {
        "eligible": False,
        "count": 0,
        "benign_count": 0,
        "attack_count": 0,
        "metrics": None,
    }
