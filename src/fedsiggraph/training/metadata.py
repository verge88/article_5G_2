"""Evaluation-only metadata loading with strict tensor alignment checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class TestMetadata:
    attack_types: tuple[str, ...]
    labels: np.ndarray


def load_test_metadata(
    path: str | Path,
    *,
    expected_labels: np.ndarray | None = None,
) -> TestMetadata:
    attack_types: list[str] = []
    labels: list[int] = []
    indices: list[int] = []

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            indices.append(int(row["event_index"]))
            labels.append(int(row["label"]))
            attack_types.append(row["attack_type"] or "normal")

    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
        raise ValueError(
            "test metadata event_index is not contiguous/aligned"
        )

    label_array = np.asarray(labels, dtype=np.int64)
    if expected_labels is not None:
        expected = np.asarray(expected_labels, dtype=np.int64)
        if expected.shape != label_array.shape:
            raise ValueError(
                "test metadata length does not match score/test tensor length"
            )
        if not np.array_equal(expected, label_array):
            raise ValueError(
                "test metadata labels do not match test tensor labels"
            )

    return TestMetadata(
        attack_types=tuple(attack_types),
        labels=label_array,
    )
