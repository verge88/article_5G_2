#!/usr/bin/env python3
"""Create a small, explicitly non-research subset of a prepared dataset.

This utility exists only for real-format CI/model smoke tests. It preserves the
original preprocessing state but marks the output manifest as debug-only so
results cannot be confused with full research experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import torch

from fedsiggraph.evaluation.eligibility import research_claim_eligibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-events", type=int, default=2000)
    parser.add_argument("--val-events", type=int, default=500)
    parser.add_argument("--calibration-events", type=int, default=500)
    parser.add_argument("--test-events", type=int, default=1000)
    return parser.parse_args()


def load_temporal(path: Path):
    return torch.load(path, weights_only=False)


def slice_temporal(data, count: int):
    if count <= 0:
        raise ValueError("subset event counts must be positive")
    return data[: min(count, int(data.num_events))]


def tensor_rate(values) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().mean().item())


def split_coverage(data) -> dict[str, float]:
    return {
        "explicit_context_history": tensor_rate(data.context_history_mask),
        "cross_protocol_transition": tensor_rate(data.cross_protocol_mask),
    }


def subset_test_metadata(
    source: Path,
    target: Path,
    *,
    count: int,
) -> None:
    with open(source, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= count:
                break
            rows.append(row)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("test metadata has no header")

    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    splits = {}
    requested = {
        "train": args.train_events,
        "val": args.val_events,
        "calibration": args.calibration_events,
        "test": args.test_events,
    }
    for name, count in requested.items():
        data = load_temporal(source / f"{name}.pt")
        subset = slice_temporal(data, count)
        torch.save(subset, output / f"{name}.pt")
        splits[name] = subset

    shutil.copy2(source / "preprocessing.json", output / "preprocessing.json")

    with open(source / "manifest.json", "r", encoding="utf-8") as handle:
        original_manifest = json.load(handle)

    metadata_name = original_manifest.get(
        "test_metadata_file", "test_metadata.csv"
    )
    subset_test_metadata(
        source / metadata_name,
        output / metadata_name,
        count=int(splits["test"].num_events),
    )

    context_coverage = {
        name: split_coverage(data) for name, data in splits.items()
    }
    manifest = dict(original_manifest)
    manifest.update(
        {
            "debug_subset": True,
            "research_metrics_valid": False,
            "debug_subset_source": str(source),
            "counts": {
                name: int(data.num_events) for name, data in splits.items()
            },
            "context_coverage": context_coverage,
            "claim_eligibility": research_claim_eligibility(
                context_coverage
            ),
        }
    )

    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
