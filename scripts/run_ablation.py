#!/usr/bin/env python3
"""Run the staged FedSigGraph ablation with identical seeds and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fedsiggraph.evaluation.aggregate import aggregate_results
from fedsiggraph.evaluation.provenance import runtime_provenance


MODEL_CHOICES = (
    "numeric_tgn",
    "semantic_tgn",
    "temporal_heat",
    "temporal_heat_aux",
    "temporal_heat_continuation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument(
        "--allow-debug-subset",
        action="store_true",
        help="Permit explicitly non-research debug subsets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def model_command(
    model: str,
    *,
    data: Path,
    output: Path,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    common = [
        "--data",
        str(data),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--neighbors",
        str(args.neighbors),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--target-fpr",
        str(args.target_fpr),
        "--test-memory",
        args.test_memory,
        "--output",
        str(output),
    ]
    if args.allow_debug_subset:
        common.append("--allow-debug-subset")

    if model == "numeric_tgn":
        return [sys.executable, "scripts/train_baseline.py", *common]

    if model == "semantic_tgn":
        return [sys.executable, "scripts/train_semantic_tgn.py", *common]

    heat = [
        sys.executable,
        "scripts/train_temporal_heat.py",
        *common,
    ]
    if model == "temporal_heat":
        return [
            *heat,
            "--continuation-loss-weight",
            "0",
            "--continuation-score-weight",
            "0",
        ]
    if model == "temporal_heat_aux":
        return [
            *heat,
            "--continuation-loss-weight",
            "0.25",
            "--continuation-score-weight",
            "0",
        ]
    if model == "temporal_heat_continuation":
        return [
            *heat,
            "--continuation-loss-weight",
            "0.25",
            "--continuation-score-weight",
            "1.0",
        ]
    raise ValueError(f"unknown model: {model}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []

    for model in args.models:
        for seed in args.seeds:
            output = output_dir / f"{model}_seed{seed}.json"
            command = model_command(
                model,
                data=data_dir,
                output=output,
                seed=seed,
                args=args,
            )
            print("$", " ".join(command), flush=True)
            result_paths.append(output)

            if args.dry_run:
                continue

            subprocess.run(command, check=True)

    if args.dry_run:
        return

    results = []
    for path in result_paths:
        with open(path, "r", encoding="utf-8") as handle:
            results.append(json.load(handle))

    summary = {
        "data": str(data_dir),
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "learning_rate": args.lr,
        "target_fpr": args.target_fpr,
        "test_memory": args.test_memory,
        "provenance": runtime_provenance(Path(__file__).resolve().parents[1]),
        "aggregate": aggregate_results(results),
    }
    with open(
        output_dir / "summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        json.dumps(summary["aggregate"], indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
