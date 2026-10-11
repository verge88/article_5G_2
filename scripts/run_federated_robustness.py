#!/usr/bin/env python3
"""Run a reproducible federated robustness experiment matrix.

The launcher compares:
- FedAvg/FedProx on clean clients;
- weighted, coordinate-median, and trimmed-mean aggregation;
- optional controlled sign-flip/model-replacement poisoning.

Each individual run is delegated to train_federated.py so training, local
calibration, and evaluation remain identical across the matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


from fedsiggraph.evaluation.provenance import runtime_provenance


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.34,
        help="0.34 trims one client per tail in a 3-client experiment.",
    )
    parser.add_argument(
        "--poison-client",
        action="append",
        default=[],
        help="Prepared client directory name. May be repeated.",
    )
    parser.add_argument(
        "--poison-mode",
        choices=["sign_flip", "model_replacement"],
        default="sign_flip",
    )
    parser.add_argument("--poison-factor", type=float, default=5.0)
    parser.add_argument("--local-adapter-dim", type=int, default=16)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--continuation-history-mode",
        choices=["immediate", "sequence"],
        default="sequence",
    )
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    *,
    seed: int,
    strategy: str,
    aggregation: str,
    poisoned: bool,
    output: Path,
) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_federated.py"),
        "--clients",
        *args.clients,
        "--strategy",
        strategy,
        "--aggregation",
        aggregation,
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--prox-mu",
        str(args.prox_mu),
        "--trim-ratio",
        str(args.trim_ratio),
        "--local-adapter-dim",
        str(args.local_adapter_dim),
        "--target-fpr",
        str(args.target_fpr),
        "--test-memory",
        args.test_memory,
        "--neighbors",
        str(args.neighbors),
        "--event-dim",
        str(args.event_dim),
        "--memory-dim",
        str(args.memory_dim),
        "--time-dim",
        str(args.time_dim),
        "--embedding-dim",
        str(args.embedding_dim),
        "--heads",
        str(args.heads),
        "--continuation-history-mode",
        args.continuation_history_mode,
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]

    if poisoned:
        if not args.poison_client:
            raise ValueError(
                "poisoned matrix entry requested without --poison-client"
            )
        for client in args.poison_client:
            command.extend(["--poison-client", client])
        command.extend(
            [
                "--poison-mode",
                args.poison_mode,
                "--poison-factor",
                str(args.poison_factor),
            ]
        )

    subprocess.run(command, cwd=ROOT, check=True)
    with open(output, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_run(payload: dict) -> dict:
    evaluations = payload.get("evaluations", [])
    pr_auc = [
        float(item["metrics"]["pr_auc"])
        for item in evaluations
        if item["metrics"].get("pr_auc") is not None
    ]
    recall = [
        float(item["metrics"]["recall"])
        for item in evaluations
        if item["metrics"].get("recall") is not None
    ]
    fpr = [
        float(item["metrics"]["false_positive_rate"])
        for item in evaluations
    ]
    return {
        "mean_client_pr_auc": mean(pr_auc),
        "mean_client_recall": mean(recall),
        "mean_client_fpr": mean(fpr),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.poison_client and len(args.clients) < 3:
        raise ValueError(
            "poisoning robustness matrix requires at least three clients; "
            "with two clients median/trimmed aggregation cannot isolate one "
            "malicious participant"
        )
    if args.poison_client:
        trim = int(len(args.clients) * args.trim_ratio)
        if trim == 0:
            raise ValueError(
                "trim-ratio would remove zero clients; increase it for a "
                "meaningful trimmed-mean robustness experiment"
            )

    matrix = [
        # Clean references:
        ("fedavg", "weighted_mean", False),
        ("fedprox", "weighted_mean", False),
    ]
    if args.poison_client:
        matrix.extend(
            [
                ("fedprox", "weighted_mean", True),
                ("fedprox", "coordinate_median", True),
                ("fedprox", "trimmed_mean", True),
            ]
        )

    runs: list[dict] = []
    for seed in args.seeds:
        for strategy, aggregation, poisoned in matrix:
            label = (
                f"{strategy}-{aggregation}-"
                f"{'poisoned' if poisoned else 'clean'}-seed{seed}"
            )
            path = output_dir / f"{label}.json"
            print(f"[robustness] {label}")
            payload = run_one(
                args,
                seed=seed,
                strategy=strategy,
                aggregation=aggregation,
                poisoned=poisoned,
                output=path,
            )
            runs.append(
                {
                    "label": label,
                    "seed": seed,
                    "strategy": strategy,
                    "aggregation": aggregation,
                    "poisoned": poisoned,
                    "result_file": path.name,
                    **summarize_run(payload),
                }
            )

    grouped: dict[str, list[dict]] = {}
    for item in runs:
        key = (
            f"{item['strategy']}|{item['aggregation']}|"
            f"{'poisoned' if item['poisoned'] else 'clean'}"
        )
        grouped.setdefault(key, []).append(item)

    summary_groups = {}
    for key, items in grouped.items():
        summary_groups[key] = {
            "runs": len(items),
            "mean_client_pr_auc": mean(
                [
                    item["mean_client_pr_auc"]
                    for item in items
                    if item["mean_client_pr_auc"] is not None
                ]
            ),
            "mean_client_recall": mean(
                [
                    item["mean_client_recall"]
                    for item in items
                    if item["mean_client_recall"] is not None
                ]
            ),
            "mean_client_fpr": mean(
                [
                    item["mean_client_fpr"]
                    for item in items
                    if item["mean_client_fpr"] is not None
                ]
            ),
        }

    summary = {
        "experiment": "federated_robustness",
        "clients": args.clients,
        "seeds": args.seeds,
        "poison_clients": args.poison_client,
        "poison_mode": args.poison_mode if args.poison_client else None,
        "poison_factor": args.poison_factor if args.poison_client else None,
        "continuation_history_mode": args.continuation_history_mode,
        "provenance": runtime_provenance(ROOT),
        "runs": runs,
        "groups": summary_groups,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
