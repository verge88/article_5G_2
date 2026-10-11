#!/usr/bin/env python3
"""Run the reproducible Granomaly NRF/SMF FedSigGraph experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


from fedsiggraph.evaluation.provenance import runtime_provenance


ROOT = Path(__file__).resolve().parents[1]

MODEL_CHOICES = (
    "numeric_tgn",
    "semantic_tgn",
    "temporal_heat",
    "temporal_heat_aux",
    "temporal_heat_continuation_immediate",
    "temporal_heat_continuation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="experiments/granomaly")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["nrf", "smf"],
        default=["nrf", "smf"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
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
        "--skip-download",
        action="store_true",
        help="Reuse captures already present under workspace/raw.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid completed model/seed result JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without running them.",
    )
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    raw = workspace / "raw"
    prepared_root = workspace / "prepared"
    results_root = workspace / "results"

    if not args.skip_download:
        run(
            [
                sys.executable,
                "scripts/fetch_granomaly.py",
                "--output",
                str(raw),
            ],
            dry_run=args.dry_run,
        )

    experiment_summaries: dict[str, dict] = {}

    for experiment in args.experiments:
        attack_file = raw / f"bridge_attack_{experiment}.csv"
        normal_file = raw / "bridge_normal.csv"
        prepared = prepared_root / experiment
        results = results_root / experiment

        run(
            [
                sys.executable,
                "scripts/prepare_granomaly.py",
                "--normal",
                str(normal_file),
                "--attack",
                str(attack_file),
                "--output",
                str(prepared),
            ],
            dry_run=args.dry_run,
        )

        ablation_command = [
            sys.executable,
            "scripts/run_ablation.py",
            "--data",
            str(prepared),
            "--output-dir",
            str(results),
            "--models",
            *args.models,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--neighbors",
            str(args.neighbors),
            "--lr",
            str(args.lr),
            "--target-fpr",
            str(args.target_fpr),
            "--test-memory",
            args.test_memory,
        ]
        if args.resume:
            ablation_command.append("--resume")
        if args.dry_run:
            ablation_command.append("--dry-run")
        run(ablation_command, dry_run=False)

        if not args.dry_run:
            experiment_summaries[experiment] = load_json(
                results / "summary.json"
            )

    if args.dry_run:
        return

    source_manifest = load_json(raw / "source_manifest.json")
    combined = {
        "suite": "granomaly_nrf_smf",
        "source": source_manifest,
        "experiments": args.experiments,
        "seeds": args.seeds,
        "models": args.models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "learning_rate": args.lr,
        "target_fpr": args.target_fpr,
        "test_memory": args.test_memory,
        "resume": args.resume,
        "provenance": runtime_provenance(ROOT),
        "results": experiment_summaries,
    }

    workspace.mkdir(parents=True, exist_ok=True)
    with open(
        workspace / "suite_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(combined, handle, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                name: summary.get("aggregate", {})
                for name, summary in experiment_summaries.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
