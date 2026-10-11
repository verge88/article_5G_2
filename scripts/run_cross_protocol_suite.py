#!/usr/bin/env python3
"""Prepare and run the FedSigGraph cross-protocol ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benign = parser.add_mutually_exclusive_group(required=True)
    benign.add_argument("--benign-csv")
    benign.add_argument("--benign-dir")
    benign.add_argument("--benign-pcap")
    test = parser.add_mutually_exclusive_group(required=True)
    test.add_argument("--test-csv")
    test.add_argument("--test-dir")
    test.add_argument("--test-pcap")

    parser.add_argument("--workspace", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--role-map")
    parser.add_argument("--test-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--attack-type")
    parser.add_argument(
        "--pcap-display-filter",
        default="http2||ngap||pfcp",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
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
        "--test-history-mode",
        choices=["reset", "auto", "continuous"],
        default="reset",
        help=(
            "Cross-source experiments default to reset so independent captures "
            "never share temporal memory or explicit context history."
        ),
    )
    parser.add_argument(
        "--allow-zero-cross-protocol",
        action="store_true",
        help=(
            "Run even when the prepared dataset cannot support the "
            "cross-protocol continuation claim."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    prepared = workspace / "prepared"
    results = workspace / "results"
    raw = workspace / "raw"

    benign_csv = args.benign_csv
    test_csv = args.test_csv

    if args.benign_pcap:
        benign_csv = str(raw / "benign_correlated.csv")
        run(
            [
                sys.executable,
                "scripts/export_correlated_pcap.py",
                "--pcap",
                args.benign_pcap,
                "--output",
                benign_csv,
                "--display-filter",
                args.pcap_display_filter,
            ],
            dry_run=args.dry_run,
        )

    if args.test_pcap:
        test_csv = str(raw / "test_correlated.csv")
        run(
            [
                sys.executable,
                "scripts/export_correlated_pcap.py",
                "--pcap",
                args.test_pcap,
                "--output",
                test_csv,
                "--display-filter",
                args.pcap_display_filter,
            ],
            dry_run=args.dry_run,
        )

    prepare = [
        sys.executable,
        "scripts/prepare_signaling_parser.py",
        "--output",
        str(prepared),
        "--deployment-id",
        args.deployment_id,
        "--test-history-mode",
        args.test_history_mode,
    ]
    if benign_csv:
        prepare += ["--benign-csv", benign_csv]
    else:
        prepare += ["--benign-dir", args.benign_dir]
    if test_csv:
        prepare += ["--test-csv", test_csv]
    else:
        prepare += ["--test-dir", args.test_dir]
    if args.role_map:
        prepare += ["--role-map", args.role_map]
    if args.test_label is not None:
        prepare += ["--test-label", str(args.test_label)]
    if args.attack_type:
        prepare += ["--attack-type", args.attack_type]

    run(prepare, dry_run=args.dry_run)

    if not args.dry_run:
        with open(prepared / "manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        eligible = bool(
            manifest.get("claim_eligibility", {}).get(
                "cross_protocol_continuation",
                False,
            )
        )
        if not eligible and not args.allow_zero_cross_protocol:
            coverage = manifest.get("context_coverage", {})
            raise SystemExit(
                "prepared dataset has insufficient explicit cross-protocol "
                "coverage for the continuation claim; use "
                "--allow-zero-cross-protocol only for non-claim debugging. "
                f"coverage={coverage}"
            )

    ablation = [
        sys.executable,
        "scripts/run_ablation.py",
        "--data",
        str(prepared),
        "--output-dir",
        str(results),
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
    if args.dry_run:
        ablation.append("--dry-run")
    run(ablation, dry_run=False)

    if args.dry_run:
        return

    with open(prepared / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(results / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    suite_summary = {
        "suite": "cross_protocol_signaling",
        "deployment_id": args.deployment_id,
        "context_coverage": manifest.get("context_coverage", {}),
        "claim_eligibility": manifest.get("claim_eligibility", {}),
        "aggregate": summary.get("aggregate", {}),
    }
    workspace.mkdir(parents=True, exist_ok=True)
    with open(
        workspace / "suite_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(suite_summary, handle, indent=2, sort_keys=True)

    print(json.dumps(suite_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
