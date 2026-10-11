#!/usr/bin/env python3
"""Train TemporalHEAT with semantic hard negatives and context continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from fedsiggraph.anomaly.calibration import CompositeScoreCalibrator
from fedsiggraph.training.metadata import load_test_metadata
from fedsiggraph.training.metrics import (
    attack_type_breakdown,
    calibrate_threshold,
    evaluate_masked_scores,
    evaluate_scores,
    temporal_duration_ns,
)
from fedsiggraph.training.temporal_heat import (
    TemporalHEATRunner,
    calibrator_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument("--token-mask-prob", type=float, default=0.05)
    parser.add_argument(
        "--continuation-loss-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--continuation-score-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cross-protocol-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--continuation-history-mode",
        choices=["immediate", "sequence"],
        default="sequence",
    )
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--result-model-name",
        help="Optional stable model label used by multi-run aggregation.",
    )
    parser.add_argument(
        "--allow-debug-subset",
        action="store_true",
        help="Permit explicitly non-research debug subsets.",
    )
    return parser.parse_args()


def load_temporal(path: Path):
    return torch.load(path, weights_only=False)


def main() -> None:
    args = parse_args()
    if args.continuation_score_weight < 0:
        raise ValueError("continuation_score_weight must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data)
    train = load_temporal(data_dir / "train.pt")
    val = load_temporal(data_dir / "val.pt")
    calibration = load_temporal(data_dir / "calibration.pt")
    test = load_temporal(data_dir / "test.pt")

    with open(
        data_dir / "preprocessing.json", "r", encoding="utf-8"
    ) as handle:
        preprocessing = json.load(handle)
    with open(
        data_dir / "manifest.json", "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("debug_subset") and not args.allow_debug_subset:
        raise RuntimeError(
            "debug_subset requires --allow-debug-subset; "
            "its metrics are not valid research results"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    capacity = int(
        train.num_events
        + val.num_events
        + calibration.num_events
        + test.num_events
    )
    runner = TemporalHEATRunner(
        train,
        preprocessing=preprocessing,
        total_capacity=max(capacity, 1),
        neighbors=args.neighbors,
        lr=args.lr,
        event_dim=args.event_dim,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        seed=args.seed,
        token_mask_prob=args.token_mask_prob,
        continuation_loss_weight=args.continuation_loss_weight,
        cross_protocol_weight=args.cross_protocol_weight,
        continuation_history_mode=args.continuation_history_mode,
        device=device,
    )

    losses: list[float] = []
    endpoint_coverages: list[float] = []
    procedure_coverages: list[float] = []
    context_coverages: list[float] = []
    cross_coverages: list[float] = []

    for epoch in range(1, args.epochs + 1):
        (
            loss,
            endpoint_coverage,
            procedure_coverage,
            context_coverage,
            cross_coverage,
        ) = runner.train_epoch(train, args.batch_size)
        losses.append(loss)
        endpoint_coverages.append(endpoint_coverage)
        procedure_coverages.append(procedure_coverage)
        context_coverages.append(context_coverage)
        cross_coverages.append(cross_coverage)
        print(
            f"epoch={epoch:03d} loss={loss:.6f} "
            f"endpoint_negative_coverage={endpoint_coverage:.4f} "
            f"procedure_negative_coverage={procedure_coverage:.4f} "
            f"context_coverage={context_coverage:.4f} "
            f"cross_protocol_coverage={cross_coverage:.4f}"
        )

    # Rebuild evaluation memory once with the *final* frozen model weights.
    # The last training epoch observed events while weights were still changing,
    # so carrying that state directly into validation would make evaluation
    # depend on optimizer trajectory rather than only on the final model.
    def warm_to_pretest_state():
        runner.reset_state()
        runner.score_stream(train, observe="always")
        runner.score_stream(val, observe="always")
        return runner.score_stream(calibration, observe="always")

    calibration_components = warm_to_pretest_state()
    if np.any(calibration_components.labels == 1):
        raise RuntimeError("calibration data contains attack labels")

    base_calibrator = CompositeScoreCalibrator.fit(
        calibration_components.base,
        calibration_components.continuation,
        calibration_components.context_mask,
        continuation_weight=0.0,
    )
    base_calibration_scores = base_calibrator.transform(
        calibration_components.base,
        calibration_components.continuation,
        calibration_components.context_mask,
    )
    base_threshold = calibrate_threshold(
        base_calibration_scores,
        args.target_fpr,
    )

    score_calibrator = CompositeScoreCalibrator.fit(
        calibration_components.base,
        calibration_components.continuation,
        calibration_components.context_mask,
        continuation_weight=args.continuation_score_weight,
    )
    calibration_scores = score_calibrator.transform(
        calibration_components.base,
        calibration_components.continuation,
        calibration_components.context_mask,
    )
    threshold = calibrate_threshold(
        calibration_scores, args.target_fpr
    )

    observe_policy = {
        "frozen": "never",
        "safe": "safe",
        "blind": "always",
    }[args.test_memory]

    if manifest.get("test_history_mode") == "reset":
        runner.reset_state()
    test_components = runner.score_stream(
        test,
        observe=observe_policy,
        calibrator=score_calibrator,
        threshold=threshold if observe_policy == "safe" else None,
    )
    test_scores = score_calibrator.transform(
        test_components.base,
        test_components.continuation,
        test_components.context_mask,
    )

    # Base-only scoring gets its own causal trajectory. This matters for the
    # safe policy because accept/reject decisions change future TGN memory.
    warm_to_pretest_state()
    if manifest.get("test_history_mode") == "reset":
        runner.reset_state()
    base_test_components = runner.score_stream(
        test,
        observe=observe_policy,
        calibrator=base_calibrator,
        threshold=base_threshold if observe_policy == "safe" else None,
    )
    base_test_scores = base_calibrator.transform(
        base_test_components.base,
        base_test_components.continuation,
        base_test_components.context_mask,
    )
    if not np.array_equal(
        base_test_components.labels,
        test_components.labels,
    ):
        raise RuntimeError("base/composite test streams are not label-aligned")

    metadata = load_test_metadata(
        data_dir / manifest.get(
            "test_metadata_file", "test_metadata.csv"
        ),
        expected_labels=test_components.labels,
    )

    duration_ns = temporal_duration_ns(
        test,
        time_unit_ns=int(manifest.get("time_unit_ns", 1_000)),
    )
    base_metrics = evaluate_scores(
        base_test_components.labels,
        base_test_scores,
        base_threshold,
        duration_ns=duration_ns,
    )
    composite_metrics = evaluate_scores(
        test_components.labels,
        test_scores,
        threshold,
        duration_ns=duration_ns,
    )
    base_context_metrics = evaluate_masked_scores(
        base_test_components.labels,
        base_test_scores,
        base_threshold,
        base_test_components.context_mask,
    )
    base_cross_metrics = evaluate_masked_scores(
        base_test_components.labels,
        base_test_scores,
        base_threshold,
        base_test_components.cross_protocol_mask,
    )
    composite_context_metrics = evaluate_masked_scores(
        test_components.labels,
        test_scores,
        threshold,
        test_components.context_mask,
    )
    composite_cross_metrics = evaluate_masked_scores(
        test_components.labels,
        test_scores,
        threshold,
        test_components.cross_protocol_mask,
    )

    result = {
        "research_metrics_valid": not bool(manifest.get("debug_subset")),
        "model": (
            args.result_model_name
            or "temporal_heat_context_continuation"
        ),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "learning_rate": args.lr,
        "event_dim": args.event_dim,
        "memory_dim": args.memory_dim,
        "time_dim": args.time_dim,
        "embedding_dim": args.embedding_dim,
        "heads": args.heads,
        "token_mask_prob": args.token_mask_prob,
        "continuation_loss_weight": args.continuation_loss_weight,
        "continuation_score_weight": args.continuation_score_weight,
        "cross_protocol_weight": args.cross_protocol_weight,
        "continuation_history_mode": args.continuation_history_mode,
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "base_threshold": base_threshold,
        "base_score_calibration": calibrator_to_dict(base_calibrator),
        "score_calibration": calibrator_to_dict(score_calibrator),
        "test_memory": args.test_memory,
        "test_history_mode": manifest.get("test_history_mode"),
        "train_loss_final": losses[-1] if losses else None,
        "endpoint_negative_coverage_final": (
            endpoint_coverages[-1] if endpoint_coverages else None
        ),
        "procedure_negative_coverage_final": (
            procedure_coverages[-1] if procedure_coverages else None
        ),
        "context_coverage_train_final": (
            context_coverages[-1] if context_coverages else None
        ),
        "cross_protocol_coverage_train_final": (
            cross_coverages[-1] if cross_coverages else None
        ),
        "context_coverage_test": (
            float(test_components.context_mask.mean())
            if test_components.context_mask.size
            else 0.0
        ),
        "cross_protocol_coverage_test": (
            float(test_components.cross_protocol_mask.mean())
            if test_components.cross_protocol_mask.size
            else 0.0
        ),
        "metrics": composite_metrics,
        "scope_metrics": {
            "explicit_context": composite_context_metrics,
            "cross_protocol": composite_cross_metrics,
        },
        "score_ablation": {
            "memory_trajectory": (
                "independent_causal_replay"
                if args.test_memory == "safe"
                else "equivalent_policy_replay"
            ),
            "base": {
                "threshold": base_threshold,
                "metrics": base_metrics,
                "explicit_context": base_context_metrics,
                "cross_protocol": base_cross_metrics,
            },
            "composite": {
                "threshold": threshold,
                "metrics": composite_metrics,
                "explicit_context": composite_context_metrics,
                "cross_protocol": composite_cross_metrics,
            },
        },
        "per_attack": attack_type_breakdown(
            metadata.attack_types,
            test_components.labels,
            test_scores,
            threshold,
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
