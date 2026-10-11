#!/usr/bin/env python3
"""Prepare 5G-Signaling-Parser exports for cross-protocol experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fedsiggraph.data import SignalingParserAdapter, fit_preprocessing
from fedsiggraph.data.pyg import to_temporal_data
from fedsiggraph.evaluation.eligibility import research_claim_eligibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benign = parser.add_mutually_exclusive_group(required=True)
    benign.add_argument("--benign-csv")
    benign.add_argument("--benign-dir")
    test = parser.add_mutually_exclusive_group(required=True)
    test.add_argument("--test-csv")
    test.add_argument("--test-dir")

    parser.add_argument("--output", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--role-map")
    parser.add_argument("--benign-pattern", default="*_m_stats.csv")
    parser.add_argument("--test-pattern", default="*_m_stats.csv")
    parser.add_argument("--test-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--attack-type")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def load_role_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("role-map must contain a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def load_events(
    adapter: SignalingParserAdapter,
    *,
    csv_path: str | None,
    directory: str | None,
    pattern: str,
    label: int | None,
    attack_type: str | None,
):
    if csv_path:
        return adapter.from_csv(
            csv_path,
            label=label,
            attack_type=attack_type,
        )
    return adapter.from_directory(
        directory,
        pattern=pattern,
        label=label,
        attack_type=attack_type,
    )


def rate(mask) -> float:
    if int(mask.numel()) == 0:
        return 0.0
    return float(mask.float().mean().item())


def coverage(data) -> dict[str, float]:
    return {
        "explicit_context_history": rate(data.context_history_mask),
        "cross_protocol_transition": rate(data.cross_protocol_mask),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    adapter = SignalingParserAdapter(
        args.deployment_id,
        role_by_entity=load_role_map(args.role_map),
    )
    benign_events = load_events(
        adapter,
        csv_path=args.benign_csv,
        directory=args.benign_dir,
        pattern=args.benign_pattern,
        label=0,
        attack_type=None,
    )
    test_events = load_events(
        adapter,
        csv_path=args.test_csv,
        directory=args.test_dir,
        pattern=args.test_pattern,
        label=args.test_label,
        attack_type=args.attack_type,
    )

    fitted = fit_preprocessing(
        benign_events,
        test_events,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    splits = fitted.splits
    development_history = splits.train + splits.val + splits.calibration

    # Separate captures may use unrelated clock epochs. Only warm test context
    # from development state when timestamps are chronologically contiguous.
    test_is_contiguous = bool(splits.test and development_history) and (
        min(event.timestamp_ns for event in splits.test)
        >= max(event.timestamp_ns for event in development_history)
    )
    test_history = development_history if test_is_contiguous else ()
    test_origin_ns = (
        fitted.time_origin_ns
        if test_is_contiguous or not splits.test
        else min(event.timestamp_ns for event in splits.test)
    )

    train_data = to_temporal_data(
        splits.train,
        vocab=fitted.vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
        time_origin_ns=fitted.time_origin_ns,
    )
    val_data = to_temporal_data(
        splits.val,
        vocab=fitted.vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
        time_origin_ns=fitted.time_origin_ns,
        history=splits.train,
    )
    calibration_data = to_temporal_data(
        splits.calibration,
        vocab=fitted.vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
        time_origin_ns=fitted.time_origin_ns,
        history=splits.train + splits.val,
    )
    test_data = to_temporal_data(
        splits.test,
        vocab=fitted.vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
        time_origin_ns=test_origin_ns,
        history=test_history,
    )

    import torch

    torch.save(train_data, output / "train.pt")
    torch.save(val_data, output / "val.pt")
    torch.save(calibration_data, output / "calibration.pt")
    torch.save(test_data, output / "test.pt")

    with open(output / "test_metadata.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_index",
                "event_id",
                "timestamp_ns",
                "label",
                "attack_type",
                "protocol",
                "relation",
                "procedure",
                "has_ue_context",
                "has_session_context",
            ],
        )
        writer.writeheader()
        for event_index, event in enumerate(splits.test):
            writer.writerow(
                {
                    "event_index": event_index,
                    "event_id": event.event_id,
                    "timestamp_ns": event.timestamp_ns,
                    "label": -1 if event.label is None else event.label,
                    "attack_type": event.attack_type or "unknown",
                    "protocol": event.protocol,
                    "relation": event.relation,
                    "procedure": event.procedure,
                    "has_ue_context": int(event.ue_context is not None),
                    "has_session_context": int(event.session_context is not None),
                }
            )

    context_coverage = {
        "train": coverage(train_data),
        "val": coverage(val_data),
        "calibration": coverage(calibration_data),
        "test": coverage(test_data),
    }
    manifest = {
        "source_adapter": "5G-Signaling-Parser",
        "deployment_id": args.deployment_id,
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "calibration": len(splits.calibration),
            "test": len(splits.test),
        },
        "test_history_mode": "continuous" if test_is_contiguous else "reset",
        "context_coverage": context_coverage,
        "claim_eligibility": research_claim_eligibility(context_coverage),
        "explicit_context_counts": {
            "benign_ue": sum(e.ue_context is not None for e in benign_events),
            "benign_session": sum(
                e.session_context is not None for e in benign_events
            ),
            "test_ue": sum(e.ue_context is not None for e in test_events),
            "test_session": sum(
                e.session_context is not None for e in test_events
            ),
        },
        "time_unit": "microseconds",
        "time_unit_ns": 1_000,
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    preprocessing = {
        "vocab": {
            "src_role": fitted.vocab.src_role.token_to_id,
            "dst_role": fitted.vocab.dst_role.token_to_id,
            "protocol": fitted.vocab.protocol.token_to_id,
            "relation": fitted.vocab.relation.token_to_id,
            "procedure": fitted.vocab.procedure.token_to_id,
            "message_type": fitted.vocab.message_type.token_to_id,
            "direction": fitted.vocab.direction.token_to_id,
        },
        "node_index": fitted.node_index.entity_to_id,
        "numeric_median": fitted.numeric_encoder.stats.median.tolist(),
        "numeric_scale": fitted.numeric_encoder.stats.scale.tolist(),
        "time_origin_ns": fitted.time_origin_ns,
        "test_time_origin_ns": test_origin_ns,
    }
    with open(output / "preprocessing.json", "w", encoding="utf-8") as handle:
        json.dump(preprocessing, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
