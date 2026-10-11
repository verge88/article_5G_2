#!/usr/bin/env python3
"""Prepare Granomaly captures for FedSigGraph experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fedsiggraph.data import GranomalyAdapter, fit_preprocessing
from fedsiggraph.data.pyg import to_temporal_data
from fedsiggraph.evaluation.eligibility import research_claim_eligibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True, help="Path to benign Granomaly CSV.")
    parser.add_argument("--attack", required=True, help="Path to attack experiment CSV.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--deployment-id", default="granomaly-open5gs")
    parser.add_argument(
        "--role-map",
        help="Optional JSON object mapping local entity/IP to NF role.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def load_role_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("role-map must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def unknown_rate(values: list[int], unk_id: int = 1) -> float:
    if not values:
        return 0.0
    return sum(value == unk_id for value in values) / len(values)


def node_unknown_rate(events, node_index) -> float:
    identities: list[str] = []
    for event in events:
        identities.extend([event.src_entity, event.dst_entity])
    if not identities:
        return 0.0
    return sum(
        node_index.encode(item) == node_index.unk_id
        for item in identities
    ) / len(identities)


def tensor_rate(values) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().mean().item())


def split_coverage(data) -> dict[str, float]:
    return {
        "explicit_context_history": tensor_rate(
            data.context_history_mask
        ),
        "cross_protocol_transition": tensor_rate(
            data.cross_protocol_mask
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = GranomalyAdapter(
        args.deployment_id,
        role_by_entity=load_role_map(args.role_map),
    )
    benign_events = adapter.from_csv(args.normal)
    test_events = adapter.from_csv(args.attack)

    fitted = fit_preprocessing(
        benign_events,
        test_events,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    splits = fitted.splits

    development_history = splits.train + splits.val + splits.calibration
    if splits.test and development_history:
        test_is_contiguous = (
            min(event.timestamp_ns for event in splits.test)
            >= max(event.timestamp_ns for event in development_history)
        )
    else:
        test_is_contiguous = False

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

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(calibration_data, output_dir / "calibration.pt")
    torch.save(test_data, output_dir / "test.pt")

    # Evaluation-only sidecar. Attack family labels never enter tensors,
    # vocabulary fitting, scaling, training, or threshold calibration.
    with open(
        output_dir / "test_metadata.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
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
                    "attack_type": event.attack_type or "normal",
                    "protocol": event.protocol,
                    "relation": event.relation,
                    "procedure": event.procedure,
                }
            )

    test_protocol_ids = [
        fitted.vocab.protocol.encode(event.protocol) for event in splits.test
    ]
    test_procedure_ids = [
        fitted.vocab.procedure.encode(event.procedure) for event in splits.test
    ]
    test_dst_role_ids = [
        fitted.vocab.dst_role.encode(event.dst_role) for event in splits.test
    ]

    preprocessing = {
        "vocab_sizes": {
            "src_role": len(fitted.vocab.src_role),
            "dst_role": len(fitted.vocab.dst_role),
            "protocol": len(fitted.vocab.protocol),
            "relation": len(fitted.vocab.relation),
            "procedure": len(fitted.vocab.procedure),
            "message_type": len(fitted.vocab.message_type),
            "direction": len(fitted.vocab.direction),
        },
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
        "node_type_by_id": list(fitted.node_index.node_type_by_id),
        "role_by_id": list(fitted.node_index.role_by_id),
        "node_type_ids": fitted.node_index.node_type_ids(),
        "num_node_types": fitted.node_index.num_node_types,
        "numeric_median": fitted.numeric_encoder.stats.median.tolist(),
        "numeric_scale": fitted.numeric_encoder.stats.scale.tolist(),
        "time_origin_ns": fitted.time_origin_ns,
        "test_time_origin_ns": test_origin_ns,
    }
    with open(
        output_dir / "preprocessing.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(preprocessing, handle, indent=2, sort_keys=True)

    context_coverage = {
        "train": split_coverage(train_data),
        "val": split_coverage(val_data),
        "calibration": split_coverage(calibration_data),
        "test": split_coverage(test_data),
    }
    claim_eligibility = research_claim_eligibility(context_coverage)

    manifest = {
        "deployment_id": args.deployment_id,
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "calibration": len(splits.calibration),
            "test": len(splits.test),
        },
        "num_train_nodes_including_unk": fitted.node_index.num_nodes,
        "num_node_types": fitted.node_index.num_node_types,
        "test_history_mode": (
            "continuous" if test_is_contiguous else "reset"
        ),
        "context_coverage": context_coverage,
        "claim_eligibility": claim_eligibility,
        "test_unknown_rates": {
            "node_identity": node_unknown_rate(
                splits.test, fitted.node_index
            ),
            "protocol": unknown_rate(test_protocol_ids),
            "procedure": unknown_rate(test_procedure_ids),
            "dst_role": unknown_rate(test_dst_role_ids),
        },
        "test_metadata_file": "test_metadata.csv",
        "context_history_size": int(
            train_data.context_sequence_mask.size(1)
        ),
        "time_unit": "microseconds",
        "time_unit_ns": 1_000,
    }
    with open(
        output_dir / "manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
