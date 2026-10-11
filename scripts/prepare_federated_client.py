#!/usr/bin/env python3
"""Prepare one 5G deployment as a federated FedSigGraph client."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fedsiggraph.data import (
    GranomalyAdapter,
    fit_federated_preprocessing,
)
from fedsiggraph.data.pyg import to_temporal_data
from fedsiggraph.evaluation.eligibility import research_claim_eligibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--role-map")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--relation-buckets", type=int, default=512)
    parser.add_argument("--procedure-buckets", type=int, default=2048)
    parser.add_argument("--message-type-buckets", type=int, default=128)
    parser.add_argument(
        "--schema-version",
        default="fedsiggraph-v1",
    )
    return parser.parse_args()


def load_role_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("role-map must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


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


def node_unknown_rate(events, node_index) -> float:
    identities: list[str] = []
    for event in events:
        identities.extend([event.src_entity, event.dst_entity])
    if not identities:
        return 0.0
    unknown = sum(
        node_index.encode(entity) == node_index.unk_id
        for entity in identities
    )
    return unknown / len(identities)


def categorical_unknown_rate(events, encoder, getter) -> float:
    events = list(events)
    if not events:
        return 0.0
    unknown = sum(
        encoder.encode(getter(event)) == encoder.unk_id
        for event in events
    )
    return unknown / len(events)


def hash_collision_stats(events, encoder, getter) -> dict[str, float | int]:
    tokens = {str(getter(event)) for event in events}
    if not tokens:
        return {
            "unique_tokens": 0,
            "unique_buckets_used": 0,
            "collisions": 0,
            "collision_rate": 0.0,
        }

    ids = {encoder.encode(token) for token in tokens}
    collisions = max(0, len(tokens) - len(ids))
    return {
        "unique_tokens": len(tokens),
        "unique_buckets_used": len(ids),
        "collisions": collisions,
        "collision_rate": collisions / len(tokens),
    }


def write_test_metadata(path: Path, events) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
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
        for event_index, event in enumerate(events):
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


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    adapter = GranomalyAdapter(
        args.client_id,
        role_by_entity=load_role_map(args.role_map),
    )
    benign_events = adapter.from_csv(args.normal)
    test_events = adapter.from_csv(args.attack)

    fitted = fit_federated_preprocessing(
        benign_events,
        test_events,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        relation_buckets=args.relation_buckets,
        procedure_buckets=args.procedure_buckets,
        message_type_buckets=args.message_type_buckets,
        schema_version=args.schema_version,
    )
    splits = fitted.splits
    vocab = fitted.vocab

    development_history = (
        splits.train + splits.val + splits.calibration
    )
    if splits.test and development_history:
        test_is_contiguous = (
            min(event.timestamp_ns for event in splits.test)
            >= max(
                event.timestamp_ns
                for event in development_history
            )
        )
    else:
        test_is_contiguous = False

    test_history = development_history if test_is_contiguous else ()
    test_origin_ns = (
        fitted.time_origin_ns
        if test_is_contiguous or not splits.test
        else min(event.timestamp_ns for event in splits.test)
    )

    build = dict(
        vocab=vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
    )
    train_data = to_temporal_data(
        splits.train,
        **build,
        time_origin_ns=fitted.time_origin_ns,
    )
    val_data = to_temporal_data(
        splits.val,
        **build,
        time_origin_ns=fitted.time_origin_ns,
        history=splits.train,
    )
    calibration_data = to_temporal_data(
        splits.calibration,
        **build,
        time_origin_ns=fitted.time_origin_ns,
        history=splits.train + splits.val,
    )
    test_data = to_temporal_data(
        splits.test,
        **build,
        time_origin_ns=test_origin_ns,
        history=test_history,
    )

    import torch

    for name, data in (
        ("train", train_data),
        ("val", val_data),
        ("calibration", calibration_data),
        ("test", test_data),
    ):
        torch.save(data, output / f"{name}.pt")

    write_test_metadata(
        output / "test_metadata.csv",
        splits.test,
    )

    preprocessing = {
        "federated": True,
        "client_id": args.client_id,
        "semantic_schema_fingerprint": vocab.fingerprint(),
        "vocab_schema": vocab.config(),
        "vocab_sizes": vocab.sizes(),
        # Everything below is private deployment-local state.
        "node_index": fitted.node_index.entity_to_id,
        "node_type_by_id": list(
            fitted.node_index.node_type_by_id
        ),
        "role_by_id": list(fitted.node_index.role_by_id),
        "node_type_ids": fitted.node_index.node_type_ids(),
        "num_node_types": fitted.node_index.num_node_types,
        "numeric_median": (
            fitted.numeric_encoder.stats.median.tolist()
        ),
        "numeric_scale": (
            fitted.numeric_encoder.stats.scale.tolist()
        ),
        "time_origin_ns": fitted.time_origin_ns,
        "test_time_origin_ns": test_origin_ns,
    }
    with open(
        output / "preprocessing.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(preprocessing, handle, indent=2, sort_keys=True)

    train_events = splits.train
    context_coverage = {
        "train": split_coverage(train_data),
        "val": split_coverage(val_data),
        "calibration": split_coverage(calibration_data),
        "test": split_coverage(test_data),
    }
    claim_eligibility = research_claim_eligibility(context_coverage)

    manifest = {
        "federated": True,
        "client_id": args.client_id,
        "semantic_schema_fingerprint": vocab.fingerprint(),
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "calibration": len(splits.calibration),
            "test": len(splits.test),
        },
        "num_local_train_nodes_including_unk": (
            fitted.node_index.num_nodes
        ),
        "num_node_types": fitted.node_index.num_node_types,
        "test_history_mode": (
            "continuous" if test_is_contiguous else "reset"
        ),
        "context_coverage": context_coverage,
        "claim_eligibility": claim_eligibility,
        "hash_collision_audit_train": {
            "relation": hash_collision_stats(
                train_events, vocab.relation, lambda event: event.relation
            ),
            "procedure": hash_collision_stats(
                train_events, vocab.procedure, lambda event: event.procedure
            ),
            "message_type": hash_collision_stats(
                train_events,
                vocab.message_type,
                lambda event: event.message_type,
            ),
        },
        "test_unknown_rates": {
            "node_identity": node_unknown_rate(
                splits.test, fitted.node_index
            ),
            "src_role": categorical_unknown_rate(
                splits.test, vocab.src_role, lambda event: event.src_role
            ),
            "dst_role": categorical_unknown_rate(
                splits.test, vocab.dst_role, lambda event: event.dst_role
            ),
            "protocol": categorical_unknown_rate(
                splits.test, vocab.protocol, lambda event: event.protocol
            ),
        },
        "test_metadata_file": "test_metadata.csv",
        "context_history_size": int(
            train_data.context_sequence_mask.size(1)
        ),
        "time_unit": "microseconds",
        "time_unit_ns": 1_000,
    }
    with open(
        output / "manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
