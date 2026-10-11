#!/usr/bin/env python3
"""Prepare 5G-Signaling-Parser exports for cross-protocol experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fedsiggraph.data import SignalingParserAdapter, fit_preprocessing
from fedsiggraph.data.signaling_parser import apply_role_map, infer_role_map
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
    parser.add_argument(
        "--infer-roles",
        action="store_true",
        help=(
            "Fill UNKNOWN_NF roles from NGAP/PFCP direction semantics. "
            "Explicit role-map/hostname roles are preserved."
        ),
    )
    parser.add_argument("--benign-pattern", default="*_m_stats.csv")
    parser.add_argument("--test-pattern", default="*_m_stats.csv")
    parser.add_argument("--test-label", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--attack-window",
        nargs=2,
        metavar=("START_SECONDS", "END_SECONDS"),
        help=(
            "Label only events inside this inclusive source-time interval as "
            "attack; events outside are benign. Cannot be combined with "
            "--test-label."
        ),
    )
    parser.add_argument("--attack-type")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--test-history-mode",
        choices=["auto", "reset", "continuous"],
        default="auto",
        help=(
            "Whether test events inherit development temporal/context history. "
            "Use reset for independent captures or cross-dataset evaluation."
        ),
    )
    return parser.parse_args()


def load_role_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("role-map must contain a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory(
    *,
    csv_path: str | None,
    directory: str | None,
    pattern: str,
) -> list[dict[str, object]]:
    if csv_path:
        paths = [Path(csv_path)]
    else:
        paths = sorted(Path(directory).glob(pattern))
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def seconds_to_ns(value: str) -> int:
    try:
        return int(Decimal(str(value)) * Decimal("1000000000"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid attack-window time: {value!r}") from exc


def apply_attack_window(
    events,
    *,
    start_ns: int,
    end_ns: int,
    attack_type: str | None,
):
    if end_ns < start_ns:
        raise ValueError("attack-window end must be >= start")
    return [
        event.synthetic(
            label=1 if start_ns <= event.timestamp_ns <= end_ns else 0,
            attack_type=(
                attack_type
                if start_ns <= event.timestamp_ns <= end_ns
                else None
            ),
        )
        for event in events
    ]


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


def coverage(data, *, sbi_protocol_id: int) -> dict[str, float]:
    cross = data.cross_protocol_mask
    sbi_cross = cross & (
        (data.protocol_id == sbi_protocol_id)
        | (data.prev_protocol_id == sbi_protocol_id)
    )
    return {
        "explicit_context_history": rate(data.context_history_mask),
        "cross_protocol_transition": rate(cross),
        "sbi_cross_protocol_transition": rate(sbi_cross),
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
    if args.attack_window is not None and args.test_label is not None:
        raise ValueError("--attack-window cannot be combined with --test-label")

    test_events = load_events(
        adapter,
        csv_path=args.test_csv,
        directory=args.test_dir,
        pattern=args.test_pattern,
        label=(0 if args.attack_window is not None else args.test_label),
        attack_type=(
            None if args.attack_window is not None else args.attack_type
        ),
    )
    if args.attack_window is not None:
        start_ns, end_ns = (
            seconds_to_ns(value) for value in args.attack_window
        )
        test_events = apply_attack_window(
            test_events,
            start_ns=start_ns,
            end_ns=end_ns,
            attack_type=args.attack_type,
        )

    inferred_roles = {"benign": {}, "test": {}}
    if args.infer_roles:
        inferred_roles["benign"] = infer_role_map(benign_events)
        benign_events = apply_role_map(
            benign_events,
            inferred_roles["benign"],
        )
        inferred_roles["test"] = infer_role_map(test_events)
        test_events = apply_role_map(
            test_events,
            inferred_roles["test"],
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
    auto_contiguous = bool(splits.test and development_history) and (
        min(event.timestamp_ns for event in splits.test)
        >= max(event.timestamp_ns for event in development_history)
    )
    if args.test_history_mode == "auto":
        test_is_contiguous = auto_contiguous
    elif args.test_history_mode == "continuous":
        if not auto_contiguous:
            raise ValueError(
                "continuous test history requested but test timestamps precede "
                "the development history"
            )
        test_is_contiguous = True
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

    sbi_protocol_id = fitted.vocab.protocol.encode("SBI")
    context_coverage = {
        "train": coverage(train_data, sbi_protocol_id=sbi_protocol_id),
        "val": coverage(val_data, sbi_protocol_id=sbi_protocol_id),
        "calibration": coverage(
            calibration_data,
            sbi_protocol_id=sbi_protocol_id,
        ),
        "test": coverage(test_data, sbi_protocol_id=sbi_protocol_id),
    }
    manifest = {
        "source_adapter": "5G-Signaling-Parser",
        "inferred_roles": inferred_roles,
        "source_files": {
            "benign": source_inventory(
                csv_path=args.benign_csv,
                directory=args.benign_dir,
                pattern=args.benign_pattern,
            ),
            "test": source_inventory(
                csv_path=args.test_csv,
                directory=args.test_dir,
                pattern=args.test_pattern,
            ),
        },
        "deployment_id": args.deployment_id,
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "calibration": len(splits.calibration),
            "test": len(splits.test),
        },
        "test_history_mode": "continuous" if test_is_contiguous else "reset",
        "test_history_mode_requested": args.test_history_mode,
        "test_label_policy": (
            {
                "mode": "attack_window",
                "start_seconds": args.attack_window[0],
                "end_seconds": args.attack_window[1],
                "attack_type": args.attack_type,
            }
            if args.attack_window is not None
            else {
                "mode": "uniform",
                "label": args.test_label,
                "attack_type": args.attack_type,
            }
        ),
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
        "context_history_size": int(
            train_data.context_sequence_mask.size(1)
        ),
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
