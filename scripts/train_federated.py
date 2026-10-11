#!/usr/bin/env python3
"""Run framework-independent FedAvg/FedProx training over prepared clients.

Each client keeps deployment-local temporal state inside its TemporalHEATRunner.
Only trainable shared parameters are exported and aggregated. After the final
round, every client performs its own benign calibration and test evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fedsiggraph.anomaly.calibration import CompositeScoreCalibrator
from fedsiggraph.federated import (
    coordinate_median,
    extract_shared_parameters,
    load_shared_parameters,
    trimmed_mean,
    weighted_average,
    poison_update,
)
from fedsiggraph.training.metadata import load_test_metadata
from fedsiggraph.training.metrics import (
    attack_type_breakdown,
    calibrate_threshold,
    evaluate_scores,
    temporal_duration_ns,
)
from fedsiggraph.training.temporal_heat import (
    TemporalHEATRunner,
    calibrator_to_dict,
)


@dataclass
class ClientState:
    name: str
    data_dir: Path
    train: object
    val: object
    calibration: object
    test: object
    preprocessing: dict
    manifest: dict
    runner: TemporalHEATRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients",
        nargs="+",
        required=True,
        help="Prepared federated client directories.",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox"],
        default="fedprox",
    )
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument(
        "--aggregation",
        choices=["weighted_mean", "coordinate_median", "trimmed_mean"],
        default="weighted_mean",
    )
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument(
        "--poison-client",
        action="append",
        default=[],
        help="Client directory name whose model update is perturbed.",
    )
    parser.add_argument(
        "--poison-mode",
        choices=["model_replacement", "sign_flip"],
        default="sign_flip",
    )
    parser.add_argument("--poison-factor", type=float, default=5.0)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--local-adapter-dim",
        type=int,
        default=16,
        help="Private residual adapter bottleneck; 0 disables personalization.",
    )
    parser.add_argument("--token-mask-prob", type=float, default=0.05)
    parser.add_argument("--continuation-loss-weight", type=float, default=0.25)
    parser.add_argument("--continuation-score-weight", type=float, default=1.0)
    parser.add_argument("--cross-protocol-weight", type=float, default=2.0)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_temporal(path: Path):
    return torch.load(path, weights_only=False)


def build_clients(args: argparse.Namespace, device: torch.device) -> list[ClientState]:
    states: list[ClientState] = []
    expected_fingerprint: str | None = None
    expected_vocab_sizes: dict[str, int] | None = None

    for index, value in enumerate(args.clients):
        data_dir = Path(value)
        train = load_temporal(data_dir / "train.pt")
        val = load_temporal(data_dir / "val.pt")
        calibration = load_temporal(data_dir / "calibration.pt")
        test = load_temporal(data_dir / "test.pt")
        preprocessing = load_json(data_dir / "preprocessing.json")
        manifest = load_json(data_dir / "manifest.json")

        if not preprocessing.get("federated"):
            raise ValueError(
                f"{data_dir} was not prepared with prepare_federated_client.py"
            )

        fingerprint = str(preprocessing["semantic_schema_fingerprint"])
        vocab_sizes = {
            key: int(size)
            for key, size in preprocessing["vocab_sizes"].items()
        }
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
            expected_vocab_sizes = vocab_sizes
        elif (
            fingerprint != expected_fingerprint
            or vocab_sizes != expected_vocab_sizes
        ):
            raise ValueError(
                "all clients must use the same federated semantic vocabulary "
                "schema and embedding sizes"
            )

        total_capacity = int(
            train.num_events + val.num_events + calibration.num_events + test.num_events
        )
        runner = TemporalHEATRunner(
            train,
            preprocessing=preprocessing,
            total_capacity=max(total_capacity, 1),
            neighbors=args.neighbors,
            lr=args.lr,
            event_dim=args.event_dim,
            memory_dim=args.memory_dim,
            time_dim=args.time_dim,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            seed=args.seed + index,
            token_mask_prob=args.token_mask_prob,
            continuation_loss_weight=args.continuation_loss_weight,
            cross_protocol_weight=args.cross_protocol_weight,
            device=device,
            local_adapter_dim=args.local_adapter_dim,
        )
        states.append(
            ClientState(
                name=data_dir.name,
                data_dir=data_dir,
                train=train,
                val=val,
                calibration=calibration,
                test=test,
                preprocessing=preprocessing,
                manifest=manifest,
                runner=runner,
            )
        )

    if len(states) < 2:
        raise ValueError("federated training requires at least two clients")
    return states


def evaluate_client(
    client: ClientState,
    global_parameters,
    args: argparse.Namespace,
) -> dict[str, object]:
    load_shared_parameters(client.runner.model, global_parameters)
    client.runner.reset_state()

    # Local benign history and calibration remain deployment-private.
    client.runner.score_stream(client.val, observe="always")
    calibration_components = client.runner.score_stream(
        client.calibration,
        observe="always",
    )
    if np.any(calibration_components.labels == 1):
        raise RuntimeError(
            f"client {client.name} calibration split contains attack labels"
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
    threshold = calibrate_threshold(calibration_scores, args.target_fpr)

    if client.manifest.get("test_history_mode") == "reset":
        client.runner.reset_state()

    observe_policy = {
        "frozen": "never",
        "safe": "safe",
        "blind": "always",
    }[args.test_memory]
    test_components = client.runner.score_stream(
        client.test,
        observe=observe_policy,
        calibrator=score_calibrator,
        threshold=threshold if observe_policy == "safe" else None,
    )
    test_scores = score_calibrator.transform(
        test_components.base,
        test_components.continuation,
        test_components.context_mask,
    )
    metadata = load_test_metadata(
        client.data_dir
        / client.manifest.get("test_metadata_file", "test_metadata.csv"),
        expected_labels=test_components.labels,
    )

    return {
        "client": client.name,
        "threshold": threshold,
        "score_calibration": calibrator_to_dict(score_calibrator),
        "test_memory": args.test_memory,
        "test_history_mode": client.manifest.get("test_history_mode"),
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
        "metrics": evaluate_scores(
            test_components.labels,
            test_scores,
            threshold,
            duration_ns=temporal_duration_ns(
                client.test,
                time_unit_ns=int(client.manifest.get("time_unit_ns", 1_000)),
            ),
        ),
        "per_attack": attack_type_breakdown(
            metadata.attack_types,
            test_components.labels,
            test_scores,
            threshold,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.local_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("rounds, local-epochs, and batch-size must be positive")
    if args.strategy == "fedprox" and args.prox_mu < 0:
        raise ValueError("prox-mu must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients = build_clients(args, device)
    known_clients = {client.name for client in clients}
    requested_poison = set(args.poison_client)
    unknown_poison = sorted(requested_poison - known_clients)
    if unknown_poison:
        raise ValueError(
            f"poison-client names do not match prepared clients: {unknown_poison}"
        )
    if requested_poison and args.poison_factor < 0:
        raise ValueError("poison-factor must be non-negative")

    global_parameters = extract_shared_parameters(clients[0].runner.model)

    history: list[dict[str, object]] = []
    for round_index in range(1, args.rounds + 1):
        client_updates = []
        client_metrics = []

        for client in clients:
            load_shared_parameters(client.runner.model, global_parameters)
            client.runner.reset_optimizer()
            losses: list[float] = []
            endpoint_coverages: list[float] = []
            procedure_coverages: list[float] = []
            context_coverages: list[float] = []
            cross_coverages: list[float] = []

            for _ in range(args.local_epochs):
                (
                    loss,
                    endpoint_coverage,
                    procedure_coverage,
                    context_coverage,
                    cross_coverage,
                ) = client.runner.train_epoch(
                    client.train,
                    batch_size=min(
                        args.batch_size,
                        max(int(client.train.num_events), 1),
                    ),
                    prox_reference=(
                        global_parameters if args.strategy == "fedprox" else None
                    ),
                    prox_mu=(
                        args.prox_mu if args.strategy == "fedprox" else 0.0
                    ),
                )
                losses.append(loss)
                endpoint_coverages.append(endpoint_coverage)
                procedure_coverages.append(procedure_coverage)
                context_coverages.append(context_coverage)
                cross_coverages.append(cross_coverage)

            shared = extract_shared_parameters(client.runner.model)
            poisoned = client.name in requested_poison
            if poisoned:
                shared = poison_update(
                    shared,
                    global_parameters,
                    mode=args.poison_mode,
                    factor=args.poison_factor,
                )
            examples = int(client.train.num_events)
            client_updates.append((shared, examples))
            client_metrics.append(
                {
                    "client": client.name,
                    "examples": examples,
                    "loss": float(losses[-1]),
                    "endpoint_negative_coverage": float(endpoint_coverages[-1]),
                    "procedure_negative_coverage": float(procedure_coverages[-1]),
                    "context_coverage": float(context_coverages[-1]),
                    "cross_protocol_coverage": float(cross_coverages[-1]),
                    "poisoned_update": poisoned,
                }
            )

        if args.aggregation == "weighted_mean":
            global_parameters = weighted_average(client_updates)
        elif args.aggregation == "coordinate_median":
            global_parameters = coordinate_median(client_updates)
        else:
            global_parameters = trimmed_mean(
                client_updates, trim_ratio=args.trim_ratio
            )

        round_summary = {
            "round": round_index,
            "clients": client_metrics,
            "weighted_mean_loss": float(
                sum(item["loss"] * item["examples"] for item in client_metrics)
                / max(sum(item["examples"] for item in client_metrics), 1)
            ),
        }
        history.append(round_summary)
        print(json.dumps(round_summary, sort_keys=True))

    evaluations = [
        evaluate_client(client, global_parameters, args)
        for client in clients
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output.with_suffix(".pt")
    torch.save(global_parameters, weights_path)

    result = {
        "model": "federated_temporal_heat",
        "strategy": args.strategy,
        "aggregation": args.aggregation,
        "trim_ratio": (
            args.trim_ratio if args.aggregation == "trimmed_mean" else None
        ),
        "prox_mu": args.prox_mu if args.strategy == "fedprox" else 0.0,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "local_adapter_dim": args.local_adapter_dim,
        "target_fpr": args.target_fpr,
        "poison_clients": sorted(requested_poison),
        "poison_mode": (
            args.poison_mode if args.poison_client else None
        ),
        "poison_factor": (
            args.poison_factor if args.poison_client else None
        ),
        "clients": [client.name for client in clients],
        "shared_parameter_file": str(weights_path),
        "history": history,
        "evaluations": evaluations,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
