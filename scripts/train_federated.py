#!/usr/bin/env python3
"""Run framework-independent FedAvg/FedProx training over prepared clients.

Each client keeps deployment-local temporal state inside its TemporalHEATRunner.
Only trainable shared parameters are exported and aggregated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fedsiggraph.federated import (
    extract_shared_parameters,
    load_shared_parameters,
    weighted_average,
)
from fedsiggraph.training.temporal_heat import TemporalHEATRunner


@dataclass
class ClientState:
    name: str
    data_dir: Path
    train: object
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
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox"],
        default="fedprox",
    )
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--token-mask-prob", type=float, default=0.05)
    parser.add_argument("--continuation-loss-weight", type=float, default=0.25)
    parser.add_argument("--cross-protocol-weight", type=float, default=2.0)
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

        runner = TemporalHEATRunner(
            train,
            preprocessing=preprocessing,
            total_capacity=max(int(train.num_events), 1),
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
        )
        states.append(
            ClientState(
                name=data_dir.name,
                data_dir=data_dir,
                train=train,
                preprocessing=preprocessing,
                manifest=manifest,
                runner=runner,
            )
        )

    if len(states) < 2:
        raise ValueError("federated training requires at least two clients")
    return states


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.local_epochs <= 0:
        raise ValueError("rounds and local-epochs must be positive")
    if args.strategy == "fedprox" and args.prox_mu < 0:
        raise ValueError("prox-mu must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients = build_clients(args, device)
    global_parameters = extract_shared_parameters(clients[0].runner.model)

    history: list[dict[str, object]] = []
    for round_index in range(1, args.rounds + 1):
        client_updates = []
        client_metrics = []

        for client in clients:
            load_shared_parameters(client.runner.model, global_parameters)
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
                    batch_size=min(512, max(int(client.train.num_events), 1)),
                    prox_reference=(
                        global_parameters if args.strategy == "fedprox" else None
                    ),
                    prox_mu=(args.prox_mu if args.strategy == "fedprox" else 0.0),
                )
                losses.append(loss)
                endpoint_coverages.append(endpoint_coverage)
                procedure_coverages.append(procedure_coverage)
                context_coverages.append(context_coverage)
                cross_coverages.append(cross_coverage)

            shared = extract_shared_parameters(client.runner.model)
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
                }
            )

        global_parameters = weighted_average(client_updates)
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output.with_suffix(".pt")
    torch.save(global_parameters, weights_path)

    result = {
        "model": "federated_temporal_heat",
        "strategy": args.strategy,
        "prox_mu": args.prox_mu if args.strategy == "fedprox" else 0.0,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "seed": args.seed,
        "device": str(device),
        "clients": [client.name for client in clients],
        "shared_parameter_file": str(weights_path),
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
