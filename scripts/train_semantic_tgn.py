#!/usr/bin/env python3
"""Train the semantic causal TGN with train-derived hard negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from fedsiggraph.models.semantic_tgn import SemanticCausalTGN
from fedsiggraph.training.history import ObservedHistoryBank
from fedsiggraph.training.metrics import (
    attack_type_breakdown,
    calibrate_threshold,
    evaluate_scores,
    temporal_duration_ns,
)
from fedsiggraph.training.metadata import load_test_metadata
from fedsiggraph.training.negatives import (
    EndpointNegativeTable,
    ProcedureNegativeTable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument("--token-mask-prob", type=float, default=0.05)
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--allow-debug-subset",
        action="store_true",
        help="Permit explicitly non-research debug subsets.",
    )
    return parser.parse_args()


def load_temporal(path: Path):
    return torch.load(path, weights_only=False)


class SemanticRunner:
    def __init__(
        self,
        train_data,
        *,
        vocab_sizes: dict[str, int],
        total_capacity: int,
        neighbors: int,
        lr: float,
        event_dim: int,
        seed: int,
        token_mask_prob: float,
        device: torch.device,
    ) -> None:
        if not (0.0 <= token_mask_prob < 1.0):
            raise ValueError("token_mask_prob must be in [0, 1)")

        self.device = device
        self.num_nodes = int(train_data.num_nodes)
        self.event_dim = event_dim
        self.total_capacity = total_capacity
        self.token_mask_prob = token_mask_prob
        self.procedure_negatives = ProcedureNegativeTable(train_data, seed)
        self.endpoint_negatives = EndpointNegativeTable(train_data, seed + 1)

        self.model = SemanticCausalTGN(
            num_nodes=self.num_nodes,
            numeric_dim=int(train_data.msg.size(-1)),
            vocab_sizes=vocab_sizes,
            event_dim=event_dim,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.neighbor_loader = LastNeighborLoader(
            self.num_nodes,
            size=neighbors,
            device=device,
        )
        self.assoc = torch.empty(
            self.num_nodes, dtype=torch.long, device=device
        )
        self.history = ObservedHistoryBank(total_capacity, event_dim)

    def reset_state(self) -> None:
        self.model.reset_state()
        self.neighbor_loader.reset_state()
        self.history = ObservedHistoryBank(
            self.total_capacity, self.event_dim
        )

    def _embedding_context(self, nodes: torch.Tensor):
        n_id, edge_index, e_id = self.neighbor_loader(nodes)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        memory, last_update = self.model.memory(n_id)
        edge_t, edge_msg = self.history.lookup(e_id, self.device)
        z = self.model.embedding(
            memory,
            last_update,
            edge_index,
            edge_t,
            edge_msg,
        )
        return z, self.assoc

    def _masked_benign_embedding(self, batch):
        if self.token_mask_prob <= 0:
            return self.model.encode_batch(batch)

        def mask(ids: torch.Tensor) -> torch.Tensor:
            out = ids.clone()
            selected = torch.rand(
                ids.shape, device=ids.device
            ) < self.token_mask_prob
            out[selected] = 1  # reserved UNK id
            return out

        return self.model.encode_batch(
            batch,
            dst_role_id=mask(batch.dst_role_id),
            relation_id=mask(batch.relation_id),
            procedure_id=mask(batch.procedure_id),
        )

    def train_epoch(
        self,
        data,
        batch_size: int,
    ) -> tuple[float, float, float]:
        self.model.train()
        self.reset_state()
        loader = TemporalDataLoader(data, batch_size=batch_size)

        total_loss = 0.0
        total_events = 0
        endpoint_negative_events = 0
        procedure_negative_events = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            src = batch.src
            pos_dst = batch.dst
            (
                neg_dst,
                neg_dst_role_id,
                endpoint_mask,
            ) = self.endpoint_negatives.sample(batch)
            nodes = torch.cat([src, pos_dst, neg_dst]).unique()
            z, assoc = self._embedding_context(nodes)

            benign_event = self._masked_benign_embedding(batch)
            endpoint_event = self.model.encode_batch(
                batch,
                dst_role_id=neg_dst_role_id,
            )
            procedure_ids, procedure_mask = self.procedure_negatives.sample(
                batch
            )
            procedure_event = self.model.encode_batch(
                batch,
                procedure_id=procedure_ids,
            )

            benign_logits = self.model.score_embeddings(
                z[assoc[src]],
                z[assoc[pos_dst]],
                benign_event,
            )
            losses = [
                self.criterion(
                    benign_logits, torch.zeros_like(benign_logits)
                ),
            ]

            if endpoint_mask.any():
                endpoint_negative_events += int(endpoint_mask.sum())
                endpoint_logits = self.model.score_embeddings(
                    z[assoc[src[endpoint_mask]]],
                    z[assoc[neg_dst[endpoint_mask]]],
                    endpoint_event[endpoint_mask],
                )
                losses.append(
                    self.criterion(
                        endpoint_logits,
                        torch.ones_like(endpoint_logits),
                    )
                )

            if procedure_mask.any():
                procedure_logits = self.model.score_embeddings(
                    z[assoc[src[procedure_mask]]],
                    z[assoc[pos_dst[procedure_mask]]],
                    procedure_event[procedure_mask],
                )
                losses.append(
                    self.criterion(
                        procedure_logits,
                        torch.ones_like(procedure_logits),
                    )
                )
                procedure_negative_events += int(procedure_mask.sum())

            loss = torch.stack(losses).mean()
            loss.backward()
            self.optimizer.step()

            # Re-encode the clean observed event using updated parameters.
            with torch.no_grad():
                observed_msg = self.model.encode_batch(batch)
                self.model.observe(
                    src, pos_dst, batch.t, observed_msg
                )
                self.neighbor_loader.insert(src, pos_dst)
                self.history.append(batch.t, observed_msg)
                self.model.memory.detach()

            n = int(batch.num_events)
            total_loss += float(loss.detach().cpu()) * n
            total_events += n

        endpoint_coverage = endpoint_negative_events / max(total_events, 1)
        procedure_coverage = procedure_negative_events / max(total_events, 1)
        return (
            total_loss / max(total_events, 1),
            endpoint_coverage,
            procedure_coverage,
        )

    @torch.no_grad()
    def score_stream(
        self,
        data,
        *,
        observe: str,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if observe not in {"always", "never", "safe"}:
            raise ValueError(f"unknown observe policy: {observe}")
        if observe == "safe" and threshold is None:
            raise ValueError("safe observation requires a threshold")

        self.model.eval()
        loader = TemporalDataLoader(data, batch_size=1)
        scores: list[float] = []
        labels: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            nodes = torch.cat([batch.src, batch.dst]).unique()
            z, assoc = self._embedding_context(nodes)
            event_embedding = self.model.encode_batch(batch)
            logits = self.model.score_embeddings(
                z[assoc[batch.src]],
                z[assoc[batch.dst]],
                event_embedding,
            )
            score = float(logits.sigmoid().item())
            scores.append(score)
            labels.append(int(batch.y.item()))

            should_observe = (
                observe == "always"
                or (observe == "safe" and score <= float(threshold))
            )
            if should_observe:
                self.model.observe(
                    batch.src,
                    batch.dst,
                    batch.t,
                    event_embedding,
                )
                self.neighbor_loader.insert(batch.src, batch.dst)
                self.history.append(batch.t, event_embedding)
                self.model.memory.detach()

        return np.asarray(scores), np.asarray(labels)


def vocab_sizes_from_preprocessing(preprocessing: dict) -> dict[str, int]:
    if "vocab_sizes" in preprocessing:
        return {
            name: int(size)
            for name, size in preprocessing["vocab_sizes"].items()
        }
    return {
        name: len(mapping)
        for name, mapping in preprocessing["vocab"].items()
    }


def main() -> None:
    args = parse_args()
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
    with open(data_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("debug_subset") and not args.allow_debug_subset:
        raise RuntimeError(
            "debug_subset requires --allow-debug-subset; "
            "its metrics are not valid research results"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capacity = int(
        train.num_events + val.num_events + calibration.num_events
        + test.num_events
    )

    runner = SemanticRunner(
        train,
        vocab_sizes=vocab_sizes_from_preprocessing(preprocessing),
        total_capacity=max(capacity, 1),
        neighbors=args.neighbors,
        lr=args.lr,
        event_dim=args.event_dim,
        seed=args.seed,
        token_mask_prob=args.token_mask_prob,
        device=device,
    )

    losses: list[float] = []
    endpoint_coverages: list[float] = []
    procedure_coverages: list[float] = []
    for epoch in range(1, args.epochs + 1):
        loss, endpoint_coverage, procedure_coverage = runner.train_epoch(
            train, args.batch_size
        )
        losses.append(loss)
        endpoint_coverages.append(endpoint_coverage)
        procedure_coverages.append(procedure_coverage)
        print(
            f"epoch={epoch:03d} loss={loss:.6f} "
            f"endpoint_negative_coverage={endpoint_coverage:.4f} "
            f"procedure_negative_coverage={procedure_coverage:.4f}"
        )

    runner.score_stream(val, observe="always")
    calibration_scores, calibration_labels = runner.score_stream(
        calibration,
        observe="always",
    )
    if np.any(calibration_labels == 1):
        raise RuntimeError("calibration data contains attack labels")
    threshold = calibrate_threshold(
        calibration_scores, args.target_fpr
    )

    if manifest.get("test_history_mode") == "reset":
        runner.reset_state()

    observe_policy = {
        "frozen": "never",
        "safe": "safe",
        "blind": "always",
    }[args.test_memory]
    test_scores, test_labels = runner.score_stream(
        test,
        observe=observe_policy,
        threshold=threshold if observe_policy == "safe" else None,
    )

    metadata = load_test_metadata(
        data_dir / manifest.get("test_metadata_file", "test_metadata.csv"),
        expected_labels=test_labels,
    )

    result = {
        "research_metrics_valid": not bool(manifest.get("debug_subset")),
        "model": "semantic_causal_tgn",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "learning_rate": args.lr,
        "event_dim": args.event_dim,
        "token_mask_prob": args.token_mask_prob,
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "test_memory": args.test_memory,
        "test_history_mode": manifest.get("test_history_mode"),
        "train_loss_final": losses[-1] if losses else None,
        "endpoint_negative_coverage_final": (
            endpoint_coverages[-1] if endpoint_coverages else None
        ),
        "procedure_negative_coverage_final": (
            procedure_coverages[-1] if procedure_coverages else None
        ),
        "metrics": evaluate_scores(
            test_labels,
            test_scores,
            threshold,
            duration_ns=temporal_duration_ns(
                test,
                time_unit_ns=int(manifest.get("time_unit_ns", 1_000)),
            ),
        ),
        "per_attack": attack_type_breakdown(
            metadata.attack_types,
            test_labels,
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
