#!/usr/bin/env python3
"""Train TemporalHEAT with semantic hard negatives and context continuation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from fedsiggraph.anomaly.calibration import CompositeScoreCalibrator
from fedsiggraph.models.continuation import continuation_nll
from fedsiggraph.models.temporal_heat import TemporalHEAT
from fedsiggraph.training.history import ObservedHeteroHistoryBank
from fedsiggraph.training.metrics import (
    attack_type_breakdown,
    calibrate_threshold,
    evaluate_scores,
    temporal_duration_ns,
)
from fedsiggraph.training.metadata import load_test_metadata
from fedsiggraph.training.negatives import (
    ProcedureNegativeTable,
    sample_negative_destination,
)


@dataclass(frozen=True, slots=True)
class StreamComponents:
    base: np.ndarray
    continuation: np.ndarray
    context_mask: np.ndarray
    cross_protocol_mask: np.ndarray
    labels: np.ndarray


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
        help="Auxiliary causal continuation loss weight; set 0 for ablation.",
    )
    parser.add_argument(
        "--continuation-score-weight",
        type=float,
        default=1.0,
        help="Continuation NLL contribution after benign robust calibration.",
    )
    parser.add_argument(
        "--cross-protocol-weight",
        type=float,
        default=2.0,
        help="Extra training weight for explicit context transitions that change protocol.",
    )
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def load_temporal(path: Path):
    return torch.load(path, weights_only=False)


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


def calibrator_to_dict(
    calibrator: CompositeScoreCalibrator,
) -> dict[str, object]:
    continuation = (
        None
        if calibrator.continuation is None
        else {
            "median": calibrator.continuation.median,
            "scale": calibrator.continuation.scale,
        }
    )
    return {
        "base": {
            "median": calibrator.base.median,
            "scale": calibrator.base.scale,
        },
        "continuation": continuation,
        "continuation_weight": calibrator.continuation_weight,
    }


class TemporalHEATRunner:
    def __init__(
        self,
        train_data,
        *,
        preprocessing: dict,
        total_capacity: int,
        neighbors: int,
        lr: float,
        event_dim: int,
        memory_dim: int,
        time_dim: int,
        embedding_dim: int,
        heads: int,
        seed: int,
        token_mask_prob: float,
        continuation_loss_weight: float,
        cross_protocol_weight: float,
        device: torch.device,
    ) -> None:
        if not (0.0 <= token_mask_prob < 1.0):
            raise ValueError("token_mask_prob must be in [0, 1)")
        if continuation_loss_weight < 0:
            raise ValueError("continuation_loss_weight must be non-negative")
        if cross_protocol_weight < 1.0:
            raise ValueError("cross_protocol_weight must be >= 1")

        self.device = device
        self.num_nodes = int(train_data.num_nodes)
        self.event_dim = event_dim
        self.total_capacity = total_capacity
        self.token_mask_prob = token_mask_prob
        self.continuation_loss_weight = continuation_loss_weight
        self.cross_protocol_weight = cross_protocol_weight

        node_types = preprocessing["node_type_ids"]
        if len(node_types) != self.num_nodes:
            raise ValueError(
                "node_type_ids length does not match prepared num_nodes"
            )
        self.node_type_ids = torch.as_tensor(
            node_types, dtype=torch.long, device=device
        )

        vocab_sizes = vocab_sizes_from_preprocessing(preprocessing)
        self.model = TemporalHEAT(
            num_nodes=self.num_nodes,
            numeric_dim=int(train_data.msg.size(-1)),
            vocab_sizes=vocab_sizes,
            num_node_types=int(preprocessing["num_node_types"]),
            num_edge_types=vocab_sizes["relation"],
            event_dim=event_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            heads=heads,
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
        self.history = ObservedHeteroHistoryBank(
            total_capacity, event_dim
        )
        self.procedure_negatives = ProcedureNegativeTable(train_data, seed)

    def reset_state(self) -> None:
        self.model.reset_state()
        self.neighbor_loader.reset_state()
        self.history = ObservedHeteroHistoryBank(
            self.total_capacity, self.event_dim
        )

    def _embedding_context(self, nodes: torch.Tensor):
        n_id, edge_index, e_id = self.neighbor_loader(nodes)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        memory, last_update = self.model.memory(n_id)
        edge_t, edge_msg, edge_type = self.history.lookup(
            e_id, self.device
        )
        node_type = self.node_type_ids[n_id]
        z = self.model.embedding(
            memory,
            last_update,
            edge_index,
            edge_t,
            edge_msg,
            node_type,
            edge_type,
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
            out[selected] = 1  # train-reserved UNK ID
            return out

        return self.model.encode_batch(
            batch,
            dst_role_id=mask(batch.dst_role_id),
            relation_id=mask(batch.relation_id),
            procedure_id=mask(batch.procedure_id),
        )

    def _continuation_nll(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        protocol_logits, procedure_logits = self.model.continuation_logits(
            z_src, z_dst, batch
        )
        return continuation_nll(
            protocol_logits,
            procedure_logits,
            protocol_target=batch.protocol_id,
            procedure_target=batch.procedure_id,
        )

    def train_epoch(
        self,
        data,
        batch_size: int,
    ) -> tuple[float, float, float, float]:
        self.model.train()
        self.reset_state()
        loader = TemporalDataLoader(data, batch_size=batch_size)

        total_loss = 0.0
        total_events = 0
        procedure_negative_events = 0
        context_events = 0
        cross_protocol_events = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            src = batch.src
            pos_dst = batch.dst
            neg_dst = sample_negative_destination(
                pos_dst, num_nodes=self.num_nodes
            )
            nodes = torch.cat([src, pos_dst, neg_dst]).unique()
            z, assoc = self._embedding_context(nodes)
            z_src = z[assoc[src]]
            z_pos_dst = z[assoc[pos_dst]]

            benign_event = self._masked_benign_embedding(batch)
            endpoint_event = self.model.encode_batch(batch)
            procedure_ids, procedure_mask = self.procedure_negatives.sample(
                batch
            )
            procedure_event = self.model.encode_batch(
                batch,
                procedure_id=procedure_ids,
            )

            benign_logits = self.model.score_embeddings(
                z_src,
                z_pos_dst,
                benign_event,
            )
            endpoint_logits = self.model.score_embeddings(
                z_src,
                z[assoc[neg_dst]],
                endpoint_event,
            )

            primary_losses = [
                self.criterion(
                    benign_logits, torch.zeros_like(benign_logits)
                ),
                self.criterion(
                    endpoint_logits, torch.ones_like(endpoint_logits)
                ),
            ]

            if procedure_mask.any():
                procedure_logits = self.model.score_embeddings(
                    z_src[procedure_mask],
                    z_pos_dst[procedure_mask],
                    procedure_event[procedure_mask],
                )
                primary_losses.append(
                    self.criterion(
                        procedure_logits,
                        torch.ones_like(procedure_logits),
                    )
                )
                procedure_negative_events += int(procedure_mask.sum())

            primary_loss = torch.stack(primary_losses).mean()
            loss = primary_loss

            context_mask = batch.context_history_mask.bool()
            cross_mask = batch.cross_protocol_mask.bool() & context_mask
            context_events += int(context_mask.sum())
            cross_protocol_events += int(cross_mask.sum())

            if (
                self.continuation_loss_weight > 0
                and context_mask.any()
            ):
                nll = self._continuation_nll(
                    z_src,
                    z_pos_dst,
                    batch,
                )
                weights = torch.ones_like(nll)
                if self.cross_protocol_weight > 1:
                    weights = torch.where(
                        cross_mask,
                        torch.full_like(
                            weights, self.cross_protocol_weight
                        ),
                        weights,
                    )
                continuation_loss = (
                    nll[context_mask] * weights[context_mask]
                ).mean()
                loss = (
                    primary_loss
                    + self.continuation_loss_weight * continuation_loss
                )

            loss.backward()
            self.optimizer.step()

            # Commit the clean current event only after every score/loss that
            # depends on pre-event state has been computed.
            with torch.no_grad():
                observed_msg = self.model.encode_batch(batch)
                self.model.observe(
                    src, pos_dst, batch.t, observed_msg
                )
                self.neighbor_loader.insert(src, pos_dst)
                self.history.append(
                    batch.t,
                    observed_msg,
                    batch.relation_id,
                )
                self.model.memory.detach()

            n = int(batch.num_events)
            total_loss += float(loss.detach().cpu()) * n
            total_events += n

        denominator = max(total_events, 1)
        return (
            total_loss / denominator,
            procedure_negative_events / denominator,
            context_events / denominator,
            cross_protocol_events / denominator,
        )

    @torch.no_grad()
    def score_stream(
        self,
        data,
        *,
        observe: str,
        calibrator: CompositeScoreCalibrator | None = None,
        threshold: float | None = None,
    ) -> StreamComponents:
        if observe not in {"always", "never", "safe"}:
            raise ValueError(f"unknown observe policy: {observe}")
        if observe == "safe" and (
            threshold is None or calibrator is None
        ):
            raise ValueError(
                "safe observation requires a composite calibrator and threshold"
            )

        self.model.eval()
        loader = TemporalDataLoader(data, batch_size=1)
        base_scores: list[float] = []
        continuation_scores: list[float] = []
        context_masks: list[bool] = []
        cross_masks: list[bool] = []
        labels: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            nodes = torch.cat([batch.src, batch.dst]).unique()
            z, assoc = self._embedding_context(nodes)
            z_src = z[assoc[batch.src]]
            z_dst = z[assoc[batch.dst]]

            event_embedding = self.model.encode_batch(batch)
            logits = self.model.score_embeddings(
                z_src,
                z_dst,
                event_embedding,
            )
            base_score = float(logits.sigmoid().item())

            has_context = bool(batch.context_history_mask.item())
            is_cross = bool(batch.cross_protocol_mask.item()) and has_context
            continuation_score = np.nan
            if has_context:
                nll = self._continuation_nll(
                    z_src,
                    z_dst,
                    batch,
                )
                continuation_score = float(nll.item())

            base_scores.append(base_score)
            continuation_scores.append(continuation_score)
            context_masks.append(has_context)
            cross_masks.append(is_cross)
            labels.append(int(batch.y.item()))

            should_observe = observe == "always"
            if observe == "safe":
                assert calibrator is not None
                composite = calibrator.transform(
                    np.asarray([base_score]),
                    np.asarray([continuation_score]),
                    np.asarray([has_context]),
                )
                should_observe = float(composite[0]) <= float(threshold)

            if should_observe:
                self.model.observe(
                    batch.src,
                    batch.dst,
                    batch.t,
                    event_embedding,
                )
                self.neighbor_loader.insert(batch.src, batch.dst)
                self.history.append(
                    batch.t,
                    event_embedding,
                    batch.relation_id,
                )
                self.model.memory.detach()

        return StreamComponents(
            base=np.asarray(base_scores, dtype=np.float64),
            continuation=np.asarray(
                continuation_scores, dtype=np.float64
            ),
            context_mask=np.asarray(context_masks, dtype=bool),
            cross_protocol_mask=np.asarray(cross_masks, dtype=bool),
            labels=np.asarray(labels, dtype=np.int64),
        )


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
    with open(data_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        device=device,
    )

    losses: list[float] = []
    procedure_coverages: list[float] = []
    context_coverages: list[float] = []
    cross_coverages: list[float] = []

    for epoch in range(1, args.epochs + 1):
        (
            loss,
            procedure_coverage,
            context_coverage,
            cross_coverage,
        ) = runner.train_epoch(train, args.batch_size)
        losses.append(loss)
        procedure_coverages.append(procedure_coverage)
        context_coverages.append(context_coverage)
        cross_coverages.append(cross_coverage)
        print(
            f"epoch={epoch:03d} loss={loss:.6f} "
            f"procedure_negative_coverage={procedure_coverage:.4f} "
            f"context_coverage={context_coverage:.4f} "
            f"cross_protocol_coverage={cross_coverage:.4f}"
        )

    runner.score_stream(val, observe="always")
    calibration_components = runner.score_stream(
        calibration,
        observe="always",
    )
    if np.any(calibration_components.labels == 1):
        raise RuntimeError("calibration data contains attack labels")

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

    if manifest.get("test_history_mode") == "reset":
        runner.reset_state()

    observe_policy = {
        "frozen": "never",
        "safe": "safe",
        "blind": "always",
    }[args.test_memory]
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

    test_context_coverage = float(
        test_components.context_mask.mean()
    ) if test_components.context_mask.size else 0.0
    test_cross_coverage = float(
        test_components.cross_protocol_mask.mean()
    ) if test_components.cross_protocol_mask.size else 0.0

    metadata = load_test_metadata(
        data_dir / manifest.get("test_metadata_file", "test_metadata.csv"),
        expected_labels=test_components.labels,
    )

    result = {
        "model": "temporal_heat_context_continuation",
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
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "score_calibration": calibrator_to_dict(score_calibrator),
        "test_memory": args.test_memory,
        "test_history_mode": manifest.get("test_history_mode"),
        "train_loss_final": losses[-1] if losses else None,
        "procedure_negative_coverage_final": (
            procedure_coverages[-1] if procedure_coverages else None
        ),
        "context_coverage_train_final": (
            context_coverages[-1] if context_coverages else None
        ),
        "cross_protocol_coverage_train_final": (
            cross_coverages[-1] if cross_coverages else None
        ),
        "context_coverage_test": test_context_coverage,
        "cross_protocol_coverage_test": test_cross_coverage,
        "metrics": evaluate_scores(
            test_components.labels,
            test_scores,
            threshold,
            duration_ns=temporal_duration_ns(test),
        ),
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
