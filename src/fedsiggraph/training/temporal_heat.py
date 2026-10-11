"""Reusable local TemporalHEAT training/evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

try:
    import torch
    from torch_geometric.loader import TemporalDataLoader
    from torch_geometric.nn.models.tgn import LastNeighborLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("TemporalHEAT training requires the ML extra") from exc

from fedsiggraph.anomaly.calibration import CompositeScoreCalibrator
from fedsiggraph.federated.parameters import proximal_penalty
from fedsiggraph.models.continuation import continuation_nll
from fedsiggraph.models.temporal_heat import TemporalHEAT
from fedsiggraph.training.history import ObservedHeteroHistoryBank
from fedsiggraph.training.negatives import (
    EndpointNegativeTable,
    ProcedureNegativeTable,
)


@dataclass(frozen=True, slots=True)
class StreamComponents:
    base: np.ndarray
    continuation: np.ndarray
    context_mask: np.ndarray
    cross_protocol_mask: np.ndarray
    labels: np.ndarray


def vocab_sizes_from_preprocessing(
    preprocessing: dict,
) -> dict[str, int]:
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
    """One deployment-local trainer whose temporal state is never federated."""

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
        self.endpoint_negatives = EndpointNegativeTable(train_data, seed + 1)

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
            out[selected] = 1
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
        *,
        prox_reference: dict[str, torch.Tensor] | None = None,
        prox_mu: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        """Train one local epoch, optionally with a FedProx shared-weight term."""
        if prox_mu < 0:
            raise ValueError("prox_mu must be non-negative")
        if prox_mu > 0 and prox_reference is None:
            raise ValueError("FedProx requires a global parameter reference")

        self.model.train()
        self.reset_state()
        loader = TemporalDataLoader(data, batch_size=batch_size)

        total_loss = 0.0
        total_events = 0
        endpoint_negative_events = 0
        procedure_negative_events = 0
        context_events = 0
        cross_protocol_events = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            src = batch.src
            pos_dst = batch.dst
            neg_dst, endpoint_mask = self.endpoint_negatives.sample(batch)
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
                z_src, z_pos_dst, benign_event
            )
            primary_losses = [
                self.criterion(
                    benign_logits, torch.zeros_like(benign_logits)
                ),
            ]

            if endpoint_mask.any():
                endpoint_logits = self.model.score_embeddings(
                    z_src[endpoint_mask],
                    z[assoc[neg_dst[endpoint_mask]]],
                    endpoint_event[endpoint_mask],
                )
                primary_losses.append(
                    self.criterion(
                        endpoint_logits,
                        torch.ones_like(endpoint_logits),
                    )
                )
                endpoint_negative_events += int(endpoint_mask.sum())

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
                    z_src, z_pos_dst, batch
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

            if prox_mu > 0:
                assert prox_reference is not None
                loss = loss + proximal_penalty(
                    self.model,
                    prox_reference,
                    mu=prox_mu,
                )

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                observed_msg = self.model.encode_batch(batch)
                self.model.observe(
                    src, pos_dst, batch.t, observed_msg
                )
                self.neighbor_loader.insert(src, pos_dst)
                self.history.append(
                    batch.t, observed_msg, batch.relation_id
                )
                self.model.memory.detach()

            n = int(batch.num_events)
            total_loss += float(loss.detach().cpu()) * n
            total_events += n

        denominator = max(total_events, 1)
        return (
            total_loss / denominator,
            endpoint_negative_events / denominator,
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
                z_src, z_dst, event_embedding
            )
            base_score = float(logits.sigmoid().item())

            has_context = bool(batch.context_history_mask.item())
            is_cross = (
                bool(batch.cross_protocol_mask.item()) and has_context
            )
            continuation_score = np.nan
            if has_context:
                nll = self._continuation_nll(
                    z_src, z_dst, batch
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
