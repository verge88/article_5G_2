"""TemporalHEAT: TGN memory with heterogeneous edge-enhanced attention."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
    from torch_geometric.nn import HEATConv, TGNMemory
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TemporalHEAT requires `pip install -e '.[ml]'`"
    ) from exc

from .continuation import ContextContinuationHead
from .semantic_tgn import SemanticAnomalyHead, SemanticEventEncoder


class LocalResidualAdapter(nn.Module):
    """Small deployment-private residual adapter for personalized federation."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("adapter bottleneck must be positive")
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.gelu(self.down(x)))


class TemporalHEATEmbedding(nn.Module):
    """Relation-aware temporal neighborhood aggregation."""

    def __init__(
        self,
        *,
        memory_dim: int,
        embedding_dim: int,
        event_dim: int,
        time_encoder: nn.Module,
        num_node_types: int,
        num_edge_types: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")

        self.time_encoder = time_encoder
        edge_dim = event_dim + time_encoder.out_channels
        per_head = embedding_dim // heads

        self.conv1 = HEATConv(
            in_channels=memory_dim,
            out_channels=per_head,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_emb_dim=16,
            edge_dim=edge_dim,
            edge_attr_emb_dim=32,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = HEATConv(
            in_channels=embedding_dim,
            out_channels=per_head,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_emb_dim=16,
            edge_dim=edge_dim,
            edge_attr_emb_dim=32,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        memory: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        edge_t: Tensor,
        edge_msg: Tensor,
        node_type: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        rel_t = last_update[edge_index[0]] - edge_t
        rel_t_enc = self.time_encoder(rel_t.to(memory.dtype))
        edge_attr = torch.cat([rel_t_enc, edge_msg], dim=-1)

        hidden = self.conv1(
            memory,
            edge_index,
            node_type,
            edge_type,
            edge_attr,
        )
        hidden = self.norm1(F.relu(hidden))
        hidden = self.conv2(
            hidden,
            edge_index,
            node_type,
            edge_type,
            edge_attr,
        )
        return self.norm2(F.relu(hidden))


class TemporalHEAT(nn.Module):
    """Semantic causal detector with relation-aware temporal graph attention."""

    def __init__(
        self,
        *,
        num_nodes: int,
        numeric_dim: int,
        vocab_sizes: Mapping[str, int],
        num_node_types: int,
        num_edge_types: int,
        event_dim: int = 64,
        memory_dim: int = 128,
        time_dim: int = 32,
        embedding_dim: int = 128,
        heads: int = 4,
        continuation_hidden_dim: int = 128,
        continuation_history_mode: str = "sequence",
        local_adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        self.event_encoder = SemanticEventEncoder(
            vocab_sizes=vocab_sizes,
            numeric_dim=numeric_dim,
            output_dim=event_dim,
        )
        self.local_adapter = (
            LocalResidualAdapter(event_dim, local_adapter_dim)
            if local_adapter_dim > 0
            else nn.Identity()
        )
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=event_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(event_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.embedding = TemporalHEATEmbedding(
            memory_dim=memory_dim,
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            time_encoder=self.memory.time_enc,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            heads=heads,
        )
        self.head = SemanticAnomalyHead(embedding_dim, event_dim)
        self.continuation_head = ContextContinuationHead(
            node_dim=embedding_dim,
            vocab_sizes=vocab_sizes,
            hidden_dim=continuation_hidden_dim,
            history_mode=continuation_history_mode,
        )

    def reset_state(self) -> None:
        self.memory.reset_state()

    def encode_batch(
        self,
        batch,
        *,
        dst_role_id: Tensor | None = None,
        relation_id: Tensor | None = None,
        procedure_id: Tensor | None = None,
    ) -> Tensor:
        encoded = self.event_encoder(
            batch.msg,
            src_role_id=batch.src_role_id,
            dst_role_id=(
                batch.dst_role_id if dst_role_id is None else dst_role_id
            ),
            protocol_id=batch.protocol_id,
            relation_id=(
                batch.relation_id if relation_id is None else relation_id
            ),
            procedure_id=(
                batch.procedure_id if procedure_id is None else procedure_id
            ),
            message_type_id=batch.message_type_id,
            direction_id=batch.direction_id,
        )
        return self.local_adapter(encoded)

    def score_embeddings(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        event_embedding: Tensor,
    ) -> Tensor:
        return self.head(z_src, z_dst, event_embedding)

    def continuation_logits(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        batch,
    ) -> tuple[Tensor, Tensor]:
        """Predict current semantics from pre-event state and previous context."""
        return self.continuation_head(
            z_src,
            z_dst,
            prev_protocol_id=batch.prev_protocol_id,
            prev_relation_id=batch.prev_relation_id,
            prev_procedure_id=batch.prev_procedure_id,
            protocol_history_id=getattr(
                batch, "context_protocol_history_id", None
            ),
            relation_history_id=getattr(
                batch, "context_relation_history_id", None
            ),
            procedure_history_id=getattr(
                batch, "context_procedure_history_id", None
            ),
            sequence_mask=getattr(batch, "context_sequence_mask", None),
        )

    def observe(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        encoded_msg: Tensor,
    ) -> None:
        self.memory.update_state(src, dst, t, encoded_msg)
