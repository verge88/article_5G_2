"""Semantic TGN baseline between topology-only TGN and TemporalHEAT."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import torch
    from torch import Tensor, nn
    from torch_geometric.nn import TGNMemory
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Semantic TGN requires `pip install -e '.[ml]'`"
    ) from exc

from fedsiggraph.baselines.tgn import GraphAttentionEmbedding


_REQUIRED_VOCABS = (
    "src_role",
    "dst_role",
    "protocol",
    "relation",
    "procedure",
    "message_type",
    "direction",
)


class SemanticEventEncoder(nn.Module):
    """Encode current 5G event semantics into a dense trainable message."""

    def __init__(
        self,
        vocab_sizes: Mapping[str, int],
        numeric_dim: int,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        missing = [name for name in _REQUIRED_VOCABS if name not in vocab_sizes]
        if missing:
            raise ValueError(f"missing vocabulary sizes: {missing}")

        self.src_role = nn.Embedding(vocab_sizes["src_role"], 8)
        self.dst_role = nn.Embedding(vocab_sizes["dst_role"], 8)
        self.protocol = nn.Embedding(vocab_sizes["protocol"], 8)
        self.relation = nn.Embedding(vocab_sizes["relation"], 16)
        self.procedure = nn.Embedding(vocab_sizes["procedure"], 24)
        self.message_type = nn.Embedding(vocab_sizes["message_type"], 8)
        self.direction = nn.Embedding(vocab_sizes["direction"], 4)

        self.numeric = nn.Sequential(
            nn.Linear(numeric_dim, 16),
            nn.ReLU(),
        )
        concatenated = 8 + 8 + 8 + 16 + 24 + 8 + 4 + 16
        self.output = nn.Sequential(
            nn.Linear(concatenated, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        numeric: Tensor,
        *,
        src_role_id: Tensor,
        dst_role_id: Tensor,
        protocol_id: Tensor,
        relation_id: Tensor,
        procedure_id: Tensor,
        message_type_id: Tensor,
        direction_id: Tensor,
    ) -> Tensor:
        pieces = [
            self.src_role(src_role_id),
            self.dst_role(dst_role_id),
            self.protocol(protocol_id),
            self.relation(relation_id),
            self.procedure(procedure_id),
            self.message_type(message_type_id),
            self.direction(direction_id),
            self.numeric(numeric),
        ]
        return self.output(torch.cat(pieces, dim=-1))


class SemanticAnomalyHead(nn.Module):
    """Score endpoint state and the semantics of the *current* event together."""

    def __init__(self, node_dim: int, event_dim: int) -> None:
        super().__init__()
        hidden = max(node_dim, event_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + event_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        event_embedding: Tensor,
    ) -> Tensor:
        return self.net(torch.cat([z_src, z_dst, event_embedding], dim=-1))


class SemanticCausalTGN(nn.Module):
    """TGN memory with a semantic current-event encoder and anomaly head."""

    def __init__(
        self,
        *,
        num_nodes: int,
        numeric_dim: int,
        vocab_sizes: Mapping[str, int],
        event_dim: int = 64,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 100,
    ) -> None:
        super().__init__()
        self.event_encoder = SemanticEventEncoder(
            vocab_sizes=vocab_sizes,
            numeric_dim=numeric_dim,
            output_dim=event_dim,
        )
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=event_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(event_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.embedding = GraphAttentionEmbedding(
            memory_dim=memory_dim,
            embedding_dim=embedding_dim,
            msg_dim=event_dim,
            time_encoder=self.memory.time_enc,
        )
        self.head = SemanticAnomalyHead(embedding_dim, event_dim)

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
        return self.event_encoder(
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

    def score_embeddings(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        event_embedding: Tensor,
    ) -> Tensor:
        return self.head(z_src, z_dst, event_embedding)

    def observe(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        encoded_msg: Tensor,
    ) -> None:
        self.memory.update_state(src, dst, t, encoded_msg)
