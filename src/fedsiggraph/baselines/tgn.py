"""Granomaly-style causal TGN baseline.

This module deliberately separates ``score`` from ``observe``. Calling ``score``
must not update temporal memory; a caller may observe only after making an
anomaly decision.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    from torch_geometric.nn import TGNMemory, TransformerConv
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
except ImportError as exc:  # pragma: no cover - only without ML extras.
    raise ImportError(
        "TGN baseline requires `pip install -e '.[ml]'`"
    ) from exc


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        embedding_dim: int,
        msg_dim: int,
        time_encoder: nn.Module,
    ) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be divisible by two")
        self.time_encoder = time_encoder
        edge_dim = msg_dim + time_encoder.out_channels
        self.conv = TransformerConv(
            memory_dim,
            embedding_dim // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        memory: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        edge_t: Tensor,
        edge_msg: Tensor,
    ) -> Tensor:
        rel_t = last_update[edge_index[0]] - edge_t
        rel_t_enc = self.time_encoder(rel_t.to(memory.dtype))
        edge_attr = torch.cat([rel_t_enc, edge_msg], dim=-1)
        return self.conv(memory, edge_index, edge_attr)


class LinkAnomalyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.src = nn.Linear(channels, channels)
        self.dst = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, 1)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        hidden = (self.src(z_src) + self.dst(z_dst)).relu()
        return self.out(hidden)


class CausalTGNBaseline(nn.Module):
    """Small TGN baseline whose state mutation is explicit."""

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 100,
    ) -> None:
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.embedding = GraphAttentionEmbedding(
            memory_dim=memory_dim,
            embedding_dim=embedding_dim,
            msg_dim=msg_dim,
            time_encoder=self.memory.time_enc,
        )
        self.head = LinkAnomalyHead(embedding_dim)

    def reset_state(self) -> None:
        self.memory.reset_state()

    def score_embeddings(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        """Return anomaly logits without mutating TGN memory."""
        return self.head(z_src, z_dst)

    def observe(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> None:
        """Update memory only after the caller has scored/accepted the event."""
        self.memory.update_state(src, dst, t, msg)
