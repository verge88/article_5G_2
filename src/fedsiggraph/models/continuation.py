"""Causal continuation prediction from pre-event graph state and prior context."""

from __future__ import annotations

from collections.abc import Mapping

try:
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("continuation head requires the ML extra") from exc


class ContextContinuationHead(nn.Module):
    """Predict the *current* protocol/procedure from only previous context.

    The current event embedding is intentionally absent from this head. Inputs
    are pre-event node embeddings plus categorical IDs from the immediately
    preceding event in the same explicitly recovered UE/session context.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        vocab_sizes: Mapping[str, int],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        for name in ("protocol", "relation", "procedure"):
            if name not in vocab_sizes:
                raise ValueError(f"missing vocabulary size: {name}")

        self.prev_protocol = nn.Embedding(vocab_sizes["protocol"], 8)
        self.prev_relation = nn.Embedding(vocab_sizes["relation"], 16)
        self.prev_procedure = nn.Embedding(vocab_sizes["procedure"], 24)

        input_dim = 2 * node_dim + 8 + 16 + 24
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        self.protocol_out = nn.Linear(
            hidden_dim, vocab_sizes["protocol"]
        )
        self.procedure_out = nn.Linear(
            hidden_dim, vocab_sizes["procedure"]
        )

    def forward(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        *,
        prev_protocol_id: Tensor,
        prev_relation_id: Tensor,
        prev_procedure_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(
            __import__("torch").cat(
                [
                    z_src,
                    z_dst,
                    self.prev_protocol(prev_protocol_id),
                    self.prev_relation(prev_relation_id),
                    self.prev_procedure(prev_procedure_id),
                ],
                dim=-1,
            )
        )
        return self.protocol_out(hidden), self.procedure_out(hidden)
