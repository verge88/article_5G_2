"""Causal continuation prediction from pre-event graph state and prior context."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("continuation head requires the ML extra") from exc


class ContextContinuationHead(nn.Module):
    """Predict the current protocol/procedure from only previous context.

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
        context_dim: int = 64,
        history_mode: str = "sequence",
    ) -> None:
        super().__init__()
        if history_mode not in {"immediate", "sequence"}:
            raise ValueError("history_mode must be 'immediate' or 'sequence'")
        self.history_mode = history_mode

        for name in ("protocol", "relation", "procedure"):
            if name not in vocab_sizes:
                raise ValueError(f"missing vocabulary size: {name}")

        self.prev_protocol = nn.Embedding(
            vocab_sizes["protocol"], 8, padding_idx=0
        )
        self.prev_relation = nn.Embedding(
            vocab_sizes["relation"], 16, padding_idx=0
        )
        self.prev_procedure = nn.Embedding(
            vocab_sizes["procedure"], 24, padding_idx=0
        )

        token_dim = 8 + 16 + 24
        self.immediate_projection = nn.Sequential(
            nn.Linear(token_dim, context_dim),
            nn.ReLU(),
            nn.LayerNorm(context_dim),
        )
        self.sequence_gru = nn.GRU(
            input_size=token_dim,
            hidden_size=context_dim,
            batch_first=True,
        )

        input_dim = 2 * node_dim + context_dim
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

    def _embed_context_tokens(
        self,
        protocol_id: Tensor,
        relation_id: Tensor,
        procedure_id: Tensor,
    ) -> Tensor:
        return torch.cat(
            [
                self.prev_protocol(protocol_id),
                self.prev_relation(relation_id),
                self.prev_procedure(procedure_id),
            ],
            dim=-1,
        )

    def _sequence_context(
        self,
        protocol_history_id: Tensor,
        relation_history_id: Tensor,
        procedure_history_id: Tensor,
        sequence_mask: Tensor,
    ) -> Tensor:
        from torch.nn.utils.rnn import pack_padded_sequence

        tokens = self._embed_context_tokens(
            protocol_history_id,
            relation_history_id,
            procedure_history_id,
        )
        lengths = sequence_mask.long().sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(
            tokens,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.sequence_gru(packed)
        return hidden[-1]

    def forward(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        *,
        prev_protocol_id: Tensor,
        prev_relation_id: Tensor,
        prev_procedure_id: Tensor,
        protocol_history_id: Tensor | None = None,
        relation_history_id: Tensor | None = None,
        procedure_history_id: Tensor | None = None,
        sequence_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        use_sequence = (
            self.history_mode == "sequence"
            and protocol_history_id is not None
            and relation_history_id is not None
            and procedure_history_id is not None
            and sequence_mask is not None
        )

        if use_sequence:
            context = self._sequence_context(
                protocol_history_id,
                relation_history_id,
                procedure_history_id,
                sequence_mask,
            )
        else:
            immediate = self._embed_context_tokens(
                prev_protocol_id,
                prev_relation_id,
                prev_procedure_id,
            )
            context = self.immediate_projection(immediate)

        hidden = self.trunk(
            torch.cat(
                [
                    z_src,
                    z_dst,
                    context,
                ],
                dim=-1,
            )
        )
        return self.protocol_out(hidden), self.procedure_out(hidden)


def continuation_nll(
    protocol_logits: Tensor,
    procedure_logits: Tensor,
    *,
    protocol_target: Tensor,
    procedure_target: Tensor,
) -> Tensor:
    """Return an un-reduced joint continuation surprise per event."""
    protocol_loss = F.cross_entropy(
        protocol_logits,
        protocol_target,
        reduction="none",
    )
    procedure_loss = F.cross_entropy(
        procedure_logits,
        procedure_target,
        reduction="none",
    )
    return 0.5 * (protocol_loss + procedure_loss)
