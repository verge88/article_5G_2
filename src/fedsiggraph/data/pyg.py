"""Conversion from canonical events to PyG :class:`TemporalData`."""

from __future__ import annotations

from typing import Iterable, Sequence

from .context import build_context_transition_features
from .features import CausalNumericFeatureEncoder
from .index import LocalNodeIndex
from .schema import CanonicalEvent
from .vocab import SemanticVocabularies


def to_temporal_data(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    vocab: SemanticVocabularies,
    node_index: LocalNodeIndex,
    numeric_encoder: CausalNumericFeatureEncoder,
    time_origin_ns: int,
    history: Sequence[CanonicalEvent] | Iterable[CanonicalEvent] = (),
    time_unit_ns: int = 1_000,
):
    """Build a continuous-time event stream.

    Categorical IDs come from a vocabulary already frozen on training data.
    Node IDs come from the local training node index; unseen entities map to
    node 0. Timestamps are shifted by a common train-time origin and expressed
    in microseconds by default, preserving real intervals without passing large
    Unix epoch values into the TGN time encoder.

    Additional ``prev_*`` tensors describe only the immediately preceding
    event in the same explicitly recovered UE/session context. Missing context
    is represented with PAD IDs and ``context_history_mask=False``.
    """
    try:
        import torch
        from torch_geometric.data import TemporalData
    except ImportError as exc:  # pragma: no cover - exercised only without ML extras.
        raise ImportError(
            "TemporalData conversion requires `pip install -e '.[ml]'`"
        ) from exc

    if time_unit_ns <= 0:
        raise ValueError("time_unit_ns must be positive")

    ordered = sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))
    history_ordered = sorted(
        history, key=lambda e: (e.timestamp_ns, e.event_id)
    )
    numeric = numeric_encoder.transform(
        ordered, history=history_ordered
    )
    transitions = build_context_transition_features(
        ordered,
        vocab=vocab,
        history=history_ordered,
    )

    src: list[int] = []
    dst: list[int] = []
    t: list[int] = []
    y: list[int] = []
    encoded_fields: dict[str, list[int]] = {
        "src_role_id": [],
        "dst_role_id": [],
        "protocol_id": [],
        "relation_id": [],
        "procedure_id": [],
        "message_type_id": [],
        "direction_id": [],
    }

    for event in ordered:
        relative_ns = event.timestamp_ns - time_origin_ns
        if relative_ns < 0:
            raise ValueError("event timestamp precedes the configured time origin")
        src.append(node_index.encode(event.src_entity))
        dst.append(node_index.encode(event.dst_entity))
        t.append(relative_ns // time_unit_ns)
        y.append(-1 if event.label is None else event.label)

        token_ids = vocab.encode_event(event)
        for key, value in token_ids.items():
            encoded_fields[key].append(value)

    kwargs = {
        key: torch.as_tensor(value, dtype=torch.long)
        for key, value in encoded_fields.items()
    }
    data = TemporalData(
        src=torch.as_tensor(src, dtype=torch.long),
        dst=torch.as_tensor(dst, dtype=torch.long),
        t=torch.as_tensor(t, dtype=torch.long),
        msg=torch.as_tensor(numeric, dtype=torch.float32),
        y=torch.as_tensor(y, dtype=torch.long),
        prev_protocol_id=torch.as_tensor(
            transitions.prev_protocol_id, dtype=torch.long
        ),
        prev_relation_id=torch.as_tensor(
            transitions.prev_relation_id, dtype=torch.long
        ),
        prev_procedure_id=torch.as_tensor(
            transitions.prev_procedure_id, dtype=torch.long
        ),
        context_history_mask=torch.as_tensor(
            transitions.history_mask, dtype=torch.bool
        ),
        cross_protocol_mask=torch.as_tensor(
            transitions.cross_protocol_mask, dtype=torch.bool
        ),
        **kwargs,
    )
    data.num_nodes = node_index.num_nodes
    data.time_origin_ns = int(time_origin_ns)
    data.time_unit_ns = int(time_unit_ns)
    return data
