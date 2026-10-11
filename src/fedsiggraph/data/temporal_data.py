"""Conversion from canonical events to PyG TemporalData.

The builder uses a node vocabulary fit only on training events. Unknown entities
at evaluation time map to a reserved node id instead of expanding the model from
future data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .encoding import SemanticEncoder
from .schema import CanonicalEvent


@dataclass(frozen=True, slots=True)
class FrozenNodeIndexer:
    entity_to_id: dict[str, int]

    @classmethod
    def fit(cls, events: Iterable[CanonicalEvent]) -> "FrozenNodeIndexer":
        entities = sorted(
            {
                entity
                for event in events
                for entity in (event.src_entity, event.dst_entity)
            }
        )
        # 0 is reserved for any entity first observed outside training.
        return cls({entity: idx + 1 for idx, entity in enumerate(entities)})

    @property
    def unknown_id(self) -> int:
        return 0

    @property
    def num_nodes(self) -> int:
        return len(self.entity_to_id) + 1

    def encode(self, entity: str) -> int:
        return self.entity_to_id.get(entity, self.unknown_id)


def build_temporal_data(
    events: Sequence[CanonicalEvent],
    *,
    encoder: SemanticEncoder,
    node_indexer: FrozenNodeIndexer,
):
    """Build a continuous-time PyG event stream.

    Timestamps are represented as integer milliseconds relative to the first
    event. This preserves real inter-event timing while keeping TGN time
    encodings numerically well-scaled compared with absolute Unix nanoseconds.
    """
    try:
        import torch
        from torch_geometric.data import TemporalData
    except ImportError as exc:  # pragma: no cover - exercised only without ML deps.
        raise ImportError(
            "TemporalData conversion requires `pip install -e '.[ml]'`"
        ) from exc

    if not events:
        raise ValueError("cannot build TemporalData from an empty sequence")

    ordered = sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))
    t0 = ordered[0].timestamp_ns

    src = []
    dst = []
    t_ms = []
    msg = []
    labels = []

    for event in ordered:
        categories = encoder.encode_categories(event)
        numeric = encoder.encode_numeric(event)

        src.append(node_indexer.encode(event.src_entity))
        dst.append(node_indexer.encode(event.dst_entity))
        t_ms.append((event.timestamp_ns - t0) // 1_000_000)

        msg.append(
            [
                float(categories["src_role"]),
                float(categories["dst_role"]),
                float(categories["protocol"]),
                float(categories["relation"]),
                float(categories["procedure"]),
                float(categories["message_type"]),
                float(categories["direction"]),
                float(numeric[0]),
            ]
        )
        labels.append(0 if event.label is None else int(event.label))

    return TemporalData(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(t_ms, dtype=torch.long),
        msg=torch.tensor(msg, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
    )
