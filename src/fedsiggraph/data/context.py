"""Causal transition features for explicit UE/session context.

These features never infer a session from timing or topology. Events without an
explicitly recovered UE/session identifier are masked out of continuation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .schema import CanonicalEvent
from .vocab import SemanticVocabularies


@dataclass(frozen=True, slots=True)
class ContextTransitionFeatures:
    prev_protocol_id: np.ndarray
    prev_relation_id: np.ndarray
    prev_procedure_id: np.ndarray
    history_mask: np.ndarray
    cross_protocol_mask: np.ndarray


def _ordered(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
) -> list[CanonicalEvent]:
    return sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))


def build_context_transition_features(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    vocab: SemanticVocabularies,
    history: Sequence[CanonicalEvent] | Iterable[CanonicalEvent] = (),
) -> ContextTransitionFeatures:
    """Encode the immediately preceding event in the same explicit context.

    ``history`` is consumed first and may contain only events that occurred
    before the target split. The current target event updates the context only
    *after* its previous-event features have been emitted.
    """
    history_events = _ordered(history)
    target_events = _ordered(events)

    last_by_context: dict[tuple[str, str], CanonicalEvent] = {}
    last_history_timestamp: int | None = None

    def keys(event: CanonicalEvent) -> tuple[tuple[str, str], ...]:
        """Return explicit context keys from most to least specific."""
        result: list[tuple[str, str]] = []
        if event.session_context:
            result.append(("session", event.session_context))
        if event.ue_context:
            result.append(("ue", event.ue_context))
        return tuple(result)

    def remember(event: CanonicalEvent) -> None:
        # Store both explicit identifiers when available. A later PFCP event
        # may expose a session identifier while the preceding SBI/NGAP event
        # exposes only the shared UE context (or vice versa).
        for key in keys(event):
            last_by_context[key] = event

    for event in history_events:
        remember(event)
        last_history_timestamp = event.timestamp_ns

    if target_events and last_history_timestamp is not None:
        if target_events[0].timestamp_ns < last_history_timestamp:
            raise ValueError(
                "context history contains an event later than the target split"
            )

    prev_protocol: list[int] = []
    prev_relation: list[int] = []
    prev_procedure: list[int] = []
    history_mask: list[bool] = []
    cross_protocol_mask: list[bool] = []

    for event in target_events:
        previous = next(
            (
                last_by_context[key]
                for key in keys(event)
                if key in last_by_context
            ),
            None,
        )

        if previous is None:
            prev_protocol.append(vocab.protocol.pad_id)
            prev_relation.append(vocab.relation.pad_id)
            prev_procedure.append(vocab.procedure.pad_id)
            history_mask.append(False)
            cross_protocol_mask.append(False)
        else:
            if previous.timestamp_ns > event.timestamp_ns:
                raise RuntimeError("future event leaked into context transition")
            prev_protocol.append(vocab.protocol.encode(previous.protocol))
            prev_relation.append(vocab.relation.encode(previous.relation))
            prev_procedure.append(vocab.procedure.encode(previous.procedure))
            history_mask.append(True)
            cross_protocol_mask.append(previous.protocol != event.protocol)

        remember(event)

    return ContextTransitionFeatures(
        prev_protocol_id=np.asarray(prev_protocol, dtype=np.int64),
        prev_relation_id=np.asarray(prev_relation, dtype=np.int64),
        prev_procedure_id=np.asarray(prev_procedure, dtype=np.int64),
        history_mask=np.asarray(history_mask, dtype=np.bool_),
        cross_protocol_mask=np.asarray(cross_protocol_mask, dtype=np.bool_),
    )
