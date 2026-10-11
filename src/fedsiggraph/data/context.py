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


DEFAULT_CONTEXT_HISTORY_SIZE = 4


@dataclass(frozen=True, slots=True)
class ContextTransitionFeatures:
    prev_protocol_id: np.ndarray
    prev_relation_id: np.ndarray
    prev_procedure_id: np.ndarray
    history_mask: np.ndarray
    cross_protocol_mask: np.ndarray
    protocol_history_id: np.ndarray
    relation_history_id: np.ndarray
    procedure_history_id: np.ndarray
    sequence_mask: np.ndarray


def _ordered(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
) -> list[CanonicalEvent]:
    return sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))


def build_context_transition_features(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    vocab: SemanticVocabularies,
    history: Sequence[CanonicalEvent] | Iterable[CanonicalEvent] = (),
    history_size: int = DEFAULT_CONTEXT_HISTORY_SIZE,
) -> ContextTransitionFeatures:
    """Encode the immediately preceding event in the same explicit context.

    ``history`` is consumed first and may contain only events that occurred
    before the target split. The current target event updates the context only
    *after* its previous-event features have been emitted.
    """
    if history_size <= 0:
        raise ValueError("history_size must be positive")

    history_events = _ordered(history)
    target_events = _ordered(events)

    history_by_context: dict[
        tuple[str, str], list[CanonicalEvent]
    ] = {}
    last_history_timestamp: int | None = None

    def keys(event: CanonicalEvent) -> tuple[tuple[str, str], ...]:
        """Return explicit context keys from most to least specific."""
        result: list[tuple[str, str]] = []
        if event.session_context:
            result.append(("session", event.session_context))
        if event.ue_context:
            result.append(("ue", event.ue_context))
        return tuple(result)

    def selected_history(event: CanonicalEvent) -> list[CanonicalEvent]:
        for key in keys(event):
            values = history_by_context.get(key)
            if values:
                return values
        return []

    def remember(event: CanonicalEvent) -> None:
        event_keys = keys(event)
        if not event_keys:
            return

        session_key = next(
            (key for key in event_keys if key[0] == "session"),
            None,
        )
        ue_key = next(
            (key for key in event_keys if key[0] == "ue"),
            None,
        )

        # When a session identifier first appears on an event that also carries
        # an already-known UE identifier, seed the new session history from the
        # causal UE history. This links NGAP/SBI history to future PFCP events
        # without modifying any previously emitted feature row.
        if (
            session_key is not None
            and session_key not in history_by_context
            and ue_key is not None
            and ue_key in history_by_context
        ):
            history_by_context[session_key] = list(
                history_by_context[ue_key][-history_size:]
            )

        if (
            ue_key is not None
            and ue_key not in history_by_context
            and session_key is not None
            and session_key in history_by_context
        ):
            history_by_context[ue_key] = list(
                history_by_context[session_key][-history_size:]
            )

        for key in event_keys:
            values = history_by_context.setdefault(key, [])
            if not values or values[-1].event_id != event.event_id:
                values.append(event)
            if len(values) > history_size:
                del values[:-history_size]

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
    protocol_history: list[list[int]] = []
    relation_history: list[list[int]] = []
    procedure_history: list[list[int]] = []
    sequence_mask: list[list[bool]] = []

    for event in target_events:
        prior_history = list(selected_history(event)[-history_size:])
        previous = prior_history[-1] if prior_history else None

        row_protocol = [vocab.protocol.pad_id] * history_size
        row_relation = [vocab.relation.pad_id] * history_size
        row_procedure = [vocab.procedure.pad_id] * history_size
        row_mask = [False] * history_size

        for index, prior in enumerate(prior_history):
            row_protocol[index] = vocab.protocol.encode(prior.protocol)
            row_relation[index] = vocab.relation.encode(prior.relation)
            row_procedure[index] = vocab.procedure.encode(prior.procedure)
            row_mask[index] = True

        protocol_history.append(row_protocol)
        relation_history.append(row_relation)
        procedure_history.append(row_procedure)
        sequence_mask.append(row_mask)

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
        protocol_history_id=np.asarray(protocol_history, dtype=np.int64),
        relation_history_id=np.asarray(relation_history, dtype=np.int64),
        procedure_history_id=np.asarray(procedure_history, dtype=np.int64),
        sequence_mask=np.asarray(sequence_mask, dtype=np.bool_),
    )
