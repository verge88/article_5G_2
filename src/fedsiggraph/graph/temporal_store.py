"""Causal event history.

The core invariant is simple: an event must be scored against state containing
only prior events, *before* the event is observed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from fedsiggraph.data.schema import CanonicalEvent


@dataclass(frozen=True, slots=True)
class TemporalContext:
    entity_events: tuple[CanonicalEvent, ...]
    session_events: tuple[CanonicalEvent, ...]

    @property
    def all_events(self) -> tuple[CanonicalEvent, ...]:
        merged = {event.event_id: event for event in self.entity_events}
        merged.update({event.event_id: event for event in self.session_events})
        return tuple(
            sorted(merged.values(), key=lambda e: (e.timestamp_ns, e.event_id))
        )


class CausalTemporalStore:
    """Bounded, deployment-local history for streaming anomaly detection."""

    def __init__(self, max_events_per_key: int = 32) -> None:
        if max_events_per_key <= 0:
            raise ValueError("max_events_per_key must be positive")
        self.max_events_per_key = max_events_per_key
        self._entity: dict[str, Deque[CanonicalEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_key)
        )
        self._context: dict[tuple[str, str], Deque[CanonicalEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_key)
        )
        self._last_timestamp_ns: int | None = None

    def context_before(self, event: CanonicalEvent) -> TemporalContext:
        """Return history available *before* ``event`` without mutating state."""
        entity_history = list(self._entity.get(event.src_entity, ()))
        entity_history.extend(self._entity.get(event.dst_entity, ()))

        # Deduplicate events touching both endpoints.
        entity_unique = {item.event_id: item for item in entity_history}
        entity_events = tuple(
            sorted(entity_unique.values(), key=lambda e: (e.timestamp_ns, e.event_id))
        )

        session_events = tuple(self._context.get(event.context_key, ()))
        for prior in (*entity_events, *session_events):
            if prior.timestamp_ns > event.timestamp_ns:
                raise RuntimeError("future event leaked into causal context")
            if prior.event_id == event.event_id:
                raise RuntimeError("current event leaked into causal context")

        return TemporalContext(
            entity_events=entity_events,
            session_events=session_events,
        )

    def observe(self, event: CanonicalEvent) -> None:
        """Commit an event to history after scoring."""
        if (
            self._last_timestamp_ns is not None
            and event.timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                "events must be observed in non-decreasing timestamp order"
            )

        self._entity[event.src_entity].append(event)
        if event.dst_entity != event.src_entity:
            self._entity[event.dst_entity].append(event)
        self._context[event.context_key].append(event)
        self._last_timestamp_ns = event.timestamp_ns

    def reset(self) -> None:
        self._entity.clear()
        self._context.clear()
        self._last_timestamp_ns = None
