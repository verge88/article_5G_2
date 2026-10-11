"""Causal numeric features with train-only robust scaling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .schema import CanonicalEvent


def _raw_causal_features(
    events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    history: Sequence[CanonicalEvent] | Iterable[CanonicalEvent] = (),
) -> np.ndarray:
    """Extract raw log-scaled features without looking into the future."""
    history_events = sorted(history, key=lambda e: (e.timestamp_ns, e.event_id))
    target_events = sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))

    last_entity: dict[str, int] = {}
    last_context: dict[tuple[str, str], int] = {}

    for event in history_events:
        last_entity[event.src_entity] = event.timestamp_ns
        last_entity[event.dst_entity] = event.timestamp_ns
        last_context[event.context_key] = event.timestamp_ns

    rows: list[list[float]] = []
    for event in target_events:
        t = event.timestamp_ns

        def delta_ms(last: int | None) -> float:
            if last is None:
                return 0.0
            delta_ns = t - last
            if delta_ns < 0:
                raise ValueError(
                    "feature history contains an event later than the target event"
                )
            return delta_ns / 1_000_000.0

        src_delta = delta_ms(last_entity.get(event.src_entity))
        dst_delta = delta_ms(last_entity.get(event.dst_entity))
        context_delta = delta_ms(last_context.get(event.context_key))

        rows.append(
            [
                math.log1p(event.packet_length),
                math.log1p(src_delta),
                math.log1p(dst_delta),
                math.log1p(context_delta),
            ]
        )

        # Update only after features for the current event have been computed.
        last_entity[event.src_entity] = t
        last_entity[event.dst_entity] = t
        last_context[event.context_key] = t

    if not rows:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class RobustFeatureStats:
    median: np.ndarray
    scale: np.ndarray


class CausalNumericFeatureEncoder:
    """Robust scaler fitted only on causal features from benign training data."""

    feature_names = (
        "log_packet_length",
        "log_delta_src_ms",
        "log_delta_dst_ms",
        "log_delta_context_ms",
    )

    def __init__(self, stats: RobustFeatureStats) -> None:
        self.stats = stats

    @classmethod
    def fit(
        cls,
        train_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
        *,
        require_benign: bool = True,
    ) -> "CausalNumericFeatureEncoder":
        events = list(train_events)
        if not events:
            raise ValueError("cannot fit numeric encoder on empty training data")
        if require_benign and any(event.label == 1 for event in events):
            raise ValueError(
                "numeric feature statistics must be fitted on benign events only"
            )

        raw = _raw_causal_features(events)
        median = np.median(raw, axis=0).astype(np.float32)
        mad = np.median(np.abs(raw - median), axis=0).astype(np.float32)
        scale = (1.4826 * mad).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(RobustFeatureStats(median=median, scale=scale))

    def transform(
        self,
        events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
        *,
        history: Sequence[CanonicalEvent] | Iterable[CanonicalEvent] = (),
    ) -> np.ndarray:
        raw = _raw_causal_features(events, history=history)
        if raw.size == 0:
            return raw
        return ((raw - self.stats.median) / self.stats.scale).astype(np.float32)
