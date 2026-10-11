"""Strict chronological splits for one-class anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import CanonicalEvent


@dataclass(frozen=True, slots=True)
class EventSplits:
    train: tuple[CanonicalEvent, ...]
    val: tuple[CanonicalEvent, ...]
    calibration: tuple[CanonicalEvent, ...]
    test: tuple[CanonicalEvent, ...]

    def assert_chronological(self) -> None:
        ordered = self.train + self.val + self.calibration
        if any(
            left.timestamp_ns > right.timestamp_ns
            for left, right in zip(ordered, ordered[1:])
        ):
            raise AssertionError("development splits are not chronological")


def _sorted(events: Iterable[CanonicalEvent]) -> list[CanonicalEvent]:
    return sorted(events, key=lambda e: (e.timestamp_ns, e.event_id))


def split_benign_development(
    benign_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    test_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> EventSplits:
    """Split a benign reference capture and keep the attack capture untouched.

    This is intentionally tailored to datasets such as Granomaly: a dedicated
    benign capture is used for model development, while the separate experiment
    capture becomes final test data. Calibration therefore never sees attacks.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    benign = _sorted(benign_events)
    test = _sorted(test_events)
    if len(benign) < 3:
        raise ValueError("at least three benign events are required")
    if any(event.label == 1 for event in benign):
        raise ValueError("development capture contains attack-labelled events")

    n = len(benign)

    def advance_past_timestamp_group(index: int) -> int:
        """Keep equal-timestamp events in the same development partition."""
        index = max(1, min(index, n - 1))
        while (
            index < n
            and benign[index - 1].timestamp_ns == benign[index].timestamp_ns
        ):
            index += 1
        return index

    train_end = advance_past_timestamp_group(int(n * train_fraction))
    val_end = advance_past_timestamp_group(
        int(n * (train_fraction + val_fraction))
    )

    if val_end <= train_end:
        val_end = advance_past_timestamp_group(train_end + 1)

    # Keep at least one calibration timestamp-group.
    if val_end >= n:
        raise ValueError(
            "timestamp groups are too large for non-empty train/val/calibration splits"
        )

    splits = EventSplits(
        train=tuple(benign[:train_end]),
        val=tuple(benign[train_end:val_end]),
        calibration=tuple(benign[val_end:]),
        test=tuple(test),
    )
    splits.assert_chronological()
    return splits
