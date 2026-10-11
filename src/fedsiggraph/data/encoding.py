"""Fit-on-train semantic encoding for canonical signaling events."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from statistics import median
from typing import Iterable

from .schema import CanonicalEvent
from .vocab import FrozenVocabulary


def _mad(values: list[float], center: float) -> float:
    deviations = [abs(value - center) for value in values]
    return median(deviations) if deviations else 0.0


@dataclass(frozen=True, slots=True)
class RobustScale:
    center: float
    scale: float

    @classmethod
    def fit(cls, values: Iterable[float]) -> "RobustScale":
        vals = [float(v) for v in values]
        if not vals:
            return cls(0.0, 1.0)
        center = float(median(vals))
        raw_mad = float(_mad(vals, center))
        scale = 1.4826 * raw_mad
        return cls(center, scale if scale > 1e-12 else 1.0)

    def transform(self, value: float) -> float:
        return (float(value) - self.center) / self.scale


@dataclass(frozen=True, slots=True)
class SemanticEncoder:
    """Feature metadata learned exclusively from benign training events."""

    src_role: FrozenVocabulary
    dst_role: FrozenVocabulary
    protocol: FrozenVocabulary
    relation: FrozenVocabulary
    procedure: FrozenVocabulary
    message_type: FrozenVocabulary
    direction: FrozenVocabulary
    packet_length_scale: RobustScale

    @classmethod
    def fit(cls, events: Iterable[CanonicalEvent]) -> "SemanticEncoder":
        benign = [event for event in events if event.label in (None, 0)]
        if not benign:
            raise ValueError("SemanticEncoder requires benign training events")
        return cls(
            src_role=FrozenVocabulary.fit(e.src_role for e in benign),
            dst_role=FrozenVocabulary.fit(e.dst_role for e in benign),
            protocol=FrozenVocabulary.fit(e.protocol for e in benign),
            relation=FrozenVocabulary.fit(e.relation for e in benign),
            procedure=FrozenVocabulary.fit(e.procedure for e in benign),
            message_type=FrozenVocabulary.fit(e.message_type for e in benign),
            direction=FrozenVocabulary.fit(e.direction for e in benign),
            packet_length_scale=RobustScale.fit(
                log1p(e.packet_length) for e in benign
            ),
        )

    def encode_categories(self, event: CanonicalEvent) -> dict[str, int]:
        return {
            "src_role": self.src_role.encode(event.src_role),
            "dst_role": self.dst_role.encode(event.dst_role),
            "protocol": self.protocol.encode(event.protocol),
            "relation": self.relation.encode(event.relation),
            "procedure": self.procedure.encode(event.procedure),
            "message_type": self.message_type.encode(event.message_type),
            "direction": self.direction.encode(event.direction),
        }

    def encode_numeric(self, event: CanonicalEvent) -> tuple[float]:
        return (
            self.packet_length_scale.transform(log1p(event.packet_length)),
        )
