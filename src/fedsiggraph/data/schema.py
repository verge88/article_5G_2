"""Canonical event representation shared by all 5G signaling adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """One time-ordered signaling event.

    Identity-bearing fields (``src_entity``, ``dst_entity``, UE/session context)
    are deployment-local. Transferable model features should primarily use
    semantic roles and procedures rather than raw identifiers.
    """

    event_id: int
    timestamp_ns: int
    deployment_id: str

    src_entity: str
    dst_entity: str
    src_type: str
    dst_type: str
    src_role: str
    dst_role: str

    protocol: str
    relation: str
    procedure: str
    message_type: str = "UNKNOWN"
    direction: str = "UNKNOWN"

    result_code: Optional[str] = None
    cause: Optional[str] = None

    ue_context: Optional[str] = None
    session_context: Optional[str] = None

    packet_length: int = 0
    response_latency_ms: Optional[float] = None

    # Evaluation-only metadata. These fields must never enter model features.
    label: Optional[int] = None
    attack_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise ValueError("event_id must be non-negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.packet_length < 0:
            raise ValueError("packet_length must be non-negative")
        if self.label not in (None, 0, 1):
            raise ValueError("label must be None, 0, or 1")
        for field_name in (
            "deployment_id",
            "src_entity",
            "dst_entity",
            "src_type",
            "dst_type",
            "src_role",
            "dst_role",
            "protocol",
            "relation",
            "procedure",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")

    @property
    def explicit_context_key(self) -> tuple[str, str] | None:
        """Return a context only when it was explicitly recovered from traffic.

        Cross-protocol objectives use this property so that missing correlation
        identifiers are masked instead of being guessed from time or topology.
        """
        if self.session_context:
            return ("session", self.session_context)
        if self.ue_context:
            return ("ue", self.ue_context)
        return None

    @property
    def context_key(self) -> tuple[str, str]:
        """Return the best local key for generic causal history features."""
        explicit = self.explicit_context_key
        if explicit is not None:
            return explicit
        return ("nf_pair", f"{self.src_entity}->{self.dst_entity}")

    def semantic_key(self) -> tuple[str, str, str, str, str]:
        """Transferable event semantics, intentionally excluding raw identities."""
        return (
            self.src_role,
            self.dst_role,
            self.protocol,
            self.relation,
            self.procedure,
        )

    def synthetic(self, **changes: object) -> "CanonicalEvent":
        """Create a synthetic variant while retaining immutable event structure."""
        return replace(self, **changes)
