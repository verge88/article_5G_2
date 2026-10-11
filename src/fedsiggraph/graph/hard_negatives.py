"""Data-driven semantic hard negatives for benign 5G signaling events."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from fedsiggraph.data.schema import CanonicalEvent


class HardNegativeSampler:
    """Generate locally plausible but semantically incompatible events.

    Compatibility tables are learned only from the benign training partition.
    This avoids constructing trivial impossible edges such as UE -> NRF over PFCP.
    """

    def __init__(
        self,
        benign_events: Iterable[CanonicalEvent],
        seed: int = 0,
    ) -> None:
        events = [event for event in benign_events if event.label in (None, 0)]
        if not events:
            raise ValueError("at least one benign training event is required")

        self._rng = random.Random(seed)
        self._procedures_by_protocol: dict[str, set[str]] = defaultdict(set)
        self._procedures_by_pair: dict[
            tuple[str, str, str], set[str]
        ] = defaultdict(set)
        self._dst_roles_by_src_proc: dict[
            tuple[str, str, str], set[str]
        ] = defaultdict(set)
        self._dst_roles_by_protocol: dict[str, set[str]] = defaultdict(set)

        for event in events:
            self._procedures_by_protocol[event.protocol].add(event.procedure)
            self._procedures_by_pair[
                (event.src_role, event.dst_role, event.protocol)
            ].add(event.procedure)
            self._dst_roles_by_src_proc[
                (event.src_role, event.procedure, event.protocol)
            ].add(event.dst_role)
            self._dst_roles_by_protocol[event.protocol].add(event.dst_role)

    def procedure_negative(self, event: CanonicalEvent) -> CanonicalEvent:
        """Keep endpoints/protocol valid but choose an unseen procedure for the pair."""
        allowed = self._procedures_by_pair[
            (event.src_role, event.dst_role, event.protocol)
        ]
        candidates = sorted(self._procedures_by_protocol[event.protocol] - allowed)
        if not candidates:
            raise LookupError(
                "no protocol-compatible procedure negative is available"
            )
        procedure = self._rng.choice(candidates)
        return event.synthetic(
            procedure=procedure,
            relation=f"SYNTHETIC::{event.protocol}::{procedure}",
            label=1,
            attack_type="synthetic_procedure_negative",
        )

    def endpoint_negative(self, event: CanonicalEvent) -> CanonicalEvent:
        """Keep procedure/protocol fixed but choose an incompatible destination role."""
        allowed = self._dst_roles_by_src_proc[
            (event.src_role, event.procedure, event.protocol)
        ]
        candidates = sorted(self._dst_roles_by_protocol[event.protocol] - allowed)
        if not candidates:
            raise LookupError(
                "no protocol-compatible endpoint negative is available"
            )
        dst_role = self._rng.choice(candidates)
        return event.synthetic(
            dst_entity=f"SYNTHETIC::{dst_role}",
            dst_role=dst_role,
            label=1,
            attack_type="synthetic_endpoint_negative",
        )
