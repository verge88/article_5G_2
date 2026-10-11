"""Data-driven semantic hard negatives for benign 5G signaling events."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from fedsiggraph.data.schema import CanonicalEvent


class HardNegativeSampler:
    """Generate locally plausible but semantically incompatible events.

    Compatibility tables are learned only from the benign training partition.
    Endpoint negatives reuse a *real training entity* of an incompatible role so
    the detector cannot solve the task merely by spotting a synthetic/unknown ID.
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
        self._entities_by_role: dict[str, set[str]] = defaultdict(set)

        for event in events:
            self._procedures_by_protocol[event.protocol].add(event.procedure)
            self._procedures_by_pair[
                (event.src_role, event.dst_role, event.protocol)
            ].add(event.procedure)
            self._dst_roles_by_src_proc[
                (event.src_role, event.procedure, event.protocol)
            ].add(event.dst_role)
            self._dst_roles_by_protocol[event.protocol].add(event.dst_role)
            self._entities_by_role[event.dst_role].add(event.dst_entity)

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
            label=1,
            attack_type="synthetic_procedure_negative",
        )

    def endpoint_negative(self, event: CanonicalEvent) -> CanonicalEvent:
        """Keep procedure/protocol fixed but choose a real entity of an incompatible role."""
        allowed = self._dst_roles_by_src_proc[
            (event.src_role, event.procedure, event.protocol)
        ]
        candidate_roles = sorted(
            role
            for role in (self._dst_roles_by_protocol[event.protocol] - allowed)
            if self._entities_by_role.get(role)
        )
        if not candidate_roles:
            raise LookupError(
                "no protocol-compatible endpoint negative is available"
            )

        dst_role = self._rng.choice(candidate_roles)
        dst_entity = self._rng.choice(sorted(self._entities_by_role[dst_role]))
        return event.synthetic(
            dst_entity=dst_entity,
            dst_role=dst_role,
            label=1,
            attack_type="synthetic_endpoint_negative",
        )
