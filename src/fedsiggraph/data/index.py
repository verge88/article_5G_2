"""Deployment-local node indexing.

The index is fitted on training identities only. Any entity that first appears
in validation/test is mapped to a dedicated unknown node in the strict
inductive setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import CanonicalEvent

UNK_ENTITY = "<UNK_ENTITY>"


@dataclass(frozen=True, slots=True)
class LocalNodeIndex:
    entity_to_id: dict[str, int]

    @classmethod
    def fit(cls, events: Iterable[CanonicalEvent]) -> "LocalNodeIndex":
        entities: set[str] = set()
        for event in events:
            entities.add(event.src_entity)
            entities.add(event.dst_entity)

        mapping = {UNK_ENTITY: 0}
        for entity in sorted(entities):
            if entity != UNK_ENTITY:
                mapping[entity] = len(mapping)
        return cls(mapping)

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def num_nodes(self) -> int:
        return len(self.entity_to_id)

    def encode(self, entity: str) -> int:
        return self.entity_to_id.get(entity, self.unk_id)
