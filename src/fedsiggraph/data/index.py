"""Deployment-local node indexing and train-fitted node metadata.

The index is fitted on training identities only. Any entity that first appears
in validation/test maps to one unknown node in the strict inductive setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import CanonicalEvent

UNK_ENTITY = "<UNK_ENTITY>"

NODE_TYPE_TO_ID = {
    "UNKNOWN": 0,
    "NF": 1,
    "gNB": 2,
    "UE": 3,
    "HOST": 4,
    "BROADCAST": 5,
}


def _merge_value(
    existing: str | None,
    new: str,
    *,
    entity: str,
    field: str,
) -> str:
    if existing is None or existing == "UNKNOWN":
        return new
    if new == "UNKNOWN" or new == existing:
        return existing
    raise ValueError(
        f"conflicting {field} metadata for {entity!r}: "
        f"{existing!r} vs {new!r}"
    )


@dataclass(frozen=True, slots=True)
class LocalNodeIndex:
    entity_to_id: dict[str, int]
    node_type_by_id: tuple[str, ...]
    role_by_id: tuple[str, ...]

    @classmethod
    def fit(cls, events: Iterable[CanonicalEvent]) -> "LocalNodeIndex":
        entity_type: dict[str, str] = {}
        entity_role: dict[str, str] = {}

        for event in events:
            for entity, node_type, role in (
                (event.src_entity, event.src_type, event.src_role),
                (event.dst_entity, event.dst_type, event.dst_role),
            ):
                entity_type[entity] = _merge_value(
                    entity_type.get(entity),
                    node_type,
                    entity=entity,
                    field="node type",
                )
                entity_role[entity] = _merge_value(
                    entity_role.get(entity),
                    role,
                    entity=entity,
                    field="role",
                )

        mapping = {UNK_ENTITY: 0}
        for entity in sorted(entity_type):
            if entity != UNK_ENTITY:
                mapping[entity] = len(mapping)

        node_types = ["UNKNOWN"] * len(mapping)
        roles = ["UNKNOWN"] * len(mapping)
        for entity, node_id in mapping.items():
            if node_id == 0:
                continue
            node_types[node_id] = entity_type.get(entity, "UNKNOWN")
            roles[node_id] = entity_role.get(entity, "UNKNOWN")

        return cls(
            entity_to_id=mapping,
            node_type_by_id=tuple(node_types),
            role_by_id=tuple(roles),
        )

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def num_nodes(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_node_types(self) -> int:
        return len(NODE_TYPE_TO_ID)

    def encode(self, entity: str) -> int:
        return self.entity_to_id.get(entity, self.unk_id)

    def node_type_id(self, node_id: int) -> int:
        node_type = self.node_type_by_id[node_id]
        return NODE_TYPE_TO_ID.get(node_type, NODE_TYPE_TO_ID["UNKNOWN"])

    def node_type_ids(self) -> list[int]:
        return [self.node_type_id(i) for i in range(self.num_nodes)]
