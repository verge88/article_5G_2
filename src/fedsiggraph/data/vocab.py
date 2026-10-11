"""Training-only categorical vocabularies.

Vocabularies are fitted on the benign training partition and then frozen.
Validation, calibration, and test events can only be transformed; unseen values
map to UNK instead of extending the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .schema import CanonicalEvent

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True, slots=True)
class FrozenVocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def fit(cls, values: Iterable[str]) -> "FrozenVocabulary":
        unique = sorted({str(value) for value in values})
        mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in unique:
            if token not in mapping:
                mapping[token] = len(mapping)
        return cls(mapping)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, value: str) -> int:
        return self.token_to_id.get(str(value), self.unk_id)

    def decode(self, index: int) -> str:
        inverse = {value: key for key, value in self.token_to_id.items()}
        return inverse.get(index, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.token_to_id)


@dataclass(frozen=True, slots=True)
class SemanticVocabularies:
    src_role: FrozenVocabulary
    dst_role: FrozenVocabulary
    protocol: FrozenVocabulary
    relation: FrozenVocabulary
    procedure: FrozenVocabulary
    message_type: FrozenVocabulary
    direction: FrozenVocabulary

    @classmethod
    def fit(
        cls,
        events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
        *,
        require_benign: bool = True,
    ) -> "SemanticVocabularies":
        materialized = list(events)
        if not materialized:
            raise ValueError("cannot fit vocabularies on an empty training set")
        if require_benign and any(event.label == 1 for event in materialized):
            raise ValueError(
                "training vocabulary must be fitted on benign events only"
            )

        def values(getter: Callable[[CanonicalEvent], str]) -> list[str]:
            return [getter(event) for event in materialized]

        return cls(
            src_role=FrozenVocabulary.fit(values(lambda e: e.src_role)),
            dst_role=FrozenVocabulary.fit(values(lambda e: e.dst_role)),
            protocol=FrozenVocabulary.fit(values(lambda e: e.protocol)),
            relation=FrozenVocabulary.fit(values(lambda e: e.relation)),
            procedure=FrozenVocabulary.fit(values(lambda e: e.procedure)),
            message_type=FrozenVocabulary.fit(values(lambda e: e.message_type)),
            direction=FrozenVocabulary.fit(values(lambda e: e.direction)),
        )

    def encode_event(self, event: CanonicalEvent) -> dict[str, int]:
        return {
            "src_role_id": self.src_role.encode(event.src_role),
            "dst_role_id": self.dst_role.encode(event.dst_role),
            "protocol_id": self.protocol.encode(event.protocol),
            "relation_id": self.relation.encode(event.relation),
            "procedure_id": self.procedure.encode(event.procedure),
            "message_type_id": self.message_type.encode(event.message_type),
            "direction_id": self.direction.encode(event.direction),
        }
