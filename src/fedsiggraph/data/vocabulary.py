"""Training-only categorical vocabularies.

A vocabulary is fit on the benign training partition and then frozen. Any token
first observed in validation, calibration, or test maps to UNK rather than
changing the feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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

    def encode(self, value: str) -> int:
        return self.token_to_id.get(str(value), self.unk_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)
