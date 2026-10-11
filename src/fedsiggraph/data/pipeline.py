"""Leakage-safe preprocessing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .features import CausalNumericFeatureEncoder
from .index import LocalNodeIndex
from .schema import CanonicalEvent
from .split import EventSplits, split_benign_development
from .vocab import SemanticVocabularies


@dataclass(frozen=True, slots=True)
class FittedPreprocessing:
    splits: EventSplits
    vocab: SemanticVocabularies
    node_index: LocalNodeIndex
    numeric_encoder: CausalNumericFeatureEncoder
    time_origin_ns: int


def fit_preprocessing(
    benign_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    test_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> FittedPreprocessing:
    """Fit every learned preprocessing component on the train split only."""
    splits = split_benign_development(
        benign_events,
        test_events,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    vocab = SemanticVocabularies.fit(splits.train)
    node_index = LocalNodeIndex.fit(splits.train)
    numeric_encoder = CausalNumericFeatureEncoder.fit(splits.train)
    origin = min(event.timestamp_ns for event in splits.train)

    return FittedPreprocessing(
        splits=splits,
        vocab=vocab,
        node_index=node_index,
        numeric_encoder=numeric_encoder,
        time_origin_ns=origin,
    )
