"""Leakage-safe preprocessing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .features import CausalNumericFeatureEncoder
from .federated_vocab import FederatedSemanticVocabularies
from .index import LocalNodeIndex
from .schema import CanonicalEvent
from .split import EventSplits, split_benign_development
from .vocab import SemanticVocabularies


SemanticSchema = SemanticVocabularies | FederatedSemanticVocabularies


@dataclass(frozen=True, slots=True)
class FittedPreprocessing:
    splits: EventSplits
    vocab: SemanticSchema
    node_index: LocalNodeIndex
    numeric_encoder: CausalNumericFeatureEncoder
    time_origin_ns: int


def _fit_common(
    benign_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    test_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    vocab: SemanticSchema,
    train_fraction: float,
    val_fraction: float,
) -> FittedPreprocessing:
    splits = split_benign_development(
        benign_events,
        test_events,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
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


def fit_preprocessing(
    benign_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    test_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> FittedPreprocessing:
    """Fit centralized vocab/scaling/index state on benign train only."""
    benign = list(benign_events)
    test = list(test_events)
    preliminary = split_benign_development(
        benign,
        test,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    vocab = SemanticVocabularies.fit(preliminary.train)
    return _fit_common(
        benign,
        test,
        vocab=vocab,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )


def fit_federated_preprocessing(
    benign_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    test_events: Sequence[CanonicalEvent] | Iterable[CanonicalEvent],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    relation_buckets: int = 512,
    procedure_buckets: int = 2048,
    message_type_buckets: int = 128,
    schema_version: str = "fedsiggraph-v1",
) -> FittedPreprocessing:
    """Fit only private client state; semantic IDs are globally deterministic."""
    vocab = FederatedSemanticVocabularies.create(
        relation_buckets=relation_buckets,
        procedure_buckets=procedure_buckets,
        message_type_buckets=message_type_buckets,
        schema_version=schema_version,
    )
    return _fit_common(
        benign_events,
        test_events,
        vocab=vocab,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
