import pytest

from fedsiggraph.data.context import build_context_transition_features
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.vocab import SemanticVocabularies


def event(
    event_id: int,
    timestamp_ns: int,
    protocol: str,
    procedure: str,
    *,
    session: str | None = "session-1",
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp_ns=timestamp_ns,
        deployment_id="toy",
        src_entity="nf-a",
        dst_entity="nf-b",
        src_type="NF",
        dst_type="NF",
        src_role="AMF",
        dst_role="SMF",
        protocol=protocol,
        relation=f"{protocol}::REL",
        procedure=procedure,
        session_context=session,
        label=0,
    )


def test_first_event_has_no_previous_context_and_second_is_shifted():
    events = [
        event(1, 100, "SBI", "Create"),
        event(2, 200, "PFCP", "Establish"),
    ]
    vocab = SemanticVocabularies.fit(events)

    features = build_context_transition_features(events, vocab=vocab)

    assert features.history_mask.tolist() == [False, True]
    assert features.prev_protocol_id[0] == vocab.protocol.pad_id
    assert features.prev_protocol_id[1] == vocab.protocol.encode("SBI")
    assert features.prev_procedure_id[1] == vocab.procedure.encode("Create")
    assert features.cross_protocol_mask.tolist() == [False, True]


def test_missing_explicit_context_is_masked_not_nf_pair_guessed():
    events = [
        event(1, 100, "SBI", "Create", session=None),
        event(2, 200, "PFCP", "Establish", session=None),
    ]
    vocab = SemanticVocabularies.fit(events)

    features = build_context_transition_features(events, vocab=vocab)

    assert features.history_mask.tolist() == [False, False]
    assert features.cross_protocol_mask.tolist() == [False, False]


def test_split_history_must_precede_target():
    history = [event(1, 300, "SBI", "Create")]
    target = [event(2, 200, "PFCP", "Establish")]
    vocab = SemanticVocabularies.fit(history + target)

    with pytest.raises(ValueError, match="later"):
        build_context_transition_features(
            target,
            vocab=vocab,
            history=history,
        )
