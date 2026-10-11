import pandas as pd

from fedsiggraph.data.context import build_context_transition_features
from fedsiggraph.data.signaling_parser import SignalingParserAdapter
from fedsiggraph.data.vocab import SemanticVocabularies


def test_enriched_signaling_csv_recovers_shared_session_context():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2"],
            "src": ["gnb1", "amf", "smf"],
            "dst": ["amf", "smf", "upf"],
            "message": [
                "InitialUeMessage, NASType.RegistrationRequest",
                "POST: /nsmf-pdusession/v1/sm-contexts",
                "SessionEstablishmentRequest",
            ],
            "ue": ["imsi-001", "imsi-001", "imsi-001"],
            "pdu_session_id": ["7", "7", "7"],
        }
    )

    events = SignalingParserAdapter("site-a").from_dataframe(df)

    assert [event.protocol for event in events] == ["NGAP", "SBI", "PFCP"]
    assert all(event.session_context is not None for event in events)
    assert len({event.session_context for event in events}) == 1
    assert all("imsi-001" not in event.session_context for event in events)
    assert all(event.explicit_context_key[0] == "session" for event in events)


def test_session_context_is_absent_when_export_has_no_session_identifier():
    df = pd.DataFrame(
        {
            "time": ["1.0"],
            "src": ["amf"],
            "dst": ["smf"],
            "message": ["POST: /nsmf-pdusession/v1/sm-contexts"],
            "ue": ["imsi-001"],
        }
    )

    event = SignalingParserAdapter("site-a").from_dataframe(df)[0]

    assert event.session_context is None
    assert event.ue_context is not None
    assert event.explicit_context_key[0] == "ue"


def test_shared_session_produces_cross_protocol_transition_features():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2"],
            "src": ["gnb1", "amf", "smf"],
            "dst": ["amf", "smf", "upf"],
            "message": [
                "InitialUeMessage, NASType.RegistrationRequest",
                "POST: /nsmf-pdusession/v1/sm-contexts",
                "SessionEstablishmentRequest",
            ],
            "ue": ["imsi-001", "imsi-001", "imsi-001"],
            "pdu_session_id": ["7", "7", "7"],
        }
    )
    events = SignalingParserAdapter("site-a").from_dataframe(df, label=0)
    vocab = SemanticVocabularies.fit(events)

    transitions = build_context_transition_features(events, vocab=vocab)

    assert transitions.history_mask.tolist() == [False, True, True]
    assert transitions.cross_protocol_mask.tolist() == [False, True, True]


def test_cross_protocol_context_falls_back_to_explicit_ue_when_session_is_partial():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2"],
            "src": ["gnb1", "amf", "smf"],
            "dst": ["amf", "smf", "upf"],
            "message": [
                "InitialUeMessage, NASType.RegistrationRequest",
                "POST: /nsmf-pdusession/v1/sm-contexts",
                "SessionEstablishmentRequest",
            ],
            "ue": ["imsi-001", "imsi-001", "imsi-001"],
            "pdu_session_id": [None, None, "7"],
        }
    )
    events = SignalingParserAdapter("site-a").from_dataframe(df, label=0)
    vocab = SemanticVocabularies.fit(events)

    transitions = build_context_transition_features(events, vocab=vocab)

    assert events[0].session_context is None
    assert events[1].session_context is None
    assert events[2].session_context is not None
    assert transitions.history_mask.tolist() == [False, True, True]
    assert transitions.cross_protocol_mask.tolist() == [False, True, True]
