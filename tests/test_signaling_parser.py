import pandas as pd

from fedsiggraph.data.context import build_context_transition_features
from fedsiggraph.data.signaling_parser import (
    SignalingParserAdapter,
    apply_role_map,
    infer_role_map,
)
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


def test_origami_compact_sbi_messages_are_not_misclassified_as_ngap():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2"],
            "src": ["amf", "ausf", "udm"],
            "dst": ["ausf", "udm", "amf"],
            "message": [
                "POST-nausf-auth",
                "GET-nudm-ueau",
                "200",
            ],
            "ue": ["imsi-001"] * 3,
        }
    )

    events = SignalingParserAdapter("origami").from_dataframe(df, label=0)

    assert [event.protocol for event in events] == ["SBI", "SBI", "SBI"]
    assert events[0].relation == "SBI:nausf"
    assert events[0].procedure == "POST:nausf-auth"
    assert events[1].relation == "SBI:nudm"
    assert events[1].procedure == "GET:nudm-ueau"
    assert events[2].direction == "RESPONSE"


def test_open5gs_daemon_hostnames_resolve_to_transferable_nf_roles():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1"],
            "src": ["open5gs-amfd", "open5gs-smfd"],
            "dst": ["open5gs-smfd", "open5gs-upfd"],
            "message": [
                "POST-nsmf-pdusession",
                "SessionEstablishmentRequest",
            ],
            "ue": ["imsi-001", "imsi-001"],
        }
    )

    events = SignalingParserAdapter("site-a").from_dataframe(df, label=0)

    assert events[0].src_role == "AMF"
    assert events[0].dst_role == "SMF"
    assert events[1].src_role == "SMF"
    assert events[1].dst_role == "UPF"


def test_role_inference_recovers_gnb_amf_smf_upf_from_n2_n4_direction():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2", "1.3"],
            "src": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "dst": ["10.0.0.2", "10.0.0.1", "10.0.0.4", "10.0.0.3"],
            "message": [
                "UNKNOWN_NGAP, NASType.RegistrationRequest",
                "UNKNOWN_NGAP, NASType.AuthenticationRequest",
                "SessionEstablishmentRequest",
                "SessionEstablishmentResponse",
            ],
            "ue": ["imsi-001"] * 4,
        }
    )
    events = SignalingParserAdapter("site-a").from_dataframe(df, label=0)

    mapping = infer_role_map(events)
    resolved = apply_role_map(events, mapping)

    assert mapping == {
        "10.0.0.1": "gNB",
        "10.0.0.2": "AMF",
        "10.0.0.3": "SMF",
        "10.0.0.4": "UPF",
    }
    assert resolved[0].src_role == "gNB"
    assert resolved[0].dst_role == "AMF"
    assert resolved[2].src_role == "SMF"
    assert resolved[2].dst_role == "UPF"


def test_inferred_role_map_never_overrides_explicit_role_metadata():
    df = pd.DataFrame(
        {
            "time": ["1.0"],
            "src": ["custom-ran"],
            "dst": ["custom-amf"],
            "message": ["UNKNOWN_NGAP, NASType.RegistrationRequest"],
            "ue": ["imsi-001"],
        }
    )
    events = SignalingParserAdapter(
        "site-a",
        role_by_entity={
            "custom-ran": "RAN_VENDOR_ROLE",
            "custom-amf": "AMF",
        },
    ).from_dataframe(df, label=0)

    inferred = infer_role_map(events)
    resolved = apply_role_map(events, inferred)

    assert inferred["custom-ran"] == "gNB"
    assert resolved[0].src_role == "RAN_VENDOR_ROLE"
    assert resolved[0].dst_role == "AMF"


def test_pfcp_role_inference_ignores_heartbeat_and_reverses_session_report():
    df = pd.DataFrame(
        {
            "time": ["1.0", "1.1", "1.2"],
            "src": ["a", "b", "upf"],
            "dst": ["b", "a", "smf"],
            "message": [
                "HeartbeatRequest",
                "HeartbeatRequest",
                "SessionReportRequest",
            ],
            "ue": ["", "", ""],
        }
    )
    events = SignalingParserAdapter("site-a").from_dataframe(df, label=0)

    mapping = infer_role_map(events)

    assert "a" not in mapping
    assert "b" not in mapping
    assert mapping["upf"] == "UPF"
    assert mapping["smf"] == "SMF"
