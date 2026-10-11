import pandas as pd

from fedsiggraph.data.granomaly import GranomalyAdapter
from fedsiggraph.data.semantics import (
    extract_sbi_context,
    normalize_sbi_path,
    parse_granomaly_semantics,
)


def test_normalize_sbi_path_redacts_dynamic_identifiers():
    path = (
        "/nudm-sdm/v2/imsi-001010000000057/sdm-subscriptions/"
        "46a58e52-5a15-41ef-b05f-f59f91ef810c"
    )
    assert normalize_sbi_path(path) == (
        "/nudm-sdm/v2/imsi-{id}/sdm-subscriptions/{uuid}"
    )


def test_extract_sbi_context_uses_only_explicit_identifiers():
    ue, session = extract_sbi_context(
        "/nudm-uecm/v1/imsi-001010000000057/"
        "registrations/smf-registrations/1"
    )
    assert ue == "imsi-001010000000057"
    assert session == "pdu-session:imsi-001010000000057:1"

    ue, session = extract_sbi_context(
        "/nsmf-pdusession/v1/sm-contexts/1341/release"
    )
    assert ue is None
    assert session == "sm-context:1341"

    assert extract_sbi_context("/nnrf-nfm/v1/nf-instances") == (None, None)


def test_pfcp_procedure_is_extracted():
    parsed = parse_granomaly_semantics(
        "eth:ethertype:ip:udp:pfcp",
        "PFCP Session Deletion Request",
    )
    assert parsed.protocol == "PFCP"
    assert parsed.relation == "PFCP::SESSION_DELETION"
    assert parsed.procedure == "Session Deletion"
    assert parsed.direction == "REQUEST"


def test_ngap_nas_semantics_are_separated():
    parsed = parse_granomaly_semantics(
        "eth:ethertype:ip:sctp:ngap:ngap:nas-5gs",
        "UplinkNASTransport, Deregistration request (UE originating)",
    )
    assert parsed.protocol == "NAS"
    assert parsed.relation == "NGAP::UplinkNASTransport"
    assert parsed.procedure == "Deregistration request (UE originating)"
    assert parsed.direction == "UPLINK"


def test_sbi_response_inherits_request_service_context_and_normalized_path():
    df = pd.DataFrame(
        {
            "frame.time_epoch": ["1.000001", "1.000002"],
            "frame.len": [197, 78],
            "ip.src": ["10.211.55.5", "10.211.55.11"],
            "srcport": [49654, 7777],
            "ip.dst": ["10.211.55.11", "10.211.55.5"],
            "dstport": [7777, 49654],
            "frame.protocols": [
                "eth:ethertype:ip:tcp:http2",
                "eth:ethertype:ip:tcp:http2",
            ],
            "combined_info": [
                "HEADERS[24857]: POST "
                "/nsmf-pdusession/v1/sm-contexts/1341/release",
                "HEADERS[24857]: 204 No Content",
            ],
            "Label": ["normal", "normal"],
        }
    )

    events = GranomalyAdapter("granomaly").from_dataframe(df)

    request, response = events
    assert request.src_role == "SMF"
    assert request.dst_role == "SCP"
    assert request.relation == "SBI::nsmf-pdusession"
    assert request.procedure == (
        "POST /nsmf-pdusession/v1/sm-contexts/{id}/release"
    )
    assert request.session_context == "sm-context:1341"

    assert response.relation == request.relation
    assert response.procedure == (
        "RESPONSE POST /nsmf-pdusession/v1/sm-contexts/{id}/release"
    )
    assert response.result_code == "204"
    assert response.session_context == request.session_context
    assert response.ue_context == request.ue_context
