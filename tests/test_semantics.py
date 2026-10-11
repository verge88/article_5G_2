import pandas as pd

from fedsiggraph.data.granomaly import GranomalyAdapter
from fedsiggraph.data.semantics import (
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


def test_sbi_response_inherits_request_service_and_normalized_path():
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
                "HEADERS[24857]: POST /nsmf-pdusession/v1/sm-contexts/1341/release",
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

    assert response.relation == request.relation
    assert response.procedure == (
        "RESPONSE POST /nsmf-pdusession/v1/sm-contexts/{id}/release"
    )
    assert response.result_code == "204"
