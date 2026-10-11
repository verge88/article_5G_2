"""Protocol-semantic parsing for Granomaly packet exports."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


_HTTP_REQUEST_RE = re.compile(
    r"HEADERS\[(?P<stream>\d+)\]:\s+"
    r"(?P<method>GET|POST|PUT|PATCH|DELETE)\s+(?P<path>\S+)",
    re.IGNORECASE,
)
_HTTP_RESPONSE_RE = re.compile(
    r"HEADERS\[(?P<stream>\d+)\]:\s+(?P<status>\d{3})\b",
    re.IGNORECASE,
)
_HTTP_HEADERS_RE = re.compile(r"HEADERS\[(?P<stream>\d+)\]", re.IGNORECASE)
_HTTP_DATA_RE = re.compile(r"DATA\[(?P<stream>\d+)\]", re.IGNORECASE)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_IMSI_RE = re.compile(r"^imsi-\d+$", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"^[0-9a-f]{12,}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedSemantics:
    protocol: str
    relation: str
    procedure: str
    message_type: str
    direction: str
    result_code: Optional[str] = None
    cause: Optional[str] = None
    stream_id: Optional[str] = None


def normalize_sbi_path(path: str) -> str:
    """Normalize high-cardinality identifiers while retaining API structure."""
    path_only = path.split("?", 1)[0]
    segments: list[str] = []
    for segment in path_only.split("/"):
        if not segment:
            continue
        lower = segment.lower()
        if _IMSI_RE.fullmatch(segment):
            segments.append("imsi-{id}")
        elif _UUID_RE.fullmatch(segment):
            segments.append("{uuid}")
        elif segment.isdigit():
            segments.append("{id}")
        elif _LONG_HEX_RE.fullmatch(segment):
            segments.append("{id}")
        elif lower.startswith("suci-"):
            segments.append("suci-{id}")
        else:
            segments.append(lower)
    return "/" + "/".join(segments)


def _sbi_service(path: str) -> str:
    first = next((part for part in path.split("/") if part), "unknown")
    return f"SBI::{first.lower()}"


def _transport_semantics(stack: str, info: str) -> ParsedSemantics:
    lower_stack = stack.lower()
    upper_info = info.upper()

    if "icmp" in lower_stack:
        return ParsedSemantics("ICMP", "TRANSPORT", "ICMP", "ICMP", "UNKNOWN")
    if "tcp" in lower_stack:
        if "[SYN" in upper_info:
            proc = "TCP_SYN"
        elif "[RST" in upper_info:
            proc = "TCP_RST"
        elif "[FIN" in upper_info:
            proc = "TCP_FIN"
        elif "[ACK" in upper_info:
            proc = "TCP_ACK"
        else:
            proc = "TCP_DATA"
        return ParsedSemantics("TCP", "TRANSPORT", proc, "TCP", "UNKNOWN")
    if "sctp" in lower_stack:
        proc = "SCTP_SACK" if info.lstrip().startswith("SACK") else "SCTP_DATA"
        return ParsedSemantics("SCTP", "TRANSPORT", proc, "SCTP", "UNKNOWN")
    if "udp" in lower_stack:
        return ParsedSemantics("UDP", "TRANSPORT", "UDP", "UDP", "UNKNOWN")
    return ParsedSemantics("OTHER", "TRANSPORT", "OTHER", "OTHER", "UNKNOWN")


def parse_granomaly_semantics(stack: str, combined_info: str) -> ParsedSemantics:
    """Parse a packet into stable 5G signaling semantics.

    Priority follows the outer signaling interface: PFCP, HTTP/2 SBI, NGAP/NAS,
    then generic transport. This avoids treating NGAP payloads carried inside
    multipart SBI messages as direct N2 traffic.
    """
    stack = str(stack or "")
    info = str(combined_info or "").strip()
    lower_stack = stack.lower()

    if "pfcp" in lower_stack:
        match = re.search(
            r"PFCP\s+(?P<procedure>.+?)\s+(?P<kind>Request|Response)\b",
            info,
            re.IGNORECASE,
        )
        if match:
            procedure = " ".join(match.group("procedure").split())
            kind = match.group("kind").upper()
            relation = "PFCP::" + procedure.upper().replace(" ", "_")
            return ParsedSemantics(
                protocol="PFCP",
                relation=relation,
                procedure=procedure,
                message_type=f"PFCP_{kind}",
                direction=kind,
            )
        return ParsedSemantics(
            "PFCP", "PFCP::UNKNOWN", "PFCP_UNKNOWN", "PFCP", "UNKNOWN"
        )

    if "http2" in lower_stack or ":http:" in lower_stack:
        request = _HTTP_REQUEST_RE.search(info)
        if request:
            method = request.group("method").upper()
            path = normalize_sbi_path(request.group("path"))
            return ParsedSemantics(
                protocol="SBI",
                relation=_sbi_service(path),
                procedure=f"{method} {path}",
                message_type="HTTP2_REQUEST",
                direction="REQUEST",
                stream_id=request.group("stream"),
            )

        response = _HTTP_RESPONSE_RE.search(info)
        if response:
            status = response.group("status")
            return ParsedSemantics(
                protocol="SBI",
                relation="SBI::RESPONSE",
                procedure=f"HTTP_{status}",
                message_type="HTTP2_RESPONSE",
                direction="RESPONSE",
                result_code=status,
                stream_id=response.group("stream"),
            )

        data = _HTTP_DATA_RE.search(info)
        if data:
            payload_kind = "JSON" if "json" in lower_stack or "JSON" in info else "DATA"
            return ParsedSemantics(
                protocol="SBI",
                relation="SBI::DATA",
                procedure=f"HTTP2_DATA_{payload_kind}",
                message_type="HTTP2_DATA",
                direction="UNKNOWN",
                stream_id=data.group("stream"),
            )

        headers = _HTTP_HEADERS_RE.search(info)
        if headers:
            return ParsedSemantics(
                protocol="SBI",
                relation="SBI::HEADERS",
                procedure="HTTP2_HEADERS",
                message_type="HTTP2_HEADERS",
                direction="UNKNOWN",
                stream_id=headers.group("stream"),
            )

        return ParsedSemantics(
            "SBI", "SBI::UNKNOWN", "HTTP2_UNKNOWN", "HTTP2", "UNKNOWN"
        )

    if "ngap" in lower_stack:
        first = info.split(",", 1)[0].strip()
        # Some captures contain SCTP control text despite an NGAP dissector stack.
        if not first or first.startswith(("SACK", "HEARTBEAT")):
            return _transport_semantics(stack, info)

        direction = "UNKNOWN"
        lower_first = first.lower()
        if lower_first.startswith(("uplink", "initialue")):
            direction = "UPLINK"
        elif lower_first.startswith("downlink"):
            direction = "DOWNLINK"

        has_nas = "nas-5gs" in lower_stack or "nas_5gs" in lower_stack
        if has_nas and "," in info:
            nas_message = info.split(",", 1)[1].strip()
            # Combined exports can contain multiple decoded messages; the first
            # NAS item is the stable causal label.
            nas_message = nas_message.split(",", 1)[0].strip()
            return ParsedSemantics(
                protocol="NAS",
                relation=f"NGAP::{first}",
                procedure=nas_message or first,
                message_type="NAS_MESSAGE",
                direction=direction,
            )

        return ParsedSemantics(
            protocol="NGAP",
            relation="NGAP",
            procedure=first,
            message_type="NGAP_MESSAGE",
            direction=direction,
        )

    if "nas-5gs" in lower_stack or "nas_5gs" in lower_stack:
        first = info.split(",", 1)[0].strip() or "NAS_UNKNOWN"
        return ParsedSemantics(
            "NAS", "NAS", first, "NAS_MESSAGE", "UNKNOWN"
        )

    return _transport_semantics(stack, info)
