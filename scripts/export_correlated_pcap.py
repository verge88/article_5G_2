#!/usr/bin/env python3
"""Export causally correlated NGAP/HTTP2/PFCP events from a PCAP.

Unlike post-hoc repository export, this script snapshots correlation state while
packets are traversed in capture order. Identifiers discovered in later packets
never retroactively annotate earlier rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Iterable

from fedsiggraph.data.causal_correlation import CausalCorrelationTracker


_SUPI = re.compile(r"(imsi-\d{5,18})", re.IGNORECASE)
_SUCI = re.compile(r"(suci-[A-Za-z0-9_-]+)", re.IGNORECASE)
_PDU_JSON = re.compile(r'"pduSessionId"\s*:\s*"?([0-9]+)"?')
_SM_CONTEXT = re.compile(r"/sm-contexts/([^/?\s]+)")

_NGAP_NAMES = {
    4: "DownlinkNASTransport",
    15: "InitialUeMessage",
    26: "PDUSessionResourceModify",
    28: "PDUSessionResourceRelease",
    29: "PDUSessionResourceSetup",
    41: "UEContextRelease",
    46: "UplinkNASTransport",
}

_NAS_NAMES = {
    0x41: "RegistrationRequest",
    0x42: "RegistrationAccept",
    0x43: "RegistrationComplete",
    0x56: "AuthenticationRequest",
    0x57: "AuthenticationResponse",
    0x67: "ULNASTransport",
    0x68: "DLNASTransport",
    0xC1: "PDUSessionEstablishmentRequest",
    0xC2: "PDUSessionEstablishmentAccept",
    0xC9: "PDUSessionModificationRequest",
    0xCB: "PDUSessionModificationCommand",
    0xD1: "PDUSessionReleaseRequest",
    0xD3: "PDUSessionReleaseCommand",
    0xD4: "PDUSessionReleaseComplete",
}

_PFCP_NAMES = {
    1: "HeartbeatRequest",
    2: "HeartbeatResponse",
    50: "SessionEstablishmentRequest",
    51: "SessionEstablishmentResponse",
    52: "SessionModificationRequest",
    53: "SessionModificationResponse",
    54: "SessionDeletionRequest",
    55: "SessionDeletionResponse",
    56: "SessionReportRequest",
    57: "SessionReportResponse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--display-filter",
        default="http2||ngap||pfcp",
        help="Wireshark display filter used by pyshark.",
    )
    return parser.parse_args()


def _field(layer, name: str):
    try:
        field_names = getattr(layer, "field_names", ())
        if name not in field_names:
            return None
        return getattr(layer, name)
    except Exception:
        return None


def _text(value) -> str | None:
    if value is None:
        return None
    try:
        text = str(value.show)
    except Exception:
        text = str(value)
    text = text.strip()
    return text if text else None


def _int_field(value) -> int | None:
    if value is None:
        return None
    raw = getattr(value, "hex_value", None)
    if raw is None:
        raw = _text(value)
    try:
        if isinstance(raw, str):
            return int(raw, 0)
        return int(raw)
    except Exception:
        return None


def _decode_hex_payload(value) -> str:
    text = _text(value) or ""
    compact = text.replace(":", "").replace(" ", "")
    if not compact:
        return ""
    try:
        return bytes.fromhex(compact).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _packet_endpoints(packet) -> tuple[str, str]:
    ip = getattr(packet, "ip", None)
    if ip is None:
        return ("UNKNOWN", "UNKNOWN")
    return (str(ip.src), str(ip.dst))


def _packet_time(packet) -> str:
    value = getattr(packet, "sniff_timestamp", None)
    if value is not None:
        return str(value)
    frame_info = getattr(packet, "frame_info", None)
    if frame_info is not None:
        epoch = getattr(frame_info, "time_epoch", None)
        if epoch is not None:
            return str(epoch)
    raise ValueError("packet has no absolute timestamp")


def _supi_from_ngap(layer) -> str | None:
    msin = _text(_field(layer, "nas_5gs_mm_suci_msin"))
    mcc = _text(_field(layer, "e212_mcc"))
    mnc = _text(_field(layer, "e212_mnc"))
    if msin and mcc and mnc:
        return f"imsi-{mcc.zfill(3)}{mnc.zfill(2)}{msin}"
    return None


def _first_present(layer, names: Iterable[str]) -> str | None:
    for name in names:
        value = _text(_field(layer, name))
        if value:
            return value
    return None


def _row(
    *,
    timestamp: str,
    src: str,
    dst: str,
    message: str,
    ue_key: str | None,
    session_key: str | None,
) -> dict[str, str]:
    return {
        "time": timestamp,
        "src": src,
        "dst": dst,
        "message": message,
        "ue": ue_key or "",
        "session_context": session_key or "",
    }


def ngap_rows(packet, tracker: CausalCorrelationTracker) -> list[dict[str, str]]:
    src, dst = _packet_endpoints(packet)
    timestamp = _packet_time(packet)
    rows: list[dict[str, str]] = []

    for layer in getattr(packet, "layers", ()):
        if "ngap_pdu" not in getattr(layer, "field_names", ()):
            continue

        procedure_code = _int_field(_field(layer, "procedureCode"))
        procedure = _NGAP_NAMES.get(
            procedure_code,
            f"NGAPProcedureCode.{procedure_code}"
            if procedure_code is not None
            else "UNKNOWN_NGAP",
        )
        nas_code = _int_field(_field(layer, "nas_5gs_mm_message_type"))
        nas_name = _NAS_NAMES.get(nas_code)
        message = (
            f"{procedure}, NASType.{nas_name}"
            if nas_name is not None
            else procedure
        )

        supi = _supi_from_ngap(layer)
        suci_fragment = _first_present(
            layer,
            (
                "nas_5gs_mm_suci_msin",
                "nas_5gs_mm_suci_scheme_id",
            ),
        )
        pdu_id = _first_present(
            layer,
            (
                "nas_5gs_sm_pdu_session_id",
                "nas_5gs_sm_pdu_session_identity",
                "pdu_session_id",
            ),
        )
        snapshot = tracker.observe(
            supi=supi,
            suci=(
                f"suci-fragment:{suci_fragment}"
                if suci_fragment and not supi
                else None
            ),
            ran_ngap_id=_text(_field(layer, "ran_ue_ngap_id")),
            amf_ngap_id=_text(_field(layer, "amf_ue_ngap_id")),
            pdu_session_id=pdu_id,
        )
        rows.append(
            _row(
                timestamp=timestamp,
                src=src,
                dst=dst,
                message=message,
                ue_key=snapshot.ue_key,
                session_key=snapshot.session_key,
            )
        )
    return rows


def http2_rows(packet, tracker: CausalCorrelationTracker) -> list[dict[str, str]]:
    src, dst = _packet_endpoints(packet)
    timestamp = _packet_time(packet)
    rows: list[dict[str, str]] = []

    for layer in getattr(packet, "layers", ()):
        fields = getattr(layer, "field_names", ())
        if "streamid" not in fields:
            continue

        path = _text(_field(layer, "headers_path"))
        method = _text(_field(layer, "headers_method"))
        status = _text(_field(layer, "headers_status"))
        decoded = _decode_hex_payload(_field(layer, "data_data"))

        message = None
        visible = " ".join(part for part in (path, decoded) if part)
        if path:
            message = f"{(method or '?').upper()}: {path}"
        elif status:
            message = status
        elif decoded:
            message = "DATA Json data field"

        if message is None:
            continue

        snapshot = tracker.observe(
            supi=_first_match(_SUPI, visible),
            suci=_first_match(_SUCI, visible),
            pdu_session_id=_first_match(_PDU_JSON, decoded),
            sm_context=_first_match(_SM_CONTEXT, path or ""),
        )
        rows.append(
            _row(
                timestamp=timestamp,
                src=src,
                dst=dst,
                message=message,
                ue_key=snapshot.ue_key,
                session_key=snapshot.session_key,
            )
        )
    return rows


def _seid_values(layer) -> tuple[str | None, str | None]:
    value = _field(layer, "seid")
    primary = _text(value)
    alternate = None
    try:
        alternate_fields = getattr(value, "alternate_fields", ())
        if alternate_fields:
            alternate = _text(alternate_fields[0])
    except Exception:
        pass
    return primary, alternate


def pfcp_rows(packet, tracker: CausalCorrelationTracker) -> list[dict[str, str]]:
    layer = getattr(packet, "pfcp", None)
    if layer is None:
        return []

    msg_type = _int_field(_field(layer, "msg_type"))
    message = _PFCP_NAMES.get(
        msg_type,
        f"PFCPType.{msg_type}" if msg_type is not None else "UNKNOWN_PFCP",
    )
    imsi = _text(_field(layer, "e212_imsi"))
    supi = f"imsi-{imsi}" if imsi and not imsi.startswith("imsi-") else imsi
    primary_seid, alternate_seid = _seid_values(layer)

    snapshot = tracker.observe(
        supi=supi,
        seid=primary_seid,
        seid_c=alternate_seid,
    )
    src, dst = _packet_endpoints(packet)
    return [
        _row(
            timestamp=_packet_time(packet),
            src=src,
            dst=dst,
            message=message,
            ue_key=snapshot.ue_key,
            session_key=snapshot.session_key,
        )
    ]


def export_pcap(
    pcap: str | Path,
    output: str | Path,
    *,
    display_filter: str = "http2||ngap||pfcp",
) -> dict[str, int]:
    try:
        import pyshark
    except ImportError as exc:
        raise RuntimeError(
            "PCAP export requires pyshark and tshark; install the pcap extra"
        ) from exc

    tracker = CausalCorrelationTracker()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"NGAP": 0, "SBI": 0, "PFCP": 0}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "src",
                "dst",
                "message",
                "ue",
                "session_context",
            ],
            delimiter=";",
        )
        writer.writeheader()

        capture = pyshark.FileCapture(
            str(pcap),
            display_filter=display_filter,
            keep_packets=False,
        )
        try:
            for packet in capture:
                rows: list[tuple[str, dict[str, str]]] = []
                if hasattr(packet, "ngap"):
                    rows.extend(
                        ("NGAP", row) for row in ngap_rows(packet, tracker)
                    )
                if hasattr(packet, "http2"):
                    rows.extend(
                        ("SBI", row) for row in http2_rows(packet, tracker)
                    )
                if hasattr(packet, "pfcp"):
                    rows.extend(
                        ("PFCP", row) for row in pfcp_rows(packet, tracker)
                    )

                for protocol, row in rows:
                    writer.writerow(row)
                    counts[protocol] += 1
        finally:
            capture.close()

    return counts


def main() -> None:
    args = parse_args()
    counts = export_pcap(
        args.pcap,
        args.output,
        display_filter=args.display_filter,
    )
    print(json.dumps({"output": args.output, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
