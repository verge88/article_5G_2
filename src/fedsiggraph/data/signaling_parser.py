"""Adapter for CSV output produced by lsinfo3/5G-Signaling-Parser.

Confirmed upstream CSV schema:
    time;src;dst;message;ue

The adapter never stores the raw UE/SUPI value in CanonicalEvent. It derives a
deployment-local pseudonym so cross-protocol context can be reconstructed
without federating subscriber identity.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

from .schema import CanonicalEvent


_REQUIRED_COLUMNS = {"time", "src", "dst", "message", "ue"}
_SESSION_COLUMNS = (
    "session_context",
    "pdu_session_id",
    "pduid",
    "sm_context",
    "smcontext",
    "seid_c",
    "seid_u",
    "seid",
)
_HTTP_METHOD = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_HTTP_STATUS = re.compile(r"^[1-5][0-9]{2}$")
_HTTP_COMPACT = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)[-_](.+)$",
    re.IGNORECASE,
)
_NAS_TYPE = re.compile(r"NASType\.([A-Za-z0-9_]+)")

_PFCP_PROCEDURES = (
    "SessionEstablishmentRequest",
    "SessionEstablishmentResponse",
    "SessionModificationRequest",
    "SessionModificationResponse",
    "SessionMangementRequest",
    "SessionMangementResponse",
    "SeesionMangementRequest",  # spelling present in upstream parser
    "SessionDeletionRequest",
    "SessionDeletionResponse",
    "Heartbeat",
)

_ROLE_ALIASES = {
    "gnb": "gNB",
    "gnb1": "gNB",
    "ran": "gNB",
    "amf": "AMF",
    "smf": "SMF",
    "upf": "UPF",
    "nrf": "NRF",
    "udm": "UDM",
    "udr": "UDR",
    "ausf": "AUSF",
    "pcf": "PCF",
    "nssf": "NSSF",
    "bsf": "BSF",
    "nef": "NEF",
}


def _epoch_to_ns(value: object) -> int:
    try:
        return int(Decimal(str(value)) * Decimal("1000000000"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid signaling-parser time value: {value!r}") from exc


def _normalize_role(entity: str, explicit: Mapping[str, str]) -> str:
    if entity in explicit:
        return str(explicit[entity])

    compact = re.sub(r"[^a-z0-9]", "", entity.lower())
    for alias, role in _ROLE_ALIASES.items():
        if compact == alias or compact.startswith(alias):
            return role
    return "UNKNOWN_NF"


def _local_pseudonym(deployment_id: str, kind: str, value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "nan"}:
        return None

    digest = hashlib.blake2b(
        f"{deployment_id}|{kind}|{raw}".encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return f"{kind}:{digest}"


def _http_semantics(message: str) -> tuple[str, str, str, str] | None:
    stripped = message.strip()
    match = _HTTP_METHOD.match(stripped)
    if match:
        method = match.group(1).upper()
        path = match.group(2).strip()
        normalized_path = path.split("?", 1)[0]
        first = next(
            (part for part in normalized_path.split("/") if part),
            "unknown-service",
        )
        return (
            "SBI",
            f"SBI:{first}",
            f"{method}:{normalized_path}",
            "REQUEST",
        )

    compact = _HTTP_COMPACT.match(stripped)
    if compact:
        method = compact.group(1).upper()
        endpoint = compact.group(2).strip()
        service = endpoint.split("-", 1)[0].split("_", 1)[0]
        return (
            "SBI",
            f"SBI:{service}",
            f"{method}:{endpoint}",
            "REQUEST",
        )

    if _HTTP_STATUS.match(stripped):
        return ("SBI", "SBI:response", f"HTTP_STATUS_{stripped}", "RESPONSE")

    if stripped == "DATA Json data field":
        return ("SBI", "SBI:data", "HTTP2_DATA", "DATA")
    return None


def _pfcp_semantics(message: str) -> tuple[str, str, str, str] | None:
    for procedure in _PFCP_PROCEDURES:
        if procedure.lower() in message.lower():
            direction = (
                "REQUEST"
                if "request" in procedure.lower()
                else "RESPONSE"
                if "response" in procedure.lower()
                else "UNKNOWN"
            )
            normalized = (
                procedure.replace("Seesion", "Session")
                .replace("Mangement", "Management")
            )
            return ("PFCP", "PFCP", normalized, direction)

    if "pfcp" in message.lower():
        return ("PFCP", "PFCP", message.strip(), "UNKNOWN")
    return None


def _ngap_semantics(message: str) -> tuple[str, str, str, str, str]:
    stripped = message.strip()
    parts = [part.strip() for part in stripped.split(",", 1)]
    procedure = parts[0] if parts and parts[0] else "UNKNOWN_NGAP"

    nas = _NAS_TYPE.search(stripped)
    message_type = nas.group(1) if nas else procedure

    direction = "UNKNOWN"
    lower = procedure.lower()
    if lower.startswith("uplink"):
        direction = "UPLINK"
    elif lower.startswith("downlink"):
        direction = "DOWNLINK"
    elif procedure == "InitialUeMessage":
        direction = "UPLINK"

    return ("NGAP", "NGAP", procedure, direction, message_type)


class SignalingParserAdapter:
    """Convert 5G-Signaling-Parser message statistics into canonical events."""

    def __init__(
        self,
        deployment_id: str,
        role_by_entity: Mapping[str, str] | None = None,
    ) -> None:
        self.deployment_id = deployment_id
        self.role_by_entity = dict(role_by_entity or {})

    def from_csv(
        self,
        path: str | Path,
        *,
        label: int | None = None,
        attack_type: str | None = None,
    ) -> list[CanonicalEvent]:
        df = pd.read_csv(path, sep=";")
        return self.from_dataframe(
            df,
            label=label,
            attack_type=attack_type,
        )

    def from_directory(
        self,
        path: str | Path,
        *,
        pattern: str = "*_m_stats.csv",
        label: int | None = None,
        attack_type: str | None = None,
    ) -> list[CanonicalEvent]:
        root = Path(path)
        events: list[CanonicalEvent] = []
        for csv_path in sorted(root.glob(pattern)):
            events.extend(
                self.from_csv(
                    csv_path,
                    label=label,
                    attack_type=attack_type,
                )
            )
        return self._renumber_and_sort(events)

    def from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        label: int | None = None,
        attack_type: str | None = None,
    ) -> list[CanonicalEvent]:
        missing = sorted(_REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(
                f"signaling-parser input is missing required columns: {missing}"
            )
        if label not in (None, 0, 1):
            raise ValueError("label must be None, 0, or 1")

        events: list[CanonicalEvent] = []
        for source_order, row in df.reset_index(drop=True).iterrows():
            src = str(row["src"]).strip()
            dst = str(row["dst"]).strip()
            message = str(row["message"]).strip()

            http = _http_semantics(message)
            pfcp = _pfcp_semantics(message)
            message_type = message

            if http is not None:
                protocol, relation, procedure, direction = http
                message_type = procedure
            elif pfcp is not None:
                protocol, relation, procedure, direction = pfcp
                message_type = procedure
            else:
                (
                    protocol,
                    relation,
                    procedure,
                    direction,
                    message_type,
                ) = _ngap_semantics(message)

            ue_context = _local_pseudonym(
                self.deployment_id,
                "ue",
                row["ue"],
            )

            # Enriched exports may expose one or more identifiers that bind
            # NGAP/NAS, SBI and PFCP messages to the same PDU/SM session.
            # Prefer a PDU/SM identifier over SEID, but never store the raw
            # value: session_context is deployment-local and pseudonymized.
            session_raw = None
            session_kind = None
            for column in _SESSION_COLUMNS:
                if column in row.index and pd.notna(row[column]):
                    value = str(row[column]).strip()
                    if value and value.lower() not in {"none", "nan"}:
                        session_raw = value
                        session_kind = column
                        break

            session_context = (
                _local_pseudonym(
                    self.deployment_id,
                    f"session:{session_kind}",
                    session_raw,
                )
                if session_raw is not None
                else None
            )

            events.append(
                CanonicalEvent(
                    event_id=int(source_order),
                    timestamp_ns=_epoch_to_ns(row["time"]),
                    deployment_id=self.deployment_id,
                    src_entity=src,
                    dst_entity=dst,
                    src_type=("gNB" if _normalize_role(src, self.role_by_entity) == "gNB" else "NF"),
                    dst_type=("gNB" if _normalize_role(dst, self.role_by_entity) == "gNB" else "NF"),
                    src_role=_normalize_role(src, self.role_by_entity),
                    dst_role=_normalize_role(dst, self.role_by_entity),
                    protocol=protocol,
                    relation=relation,
                    procedure=procedure,
                    message_type=message_type,
                    direction=direction,
                    ue_context=ue_context,
                    session_context=session_context,
                    label=label,
                    attack_type=(attack_type if label == 1 else None),
                )
            )

        return self._renumber_and_sort(events)

    @staticmethod
    def _renumber_and_sort(
        events: Iterable[CanonicalEvent],
    ) -> list[CanonicalEvent]:
        ordered = sorted(
            events,
            key=lambda event: (event.timestamp_ns, event.event_id),
        )
        return [
            event.synthetic(event_id=index)
            for index, event in enumerate(ordered)
        ]
