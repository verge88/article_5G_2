"""Adapter from Granomaly CSVs to semantic :class:`CanonicalEvent`.

Unlike the original Granomaly loader, this adapter keeps real timestamps and
extracts stable procedure semantics from ``combined_info``.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .schema import CanonicalEvent
from .semantics import ParsedSemantics, parse_granomaly_semantics


_REQUIRED_COLUMNS = {
    "ip.src",
    "ip.dst",
    "frame.time_epoch",
    "frame.protocols",
}

# Published QEMU inventory in TF0x42/granomaly/framework/ansible/hosts-qemu.yml.
# User-supplied mappings override these values.
GRANOMALY_QEMU_ROLE_MAP = {
    "10.211.55.2": "gNB",
    "10.211.55.3": "UE",
    "10.211.55.4": "AMF",
    "10.211.55.5": "SMF",
    "10.211.55.6": "UPF",
    "10.211.55.7": "NRF",
    "10.211.55.8": "UDM",
    "10.211.55.9": "AUSF",
    "10.211.55.10": "UDR",
    "10.211.55.11": "SCP",
    "10.211.55.12": "PCF",
    "10.211.55.13": "BSF",
    "10.211.55.14": "NSSF",
}

_NF_ROLES = {
    "AMF",
    "SMF",
    "UPF",
    "NRF",
    "UDM",
    "AUSF",
    "UDR",
    "SCP",
    "PCF",
    "BSF",
    "NSSF",
}


def _epoch_to_ns(value: object) -> int:
    try:
        return int(Decimal(str(value)) * Decimal("1000000000"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid frame.time_epoch value: {value!r}") from exc


def _safe_text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _port_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "?"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def _entity_type(role: str, entity: str) -> str:
    if role in _NF_ROLES:
        return "NF"
    if role == "gNB":
        return "gNB"
    if role == "UE":
        return "UE"
    if entity.endswith(".255"):
        return "BROADCAST"
    return "HOST"


def _connection_key(row: pd.Series, src: str, dst: str) -> tuple[str, str]:
    src_endpoint = f"{src}:{_port_text(row.get('srcport'))}"
    dst_endpoint = f"{dst}:{_port_text(row.get('dstport'))}"
    return tuple(sorted((src_endpoint, dst_endpoint)))


def _resolve_sbi_stream(
    semantics: ParsedSemantics,
    *,
    stream_state: dict[tuple[tuple[str, str], str], tuple[str, str]],
    connection: tuple[str, str],
) -> ParsedSemantics:
    """Attach response/DATA frames to the normalized request on the same stream."""
    if semantics.protocol != "SBI" or semantics.stream_id is None:
        return semantics

    key = (connection, semantics.stream_id)
    if semantics.message_type == "HTTP2_REQUEST":
        stream_state[key] = (semantics.relation, semantics.procedure)
        return semantics

    request = stream_state.get(key)
    if request is None:
        return semantics

    relation, request_procedure = request
    if semantics.message_type == "HTTP2_RESPONSE":
        return replace(
            semantics,
            relation=relation,
            procedure=f"RESPONSE {request_procedure}",
        )
    if semantics.message_type in {"HTTP2_DATA", "HTTP2_HEADERS"}:
        return replace(
            semantics,
            relation=relation,
            procedure=f"{semantics.message_type} {request_procedure}",
        )
    return semantics


class GranomalyAdapter:
    """Convert Granomaly rows to chronologically sorted canonical events."""

    def __init__(
        self,
        deployment_id: str,
        role_by_entity: Optional[Mapping[str, str]] = None,
        *,
        use_published_qemu_roles: bool = True,
    ) -> None:
        self.deployment_id = deployment_id
        roles = dict(GRANOMALY_QEMU_ROLE_MAP if use_published_qemu_roles else {})
        roles.update(dict(role_by_entity or {}))
        self.role_by_entity = roles

    def from_csv(self, path: str | Path) -> list[CanonicalEvent]:
        return self.from_dataframe(pd.read_csv(path))

    def from_dataframe(self, df: pd.DataFrame) -> list[CanonicalEvent]:
        missing = sorted(_REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Granomaly input is missing required columns: {missing}")

        work = df.copy()
        work["_source_order"] = range(len(work))
        work["_timestamp_ns"] = work["frame.time_epoch"].map(_epoch_to_ns)
        work = work.sort_values(
            ["_timestamp_ns", "_source_order"], kind="stable"
        ).reset_index(drop=True)

        stream_state: dict[
            tuple[tuple[str, str], str], tuple[str, str]
        ] = {}
        events: list[CanonicalEvent] = []

        for event_id, row in work.iterrows():
            src = str(row["ip.src"])
            dst = str(row["ip.dst"])
            stack = _safe_text(row["frame.protocols"])
            info = _safe_text(row.get("combined_info"))

            semantics = parse_granomaly_semantics(stack, info)
            semantics = _resolve_sbi_stream(
                semantics,
                stream_state=stream_state,
                connection=_connection_key(row, src, dst),
            )

            label_name = (
                str(row["Label"])
                if "Label" in row and pd.notna(row["Label"])
                else "normal"
            )
            binary_label = 0 if label_name.lower() == "normal" else 1

            packet_length = 0
            if "frame.len" in row and pd.notna(row["frame.len"]):
                packet_length = int(row["frame.len"])

            src_role = self.role_by_entity.get(src, "UNKNOWN")
            dst_role = self.role_by_entity.get(dst, "UNKNOWN")

            events.append(
                CanonicalEvent(
                    event_id=int(event_id),
                    timestamp_ns=int(row["_timestamp_ns"]),
                    deployment_id=self.deployment_id,
                    src_entity=src,
                    dst_entity=dst,
                    src_type=_entity_type(src_role, src),
                    dst_type=_entity_type(dst_role, dst),
                    src_role=src_role,
                    dst_role=dst_role,
                    protocol=semantics.protocol,
                    relation=semantics.relation,
                    procedure=semantics.procedure,
                    message_type=semantics.message_type,
                    direction=semantics.direction,
                    result_code=semantics.result_code,
                    cause=semantics.cause,
                    packet_length=packet_length,
                    label=binary_label,
                    attack_type=None if binary_label == 0 else label_name,
                )
            )
        return events

    @staticmethod
    def benign(events: Iterable[CanonicalEvent]) -> list[CanonicalEvent]:
        return [event for event in events if event.label in (None, 0)]
