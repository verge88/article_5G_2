"""Adapter from Granomaly packet CSVs to :class:`CanonicalEvent`.

The original Granomaly loader uses the DataFrame row index as time. This adapter
uses ``frame.time_epoch`` and converts it to integer nanoseconds, preserving
causal ordering without floating-point timestamp drift.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .schema import CanonicalEvent


_REQUIRED_COLUMNS = {
    "ip.src",
    "ip.dst",
    "frame.time_epoch",
    "frame.protocols",
}


def _epoch_to_ns(value: object) -> int:
    try:
        return int(Decimal(str(value)) * Decimal("1000000000"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid frame.time_epoch value: {value!r}") from exc


def _protocol_family(stack: str) -> str:
    """Extract a conservative protocol family from a Wireshark protocol stack."""
    tokens = [token.strip().lower() for token in str(stack).split(":") if token.strip()]
    for candidate, family in (
        ("ngap", "NGAP"),
        ("nas-5gs", "NAS"),
        ("nas_5gs", "NAS"),
        ("pfcp", "PFCP"),
        ("http2", "SBI"),
        ("http", "SBI"),
    ):
        if candidate in tokens:
            return family
    return tokens[-1].upper() if tokens else "UNKNOWN"


class GranomalyAdapter:
    """Convert Granomaly CSV rows to chronologically sorted canonical events."""

    def __init__(
        self,
        deployment_id: str,
        role_by_entity: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.deployment_id = deployment_id
        self.role_by_entity = dict(role_by_entity or {})

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

        events: list[CanonicalEvent] = []
        for event_id, row in work.iterrows():
            src = str(row["ip.src"])
            dst = str(row["ip.dst"])
            stack = str(row["frame.protocols"])
            label_name = (
                str(row["Label"])
                if "Label" in row and pd.notna(row["Label"])
                else "normal"
            )
            binary_label = 0 if label_name.lower() == "normal" else 1

            packet_length = 0
            if "frame.len" in row and pd.notna(row["frame.len"]):
                packet_length = int(row["frame.len"])

            events.append(
                CanonicalEvent(
                    event_id=int(event_id),
                    timestamp_ns=int(row["_timestamp_ns"]),
                    deployment_id=self.deployment_id,
                    src_entity=src,
                    dst_entity=dst,
                    src_type="NF",
                    dst_type="NF",
                    src_role=self.role_by_entity.get(src, "UNKNOWN_NF"),
                    dst_role=self.role_by_entity.get(dst, "UNKNOWN_NF"),
                    protocol=_protocol_family(stack),
                    relation=stack,
                    # Granomaly does not expose parsed 5G procedures in its minimal
                    # CSV schema. Keep the stack as a lossless baseline token.
                    procedure=stack,
                    packet_length=packet_length,
                    label=binary_label,
                    attack_type=None if binary_label == 0 else label_name,
                )
            )
        return events

    @staticmethod
    def benign(events: Iterable[CanonicalEvent]) -> list[CanonicalEvent]:
        return [event for event in events if event.label in (None, 0)]
