"""Causal UE/session identifier correlation.

The tracker is intentionally independent of pyshark. A PCAP adapter feeds only
identifiers visible in the *current* packet and receives an immutable snapshot.
Later packets may add aliases, but previously emitted snapshots never change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    return text


@dataclass(frozen=True, slots=True)
class CorrelationSnapshot:
    """Correlation state available immediately after the current event."""

    ue_key: str | None
    session_key: str | None


class CausalCorrelationTracker:
    """Link identifiers only when a current event provides causal evidence."""

    UE_KINDS = ("supi", "suci", "ran_ngap_id", "amf_ngap_id")
    SESSION_KINDS = ("pdu_session_id", "sm_context", "seid_c", "seid_u", "seid")

    def __init__(self) -> None:
        self._ue_alias: dict[tuple[str, str], str] = {}
        self._session_alias: dict[tuple[str, str], str] = {}
        self._session_ue: dict[str, str] = {}
        self._next_ue = 1
        self._next_session = 1

    def _new_ue(self) -> str:
        key = f"ue-{self._next_ue}"
        self._next_ue += 1
        return key

    def _new_session(self) -> str:
        key = f"session-{self._next_session}"
        self._next_session += 1
        return key

    @staticmethod
    def _items(kinds: Iterable[str], values: dict[str, object | None]):
        for kind in kinds:
            value = _clean(values.get(kind))
            if value is not None:
                yield kind, value

    def _merge_ue_keys(self, keep: str, remove: str) -> None:
        if keep == remove:
            return
        for alias, key in list(self._ue_alias.items()):
            if key == remove:
                self._ue_alias[alias] = keep
        for session, key in list(self._session_ue.items()):
            if key == remove:
                self._session_ue[session] = keep

    def _merge_session_keys(self, keep: str, remove: str) -> None:
        if keep == remove:
            return
        for alias, key in list(self._session_alias.items()):
            if key == remove:
                self._session_alias[alias] = keep

        keep_ue = self._session_ue.get(keep)
        remove_ue = self._session_ue.get(remove)
        if keep_ue is None and remove_ue is not None:
            self._session_ue[keep] = remove_ue
        self._session_ue.pop(remove, None)

    def observe(
        self,
        *,
        supi: object | None = None,
        suci: object | None = None,
        ran_ngap_id: object | None = None,
        amf_ngap_id: object | None = None,
        pdu_session_id: object | None = None,
        sm_context: object | None = None,
        seid_c: object | None = None,
        seid_u: object | None = None,
        seid: object | None = None,
    ) -> CorrelationSnapshot:
        """Observe identifiers visible in one event and return its causal context."""
        values = {
            "supi": supi,
            "suci": suci,
            "ran_ngap_id": ran_ngap_id,
            "amf_ngap_id": amf_ngap_id,
            "pdu_session_id": pdu_session_id,
            "sm_context": sm_context,
            "seid_c": seid_c,
            "seid_u": seid_u,
            "seid": seid,
        }
        ue_items = list(self._items(self.UE_KINDS, values))
        session_items = list(self._items(self.SESSION_KINDS, values))

        existing_ues = [
            self._ue_alias[item]
            for item in ue_items
            if item in self._ue_alias
        ]
        existing_sessions = [
            self._session_alias[item]
            for item in session_items
            if item in self._session_alias
        ]

        # A known session may identify the UE even when the current PFCP/SBI
        # message carries no SUPI/RAN identifier.
        session_ue_candidates = [
            self._session_ue[key]
            for key in existing_sessions
            if key in self._session_ue
        ]

        ue_key = (
            existing_ues[0]
            if existing_ues
            else session_ue_candidates[0]
            if session_ue_candidates
            else self._new_ue()
            if ue_items
            else None
        )
        if ue_key is not None:
            for other in {*existing_ues, *session_ue_candidates}:
                self._merge_ue_keys(ue_key, other)
            for item in ue_items:
                self._ue_alias[item] = ue_key

        session_key = (
            existing_sessions[0]
            if existing_sessions
            else self._new_session()
            if session_items
            else None
        )
        if session_key is not None:
            for other in set(existing_sessions):
                self._merge_session_keys(session_key, other)
            for item in session_items:
                self._session_alias[item] = session_key

            linked_ue = self._session_ue.get(session_key)
            if ue_key is None and linked_ue is not None:
                ue_key = linked_ue
            elif ue_key is not None:
                if linked_ue is not None and linked_ue != ue_key:
                    self._merge_ue_keys(ue_key, linked_ue)
                self._session_ue[session_key] = ue_key

        return CorrelationSnapshot(
            ue_key=ue_key,
            session_key=session_key,
        )
