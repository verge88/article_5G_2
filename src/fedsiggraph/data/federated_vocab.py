"""Stable semantic IDs for federated clients without vocabulary exchange."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .schema import CanonicalEvent
from .vocab import PAD_TOKEN, UNK_TOKEN


@dataclass(frozen=True, slots=True)
class FixedVocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def from_tokens(cls, tokens: tuple[str, ...]) -> "FixedVocabulary":
        mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in tokens:
            if token not in mapping:
                mapping[token] = len(mapping)
        return cls(mapping)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, value: str) -> int:
        return self.token_to_id.get(str(value), self.unk_id)

    def __len__(self) -> int:
        return len(self.token_to_id)


@dataclass(frozen=True, slots=True)
class StableHashVocabulary:
    """Deterministically map arbitrary semantic tokens into shared buckets."""

    namespace: str
    buckets: int

    def __post_init__(self) -> None:
        if self.buckets < 2:
            raise ValueError("hash vocabulary requires at least two buckets")

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, value: str) -> int:
        token = str(value)
        if not token or token == UNK_TOKEN:
            return self.unk_id
        payload = f"{self.namespace}\0{token}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % self.buckets
        return 2 + bucket

    def __len__(self) -> int:
        return self.buckets + 2


ROLE_TOKENS = (
    "UNKNOWN",
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
    "gNB",
    "UE",
)

PROTOCOL_TOKENS = (
    "UNKNOWN",
    "SBI",
    "PFCP",
    "NGAP",
    "NAS",
    "TCP",
    "SCTP",
    "UDP",
    "ICMP",
    "OTHER",
)

DIRECTION_TOKENS = (
    "UNKNOWN",
    "REQUEST",
    "RESPONSE",
    "UPLINK",
    "DOWNLINK",
)


@dataclass(frozen=True, slots=True)
class FederatedSemanticVocabularies:
    """Common ID space requiring no client token-list exchange."""

    src_role: FixedVocabulary
    dst_role: FixedVocabulary
    protocol: FixedVocabulary
    relation: StableHashVocabulary
    procedure: StableHashVocabulary
    message_type: StableHashVocabulary
    direction: FixedVocabulary

    @classmethod
    def create(
        cls,
        *,
        relation_buckets: int = 512,
        procedure_buckets: int = 2048,
        message_type_buckets: int = 128,
        schema_version: str = "fedsiggraph-v1",
    ) -> "FederatedSemanticVocabularies":
        role = FixedVocabulary.from_tokens(ROLE_TOKENS)
        return cls(
            src_role=role,
            dst_role=role,
            protocol=FixedVocabulary.from_tokens(PROTOCOL_TOKENS),
            relation=StableHashVocabulary(
                f"{schema_version}:relation", relation_buckets
            ),
            procedure=StableHashVocabulary(
                f"{schema_version}:procedure", procedure_buckets
            ),
            message_type=StableHashVocabulary(
                f"{schema_version}:message_type", message_type_buckets
            ),
            direction=FixedVocabulary.from_tokens(DIRECTION_TOKENS),
        )

    def encode_event(self, event: CanonicalEvent) -> dict[str, int]:
        return {
            "src_role_id": self.src_role.encode(event.src_role),
            "dst_role_id": self.dst_role.encode(event.dst_role),
            "protocol_id": self.protocol.encode(event.protocol),
            "relation_id": self.relation.encode(event.relation),
            "procedure_id": self.procedure.encode(event.procedure),
            "message_type_id": self.message_type.encode(event.message_type),
            "direction_id": self.direction.encode(event.direction),
        }

    def sizes(self) -> dict[str, int]:
        return {
            "src_role": len(self.src_role),
            "dst_role": len(self.dst_role),
            "protocol": len(self.protocol),
            "relation": len(self.relation),
            "procedure": len(self.procedure),
            "message_type": len(self.message_type),
            "direction": len(self.direction),
        }

    def config(self) -> dict[str, object]:
        return {
            "kind": "stable_hash_v1",
            "role_tokens": list(ROLE_TOKENS),
            "protocol_tokens": list(PROTOCOL_TOKENS),
            "direction_tokens": list(DIRECTION_TOKENS),
            "relation": {
                "namespace": self.relation.namespace,
                "buckets": self.relation.buckets,
            },
            "procedure": {
                "namespace": self.procedure.namespace,
                "buckets": self.procedure.buckets,
            },
            "message_type": {
                "namespace": self.message_type.namespace,
                "buckets": self.message_type.buckets,
            },
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.config(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
