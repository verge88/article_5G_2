import gzip
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parse_attack_pcaps",
    ROOT / "scripts" / "parse_attack_pcaps.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_materialize_capture_decompresses_gzip(tmp_path):
    source = tmp_path / "example.pcap.gz"
    payload = b"pcap-bytes"
    with gzip.open(source, "wb") as handle:
        handle.write(payload)

    materialized, temporary = MODULE.materialize_capture(
        source,
        tmp_path / "out",
        dry_run=False,
    )

    assert temporary == materialized
    assert materialized.name == "example.pcap"
    assert materialized.read_bytes() == payload
