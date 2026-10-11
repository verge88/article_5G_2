import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_signaling_parser",
    ROOT / "scripts" / "prepare_signaling_parser.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_inventory_records_name_size_and_sha256(tmp_path):
    source = tmp_path / "capture_m_stats.csv"
    payload = b"time;src;dst;message;ue\n1;amf;smf;POST: /x;imsi-1\n"
    source.write_bytes(payload)

    inventory = MODULE.source_inventory(
        csv_path=str(source),
        directory=None,
        pattern="*_m_stats.csv",
    )

    assert inventory == [
        {
            "name": "capture_m_stats.csv",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    assert str(tmp_path) not in repr(inventory)
