import gzip
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_correlated_pcap",
    ROOT / "scripts" / "export_correlated_pcap.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_materialized_capture_decompresses_and_cleans_up(tmp_path):
    source = tmp_path / "example.pcap.gz"
    payload = b"pcap-payload"
    with gzip.open(source, "wb") as handle:
        handle.write(payload)

    with MODULE.materialized_capture(source) as capture:
        capture = Path(capture)
        assert capture.read_bytes() == payload
        temporary = capture

    assert not temporary.exists()
