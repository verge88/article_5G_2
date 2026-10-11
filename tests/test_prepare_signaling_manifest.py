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


def test_attack_window_labels_only_events_inside_interval():
    from fedsiggraph.data.schema import CanonicalEvent

    def event(event_id, seconds):
        return CanonicalEvent(
            event_id=event_id,
            timestamp_ns=int(seconds * 1_000_000_000),
            deployment_id="site-a",
            src_entity="amf",
            dst_entity="smf",
            src_type="NF",
            dst_type="NF",
            src_role="AMF",
            dst_role="SMF",
            protocol="SBI",
            relation="Nsmf_PDUSession",
            procedure="CreateSMContext",
            label=0,
        )

    labeled = MODULE.apply_attack_window(
        [event(1, 1), event(2, 2), event(3, 3)],
        start_ns=MODULE.seconds_to_ns("1.5"),
        end_ns=MODULE.seconds_to_ns("2.5"),
        attack_type="windowed-attack",
    )

    assert [item.label for item in labeled] == [0, 1, 0]
    assert [item.attack_type for item in labeled] == [
        None,
        "windowed-attack",
        None,
    ]
