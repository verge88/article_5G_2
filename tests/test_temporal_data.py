import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fedsiggraph.data.encoding import SemanticEncoder
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.temporal_data import FrozenNodeIndexer, build_temporal_data


def event(i, ts, src, dst, procedure):
    return CanonicalEvent(
        event_id=i,
        timestamp_ns=ts,
        deployment_id="site-a",
        src_entity=src,
        dst_entity=dst,
        src_type="NF",
        dst_type="NF",
        src_role="AMF" if src.startswith("amf") else "SMF",
        dst_role="SMF" if dst.startswith("smf") else "UPF",
        protocol="SBI",
        relation=procedure,
        procedure=procedure,
        packet_length=128,
        label=0,
    )


def test_temporal_data_uses_relative_real_time_and_unknown_node():
    train = [
        event(1, 1_000_000_000, "amf-1", "smf-1", "P1"),
        event(2, 1_010_000_000, "amf-1", "smf-1", "P2"),
    ]
    encoder = SemanticEncoder.fit(train)
    nodes = FrozenNodeIndexer.fit(train)

    evaluation = [
        train[0],
        event(3, 1_025_000_000, "amf-new", "smf-1", "P1"),
    ]
    data = build_temporal_data(
        evaluation,
        encoder=encoder,
        node_indexer=nodes,
    )

    assert data.t.tolist() == [0, 25]
    assert data.src[1].item() == nodes.unknown_id
    assert data.dst[1].item() != nodes.unknown_id
    assert data.msg.shape == (2, 8)
