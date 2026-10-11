import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fedsiggraph.baselines.tgn import CausalTGNBaseline
from fedsiggraph.data.features import CausalNumericFeatureEncoder
from fedsiggraph.data.index import LocalNodeIndex
from fedsiggraph.data.pyg import to_temporal_data
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.vocab import SemanticVocabularies


def event(event_id: int, timestamp_ns: int) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        timestamp_ns=timestamp_ns,
        deployment_id="toy",
        src_entity="amf-1",
        dst_entity="smf-1",
        src_type="NF",
        dst_type="NF",
        src_role="AMF",
        dst_role="SMF",
        protocol="SBI",
        relation="Nsmf_PDUSession",
        procedure="CreateSMContext",
        packet_length=256,
        label=0,
    )


def test_temporal_data_and_tgn_baseline_construct():
    events = [event(1, 1_000_000), event(2, 2_000_000)]
    vocab = SemanticVocabularies.fit(events)
    node_index = LocalNodeIndex.fit(events)
    numeric = CausalNumericFeatureEncoder.fit(events)

    data = to_temporal_data(
        events,
        vocab=vocab,
        node_index=node_index,
        numeric_encoder=numeric,
        time_origin_ns=events[0].timestamp_ns,
    )

    assert data.num_events == 2
    assert data.msg.shape == (2, 4)
    assert data.procedure_id.shape == (2,)
    assert data.t.tolist() == [0, 1000]

    model = CausalTGNBaseline(
        num_nodes=data.num_nodes,
        msg_dim=data.msg.size(-1),
        memory_dim=16,
        time_dim=16,
        embedding_dim=16,
    )
    logits = model.score_embeddings(
        torch.zeros((1, 16)),
        torch.zeros((1, 16)),
    )
    assert logits.shape == (1, 1)
