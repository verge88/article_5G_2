import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import TemporalDataLoader

from fedsiggraph.baselines.tgn import CausalTGNBaseline
from fedsiggraph.data.features import CausalNumericFeatureEncoder
from fedsiggraph.data.index import LocalNodeIndex
from fedsiggraph.data.pyg import to_temporal_data
from fedsiggraph.data.schema import CanonicalEvent
from fedsiggraph.data.vocab import SemanticVocabularies
from fedsiggraph.models.semantic_tgn import SemanticCausalTGN
from fedsiggraph.models.temporal_heat import TemporalHEAT


def event(
    event_id: int,
    timestamp_ns: int,
    *,
    procedure: str = "CreateSMContext",
) -> CanonicalEvent:
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
        procedure=procedure,
        message_type="HTTP2_REQUEST",
        direction="REQUEST",
        packet_length=256,
        label=0,
    )


def build_toy_data():
    events = [
        event(1, 1_000_000),
        event(2, 2_000_000, procedure="UpdateSMContext"),
    ]
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
    return data, vocab, node_index


def vocab_sizes(vocab):
    return {
        "src_role": len(vocab.src_role),
        "dst_role": len(vocab.dst_role),
        "protocol": len(vocab.protocol),
        "relation": len(vocab.relation),
        "procedure": len(vocab.procedure),
        "message_type": len(vocab.message_type),
        "direction": len(vocab.direction),
    }


def test_temporal_data_and_tgn_baseline_construct():
    data, _, _ = build_toy_data()

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


def test_temporal_loader_preserves_semantic_attributes():
    data, vocab, _ = build_toy_data()
    batch = next(iter(TemporalDataLoader(data, batch_size=1)))

    assert batch.protocol_id.shape == (1,)
    assert batch.relation_id.shape == (1,)
    assert batch.procedure_id.shape == (1,)
    assert batch.src_role_id.shape == (1,)
    assert batch.dst_role_id.shape == (1,)

    model = SemanticCausalTGN(
        num_nodes=data.num_nodes,
        numeric_dim=data.msg.size(-1),
        vocab_sizes=vocab_sizes(vocab),
        event_dim=16,
        memory_dim=16,
        time_dim=16,
        embedding_dim=16,
    )
    encoded = model.encode_batch(batch)
    assert encoded.shape == (1, 16)

    logits = model.score_embeddings(
        torch.zeros((1, 16)),
        torch.zeros((1, 16)),
        encoded,
    )
    assert logits.shape == (1, 1)


def test_temporal_heat_accepts_node_relation_and_edge_features():
    data, vocab, node_index = build_toy_data()
    model = TemporalHEAT(
        num_nodes=data.num_nodes,
        numeric_dim=data.msg.size(-1),
        vocab_sizes=vocab_sizes(vocab),
        num_node_types=node_index.num_node_types,
        num_edge_types=len(vocab.relation),
        event_dim=16,
        memory_dim=16,
        time_dim=8,
        embedding_dim=16,
        heads=4,
    )

    memory = torch.randn((2, 16))
    last_update = torch.tensor([1000, 1000], dtype=torch.long)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_t = torch.tensor([500], dtype=torch.long)
    edge_msg = torch.randn((1, 16))
    node_type = torch.tensor([1, 1], dtype=torch.long)
    edge_type = torch.tensor([2], dtype=torch.long)

    out = model.embedding(
        memory,
        last_update,
        edge_index,
        edge_t,
        edge_msg,
        node_type,
        edge_type,
    )
    assert out.shape == (2, 16)
