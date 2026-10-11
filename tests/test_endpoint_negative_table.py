import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import TemporalData

from fedsiggraph.training.negatives import EndpointNegativeTable


def test_endpoint_negative_uses_known_node_of_incompatible_role():
    # Nodes: 1=AMF, 2=SMF, 3=UDM.
    data = TemporalData(
        src=torch.tensor([1, 1, 1], dtype=torch.long),
        dst=torch.tensor([2, 2, 3], dtype=torch.long),
        t=torch.tensor([0, 1, 2], dtype=torch.long),
        msg=torch.zeros((3, 1)),
        y=torch.zeros(3, dtype=torch.long),
        src_role_id=torch.tensor([10, 10, 10], dtype=torch.long),
        dst_role_id=torch.tensor([20, 20, 30], dtype=torch.long),
        protocol_id=torch.tensor([5, 5, 5], dtype=torch.long),
        procedure_id=torch.tensor([100, 100, 200], dtype=torch.long),
    )

    sampler = EndpointNegativeTable(data, seed=7)
    batch = data[0:1]
    neg_dst, neg_role, mask = sampler.sample(batch)

    assert mask.tolist() == [True]
    assert neg_dst.tolist() == [3]
    assert neg_role.tolist() == [30]
    assert neg_dst.item() != batch.dst.item()
    assert neg_role.item() != batch.dst_role_id.item()



def test_endpoint_negative_does_not_mix_src_and_dst_role_vocabularies():
    # Node 2 appears as destination role-id 20 and later as source role-id 99.
    # The two IDs live in different vocabularies and must not make node 2
    # appear semantically ambiguous for destination sampling.
    data = TemporalData(
        src=torch.tensor([1, 1, 2], dtype=torch.long),
        dst=torch.tensor([2, 3, 3], dtype=torch.long),
        t=torch.tensor([0, 1, 2], dtype=torch.long),
        msg=torch.zeros((3, 1)),
        y=torch.zeros(3, dtype=torch.long),
        src_role_id=torch.tensor([10, 10, 99], dtype=torch.long),
        dst_role_id=torch.tensor([20, 30, 30], dtype=torch.long),
        protocol_id=torch.tensor([5, 5, 5], dtype=torch.long),
        procedure_id=torch.tensor([100, 200, 300], dtype=torch.long),
    )

    sampler = EndpointNegativeTable(data, seed=1)

    assert 2 in sampler.nodes_by_role[20]
