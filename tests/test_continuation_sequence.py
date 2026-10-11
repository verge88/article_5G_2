import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fedsiggraph.models.continuation import ContextContinuationHead


VOCAB = {
    "protocol": 8,
    "relation": 8,
    "procedure": 8,
}


def _run(head, *, first_protocol: int):
    z_src = torch.zeros((1, 4))
    z_dst = torch.zeros((1, 4))

    return head(
        z_src,
        z_dst,
        prev_protocol_id=torch.tensor([3]),
        prev_relation_id=torch.tensor([3]),
        prev_procedure_id=torch.tensor([3]),
        protocol_history_id=torch.tensor([[first_protocol, 3]]),
        relation_history_id=torch.tensor([[2, 3]]),
        procedure_history_id=torch.tensor([[2, 3]]),
        sequence_mask=torch.tensor([[True, True]]),
    )


def test_sequence_mode_uses_older_history_but_immediate_mode_does_not():
    torch.manual_seed(123)

    sequence = ContextContinuationHead(
        node_dim=4,
        vocab_sizes=VOCAB,
        hidden_dim=16,
        context_dim=8,
        history_mode="sequence",
    )
    sequence.eval()

    seq_a = _run(sequence, first_protocol=2)
    seq_b = _run(sequence, first_protocol=5)

    assert not torch.allclose(seq_a[0], seq_b[0])
    assert not torch.allclose(seq_a[1], seq_b[1])

    immediate = ContextContinuationHead(
        node_dim=4,
        vocab_sizes=VOCAB,
        hidden_dim=16,
        context_dim=8,
        history_mode="immediate",
    )
    immediate.eval()

    imm_a = _run(immediate, first_protocol=2)
    imm_b = _run(immediate, first_protocol=5)

    torch.testing.assert_close(imm_a[0], imm_b[0])
    torch.testing.assert_close(imm_a[1], imm_b[1])
