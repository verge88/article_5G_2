import pytest

torch = pytest.importorskip("torch")
from torch import nn

from fedsiggraph.federated.aggregation import weighted_average
from fedsiggraph.federated.parameters import (
    extract_shared_parameters,
    load_shared_parameters,
    proximal_penalty,
)


class ToyStatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        self.local_adapter = nn.Linear(2, 2, bias=False)
        self.register_buffer("temporal_memory", torch.ones(3, 2))


def test_shared_parameter_roundtrip_preserves_private_parameter_and_buffer():
    model = ToyStatefulModel()
    shared = extract_shared_parameters(model)

    assert "shared.weight" in shared
    assert "local_adapter.weight" not in shared
    assert "temporal_memory" not in shared

    private_before = model.local_adapter.weight.detach().clone()
    memory_before = model.temporal_memory.detach().clone()

    updated = {
        name: tensor + 1.0 for name, tensor in shared.items()
    }
    load_shared_parameters(model, updated)

    torch.testing.assert_close(
        model.local_adapter.weight, private_before
    )
    torch.testing.assert_close(model.temporal_memory, memory_before)
    torch.testing.assert_close(
        model.shared.weight.detach().cpu(),
        updated["shared.weight"],
    )


def test_fedprox_penalty_uses_shared_parameters_only():
    model = ToyStatefulModel()
    reference = extract_shared_parameters(model)

    model.local_adapter.weight.data.add_(100.0)
    assert float(proximal_penalty(model, reference, mu=0.5)) == 0.0

    model.shared.weight.data.add_(1.0)
    assert float(proximal_penalty(model, reference, mu=0.5)) > 0.0


def test_weighted_average_uses_example_counts():
    a = {"w": torch.tensor([0.0, 2.0])}
    b = {"w": torch.tensor([2.0, 4.0])}

    result = weighted_average([(a, 1), (b, 3)])

    torch.testing.assert_close(
        result["w"], torch.tensor([1.5, 3.5])
    )
