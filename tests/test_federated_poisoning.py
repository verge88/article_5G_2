import pytest

torch = pytest.importorskip("torch")

from fedsiggraph.federated.poisoning import poison_update


def test_model_replacement_scales_delta_from_global_reference():
    global_state = {"w": torch.tensor([1.0, 2.0])}
    local_state = {"w": torch.tensor([2.0, 4.0])}

    poisoned = poison_update(
        local_state,
        global_state,
        mode="model_replacement",
        factor=3.0,
    )

    assert torch.allclose(poisoned["w"], torch.tensor([4.0, 8.0]))


def test_sign_flip_reverses_delta():
    global_state = {"w": torch.tensor([1.0])}
    local_state = {"w": torch.tensor([3.0])}

    poisoned = poison_update(
        local_state,
        global_state,
        mode="sign_flip",
        factor=2.0,
    )

    assert poisoned["w"].item() == pytest.approx(-3.0)
