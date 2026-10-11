"""Framework-independent aggregation used by Flower adapters and tests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("federated aggregation requires PyTorch") from exc


def weighted_average(
    updates: Sequence[tuple[Mapping[str, Tensor], int]],
) -> "OrderedDict[str, Tensor]":
    """FedAvg over identically named trainable tensors."""
    if not updates:
        raise ValueError("at least one client update is required")

    total_examples = sum(int(count) for _, count in updates)
    if total_examples <= 0:
        raise ValueError("sum of client example counts must be positive")

    reference_keys = tuple(updates[0][0].keys())
    reference_set = set(reference_keys)
    output: "OrderedDict[str, Tensor]" = OrderedDict()

    for client, count in updates:
        if count < 0:
            raise ValueError("client example count cannot be negative")
        if set(client) != reference_set:
            raise ValueError("all client updates must have identical keys")

    for name in reference_keys:
        reference = updates[0][0][name]
        accumulator = torch.zeros_like(
            reference,
            dtype=(
                reference.dtype
                if reference.is_floating_point()
                else torch.float32
            ),
            device="cpu",
        )

        for client, count in updates:
            value = client[name].detach().cpu()
            if value.shape != reference.shape:
                raise ValueError(f"shape mismatch for {name!r}")
            accumulator += value.to(accumulator.dtype) * (
                float(count) / float(total_examples)
            )

        output[name] = accumulator.to(reference.dtype)

    return output
