"""Framework-independent aggregation used by Flower adapters and tests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("federated aggregation requires PyTorch") from exc


def _validate_updates(
    updates: Sequence[tuple[Mapping[str, Tensor], int]],
) -> tuple[tuple[str, ...], int]:
    if not updates:
        raise ValueError("at least one client update is required")

    reference_keys = tuple(updates[0][0].keys())
    reference_set = set(reference_keys)
    total_examples = 0

    for client, count in updates:
        if count < 0:
            raise ValueError("client example count cannot be negative")
        if set(client) != reference_set:
            raise ValueError("all client updates must have identical keys")
        total_examples += int(count)

        for name in reference_keys:
            if client[name].shape != updates[0][0][name].shape:
                raise ValueError(f"shape mismatch for {name!r}")

    return reference_keys, total_examples


def weighted_average(
    updates: Sequence[tuple[Mapping[str, Tensor], int]],
) -> "OrderedDict[str, Tensor]":
    """FedAvg over identically named trainable tensors."""
    reference_keys, total_examples = _validate_updates(updates)
    if total_examples <= 0:
        raise ValueError("sum of client example counts must be positive")

    output: "OrderedDict[str, Tensor]" = OrderedDict()
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
            accumulator += value.to(accumulator.dtype) * (
                float(count) / float(total_examples)
            )
        output[name] = accumulator.to(reference.dtype)
    return output


def coordinate_median(
    updates: Sequence[tuple[Mapping[str, Tensor], int]],
) -> "OrderedDict[str, Tensor]":
    """Coordinate-wise median over client model tensors.

    Example counts are validated but intentionally ignored: robust aggregation
    treats each client update as one vote so a poisoned high-volume client
    cannot dominate merely through its sample count.
    """
    reference_keys, _ = _validate_updates(updates)
    output: "OrderedDict[str, Tensor]" = OrderedDict()

    for name in reference_keys:
        reference = updates[0][0][name]
        stacked = torch.stack(
            [client[name].detach().cpu().to(torch.float32) for client, _ in updates],
            dim=0,
        )
        median = stacked.median(dim=0).values
        output[name] = median.to(reference.dtype)
    return output


def trimmed_mean(
    updates: Sequence[tuple[Mapping[str, Tensor], int]],
    *,
    trim_ratio: float = 0.2,
) -> "OrderedDict[str, Tensor]":
    """Coordinate-wise trimmed mean over equally weighted client tensors."""
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must lie in [0, 0.5)")
    reference_keys, _ = _validate_updates(updates)
    n_clients = len(updates)
    trim = int(n_clients * trim_ratio)
    if trim_ratio > 0 and trim == 0:
        raise ValueError(
            "trim_ratio has no effect for this number of clients; "
            "increase the ratio or add more clients"
        )
    if 2 * trim >= n_clients:
        raise ValueError("trim_ratio removes all client updates")

    output: "OrderedDict[str, Tensor]" = OrderedDict()
    for name in reference_keys:
        reference = updates[0][0][name]
        stacked = torch.stack(
            [client[name].detach().cpu().to(torch.float32) for client, _ in updates],
            dim=0,
        )
        sorted_values, _ = torch.sort(stacked, dim=0)
        kept = sorted_values[trim:n_clients - trim] if trim else sorted_values
        output[name] = kept.mean(dim=0).to(reference.dtype)
    return output
