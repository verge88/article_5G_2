"""Shared/private parameter partition for stateful temporal federation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("federated parameter utilities require PyTorch") from exc


DEFAULT_PRIVATE_PREFIXES = (
    "local_adapter.",
)


def shared_parameter_names(
    model: nn.Module,
    *,
    private_prefixes: Sequence[str] = DEFAULT_PRIVATE_PREFIXES,
) -> tuple[str, ...]:
    """Return trainable names eligible for federation.

    ``named_parameters()`` is intentionally used instead of
    ``state_dict()``. TGNMemory keeps deployment history such as node memory
    and last-update tensors as buffers; those buffers must never be sent to the
    server. Trainable temporal encoder weights remain shareable.
    """
    return tuple(
        name
        for name, _ in model.named_parameters()
        if not any(name.startswith(prefix) for prefix in private_prefixes)
    )


def extract_shared_parameters(
    model: nn.Module,
    *,
    private_prefixes: Sequence[str] = DEFAULT_PRIVATE_PREFIXES,
) -> "OrderedDict[str, Tensor]":
    allowed = set(
        shared_parameter_names(
            model, private_prefixes=private_prefixes
        )
    )
    return OrderedDict(
        (name, parameter.detach().cpu().clone())
        for name, parameter in model.named_parameters()
        if name in allowed
    )


@torch.no_grad()
def load_shared_parameters(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
    *,
    private_prefixes: Sequence[str] = DEFAULT_PRIVATE_PREFIXES,
    strict: bool = True,
) -> None:
    local = dict(model.named_parameters())
    expected = set(
        shared_parameter_names(
            model, private_prefixes=private_prefixes
        )
    )
    provided = set(parameters)

    if strict:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            raise ValueError(
                f"shared parameter mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )

    for name, value in parameters.items():
        if name not in expected:
            if strict:
                raise ValueError(f"parameter {name!r} is not shareable")
            continue

        destination = local[name]
        tensor = value.to(
            device=destination.device,
            dtype=destination.dtype,
        )
        if tensor.shape != destination.shape:
            raise ValueError(
                f"shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(destination.shape)}"
            )
        destination.copy_(tensor)


def proximal_penalty(
    model: nn.Module,
    reference: Mapping[str, Tensor],
    *,
    mu: float,
    private_prefixes: Sequence[str] = DEFAULT_PRIVATE_PREFIXES,
) -> Tensor:
    """Return the FedProx term μ/2 ||w-w_global||² over shared weights only."""
    if mu < 0:
        raise ValueError("mu must be non-negative")

    local = dict(model.named_parameters())
    expected = set(
        shared_parameter_names(
            model, private_prefixes=private_prefixes
        )
    )
    missing = expected - set(reference)
    if missing:
        raise ValueError(
            f"FedProx reference is missing shared parameters: {sorted(missing)}"
        )

    penalty: Tensor | None = None
    for name in sorted(expected):
        parameter = local[name]
        target = reference[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        term = (parameter - target).pow(2).sum()
        penalty = term if penalty is None else penalty + term

    if penalty is None:
        # Preserve a valid tensor result for unusual parameter-free models.
        first = next(model.buffers(), None)
        device = first.device if first is not None else torch.device("cpu")
        penalty = torch.zeros((), device=device)

    return 0.5 * mu * penalty
