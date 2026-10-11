"""Controlled model-update perturbations for federated robustness experiments."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

try:
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("federated poisoning utilities require PyTorch") from exc


def poison_update(
    local: Mapping[str, Tensor],
    reference: Mapping[str, Tensor],
    *,
    mode: str,
    factor: float,
) -> "OrderedDict[str, Tensor]":
    """Perturb one client update relative to the current global reference.

    model_replacement amplifies the local delta:
        new = global + factor * (local - global)

    sign_flip reverses and amplifies the local delta:
        new = global - factor * (local - global)

    Intended only for controlled robustness evaluation.
    """
    if factor < 0:
        raise ValueError("factor must be non-negative")
    if mode not in {"model_replacement", "sign_flip"}:
        raise ValueError(f"unsupported poisoning mode: {mode!r}")
    if set(local) != set(reference):
        raise ValueError("local and reference parameters must have identical keys")

    direction = 1.0 if mode == "model_replacement" else -1.0
    output: "OrderedDict[str, Tensor]" = OrderedDict()

    for name in local:
        local_tensor = local[name].detach().cpu()
        global_tensor = reference[name].detach().cpu()
        if local_tensor.shape != global_tensor.shape:
            raise ValueError(f"shape mismatch for {name!r}")
        delta = local_tensor.to(dtype=global_tensor.dtype) - global_tensor
        output[name] = global_tensor + direction * float(factor) * delta

    return output
