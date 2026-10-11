"""Federated utilities that keep temporal deployment state local."""

from .aggregation import coordinate_median, trimmed_mean, weighted_average
from .poisoning import poison_update
from .parameters import (
    extract_shared_parameters,
    load_shared_parameters,
    proximal_penalty,
    shared_parameter_names,
)

__all__ = [
    "weighted_average",
    "coordinate_median",
    "trimmed_mean",
    "extract_shared_parameters",
    "load_shared_parameters",
    "proximal_penalty",
    "shared_parameter_names",
    "poison_update",
]
