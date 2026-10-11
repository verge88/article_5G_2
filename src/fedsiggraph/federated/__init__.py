"""Federated utilities that keep temporal deployment state local."""

from .aggregation import weighted_average
from .parameters import (
    extract_shared_parameters,
    load_shared_parameters,
    proximal_penalty,
    shared_parameter_names,
)

__all__ = [
    "weighted_average",
    "extract_shared_parameters",
    "load_shared_parameters",
    "proximal_penalty",
    "shared_parameter_names",
]
