"""Train-only tensor hard negatives shared by temporal models."""

from __future__ import annotations

import random

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("negative sampling requires the ML extra") from exc


class ProcedureNegativeTable:
    """Sample a procedure seen in the same protocol but not for this role pair."""

    def __init__(self, train_data, seed: int) -> None:
        self.rng = random.Random(seed)
        protocol_procedures: dict[int, set[int]] = {}
        pair_procedures: dict[tuple[int, int, int], set[int]] = {}

        rows = zip(
            train_data.src_role_id.tolist(),
            train_data.dst_role_id.tolist(),
            train_data.protocol_id.tolist(),
            train_data.procedure_id.tolist(),
        )
        for src_role, dst_role, protocol, procedure in rows:
            protocol_procedures.setdefault(protocol, set()).add(procedure)
            pair_procedures.setdefault(
                (src_role, dst_role, protocol), set()
            ).add(procedure)

        self.candidates: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for key, seen_for_pair in pair_procedures.items():
            protocol = key[2]
            self.candidates[key] = tuple(
                sorted(protocol_procedures.get(protocol, set()) - seen_for_pair)
            )

    def sample(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        src_roles = batch.src_role_id.detach().cpu().tolist()
        dst_roles = batch.dst_role_id.detach().cpu().tolist()
        protocols = batch.protocol_id.detach().cpu().tolist()
        actual = batch.procedure_id.detach().cpu().tolist()

        sampled: list[int] = []
        valid: list[bool] = []
        for src_role, dst_role, protocol, current in zip(
            src_roles, dst_roles, protocols, actual
        ):
            options = self.candidates.get(
                (src_role, dst_role, protocol), ()
            )
            if options:
                sampled.append(self.rng.choice(options))
                valid.append(True)
            else:
                sampled.append(current)
                valid.append(False)

        return (
            torch.as_tensor(
                sampled, dtype=torch.long, device=batch.procedure_id.device
            ),
            torch.as_tensor(
                valid, dtype=torch.bool, device=batch.procedure_id.device
            ),
        )


def sample_negative_destination(
    pos_dst: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Sample a train-known node ID different from the positive destination."""
    if num_nodes < 3:
        raise ValueError(
            "endpoint negatives require at least two train-known nodes"
        )

    neg = torch.randint(
        1,
        num_nodes,
        pos_dst.shape,
        device=pos_dst.device,
    )
    collision = neg == pos_dst
    if collision.any():
        # When there is more than one known node, increment modulo the known
        # ID range until the collision is removed.
        shifted = ((neg[collision] - 1 + 1) % (num_nodes - 1)) + 1
        neg = neg.clone()
        neg[collision] = shifted
    return neg
