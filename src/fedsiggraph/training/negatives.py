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


class EndpointNegativeTable:
    """Sample a real train-known node with a semantically incompatible role.

    Candidate roles must occur as destinations for the same protocol somewhere
    in benign training data, but must not be valid for the current
    (src-role, procedure, protocol) context. This prevents unknown/synthetic node
    identity from becoming a shortcut feature.
    """

    def __init__(self, train_data, seed: int) -> None:
        self.rng = random.Random(seed)

        # Endpoint negatives replace *destination* nodes, so role IDs must come
        # exclusively from the destination-role vocabulary. src_role_id and
        # dst_role_id are intentionally separate vocabularies and their integer
        # IDs are not comparable even when they encode the same textual role.
        destination_roles: dict[int, set[int]] = {}
        for node, role in zip(
            train_data.dst.tolist(), train_data.dst_role_id.tolist()
        ):
            if node != 0:
                destination_roles.setdefault(int(node), set()).add(int(role))

        role_nodes: dict[int, set[int]] = {}
        for node, roles in destination_roles.items():
            # Exclude only genuinely ambiguous destination-role assignments.
            if len(roles) != 1:
                continue
            role = next(iter(roles))
            role_nodes.setdefault(role, set()).add(node)
        self.nodes_by_role: dict[int, tuple[int, ...]] = {
            role: tuple(sorted(nodes)) for role, nodes in role_nodes.items()
        }

        protocol_dst_roles: dict[int, set[int]] = {}
        allowed_roles: dict[tuple[int, int, int], set[int]] = {}
        for src_role, dst_role, protocol, procedure in zip(
            train_data.src_role_id.tolist(),
            train_data.dst_role_id.tolist(),
            train_data.protocol_id.tolist(),
            train_data.procedure_id.tolist(),
        ):
            protocol = int(protocol)
            dst_role = int(dst_role)
            protocol_dst_roles.setdefault(protocol, set()).add(dst_role)
            allowed_roles.setdefault(
                (int(src_role), int(procedure), protocol), set()
            ).add(dst_role)

        self.candidate_roles: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for key, allowed in allowed_roles.items():
            protocol = key[2]
            candidates = [
                role
                for role in protocol_dst_roles.get(protocol, set()) - allowed
                if self.nodes_by_role.get(role)
            ]
            self.candidate_roles[key] = tuple(sorted(candidates))

    def sample(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        src_roles = batch.src_role_id.detach().cpu().tolist()
        procedures = batch.procedure_id.detach().cpu().tolist()
        protocols = batch.protocol_id.detach().cpu().tolist()
        actual_dst = batch.dst.detach().cpu().tolist()

        sampled: list[int] = []
        valid: list[bool] = []
        for src_role, procedure, protocol, current_dst in zip(
            src_roles, procedures, protocols, actual_dst
        ):
            roles = self.candidate_roles.get(
                (int(src_role), int(procedure), int(protocol)), ()
            )
            if not roles:
                sampled.append(int(current_dst))
                valid.append(False)
                continue

            role = self.rng.choice(roles)
            nodes = [
                node
                for node in self.nodes_by_role[role]
                if node != int(current_dst)
            ]
            if not nodes:
                sampled.append(int(current_dst))
                valid.append(False)
                continue

            sampled.append(self.rng.choice(nodes))
            valid.append(True)

        return (
            torch.as_tensor(sampled, dtype=torch.long, device=batch.dst.device),
            torch.as_tensor(valid, dtype=torch.bool, device=batch.dst.device),
        )


def sample_negative_destination(
    pos_dst: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Legacy random baseline sampler using only train-known node IDs."""
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
        shifted = ((neg[collision] - 1 + 1) % (num_nodes - 1)) + 1
        neg = neg.clone()
        neg[collision] = shifted
    return neg
