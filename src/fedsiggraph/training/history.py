"""Edge feature banks aligned with PyG LastNeighborLoader edge IDs."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("training utilities require the ML extra") from exc


class ObservedHistoryBank:
    """Store only events actually committed to temporal memory.

    This is necessary for safe/frozen evaluation: LastNeighborLoader assigns
    edge IDs according to insertions, not original dataset row positions.
    """

    def __init__(self, capacity: int, msg_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if msg_dim <= 0:
            raise ValueError("msg_dim must be positive")
        self.t = torch.empty(capacity, dtype=torch.long)
        self.msg = torch.empty((capacity, msg_dim), dtype=torch.float32)
        self.size = 0

    def append(self, t, msg) -> None:
        n = int(t.numel())
        if self.size + n > self.t.size(0):
            raise RuntimeError("history bank capacity exceeded")
        self.t[self.size : self.size + n] = t.detach().cpu()
        self.msg[self.size : self.size + n] = msg.detach().cpu()
        self.size += n

    def _index(self, e_id):
        index = e_id.detach().cpu()
        if index.numel() and int(index.max()) >= self.size:
            raise RuntimeError("neighbor loader requested an unobserved edge ID")
        return index

    def lookup(self, e_id, device):
        index = self._index(e_id)
        return self.t[index].to(device), self.msg[index].to(device)


class ObservedHeteroHistoryBank(ObservedHistoryBank):
    """Observed history plus relation/edge type for HEAT aggregation."""

    def __init__(self, capacity: int, msg_dim: int) -> None:
        super().__init__(capacity, msg_dim)
        self.edge_type = torch.empty(capacity, dtype=torch.long)

    def append(self, t, msg, edge_type) -> None:
        n = int(t.numel())
        start = self.size
        super().append(t, msg)
        self.edge_type[start : start + n] = edge_type.detach().cpu()

    def lookup(self, e_id, device):
        index = self._index(e_id)
        return (
            self.t[index].to(device),
            self.msg[index].to(device),
            self.edge_type[index].to(device),
        )
