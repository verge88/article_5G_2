#!/usr/bin/env python3
"""Train and evaluate the causal Granomaly-style TGN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from fedsiggraph.baselines.tgn import CausalTGNBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared dataset directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=1e-3)
    parser.add_argument(
        "--test-memory",
        choices=["frozen", "safe", "blind"],
        default="safe",
        help="How test events are allowed to enter temporal memory.",
    )
    parser.add_argument("--output", help="Optional JSON result path.")
    return parser.parse_args()


def load_temporal(path: Path):
    # PyTorch >=2.6 defaults to weights_only=True. TemporalData is a trusted
    # local experiment artifact, so explicitly opt into object deserialization.
    return torch.load(path, weights_only=False)


class ObservedHistoryBank:
    """Edge features indexed exactly like LastNeighborLoader insertion IDs."""

    def __init__(self, capacity: int, msg_dim: int) -> None:
        self.t = torch.empty(capacity, dtype=torch.long)
        self.msg = torch.empty((capacity, msg_dim), dtype=torch.float32)
        self.size = 0

    def append(self, t: torch.Tensor, msg: torch.Tensor) -> None:
        n = int(t.numel())
        if self.size + n > self.t.size(0):
            raise RuntimeError("history bank capacity exceeded")
        self.t[self.size : self.size + n] = t.detach().cpu()
        self.msg[self.size : self.size + n] = msg.detach().cpu()
        self.size += n

    def lookup(
        self,
        e_id: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = e_id.detach().cpu()
        if index.numel() and int(index.max()) >= self.size:
            raise RuntimeError("neighbor loader requested an unobserved edge ID")
        return self.t[index].to(device), self.msg[index].to(device)


class Runner:
    def __init__(
        self,
        train_data,
        *,
        total_capacity: int,
        neighbors: int,
        lr: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.num_nodes = int(train_data.num_nodes)
        if self.num_nodes < 2:
            raise ValueError("TGN baseline requires at least two indexed nodes")

        self.model = CausalTGNBaseline(
            num_nodes=self.num_nodes,
            msg_dim=int(train_data.msg.size(-1)),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.neighbor_loader = LastNeighborLoader(
            self.num_nodes,
            size=neighbors,
            device=device,
        )
        self.assoc = torch.empty(
            self.num_nodes,
            dtype=torch.long,
            device=device,
        )
        self.total_capacity = total_capacity
        self.msg_dim = int(train_data.msg.size(-1))
        self.history = ObservedHistoryBank(total_capacity, self.msg_dim)

    def reset_state(self) -> None:
        self.model.reset_state()
        self.neighbor_loader.reset_state()
        self.history = ObservedHistoryBank(self.total_capacity, self.msg_dim)

    def _embedding_context(
        self,
        nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_id, edge_index, e_id = self.neighbor_loader(nodes)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        memory, last_update = self.model.memory(n_id)
        edge_t, edge_msg = self.history.lookup(e_id, self.device)
        z = self.model.embedding(
            memory,
            last_update,
            edge_index,
            edge_t,
            edge_msg,
        )
        return z, self.assoc

    @staticmethod
    def _avoid_positive_destination(
        neg_dst: torch.Tensor,
        pos_dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        collision = neg_dst == pos_dst
        if collision.any():
            # IDs 1..num_nodes-1 are train-known nodes; node 0 is UNK.
            shifted = (neg_dst[collision] % (num_nodes - 1)) + 1
            neg_dst = neg_dst.clone()
            neg_dst[collision] = shifted
        return neg_dst

    def train_epoch(self, data, batch_size: int) -> float:
        self.model.train()
        self.reset_state()
        loader = TemporalDataLoader(data, batch_size=batch_size)

        total_loss = 0.0
        total_events = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            src = batch.src
            pos_dst = batch.dst
            neg_dst = torch.randint(
                low=1,
                high=self.num_nodes,
                size=pos_dst.shape,
                device=self.device,
            )
            neg_dst = self._avoid_positive_destination(
                neg_dst,
                pos_dst,
                self.num_nodes,
            )

            nodes = torch.cat([src, pos_dst, neg_dst]).unique()
            z, assoc = self._embedding_context(nodes)

            benign_logits = self.model.score_embeddings(
                z[assoc[src]],
                z[assoc[pos_dst]],
            )
            anomaly_logits = self.model.score_embeddings(
                z[assoc[src]],
                z[assoc[neg_dst]],
            )

            loss = self.criterion(
                benign_logits,
                torch.zeros_like(benign_logits),
            )
            loss = loss + self.criterion(
                anomaly_logits,
                torch.ones_like(anomaly_logits),
            )
            loss.backward()
            self.optimizer.step()

            # Causal invariant: commit the positive event only after scoring.
            with torch.no_grad():
                self.model.observe(src, pos_dst, batch.t, batch.msg)
                self.neighbor_loader.insert(src, pos_dst)
                self.history.append(batch.t, batch.msg)
                self.model.memory.detach()

            n = int(batch.num_events)
            total_loss += float(loss.detach().cpu()) * n
            total_events += n

        return total_loss / max(total_events, 1)

    @torch.no_grad()
    def score_stream(
        self,
        data,
        *,
        observe: str,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if observe not in {"always", "never", "safe"}:
            raise ValueError(f"unknown observe policy: {observe}")
        if observe == "safe" and threshold is None:
            raise ValueError("safe observation requires a threshold")

        self.model.eval()
        loader = TemporalDataLoader(data, batch_size=1)
        scores: list[float] = []
        labels: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            nodes = torch.cat([batch.src, batch.dst]).unique()
            z, assoc = self._embedding_context(nodes)
            logits = self.model.score_embeddings(
                z[assoc[batch.src]],
                z[assoc[batch.dst]],
            )
            score = float(logits.sigmoid().item())
            scores.append(score)
            labels.append(int(batch.y.item()))

            should_observe = (
                observe == "always"
                or (observe == "safe" and score <= float(threshold))
            )
            if should_observe:
                self.model.observe(batch.src, batch.dst, batch.t, batch.msg)
                self.neighbor_loader.insert(batch.src, batch.dst)
                self.history.append(batch.t, batch.msg)
                self.model.memory.detach()

        return np.asarray(scores), np.asarray(labels)


def calibrate_threshold(scores: np.ndarray, target_fpr: float) -> float:
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be in (0, 1)")
    if scores.size == 0:
        raise ValueError("calibration split is empty")
    return float(
        np.quantile(scores, 1.0 - target_fpr, method="higher")
    )


def safe_auc(metric, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(metric(y_true, y_score))


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    data,
) -> dict[str, float | int | None]:
    pred = (y_score > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    duration_hours = 0.0
    if data.num_events > 1:
        duration_units = int(data.t.max()) - int(data.t.min())
        duration_ns = duration_units * int(data.time_unit_ns)
        duration_hours = duration_ns / 1e9 / 3600.0

    return {
        "pr_auc": safe_auc(average_precision_score, y_true, y_score),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_score),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_alarms_per_hour": (
            float(fp / duration_hours) if duration_hours > 0 else None
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    train = load_temporal(data_dir / "train.pt")
    val = load_temporal(data_dir / "val.pt")
    calibration = load_temporal(data_dir / "calibration.pt")
    test = load_temporal(data_dir / "test.pt")

    with open(data_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capacity = int(
        train.num_events + val.num_events + calibration.num_events + test.num_events
    )
    runner = Runner(
        train,
        total_capacity=max(capacity, 1),
        neighbors=args.neighbors,
        lr=args.lr,
        device=device,
    )

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        loss = runner.train_epoch(train, args.batch_size)
        losses.append(loss)
        print(f"epoch={epoch:03d} loss={loss:.6f}")

    # The final training epoch leaves memory in train state. Validation is a
    # benign temporal warm-up only; it is not used to tune the threshold.
    runner.score_stream(val, observe="always")
    calibration_scores, calibration_labels = runner.score_stream(
        calibration,
        observe="always",
    )
    if np.any(calibration_labels == 1):
        raise RuntimeError("calibration data contains attack labels")
    threshold = calibrate_threshold(calibration_scores, args.target_fpr)

    if manifest.get("test_history_mode") == "reset":
        runner.reset_state()

    observe_policy = {
        "frozen": "never",
        "safe": "safe",
        "blind": "always",
    }[args.test_memory]
    test_scores, test_labels = runner.score_stream(
        test,
        observe=observe_policy,
        threshold=threshold if observe_policy == "safe" else None,
    )

    result = {
        "model": "causal_tgn_baseline",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "learning_rate": args.lr,
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "test_memory": args.test_memory,
        "test_history_mode": manifest.get("test_history_mode"),
        "train_loss_final": losses[-1] if losses else None,
        "metrics": evaluate(test_labels, test_scores, threshold, test),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
