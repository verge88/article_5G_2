from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "srcport",
    "ip.dst",
    "dstport",
    "frame.protocols",
    "combined_info",
    "Label",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normal_rows(count: int = 30) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patterns = [
        (
            "10.211.55.5",
            40000,
            "10.211.55.11",
            7777,
            "eth:ethertype:ip:tcp:http2",
            "HEADERS[{stream}]: POST "
            "/nsmf-pdusession/v1/sm-contexts/1/release",
        ),
        (
            "10.211.55.11",
            7777,
            "10.211.55.5",
            40000,
            "eth:ethertype:ip:tcp:http2",
            "HEADERS[{stream}]: 204 No Content",
        ),
        (
            "10.211.55.4",
            40001,
            "10.211.55.11",
            7777,
            "eth:ethertype:ip:tcp:http2",
            "HEADERS[{stream}]: GET "
            "/nudm-sdm/v2/imsi-001010000000001/am-data",
        ),
        (
            "10.211.55.11",
            7777,
            "10.211.55.4",
            40001,
            "eth:ethertype:ip:tcp:http2",
            "HEADERS[{stream}]: 200 OK",
        ),
        (
            "10.211.55.5",
            8805,
            "10.211.55.6",
            8805,
            "eth:ethertype:ip:udp:pfcp",
            "PFCP Session Establishment Request",
        ),
        (
            "10.211.55.6",
            8805,
            "10.211.55.5",
            8805,
            "eth:ethertype:ip:udp:pfcp",
            "PFCP Session Establishment Response",
        ),
    ]

    for i in range(count):
        src, srcport, dst, dstport, stack, info = patterns[i % len(patterns)]
        # Pair request/response stream IDs; later repetitions use new streams.
        pair = i // 2
        rows.append(
            {
                "frame.time_epoch": f"{1.0 + i * 0.01:.6f}",
                "frame.len": 100 + i,
                "ip.src": src,
                "srcport": srcport,
                "ip.dst": dst,
                "dstport": dstport,
                "frame.protocols": stack,
                "combined_info": info.format(stream=2 * pair + 1),
                "Label": "normal",
            }
        )
    return rows


def attack_rows() -> list[dict[str, object]]:
    rows = normal_rows(5)
    for i, row in enumerate(rows):
        row["frame.time_epoch"] = f"{2.0 + i * 0.01:.6f}"

    for i in range(5):
        rows.append(
            {
                "frame.time_epoch": f"{2.1 + i * 0.01:.6f}",
                "frame.len": 60,
                "ip.src": "10.211.55.17",
                "srcport": 51000 + i,
                "ip.dst": "10.211.55.4",
                "dstport": 7777 + i,
                "frame.protocols": "eth:ethertype:ip:tcp",
                "combined_info": (
                    f"{51000 + i} → {7777 + i} [SYN] Seq=0 Win=64240"
                ),
                "Label": "portscan",
            }
        )
    return rows


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def test_temporal_heat_end_to_end(tmp_path: Path):
    normal = tmp_path / "normal.csv"
    attack = tmp_path / "attack.csv"
    prepared = tmp_path / "prepared"
    result = tmp_path / "result.json"

    write_csv(normal, normal_rows())
    write_csv(attack, attack_rows())

    run(
        [
            sys.executable,
            "scripts/prepare_granomaly.py",
            "--normal",
            str(normal),
            "--attack",
            str(attack),
            "--output",
            str(prepared),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/train_temporal_heat.py",
            "--data",
            str(prepared),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--neighbors",
            "4",
            "--event-dim",
            "16",
            "--memory-dim",
            "16",
            "--time-dim",
            "8",
            "--embedding-dim",
            "16",
            "--heads",
            "4",
            "--target-fpr",
            "0.2",
            "--test-memory",
            "safe",
            "--output",
            str(result),
        ]
    )

    with open(result, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["model"] == "temporal_heat_context_continuation"
    assert payload["metrics"]["tp"] + payload["metrics"]["fn"] == 5
    assert payload["per_attack"]["portscan"]["count"] == 5
    assert 0.0 <= payload["context_coverage_test"] <= 1.0

    with open(
        prepared / "manifest.json", "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    assert manifest["test_metadata_file"] == "test_metadata.csv"
    assert "context_coverage" in manifest
