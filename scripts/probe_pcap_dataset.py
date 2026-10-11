#!/usr/bin/env python3
"""Probe raw PCAPs for FedSigGraph protocol and explicit-context coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from fedsiggraph.data.context import build_context_transition_features
from fedsiggraph.data.signaling_parser import SignalingParserAdapter
from fedsiggraph.data.vocab import SemanticVocabularies


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcaps", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployment-prefix", default="probe")
    parser.add_argument(
        "--display-filter",
        default="http2||ngap||pfcp",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def ratio(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(bool(value) for value in values) / len(values)


def probe_csv(path: Path, *, deployment_id: str) -> dict:
    events = SignalingParserAdapter(deployment_id).from_csv(
        path,
        label=None,
    )
    if not events:
        return {
            "events": 0,
            "protocols": {},
            "ue_context_fraction": 0.0,
            "session_context_fraction": 0.0,
            "explicit_context_history_fraction": 0.0,
            "cross_protocol_transition_fraction": 0.0,
            "candidate_cross_protocol_context": False,
        }

    vocab = SemanticVocabularies.fit(
        events,
        require_benign=False,
    )
    transitions = build_context_transition_features(
        events,
        vocab=vocab,
    )
    protocols = Counter(event.protocol for event in events)
    cross_fraction = ratio(transitions.cross_protocol_mask)
    history_fraction = ratio(transitions.history_mask)

    return {
        "events": len(events),
        "protocols": dict(sorted(protocols.items())),
        "ue_context_fraction": ratio(
            event.ue_context is not None for event in events
        ),
        "session_context_fraction": ratio(
            event.session_context is not None for event in events
        ),
        "explicit_context_history_fraction": history_fraction,
        "cross_protocol_transition_fraction": cross_fraction,
        "candidate_cross_protocol_context": (
            len(protocols) >= 2
            and history_fraction > 0.0
            and cross_fraction > 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for index, raw_path in enumerate(args.pcaps):
        pcap = Path(raw_path)
        correlated = output_dir / f"{pcap.stem}_correlated.csv"

        command = [
            sys.executable,
            "scripts/export_correlated_pcap.py",
            "--pcap",
            str(pcap),
            "--output",
            str(correlated),
            "--display-filter",
            args.display_filter,
        ]
        print("$", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

        summary[str(pcap)] = probe_csv(
            correlated,
            deployment_id=f"{args.deployment_prefix}-{index}",
        )

    rendered = {
        "captures": summary,
        "candidate_count": sum(
            item["candidate_cross_protocol_context"]
            for item in summary.values()
        ),
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rendered, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(rendered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
