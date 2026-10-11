#!/usr/bin/env python3
"""Inspect parsed signaling exports before expensive ML preprocessing."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from fedsiggraph.data.context import build_context_transition_features
from fedsiggraph.data.signaling_parser import SignalingParserAdapter
from fedsiggraph.data.vocab import SemanticVocabularies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--directory")
    parser.add_argument("--pattern", default="*_m_stats.csv")
    parser.add_argument("--deployment-id", default="inspection")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SignalingParserAdapter(args.deployment_id)
    if args.csv:
        events = adapter.from_csv(args.csv, label=0)
        source_files = [Path(args.csv).name]
    else:
        root = Path(args.directory)
        source_files = [path.name for path in sorted(root.glob(args.pattern))]
        events = adapter.from_directory(
            root,
            pattern=args.pattern,
            label=0,
        )

    if not events:
        raise RuntimeError("no signaling events parsed from source")

    vocab = SemanticVocabularies.fit(events)
    transitions = build_context_transition_features(events, vocab=vocab)

    event_count = len(events)
    context_history = int(transitions.history_mask.sum())
    cross_protocol = int(transitions.cross_protocol_mask.sum())

    result = {
        "source_files": source_files,
        "event_count": event_count,
        "protocol_counts": dict(
            sorted(Counter(event.protocol for event in events).items())
        ),
        "src_role_counts": dict(
            sorted(Counter(event.src_role for event in events).items())
        ),
        "dst_role_counts": dict(
            sorted(Counter(event.dst_role for event in events).items())
        ),
        "explicit_ue_context_events": sum(
            event.ue_context is not None for event in events
        ),
        "explicit_session_context_events": sum(
            event.session_context is not None for event in events
        ),
        "explicit_context_history_events": context_history,
        "cross_protocol_transition_events": cross_protocol,
        "explicit_context_history_fraction": context_history / event_count,
        "cross_protocol_transition_fraction": cross_protocol / event_count,
        "unknown_src_role_fraction": (
            sum(event.src_role == "UNKNOWN_NF" for event in events)
            / event_count
        ),
        "unknown_dst_role_fraction": (
            sum(event.dst_role == "UNKNOWN_NF" for event in events)
            / event_count
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
