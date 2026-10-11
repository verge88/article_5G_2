#!/usr/bin/env python3
"""Bridge to the pinned 5G-Signaling-Parser without its hardcoded hosts path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-dir", required=True)
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--hosts",
        help="Optional Wireshark-style hosts file used to resolve NF names.",
    )
    return parser.parse_args()


def deduplicate(messages):
    seen = set()
    result = []
    for message in messages:
        key = (
            str(message.frametime),
            message.src,
            message.dest,
            str(message.info),
        )
        if key not in seen:
            seen.add(key)
            result.append(message)
    return result


def main() -> None:
    args = parse_args()
    parser_dir = Path(args.parser_dir).resolve()
    pcap = Path(args.pcap).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(parser_dir))

    from dialogparser.sharkparser import SharkParser
    from dialogparser.utils import parse_hosts_file

    hosts = parse_hosts_file(args.hosts) if args.hosts else None
    parser = SharkParser(str(pcap), hosts)
    parser.parse_messages("nrf")

    exported = 0
    for dialog_bin in parser.repo.uebinlist:
        ue_id = str(dialog_bin.ue.supi)
        lines = ["time;src;dst;message;ue"]
        for dialog in dialog_bin.dialogs:
            for message in deduplicate(dialog.messages):
                lines.append(
                    ";".join(
                        [
                            str(message.frametime),
                            str(message.src),
                            str(message.dest),
                            str(message.info),
                            ue_id,
                        ]
                    )
                )

        stem = pcap.name.rsplit(".", 1)[0]
        target = output / f"{stem}_{ue_id}_m_stats.csv"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1

    print(f"exported_ue_csvs={exported}")


if __name__ == "__main__":
    main()
