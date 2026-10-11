#!/usr/bin/env python3
"""Parse attack PCAPs with a pinned 5G-Signaling-Parser checkout."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys


PARSER_REPOSITORY = "https://github.com/lsinfo3/5G-Signaling-Parser.git"
PARSER_REVISION = "e48427c4ede8bde673324a78979fc20702d2c29b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--parser-dir",
        default=".cache/5G-Signaling-Parser",
        help="Pinned parser checkout location.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pcap*",
        help="Glob pattern below --pcap-dir.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with parser dependencies installed.",
    )
    parser.add_argument(
        "--hosts",
        help="Optional Wireshark-style hosts file for NF name resolution.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Clone/checkout the pinned parser revision. Dependency installation "
            "is intentionally not automatic because pyshark requires tshark."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None, dry_run: bool) -> None:
    shown = " ".join(command)
    print(f"$ {shown}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def ensure_parser(
    parser_dir: Path,
    *,
    bootstrap: bool,
    dry_run: bool,
) -> None:
    if bootstrap and not parser_dir.exists():
        parser_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", PARSER_REPOSITORY, str(parser_dir)],
            cwd=None,
            dry_run=dry_run,
        )

    if not parser_dir.exists() and not dry_run:
        raise FileNotFoundError(
            f"parser checkout not found: {parser_dir}; pass --bootstrap"
        )

    if bootstrap or parser_dir.exists():
        run(
            ["git", "fetch", "--all", "--tags"],
            cwd=parser_dir,
            dry_run=dry_run,
        )
        run(
            ["git", "checkout", "--detach", PARSER_REVISION],
            cwd=parser_dir,
            dry_run=dry_run,
        )


def main() -> None:
    args = parse_args()
    pcap_dir = Path(args.pcap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser_dir = Path(args.parser_dir)

    ensure_parser(
        parser_dir,
        bootstrap=args.bootstrap,
        dry_run=args.dry_run,
    )

    pcaps = sorted(pcap_dir.glob(args.pattern))
    if not pcaps and not args.dry_run:
        raise FileNotFoundError(
            f"no PCAP files matching {args.pattern!r} under {pcap_dir}"
        )

    manifest = {
        "parser_repository": PARSER_REPOSITORY,
        "parser_revision": PARSER_REVISION,
        "hosts_file": Path(args.hosts).name if args.hosts else None,
        "captures": [],
    }

    for pcap in pcaps:
        source_pcap = pcap
        temporary_pcap: Path | None = None

        if pcap.suffix == ".gz":
            # Zenodo ORIGAMI distributes its small example capture as
            # example.pcap.gz. PyShark expects a capture file, so materialize a
            # temporary decompressed copy and remove it after parsing.
            temporary_root = output_dir / ".decompressed"
            temporary_root.mkdir(parents=True, exist_ok=True)
            temporary_pcap = temporary_root / pcap.stem
            if args.dry_run:
                print(f"plan gunzip {pcap} -> {temporary_pcap}")
            else:
                with gzip.open(pcap, "rb") as src, open(
                    temporary_pcap, "wb"
                ) as dst:
                    shutil.copyfileobj(src, dst)
            source_pcap = temporary_pcap

        capture_name = source_pcap.stem
        capture_out = output_dir / capture_name
        bridge = Path(__file__).with_name(
            "signaling_parser_bridge.py"
        ).resolve()
        command = [
            args.python,
            str(bridge),
            "--parser-dir",
            str(parser_dir.resolve()),
            "--pcap",
            str(source_pcap.resolve()),
            "--output-dir",
            str(capture_out.resolve()),
        ]
        if args.hosts:
            command += ["--hosts", str(Path(args.hosts).resolve())]

        try:
            run(
                command,
                cwd=None,
                dry_run=args.dry_run,
            )
        finally:
            if (
                temporary_pcap is not None
                and temporary_pcap.exists()
                and not args.dry_run
            ):
                temporary_pcap.unlink()

        manifest["captures"].append(
            {
                "pcap": str(pcap),
                "output": str(capture_out),
                "compressed": pcap.suffix == ".gz",
            }
        )

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "parser_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
