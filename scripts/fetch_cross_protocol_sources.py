#!/usr/bin/env python3
"""Download pinned benign signaling and attack PCAP sources for experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ORIGAMI_RECORD = "19605370"
ORIGAMI_FILES = {
    "README.md": {
        "url": f"https://zenodo.org/records/{ORIGAMI_RECORD}/files/README.md?download=1",
        "md5": "6ce6848fe2c21963c4e9c3911009ef84",
    },
    "example.csv": {
        "url": f"https://zenodo.org/records/{ORIGAMI_RECORD}/files/example.csv?download=1",
        "md5": "2442ec8e4da942781adeb8a0a3c61979",
    },
    "example.pcap.gz": {
        "url": f"https://zenodo.org/records/{ORIGAMI_RECORD}/files/example.pcap.gz?download=1",
        "md5": "b7c63bf275cc2e19637931dda7986cb2",
    },
}

DLTEAM_REVISION = "e71267ca84f6f38ab39f830a2c515555bb5f5898"
DLTEAM_PCAPS = (
    "007a-fuzz-open5gs.pcapng",
    "007b-fuzz-open5gs.pcapng",
    "007d-fuzz-open5gs.pcapng",
    "007e-fuzz-open5gs.pcapng",
    "007f-fuzz-open5gs.pcapng",
    "007g-fuzz-open5gs.pcapng",
    "007h-fuzz-open5gs.pcapng",
)
DLTEAM_RAW_BASE = "https://raw.githubusercontent.com/DLTeamTUC/5GDatasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/cross-protocol")
    parser.add_argument(
        "--origami",
        choices=["none", "readme", "pcap", "csv", "example"],
        default="readme",
        help=(
            "Download only metadata, or the 125 MB example CSV and small example PCAP."
        ),
    )
    parser.add_argument(
        "--dlteam-fuzz",
        action="store_true",
        help="Download pinned Open5GS NAS-fuzzing PCAP captures from DLTeamTUC.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without network access.",
    )
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "FedSigGraph-5GC-research"})
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with urlopen(request, timeout=180) as response, open(
            temporary, "wb"
        ) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def fetch_file(
    *,
    url: str,
    target: Path,
    force: bool,
    dry_run: bool,
    expected_md5: str | None = None,
) -> dict:
    print(f"{'plan' if dry_run else 'fetch'} {url} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        if force or not target.exists():
            download(url, target)
        if expected_md5 is not None:
            actual_md5 = digest(target, "md5")
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for {target}: {actual_md5} != {expected_md5}"
                )
        return {
            "bytes": target.stat().st_size,
            "sha256": digest(target, "sha256"),
            "md5": digest(target, "md5"),
            "url": url,
        }

    return {
        "url": url,
        "expected_md5": expected_md5,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    manifest = {
        "origami": {
            "record": ORIGAMI_RECORD,
            "files": {},
        },
        "dlteam": {
            "repository": "DLTeamTUC/5GDatasets",
            "revision": DLTEAM_REVISION,
            "files": {},
        },
    }

    origami_names: list[str] = []
    if args.origami == "readme":
        origami_names = ["README.md"]
    elif args.origami == "pcap":
        origami_names = ["README.md", "example.pcap.gz"]
    elif args.origami == "csv":
        origami_names = ["README.md", "example.csv"]
    elif args.origami == "example":
        origami_names = ["README.md", "example.csv", "example.pcap.gz"]

    for name in origami_names:
        spec = ORIGAMI_FILES[name]
        manifest["origami"]["files"][name] = fetch_file(
            url=spec["url"],
            target=output / "origami" / name,
            force=args.force,
            dry_run=args.dry_run,
            expected_md5=spec["md5"],
        )

    if args.dlteam_fuzz:
        for name in DLTEAM_PCAPS:
            url = (
                f"{DLTEAM_RAW_BASE}/{DLTEAM_REVISION}/pcap/{name}"
            )
            manifest["dlteam"]["files"][name] = fetch_file(
                url=url,
                target=output / "dlteam" / name,
                force=args.force,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output.mkdir(parents=True, exist_ok=True)
    with open(
        output / "source_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
