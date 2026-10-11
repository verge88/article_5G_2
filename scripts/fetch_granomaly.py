#!/usr/bin/env python3
"""Download the exact upstream Granomaly CSV revision used by experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_REVISION = "51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7"
FILES = (
    "bridge_normal.csv",
    "bridge_attack_nrf.csv",
    "bridge_attack_smf.csv",
)
RAW_BASE = "https://raw.githubusercontent.com/TF0x42/granomaly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/granomaly")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Granomaly commit SHA.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing CSV files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "FedSigGraph-5GC-research"},
    )
    with urlopen(request, timeout=120) as response, open(
        target, "wb"
    ) as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_repository": "TF0x42/granomaly",
        "revision": args.revision,
        "files": {},
    }

    for filename in FILES:
        target = output / filename
        if target.exists() and not args.force:
            print(f"reuse {target}")
        else:
            url = (
                f"{RAW_BASE}/{args.revision}/datasets/{filename}"
            )
            temporary = target.with_suffix(target.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            print(f"download {url}")
            try:
                download(url, temporary)
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()

        manifest["files"][filename] = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }

    with open(
        output / "source_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
