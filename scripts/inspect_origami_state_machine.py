#!/usr/bin/env python3
"""Inspect ORIGAMI state-machine CSV as auxiliary evidence, not graph input."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd


_REQUIRED = {"node", "servicetime", "id", "run", "start", "end", "ue"}
_HTTP_METHOD = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)[-_: ]",
    re.IGNORECASE,
)
_HTTP_STATUS = re.compile(r"^[1-5][0-9]{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def protocol_family(identifier: object) -> str:
    value = str(identifier).strip()
    lower = value.lower()
    if _HTTP_METHOD.match(value) or _HTTP_STATUS.match(value):
        return "SBI"
    if "pfcp" in lower or "sessionestablishment" in lower:
        return "PFCP"
    return "NGAP_OR_NAS"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv, sep=";")
    missing = sorted(_REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(
            f"ORIGAMI state-machine CSV missing columns: {missing}"
        )

    family = df["id"].map(protocol_family)
    result = {
        "schema": "origami_state_machine",
        "graph_eligible": False,
        "reason": (
            "Rows identify the processing node but not both signaling "
            "endpoints; converting them to src/dst edges would invent topology."
        ),
        "rows": int(len(df)),
        "unique_runs": int(df["run"].nunique(dropna=True)),
        "unique_ues": int(df["ue"].nunique(dropna=True)),
        "protocol_family_counts": {
            key: int(value)
            for key, value in sorted(family.value_counts().items())
        },
        "top_nodes": {
            str(key): int(value)
            for key, value in df["node"].value_counts().head(args.top).items()
        },
        "top_event_ids": {
            str(key): int(value)
            for key, value in df["id"].value_counts().head(args.top).items()
        },
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
