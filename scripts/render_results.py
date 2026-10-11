#!/usr/bin/env python3
"""Render FedSigGraph ablation summary as publication-ready tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--output")
    return parser.parse_args()


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _mean_std(summary: dict | None) -> str:
    if not summary or not summary.get("n"):
        return "n/a"
    mean = _number(summary.get("mean"))
    std = _number(summary.get("std"))
    return f"{mean} ± {std}"


def _metric(model: dict, name: str) -> str:
    return _mean_std(model.get("metrics", {}).get(name))


def _cross_metric(model: dict, name: str) -> str:
    scope = model.get("scope_metrics", {}).get("cross_protocol", {})
    return _mean_std(scope.get("metrics", {}).get(name))


def _gain(model: dict, scope: str, name: str) -> tuple[str, str, str]:
    value = (
        model.get("continuation_gain", {})
        .get(scope, {})
        .get(name, {})
    )
    return (
        _mean_std(value),
        _number(value.get("positive_fraction")),
        _number(value.get("exact_sign_flip_pvalue")),
    )


def markdown_table(aggregate: dict) -> str:
    rows = [
        "| Model | PR-AUC | Recall | FPR | Cross-protocol PR-AUC | Cross-protocol Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in aggregate.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    _metric(model, "pr_auc"),
                    _metric(model, "recall"),
                    _metric(model, "false_positive_rate"),
                    _cross_metric(model, "pr_auc"),
                    _cross_metric(model, "recall"),
                ]
            )
            + " |"
        )

    gain_rows = [
        "",
        "### Paired continuation contribution",
        "",
        "| Model | Scope | Δ PR-AUC | Positive seeds | Exact sign-flip p | Δ Recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, model in aggregate.items():
        gains = model.get("continuation_gain", {})
        if not gains:
            continue
        for scope in ("overall", "cross_protocol"):
            pr_gain, positive, pvalue = _gain(model, scope, "pr_auc")
            recall_gain, _, _ = _gain(model, scope, "recall")
            if pr_gain == "n/a" and recall_gain == "n/a":
                continue
            gain_rows.append(
                f"| {name} | {scope} | {pr_gain} | {positive} | "
                f"{pvalue} | {recall_gain} |"
            )

    return "\n".join(rows + gain_rows) + "\n"


def latex_table(aggregate: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{FedSigGraph-5GC anomaly-detection ablation. Values are mean $\pm$ standard deviation across matched seeds.}",
        r"\label{tab:fedsiggraph-ablation}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Model & PR-AUC & Recall & FPR & Cross-protocol PR-AUC & Cross-protocol Recall \\",
        r"\hline",
    ]
    for name, model in aggregate.items():
        cells = [
            name.replace("_", r"\_"),
            _metric(model, "pr_auc"),
            _metric(model, "recall"),
            _metric(model, "false_positive_rate"),
            _cross_metric(model, "pr_auc"),
            _cross_metric(model, "recall"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"% Paired continuation gains",
        ]
    )
    for name, model in aggregate.items():
        for scope in ("overall", "cross_protocol"):
            gain, positive, pvalue = _gain(model, scope, "pr_auc")
            if gain == "n/a":
                continue
            lines.append(
                "% "
                + f"{name} {scope}: delta PR-AUC={gain}, "
                + f"positive_fraction={positive}, p={pvalue}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    # run_ablation summary stores aggregate at top level; suite summaries may
    # contain one result object per experiment. Fail loudly rather than guessing.
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(
            "summary does not contain a top-level aggregate object; render one "
            "experiment's run_ablation summary.json at a time"
        )

    rendered = (
        markdown_table(aggregate)
        if args.format == "markdown"
        else latex_table(aggregate)
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
