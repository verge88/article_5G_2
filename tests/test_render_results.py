import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_render_results_markdown_and_latex(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "aggregate": {
                    "temporal_heat_continuation": {
                        "metrics": {
                            "pr_auc": {"n": 3, "mean": 0.8, "std": 0.01},
                            "recall": {"n": 3, "mean": 0.7, "std": 0.02},
                            "false_positive_rate": {
                                "n": 3,
                                "mean": 0.001,
                                "std": 0.0001,
                            },
                        },
                        "scope_metrics": {
                            "cross_protocol": {
                                "metrics": {
                                    "pr_auc": {
                                        "n": 3,
                                        "mean": 0.9,
                                        "std": 0.02,
                                    },
                                    "recall": {
                                        "n": 3,
                                        "mean": 0.85,
                                        "std": 0.03,
                                    },
                                }
                            }
                        },
                        "continuation_gain": {
                            "overall": {
                                "pr_auc": {
                                    "n": 3,
                                    "mean": 0.1,
                                    "std": 0.01,
                                    "positive_fraction": 1.0,
                                    "exact_sign_flip_pvalue": 0.25,
                                },
                                "recall": {
                                    "n": 3,
                                    "mean": 0.08,
                                    "std": 0.02,
                                    "positive_fraction": 1.0,
                                    "exact_sign_flip_pvalue": 0.25,
                                },
                            },
                            "cross_protocol": {
                                "pr_auc": {
                                    "n": 3,
                                    "mean": 0.15,
                                    "std": 0.02,
                                    "positive_fraction": 1.0,
                                    "exact_sign_flip_pvalue": 0.25,
                                },
                                "recall": {
                                    "n": 3,
                                    "mean": 0.2,
                                    "std": 0.03,
                                    "positive_fraction": 1.0,
                                    "exact_sign_flip_pvalue": 0.25,
                                },
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    md = subprocess.run(
        [
            sys.executable,
            "scripts/render_results.py",
            "--summary",
            str(summary),
            "--format",
            "markdown",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Cross-protocol PR-AUC" in md
    assert "Paired continuation contribution" in md
    assert "0.1000 ± 0.0100" in md

    tex = subprocess.run(
        [
            sys.executable,
            "scripts/render_results.py",
            "--summary",
            str(summary),
            "--format",
            "latex",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "\\begin{table*}" in tex
    assert "temporal\\_heat\\_continuation" in tex
    assert "delta PR-AUC=0.1500 ± 0.0200" in tex
