import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_ablation_dry_run_labels_immediate_and_sequence_variants(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation.py",
            "--data",
            "prepared",
            "--output-dir",
            str(tmp_path),
            "--models",
            "temporal_heat_continuation_immediate",
            "temporal_heat_continuation",
            "--seeds",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout

    assert (
        "--result-model-name temporal_heat_continuation_immediate"
        in output
    )
    assert "--continuation-history-mode immediate" in output
    assert "--result-model-name temporal_heat_continuation " in output
    assert "--continuation-history-mode sequence" in output


def test_ablation_resume_reuses_only_matching_result(tmp_path):
    output = tmp_path / "numeric_tgn_seed1.json"
    output.write_text(
        json.dumps(
            {
                "model": "numeric_tgn",
                "seed": 1,
                "metrics": {"pr_auc": 0.5},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation.py",
            "--data",
            "prepared",
            "--output-dir",
            str(tmp_path),
            "--models",
            "numeric_tgn",
            "--seeds",
            "1",
            "--resume",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[resume] reuse" in result.stdout
    assert "scripts/train_baseline.py" not in result.stdout
