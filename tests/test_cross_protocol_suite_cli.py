from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_cross_protocol_suite_dry_run_accepts_raw_pcaps(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cross_protocol_suite.py",
            "--benign-pcap",
            "benign.pcap",
            "--test-pcap",
            "attack.pcap",
            "--deployment-id",
            "site-a",
            "--workspace",
            str(tmp_path / "experiment"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert "scripts/export_correlated_pcap.py" in output
    assert "benign_correlated.csv" in output
    assert "test_correlated.csv" in output
    assert "scripts/prepare_signaling_parser.py" in output
    assert "scripts/run_ablation.py" in output
