"""Reproducibility metadata for experiment result artifacts."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _git_commit(repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def runtime_provenance(repo_root: Path | None = None) -> dict[str, object]:
    """Return stable software provenance without machine-identifying metadata."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import torch_geometric
        versions["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass

    return {
        "repository_commit": _git_commit(repo_root),
        "python_executable_version": sys.version.split()[0],
        "software_versions": versions,
    }
