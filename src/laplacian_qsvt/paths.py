"""Repository-relative paths shared by command-line entry points."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PHASES_DIR = OUTPUTS_DIR / "phases"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CIRCUITS_DIR = OUTPUTS_DIR / "circuits"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"


def ensure_parent(path: Path) -> Path:
    """Create an output file's parent directory and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
