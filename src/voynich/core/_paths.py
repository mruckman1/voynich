"""Path resolution for data and results directories.

Walks up from this file to the project root (directory containing pyproject.toml).
All other modules should use these helpers instead of __file__-relative paths.
"""

from pathlib import Path

def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")

PROJECT_ROOT = _find_project_root()


def data_dir() -> Path:
    """Return the top-level data/ directory."""
    return PROJECT_ROOT / "data"


def json_dir() -> Path:
    """Return data/json/ where pre-built JSON files live."""
    return PROJECT_ROOT / "data" / "json"


def corpus_dir() -> Path:
    """Return data/corpus/ where IVTFF transcriptions live."""
    return PROJECT_ROOT / "data" / "corpus"


def results_dir() -> Path:
    """Return the top-level results/ directory."""
    return PROJECT_ROOT / "results"
