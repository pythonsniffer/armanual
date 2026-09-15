"""Canonical filesystem locations. Keeps asset paths out of every other module."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"
SO101_XML = ASSETS / "so101" / "so101.xml"
CONFIGS = REPO_ROOT / "configs"
OUTPUTS = REPO_ROOT / "outputs"


def ensure_outputs(*parts: str) -> Path:
    """Return (and create) an output directory under ``outputs/``."""
    path = OUTPUTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
