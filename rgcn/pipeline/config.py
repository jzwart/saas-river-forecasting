"""Config loading + repo-relative path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# rgcn/pipeline/config.py -> repo root is two parents up from the package dir.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "rgcn" / "config.yml"


class Config:
    """Thin wrapper over the parsed YAML that resolves ``paths`` to absolute
    paths under the repo root."""

    def __init__(self, raw: dict[str, Any], repo_root: Path = REPO_ROOT):
        self.raw = raw
        self.repo_root = repo_root

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a key under ``paths:`` to an absolute path."""
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else self.repo_root / p


def load_config(path: str | Path | None = None) -> Config:
    """Load a config file. Resolution order: explicit ``path`` argument, the
    RGCN_CONFIG env var (lets every pipeline module run against a split/ablation
    variant without signature changes), then rgcn/config.yml."""
    if path is None:
        path = os.environ.get("RGCN_CONFIG") or DEFAULT_CONFIG
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)
