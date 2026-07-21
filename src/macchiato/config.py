"""Configuration loading and project-path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML config and return it with the inferred project root."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    project_root = config_path.parent.parent
    return config, project_root


def project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path
