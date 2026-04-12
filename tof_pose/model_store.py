from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path


class ModelStoreError(RuntimeError):
    """Model bootstrap failed."""


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PACKAGE_ROOT / "models"
DEFAULT_MANIFEST = DEFAULT_MODELS_DIR / "MANIFEST.json"


def load_manifest(path: Path | None = None) -> dict[str, dict[str, str]]:
    manifest_path = path or DEFAULT_MANIFEST
    if not manifest_path.exists():
        raise ModelStoreError(f"Model manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def resolve_model_path(alias: str, models_dir: Path | None = None) -> Path:
    manifest = load_manifest()
    if alias not in manifest:
        raise ModelStoreError(f"Unknown model alias: {alias}")
    directory = models_dir or DEFAULT_MODELS_DIR
    return directory / manifest[alias]["filename"]


def download_model(alias: str, models_dir: Path | None = None, force: bool = False) -> Path:
    manifest = load_manifest()
    if alias not in manifest:
        raise ModelStoreError(f"Unknown model alias: {alias}")

    entry = manifest[alias]
    target_dir = models_dir or DEFAULT_MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / entry["filename"]
    if target_path.exists() and not force:
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".download")
    request = urllib.request.Request(
        entry["url"],
        headers={"User-Agent": "tof-pose-model-fetcher/0.1"},
    )
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except Exception as exc:  # pragma: no cover - network dependent
        raise ModelStoreError(f"Failed to download {alias} from {entry['url']}: {exc}") from exc

    tmp_path.replace(target_path)
    return target_path
