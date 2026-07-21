"""Scene-aware sampling helpers."""

from __future__ import annotations

from collections import Counter


def inverse_scene_weights(records: list[dict[str, str]]) -> list[float]:
    """Give each scene equal expected mass under weighted sampling."""

    counts = Counter(record["scene_id"] for record in records)
    return [1.0 / counts[record["scene_id"]] for record in records]
