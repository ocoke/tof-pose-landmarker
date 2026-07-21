from __future__ import annotations

import csv
from pathlib import Path

from macchiato.adapters.arducam_adapter import assign_scene_ids, assign_scene_splits


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_scene_boundary_is_strictly_greater_than_gap() -> None:
    assert assign_scene_ids([0, 180, 361], gap_seconds=180) == ["scene_000", "scene_000", "scene_001"]


def test_seeded_scene_split_is_exact_and_disjoint() -> None:
    scenes = [f"scene_{index:03d}" for index in range(10)]
    mapping = assign_scene_splits(scenes, seed=42, train_scene_count=6, val_scene_count=2)
    assert {scene for scene, split in mapping.items() if split == "train"} == {
        "scene_007", "scene_003", "scene_002", "scene_008", "scene_005", "scene_006"
    }
    assert {scene for scene, split in mapping.items() if split == "val"} == {"scene_009", "scene_004"}
    assert {scene for scene, split in mapping.items() if split == "test"} == {"scene_000", "scene_001"}


def test_tracked_experiment_b_manifests_have_expected_counts_and_no_scene_leakage() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "manifests"
    expected = {"train": 2370, "val": 732, "test": 793}
    scenes: dict[str, set[str]] = {}
    for split, count in expected.items():
        rows = _rows(root / f"{split}.csv")
        assert len(rows) == count
        assert all(row["eligible"] == "true" and row["split"] == split for row in rows)
        scenes[split] = {row["scene_id"] for row in rows}
    assert scenes["train"].isdisjoint(scenes["val"])
    assert scenes["train"].isdisjoint(scenes["test"])
    assert scenes["val"].isdisjoint(scenes["test"])
    all_rows = _rows(root / "all_samples.csv")
    assert len(all_rows) == 4109
    assert sum(row["eligible"] == "true" for row in all_rows) == 3895
