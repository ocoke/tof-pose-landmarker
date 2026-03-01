import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

DEFAULT_MANIFEST = "manifest_v7.jsonl"


def _read_scene_map(scene_map_path: Optional[str]) -> Dict[str, str]:
    if not scene_map_path:
        return {}
    with open(scene_map_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    raise ValueError("Scene map must be a JSON object mapping sample_id to scene_id.")


def infer_session_ids(sample_ids: Iterable[str], gap_seconds: int = 10) -> Dict[str, str]:
    parsed = []
    fallback = {}
    for sample_id in sorted({str(s) for s in sample_ids}):
        try:
            parsed.append((int(sample_id), sample_id))
        except ValueError:
            fallback[sample_id] = f"session_{sample_id}"

    if not parsed:
        return fallback

    parsed.sort()
    session_idx = 0
    previous_ts = None
    mapping: Dict[str, str] = {}
    for ts, sample_id in parsed:
        if previous_ts is None or (ts - previous_ts) > gap_seconds:
            session_idx += 1
        mapping[sample_id] = f"session_{session_idx:04d}"
        previous_ts = ts

    mapping.update(fallback)
    return mapping


def _keypoints_path(data_dir: str, sample_id: str) -> Path:
    return Path(data_dir) / "pose_coco17" / f"{sample_id}.json"


def _depth_path(data_dir: str, sample_id: str) -> Path:
    return Path(data_dir) / "depth" / f"{sample_id}.npy"


def _confidence_path(data_dir: str, sample_id: str) -> Path:
    return Path(data_dir) / "confidence" / f"{sample_id}.npy"


def _load_npy_checked(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing {label} file: {path}")
    try:
        array = np.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unreadable {label} file '{path.name}': {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"expected 2D {label} map in '{path.name}', got shape {array.shape}")
    return array


def _load_keypoints_checked(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing pose label file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unreadable pose label file '{path.name}': {exc}") from exc

    keypoints = payload.get("keypoints")
    if keypoints is None:
        raise ValueError(f"missing 'keypoints' in pose label file '{path.name}'")
    keypoints = np.array(keypoints, dtype=np.float32)
    if keypoints.shape != (17, 2):
        raise ValueError(f"expected keypoints shape (17, 2) in '{path.name}', got {tuple(keypoints.shape)}")
    return keypoints


def _compute_validity(data_dir: str, sample_id: str) -> Dict[str, Any]:
    depth = _load_npy_checked(_depth_path(data_dir, sample_id), "depth")
    confidence = _load_npy_checked(_confidence_path(data_dir, sample_id), "confidence")
    if confidence.shape != depth.shape:
        raise ValueError(
            f"confidence shape {confidence.shape} does not match depth shape {depth.shape} for sample {sample_id}"
        )
    height, width = depth.shape
    keypoints = _load_keypoints_checked(_keypoints_path(data_dir, sample_id))

    valid = (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] < width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] < height)
    )
    num_valid = int(valid.sum())
    drop_reason = "too_few_valid_keypoints" if num_valid < 12 else None
    return {
        "valid_kpt_mask": valid.astype(int).tolist(),
        "num_valid_kpts": num_valid,
        "drop_reason": drop_reason,
    }


def _skipped_output_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_skipped.jsonl"))


def _save_jsonl_rows(rows: List[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sanitize_records(
    data_dir: str,
    records: List[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_records: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for record in records:
        sample_id = record["sample_id"]
        try:
            validity = _compute_validity(data_dir, sample_id)
        except Exception as exc:  # noqa: BLE001
            skipped = {
                "sample_id": sample_id,
                "scene_id": record.get("scene_id", "scene_0"),
                "session_id": record.get("session_id", "session_0001"),
                "split": record.get("split", "train_core"),
                "error": str(exc),
            }
            skipped_rows.append(skipped)
            if verbose:
                print(f"[WARN] Skipping broken sample {sample_id}: {exc}")
            continue

        clean_record = dict(record)
        clean_record.update(validity)
        clean_records.append(clean_record)

    return clean_records, skipped_rows


def _assign_scene_splits(scene_sessions: Dict[str, List[str]]) -> Dict[str, str]:
    scenes = sorted(scene_sessions)
    split_by_session: Dict[str, str] = {}

    if not scenes:
        return split_by_session

    if len(scenes) == 1:
        sessions = sorted(scene_sessions[scenes[0]])
        if len(sessions) <= 1:
            for session_id in sessions:
                split_by_session[session_id] = "train_core"
        else:
            for session_id in sessions[:-1]:
                split_by_session[session_id] = "train_core"
            split_by_session[sessions[-1]] = "val_seen"
        return split_by_session

    if len(scenes) == 2:
        train_scene, unseen_scene = scenes
        seen_sessions = sorted(scene_sessions[train_scene])
        if len(seen_sessions) <= 1:
            for session_id in seen_sessions:
                split_by_session[session_id] = "train_core"
        else:
            for session_id in seen_sessions[:-1]:
                split_by_session[session_id] = "train_core"
            split_by_session[seen_sessions[-1]] = "val_seen"
        for session_id in scene_sessions[unseen_scene]:
            split_by_session[session_id] = "val_unseen"
        return split_by_session

    test_scene = scenes[-1]
    val_unseen_scene = scenes[-2]
    train_scenes = scenes[:-2]

    for session_id in scene_sessions[test_scene]:
        split_by_session[session_id] = "test_unseen"
    for session_id in scene_sessions[val_unseen_scene]:
        split_by_session[session_id] = "val_unseen"

    for scene_id in train_scenes:
        sessions = sorted(scene_sessions[scene_id])
        if len(sessions) <= 1:
            split_by_session[sessions[0]] = "train_core"
            continue
        for session_id in sessions[:-1]:
            split_by_session[session_id] = "train_core"
        split_by_session[sessions[-1]] = "val_seen"

    return split_by_session


def build_manifest(
    data_dir: str,
    scene_map_path: Optional[str] = None,
    output_path: Optional[str] = None,
    gap_seconds: int = 10,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    depth_dir = Path(data_dir) / "depth"
    sample_ids = sorted(path.stem for path in depth_dir.glob("*.npy"))
    scene_map = _read_scene_map(scene_map_path)
    session_map = infer_session_ids(sample_ids, gap_seconds=gap_seconds)

    records: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    scene_sessions: Dict[str, List[str]] = defaultdict(list)
    for sample_id in sample_ids:
        session_id = session_map[sample_id]
        scene_id = scene_map.get(sample_id, "scene_0")

        try:
            validity = _compute_validity(data_dir, sample_id)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "session_id": session_id,
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"[WARN] Skipping broken sample {sample_id}: {exc}")
            continue

        if session_id not in scene_sessions[scene_id]:
            scene_sessions[scene_id].append(session_id)
        records.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "session_id": session_id,
                **validity,
            }
        )

    split_by_session = _assign_scene_splits(scene_sessions)
    for record in records:
        record["split"] = split_by_session.get(record["session_id"], "train_core")

    if output_path:
        save_manifest(records, output_path)
        skipped_path = _skipped_output_path(output_path)
        if skipped_rows:
            _save_jsonl_rows(skipped_rows, skipped_path)
        elif os.path.exists(skipped_path):
            os.remove(skipped_path)
    return records


def save_manifest(records: List[Dict[str, Any]], output_path: str) -> None:
    _save_jsonl_rows(records, output_path)


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_or_load_manifest(
    data_dir: str,
    manifest_path: Optional[str] = None,
    scene_map_path: Optional[str] = None,
    gap_seconds: int = 10,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    manifest_path = manifest_path or os.path.join(data_dir, DEFAULT_MANIFEST)
    if not rebuild and os.path.exists(manifest_path):
        records = load_manifest(manifest_path)
        clean_records, skipped_rows = _sanitize_records(data_dir, records, verbose=False)
        if len(clean_records) != len(records):
            save_manifest(clean_records, manifest_path)
            skipped_path = _skipped_output_path(manifest_path)
            if skipped_rows:
                _save_jsonl_rows(skipped_rows, skipped_path)
        return clean_records
    return build_manifest(
        data_dir=data_dir,
        scene_map_path=scene_map_path,
        output_path=manifest_path,
        gap_seconds=gap_seconds,
    )


def get_records_for_split(records: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("split") == split]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v7 manifest for scene-aware training.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root.")
    parser.add_argument("--output", type=str, default=None, help="Manifest output path.")
    parser.add_argument("--scene-map", type=str, default=None, help="Optional JSON mapping sample_id to scene_id.")
    parser.add_argument("--gap-seconds", type=int, default=10, help="Gap threshold used to infer session ids.")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.data, DEFAULT_MANIFEST)
    records = build_manifest(
        args.data,
        scene_map_path=args.scene_map,
        output_path=output_path,
        gap_seconds=args.gap_seconds,
        verbose=True,
    )

    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["split"]] += 1

    print(f"Wrote {len(records)} manifest rows to {output_path}")
    skipped_path = _skipped_output_path(output_path)
    if os.path.exists(skipped_path):
        skipped_count = sum(1 for _ in open(skipped_path, "r", encoding="utf-8"))
        print(f"Skipped {skipped_count} broken samples. Details: {skipped_path}")
    for split_name in sorted(counts):
        print(f"{split_name}: {counts[split_name]}")


if __name__ == "__main__":
    main()
