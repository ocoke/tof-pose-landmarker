"""Shared records for converting source datasets into Macchiato samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ManifestRecord:
    """One timestamp-aligned Arducam sample."""

    timestamp: int
    datetime_utc: str
    scene_id: str
    split: str
    eligible: bool
    valid_keypoints: int
    height: int | str
    width: int | str
    depth_path: str
    confidence_path: str
    pose_path: str
    drop_reason: str = ""

    def to_row(self) -> dict[str, object]:
        """Convert to a CSV-ready dictionary with stable boolean spelling."""

        row = asdict(self)
        row["eligible"] = "true" if self.eligible else "false"
        return row


MANIFEST_COLUMNS = list(ManifestRecord.__dataclass_fields__)
