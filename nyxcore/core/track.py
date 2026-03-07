from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class WarningCode(str, Enum):
    missing_title = "missing_title"
    missing_artist = "missing_artist"
    missing_album = "missing_album"
    missing_cover_art = "missing_cover_art"
    duration_unavailable = "duration_unavailable"
    bitrate_unavailable = "bitrate_unavailable"
    low_bitrate = "low_bitrate"
    filename_youtube_noise = "filename_youtube_noise"
    filename_brackets_noise = "filename_brackets_noise"
    filename_feat_pattern = "filename_feat_pattern"
    read_error = "read_error"
    tag_parse_error = "tag_parse_error"
    possible_duplicate = "possible_duplicate"


@dataclass(slots=True)
class TrackRecord:
    path: str
    file_size_bytes: int
    mtime_iso: str
    tags: dict[str, str | None]
    has_cover_art: bool
    duration_seconds: float | None
    warnings: list[WarningCode] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = [code.value for code in self.warnings]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrackRecord":
        warnings = [WarningCode(item) for item in data.get("warnings", [])]
        return cls(
            path=str(data["path"]),
            file_size_bytes=int(data.get("file_size_bytes", 0)),
            mtime_iso=str(data.get("mtime_iso", "")),
            tags=dict(data.get("tags", {})),
            has_cover_art=bool(data.get("has_cover_art", False)),
            duration_seconds=data.get("duration_seconds"),
            warnings=warnings,
        )


def warn(track: TrackRecord, code: WarningCode) -> None:
    if code not in track.warnings:
        track.warnings.append(code)
