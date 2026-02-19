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


def warn(track: TrackRecord, code: WarningCode) -> None:
    if code not in track.warnings:
        track.warnings.append(code)

