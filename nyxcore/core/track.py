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
    # Transient metadata from this scan. Incremental state keeps its existing format.
    scanned_bitrate_bps: int | None = field(default=None, repr=False, compare=False)
    # Keep full tag values for organization decisions without changing serialized scan records.
    organization_tags: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False, compare=False)
    scanned_stat: tuple[int, int, int, int] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        del data["scanned_bitrate_bps"]
        del data["organization_tags"]
        del data["scanned_stat"]
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
