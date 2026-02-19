from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .track import TrackRecord, WarningCode


def to_iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def normalize_tag_value(value: object) -> str | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    text = str(value).strip()
    return text or None


def warning_for_missing_tags(tags: dict[str, str | None]) -> list[WarningCode]:
    warnings: list[WarningCode] = []
    if not tags.get("title"):
        warnings.append(WarningCode.missing_title)
    if not tags.get("artist"):
        warnings.append(WarningCode.missing_artist)
    if not tags.get("album"):
        warnings.append(WarningCode.missing_album)
    return warnings


def compute_stats(records: Iterable[TrackRecord]) -> dict:
    recs = list(records)
    artists = Counter()
    albums = Counter()
    problematic: list[dict[str, object]] = []

    missing_title = 0
    missing_artist = 0
    missing_album = 0
    cover_art_present = 0
    cover_art_missing = 0

    for rec in recs:
        artist = rec.tags.get("artist") or "UNKNOWN"
        album = rec.tags.get("album") or "UNKNOWN"
        artists[artist] += 1
        albums[album] += 1

        if rec.has_cover_art:
            cover_art_present += 1
        else:
            cover_art_missing += 1

        if WarningCode.missing_title in rec.warnings:
            missing_title += 1
        if WarningCode.missing_artist in rec.warnings:
            missing_artist += 1
        if WarningCode.missing_album in rec.warnings:
            missing_album += 1
        if rec.warnings:
            problematic.append(
                {"path": rec.path, "warnings": [warning.value for warning in rec.warnings]}
            )

    return {
        "total_tracks": len(recs),
        "missing_title": missing_title,
        "missing_artist": missing_artist,
        "missing_album": missing_album,
        "cover_art_present": cover_art_present,
        "cover_art_missing": cover_art_missing,
        "top_artists": artists.most_common(15),
        "top_albums": albums.most_common(15),
        "problematic_tracks_preview": problematic[:30],
    }


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
