from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import HeaderNotFoundError

from .track import TrackRecord, WarningCode, warn
from .utils import compute_stats, normalize_tag_value, to_iso_utc, warning_for_missing_tags

LOW_BITRATE_BPS = 128_000
YOUTUBE_NOISE_PATTERN = re.compile(r"\b(official|lyrics?|audio|video|hd|4k)\b", re.IGNORECASE)
FEAT_PATTERN = re.compile(r"\b(feat\.?|ft\.?|featuring)\b", re.IGNORECASE)


def _extract_tags(audio: object) -> dict[str, str | None]:
    tags = getattr(audio, "tags", None) or {}
    get = tags.get
    return {
        "title": normalize_tag_value(get("title")),
        "artist": normalize_tag_value(get("artist")),
        "album": normalize_tag_value(get("album")),
        "albumartist": normalize_tag_value(get("albumartist")),
        "tracknumber": normalize_tag_value(get("tracknumber")),
        "date": normalize_tag_value(get("date")) or normalize_tag_value(get("year")),
        "genre": normalize_tag_value(get("genre")),
    }


def _has_cover_art(path: Path) -> bool:
    try:
        id3 = ID3(path)
        return bool(id3.getall("APIC"))
    except ID3NoHeaderError:
        return False
    except Exception:
        return False


def _has_brackets_noise(name: str) -> bool:
    return ("(" in name and ")" in name) or ("[" in name and "]" in name)


def scan_music_folder(
    root: Path,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[list[TrackRecord], dict]:
    records: list[TrackRecord] = []
    mp3_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3"]
    fingerprints: set[tuple[str, int]] = set()

    total = len(mp3_files)
    for idx, path in enumerate(mp3_files, start=1):
        tags: dict[str, str | None] = {
            "title": None,
            "artist": None,
            "album": None,
            "albumartist": None,
            "tracknumber": None,
            "date": None,
            "genre": None,
        }
        duration_seconds: float | None = None
        file_size = 0
        mtime_iso = ""

        record = TrackRecord(
            path=str(path),
            file_size_bytes=0,
            mtime_iso="",
            tags=tags,
            has_cover_art=False,
            duration_seconds=None,
            warnings=[],
        )

        try:
            stat = path.stat()
            file_size = stat.st_size
            mtime_iso = to_iso_utc(stat.st_mtime)
        except OSError:
            warn(record, WarningCode.read_error)

        try:
            audio = MutagenFile(path, easy=True)
            if audio is None:
                warn(record, WarningCode.read_error)
            else:
                tags = _extract_tags(audio)
                info = getattr(audio, "info", None)
                if info and getattr(info, "length", None) is not None:
                    duration_seconds = round(float(info.length), 3)
                else:
                    warn(record, WarningCode.duration_unavailable)
                bitrate = getattr(info, "bitrate", None) if info is not None else None
                if bitrate is None:
                    warn(record, WarningCode.bitrate_unavailable)
                elif int(bitrate) < LOW_BITRATE_BPS:
                    warn(record, WarningCode.low_bitrate)
        except UnicodeDecodeError:
            warn(record, WarningCode.tag_parse_error)
        except (HeaderNotFoundError, OSError):
            warn(record, WarningCode.read_error)
        except Exception:
            warn(record, WarningCode.tag_parse_error)

        record.tags = tags
        record.duration_seconds = duration_seconds
        record.file_size_bytes = file_size
        record.mtime_iso = mtime_iso

        for code in warning_for_missing_tags(tags):
            warn(record, code)

        has_cover_art = _has_cover_art(path)
        record.has_cover_art = has_cover_art
        if not has_cover_art:
            warn(record, WarningCode.missing_cover_art)

        filename = path.stem
        if YOUTUBE_NOISE_PATTERN.search(filename):
            warn(record, WarningCode.filename_youtube_noise)
        if _has_brackets_noise(filename):
            warn(record, WarningCode.filename_brackets_noise)
        if FEAT_PATTERN.search(filename):
            warn(record, WarningCode.filename_feat_pattern)

        fingerprint = (path.name.lower(), file_size)
        if fingerprint in fingerprints:
            warn(record, WarningCode.possible_duplicate)
        else:
            fingerprints.add(fingerprint)

        record.warnings.sort(key=lambda code: code.value)
        records.append(record)
        if on_progress is not None:
            on_progress(idx, total, path)

    return records, compute_stats(records)
