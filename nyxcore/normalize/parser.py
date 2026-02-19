from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from mutagen import File as MutagenFile

from .rules import (
    clean_artist_hygiene,
    cleaned_filename_stem,
    collapse_ws,
    is_artist_hygiene_applicable,
    is_missing,
    normalize_artist_name,
    parse_artist_title_from_filename,
    propose_album,
)


@dataclass(slots=True)
class NormalizePreviewRecord:
    path: str
    current_title: str | None
    current_artist: str | None
    current_album: str | None
    proposed_title: str | None
    proposed_artist: str | None
    proposed_album: str | None
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    would_change: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _tag_value(tags: object, key: str) -> str | None:
    if tags is None:
        return None
    value = tags.get(key)
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    text = collapse_ws(str(value))
    return text or None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _build_artist_hygiene_record(
    *,
    path: Path,
    current_title: str | None,
    current_artist: str | None,
    current_album: str | None,
) -> NormalizePreviewRecord:
    reasons: list[str] = []
    proposed_artist = current_artist
    confidence = 0.0
    would_change = False

    if current_artist and is_artist_hygiene_applicable(current_artist):
        cleaned_artist, hygiene_reasons = clean_artist_hygiene(current_artist)
        if hygiene_reasons and cleaned_artist and cleaned_artist != collapse_ws(current_artist):
            proposed_artist = cleaned_artist
            reasons.extend(hygiene_reasons)
            confidence = 0.95
            would_change = True

    return NormalizePreviewRecord(
        path=str(path),
        current_title=current_title,
        current_artist=current_artist,
        current_album=current_album,
        proposed_title=current_title,
        proposed_artist=proposed_artist,
        proposed_album=current_album,
        reasons=sorted(set(reasons)),
        confidence=confidence,
        would_change=would_change,
    )


def build_normalize_preview(root: Path, *, strategy: str = "smart") -> list[NormalizePreviewRecord]:
    records: list[NormalizePreviewRecord] = []
    mp3_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3"]

    for path in mp3_files:
        reasons: list[str] = []
        current_title: str | None = None
        current_artist: str | None = None
        current_album: str | None = None

        try:
            audio = MutagenFile(path, easy=True)
            tags = getattr(audio, "tags", None) if audio is not None else None
            current_title = _tag_value(tags, "title")
            current_artist = _tag_value(tags, "artist")
            current_album = _tag_value(tags, "album")
        except Exception:
            tags = None

        if strategy == "artist_hygiene":
            records.append(
                _build_artist_hygiene_record(
                    path=path,
                    current_title=current_title,
                    current_artist=current_artist,
                    current_album=current_album,
                )
            )
            continue

        cleaned_stem, filename_reasons = cleaned_filename_stem(path.stem)
        reasons.extend(filename_reasons)

        parsed_artist, parsed_title, parsed_ok = parse_artist_title_from_filename(cleaned_stem)
        if parsed_ok:
            reasons.append("parsed_artist_title")

        current_artist_norm, artist_reasons = normalize_artist_name(current_artist)
        reasons.extend(artist_reasons)
        parsed_artist_norm, parsed_artist_reasons = normalize_artist_name(parsed_artist)
        reasons.extend(parsed_artist_reasons)

        if not is_missing(current_title):
            proposed_title = collapse_ws(current_title or "")
        elif parsed_ok and parsed_title:
            proposed_title = parsed_title
        else:
            proposed_title = cleaned_stem or path.stem

        if not is_missing(current_artist_norm):
            proposed_artist = current_artist_norm
        elif parsed_ok and not is_missing(parsed_artist_norm):
            proposed_artist = parsed_artist_norm
        else:
            proposed_artist = "UNKNOWN"

        proposed_album, album_reason = propose_album(current_album, cleaned_stem, strategy)
        if album_reason:
            reasons.append(album_reason)

        confidence = 0.5
        if parsed_ok:
            confidence += 0.2
        if filename_reasons and not parsed_ok:
            confidence += 0.1
        if not parsed_ok and is_missing(current_title) and is_missing(current_artist_norm):
            confidence -= 0.2
            reasons.append("ambiguous_filename")
        confidence = round(_clamp(confidence), 3)

        current_title_cmp = None if is_missing(current_title) else collapse_ws(current_title or "")
        current_artist_cmp = None if is_missing(current_artist_norm) else collapse_ws(current_artist_norm or "")
        current_album_cmp = None if is_missing(current_album) else collapse_ws(current_album or "")
        proposed_title_cmp = None if is_missing(proposed_title) else collapse_ws(proposed_title or "")
        proposed_artist_cmp = None if is_missing(proposed_artist) else collapse_ws(proposed_artist or "")
        proposed_album_cmp = None if is_missing(proposed_album) else collapse_ws(proposed_album or "")
        would_change = (
            current_title_cmp != proposed_title_cmp
            or current_artist_cmp != proposed_artist_cmp
            or current_album_cmp != proposed_album_cmp
        )

        record = NormalizePreviewRecord(
            path=str(path),
            current_title=current_title,
            current_artist=current_artist,
            current_album=current_album,
            proposed_title=proposed_title,
            proposed_artist=proposed_artist,
            proposed_album=proposed_album,
            reasons=sorted(set(reasons)),
            confidence=confidence,
            would_change=would_change,
        )
        records.append(record)

    return records
