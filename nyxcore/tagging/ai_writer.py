from __future__ import annotations

from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError, TXXX

_FIELD_TO_DESC = {
    "energy": "NYX_ENERGY",
    "bpm": "NYX_BPM",
    "tags": "NYX_TAGS",
    "genre": "NYX_GENRE_TOP",
}


def get_existing_nyx_fields(path: Path) -> dict[str, str]:
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        return {}

    existing: dict[str, str] = {}
    for field, desc in _FIELD_TO_DESC.items():
        frames = id3.getall("TXXX")
        for frame in frames:
            if frame.desc == desc and frame.text:
                existing[field] = str(frame.text[0])
                break
    return existing


def write_ai_txxx(
    path: Path,
    *,
    energy: float | None,
    bpm: float | None,
    tags: list[str] | None,
    genre_top: str | None,
    fields: list[str],
    force: bool = False,
) -> tuple[list[str], list[str]]:
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        id3 = ID3()

    normalized_tags: list[str] = []
    if tags:
        for tag in tags:
            value = str(tag).strip()
            if value:
                normalized_tags.append(value)

    normalized_genre = None if genre_top is None else str(genre_top).strip()
    if normalized_genre == "":
        normalized_genre = None

    values = {
        "energy": None if energy is None else f"{energy:.1f}",
        "bpm": None if bpm is None else f"{int(round(float(bpm)))}",
        "tags": None if not normalized_tags else ";".join(normalized_tags),
        "genre": normalized_genre,
    }

    written: list[str] = []
    skipped_existing: list[str] = []
    frames = id3.getall("TXXX")
    existing_by_desc: dict[str, TXXX] = {frame.desc: frame for frame in frames if frame.desc}

    for field in fields:
        desc = _FIELD_TO_DESC[field]
        value = values[field]
        if value is None:
            continue
        if not force and desc in existing_by_desc:
            skipped_existing.append(field)
            continue
        id3.delall(f"TXXX:{desc}")
        id3.add(TXXX(encoding=3, desc=desc, text=[value]))
        written.append(field)

    if written:
        id3.save(path)
    return written, skipped_existing
