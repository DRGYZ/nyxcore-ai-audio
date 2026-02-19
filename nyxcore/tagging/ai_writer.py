from __future__ import annotations

from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError, TXXX

_FIELD_TO_DESC = {
    "energy": "NYX_ENERGY",
    "bpm": "NYX_BPM",
    "tags": "NYX_TAGS",
    "genre": "NYX_GENRE_TOP",
}

_JUDGE_FIELD_TO_DESC = {
    "tags": "NYX_TAGS",
    "genre": "NYX_GENRE_TOP",
    "conf": "NYX_CONF",
    "judge": "NYX_JUDGE",
    "reason": "NYX_JUDGE_REASON",
}
_REASON_CONNECTOR_TAILS = {"but", "and", "or", "with", "because", "so", "which"}


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


def get_existing_nyx_judge_fields(path: Path) -> dict[str, str]:
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        return {}
    existing: dict[str, str] = {}
    frames = id3.getall("TXXX")
    for field, desc in _JUDGE_FIELD_TO_DESC.items():
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


def write_judge_txxx(
    path: Path,
    *,
    tags: list[str] | None,
    genre_top: str | None,
    conf: float | None,
    judge: str | None,
    reason: str | None,
    fields: list[str],
    force: bool = False,
) -> tuple[list[str], list[str]]:
    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        id3 = ID3()

    clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
    clean_genre = None if genre_top is None else str(genre_top).strip()
    if clean_genre == "":
        clean_genre = None
    clean_reason = None if reason is None else str(reason).strip()
    if clean_reason == "":
        clean_reason = None
    if clean_reason is not None and len(clean_reason) > 120:
        clipped = clean_reason[:120]
        clean_reason = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    if clean_reason is not None:
        r = clean_reason.strip()
        while r:
            words = r.split()
            if not words:
                break
            if words[-1].lower().strip(".,;:!?") in _REASON_CONNECTOR_TAILS:
                r = " ".join(words[:-1]).strip()
                continue
            break
        clean_reason = r.rstrip(" ,.;:!?")
        if clean_reason == "":
            clean_reason = None

    values = {
        "tags": None if not clean_tags else ";".join(clean_tags),
        "genre": clean_genre,
        "conf": None if conf is None else f"{float(conf):.2f}",
        "judge": None if judge is None or str(judge).strip() == "" else str(judge).strip(),
        "reason": clean_reason,
    }

    frames = id3.getall("TXXX")
    existing_by_desc: dict[str, TXXX] = {frame.desc: frame for frame in frames if frame.desc}
    written: list[str] = []
    skipped_existing: list[str] = []

    for field in fields:
        if field not in _JUDGE_FIELD_TO_DESC:
            continue
        desc = _JUDGE_FIELD_TO_DESC[field]
        value = values.get(field)
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
