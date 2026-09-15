from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nyxcore.core.text import normalize_match_text, tokenize_match_text
from nyxcore.core.track import TrackRecord


@dataclass(frozen=True, slots=True)
class SearchResult:
    record: TrackRecord
    filename: str
    match_fields: tuple[str, ...]


def _search_fields(record: TrackRecord) -> dict[str, str]:
    path = Path(record.path)
    return {
        "filename": path.name,
        "title": record.tags.get("title") or "",
        "artist": record.tags.get("artist") or "",
        "album": record.tags.get("album") or "",
        "albumartist": record.tags.get("albumartist") or "",
        "genre": record.tags.get("genre") or "",
        "path": record.path,
    }


def search_tracks(
    records: Iterable[TrackRecord],
    query: str,
    *,
    limit: int = 20,
) -> tuple[int, list[SearchResult]]:
    """Return deterministic, ranked, read-only matches for all query tokens."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    normalized_query = normalize_match_text(query)
    query_tokens = tokenize_match_text(query)
    if not normalized_query or not query_tokens:
        return 0, []

    ranked: list[tuple[tuple[int, int, str, str, str], SearchResult]] = []
    for record in records:
        raw_fields = _search_fields(record)
        normalized_fields = {name: normalize_match_text(value) for name, value in raw_fields.items()}
        combined = " ".join(value for value in normalized_fields.values() if value)
        if not all(token in combined for token in query_tokens):
            continue

        match_fields = tuple(
            name
            for name in ("title", "artist", "album", "albumartist", "genre", "filename", "path")
            if normalized_fields[name] and any(token in normalized_fields[name] for token in query_tokens)
        )
        stem = normalize_match_text(Path(record.path).stem)
        title = normalized_fields["title"]
        filename = normalized_fields["filename"]

        if normalized_query in {title, stem, filename}:
            rank_bucket = 0
        elif title.startswith(normalized_query) or stem.startswith(normalized_query):
            rank_bucket = 1
        elif any(value.startswith(normalized_query) for value in normalized_fields.values() if value):
            rank_bucket = 2
        else:
            rank_bucket = 3

        result = SearchResult(
            record=record,
            filename=Path(record.path).name,
            match_fields=match_fields,
        )
        sort_key = (
            rank_bucket,
            -len(match_fields),
            title or stem,
            normalized_fields["artist"],
            normalize_match_text(record.path),
        )
        ranked.append((sort_key, result))

    ranked.sort(key=lambda item: item[0])
    return len(ranked), [result for _sort_key, result in ranked[:limit]]
