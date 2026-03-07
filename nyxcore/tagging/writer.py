from __future__ import annotations

import shutil
from pathlib import Path

from mutagen import File as MutagenFile


class TagWriteError(RuntimeError):
    """Raised when basic metadata cannot be safely written for a file."""


def _load_mutagen_writer(path: Path):
    audio = MutagenFile(path, easy=True)
    if audio is None:
        raise TagWriteError(f"Unsupported or unreadable audio file: {path}")
    if audio.tags is None:
        add_tags = getattr(audio, "add_tags", None)
        if add_tags is None:
            raise TagWriteError(f"Audio format does not support safe generic tag writing: {path.suffix.lower()}")
        try:
            add_tags()
        except Exception as exc:
            raise TagWriteError(
                f"Failed to initialize tags for audio format: {path.suffix.lower()}"
            ) from exc
    return audio


def backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / src.name
    if destination.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while True:
            candidate = backup_dir / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                destination = candidate
                break
            i += 1
    shutil.copy2(src, destination)
    return destination


def write_tags(
    path: Path,
    title: str | None,
    artist: str | None,
    album: str | None,
    fields: list[str],
) -> None:
    audio = _load_mutagen_writer(path)

    if "title" in fields and title and title.upper() != "UNKNOWN":
        audio["title"] = [title]
    if "artist" in fields and artist and artist.upper() != "UNKNOWN":
        audio["artist"] = [artist]
    if "album" in fields and album and album.upper() != "UNKNOWN":
        audio["album"] = [album]

    audio.save()


def write_basic_tags(path: Path, *, title: str | None, artist: str | None, album: str | None) -> None:
    write_tags(path, title=title, artist=artist, album=album, fields=["title", "artist", "album"])
