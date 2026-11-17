from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, ID3NoHeaderError, TALB, TIT2, TPE1
from mutagen.mp3 import EasyMP3


class TagWriteError(RuntimeError):
    """Raised when basic metadata cannot be safely written for a file."""


def _sanitize_backup_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "backup"


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


def plan_backup_path(src: Path, backup_dir: Path) -> Path:
    label_parts = [part for part in (src.parent.name, src.stem) if part]
    label = "__".join(_sanitize_backup_label(part) for part in label_parts if part) or "backup"
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    destination = backup_dir / f"{label}.{digest}{src.suffix}"
    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix
        i = 1
        while True:
            candidate = backup_dir / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                destination = candidate
                break
            i += 1
    return destination


def backup_file(src: Path, backup_dir: Path, *, destination: Path | None = None) -> Path:
    destination = plan_backup_path(src, backup_dir) if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with src.open("rb") as source_stream, destination.open("xb") as destination_stream:
            created = True
            shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        shutil.copystat(src, destination)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination


def write_tags(
    path: Path,
    title: str | None,
    artist: str | None,
    album: str | None,
    fields: list[str],
    *,
    preserve_id3: bool = False,
) -> None:
    audio = _load_mutagen_writer(path)
    values = {"title": title, "artist": artist, "album": album}
    if preserve_id3 and isinstance(audio, EasyMP3):
        # Keep the existing ID3 revision and leave frames outside these fields alone.
        try:
            tags = ID3(path, translate=False)
        except ID3NoHeaderError:
            tags = ID3()
        for field, frame in (("title", TIT2), ("artist", TPE1), ("album", TALB)):
            value = values[field]
            if field in fields and value and value.upper() != "UNKNOWN":
                tags.setall(frame.__name__, [frame(encoding=1, text=[value])])
        tags.save(path, v2_version=3 if tags.version[1] == 3 else 4)
        return

    if "title" in fields and title and title.upper() != "UNKNOWN":
        audio["title"] = [title]
    if "artist" in fields and artist and artist.upper() != "UNKNOWN":
        audio["artist"] = [artist]
    if "album" in fields and album and album.upper() != "UNKNOWN":
        audio["album"] = [album]

    audio.save()


def verify_written_tags(path: Path, values: dict[str, str | None]) -> None:
    audio = MutagenFile(path, easy=True)
    if audio is None or audio.tags is None:
        raise TagWriteError("The file could not be read after writing its metadata.")
    for field, value in values.items():
        if audio.tags.get(field) != [value]:
            raise TagWriteError(f"The written {field} did not match the approved value.")


def write_basic_tags(path: Path, *, title: str | None, artist: str | None, album: str | None) -> None:
    write_tags(path, title=title, artist=artist, album=album, fields=["title", "artist", "album"])
