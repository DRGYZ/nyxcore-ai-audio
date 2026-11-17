from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, ID3NoHeaderError, TALB, TIT2, TPE1
from mutagen.mp3 import EasyMP3
from mutagen.mp4 import MP4, MP4Cover


class TagWriteError(RuntimeError):
    """Raised when basic metadata cannot be safely written for a file."""


@dataclass(frozen=True)
class ArtworkEdit:
    data: bytes
    digest: str
    width: int
    height: int
    replace_existing: bool


def front_covers(path: Path) -> list[bytes]:
    """Read native front covers while leaving other picture types alone."""
    if path.suffix.lower() == ".mp3":
        try:
            return [item.data for item in ID3(path, translate=False).getall("APIC") if item.type == 3]
        except ID3NoHeaderError:
            return []
    if path.suffix.lower() == ".flac":
        return [picture.data for picture in FLAC(path).pictures if picture.type == 3]
    if path.suffix.lower() == ".m4a":
        return [bytes(cover) for cover in (MP4(path).tags or {}).get("covr", [])]
    return []


def _check_artwork(path: Path, artwork: ArtworkEdit) -> None:
    if hashlib.sha256(artwork.data).hexdigest() != artwork.digest:
        raise TagWriteError("Artwork differs from the approved content hash.")
    covers = front_covers(path)
    if len(covers) > 1:
        raise TagWriteError("More than one native front cover is present. Artwork editing is unavailable.")
    if covers and not artwork.replace_existing:
        raise TagWriteError("Replacing the existing front cover requires explicit approval.")


def verify_written_artwork(path: Path, artwork: ArtworkEdit) -> None:
    covers = front_covers(path)
    if len(covers) != 1 or hashlib.sha256(covers[0]).hexdigest() != artwork.digest:
        raise TagWriteError("The embedded front cover did not match the approved image.")


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
    artwork: ArtworkEdit | None = None,
) -> None:
    if artwork is not None:
        _check_artwork(path, artwork)
    audio = _load_mutagen_writer(path)
    values = {"title": title, "artist": artist, "album": album}
    if (preserve_id3 or artwork is not None) and isinstance(audio, EasyMP3):
        # Keep the existing ID3 revision and leave frames outside these fields alone.
        try:
            tags = ID3(path, translate=False)
        except ID3NoHeaderError:
            tags = ID3()
        for field, frame in (("title", TIT2), ("artist", TPE1), ("album", TALB)):
            value = values[field]
            if field in fields and value and value.upper() != "UNKNOWN":
                tags.setall(frame.__name__, [frame(encoding=1, text=[value])])
        if artwork is not None:
            for picture in tags.getall("APIC"):
                if picture.type == 3:
                    del tags[picture.HashKey]
            label = "Front cover"
            while f"APIC:{label}" in tags:
                label += " "
            tags.add(APIC(encoding=1, mime="image/jpeg", type=3, desc=label, data=artwork.data))
        # Preserve existing NUL-separated lists instead of flattening them with slashes.
        tags.save(path, v2_version=3 if tags.version[1] == 3 else 4, v23_sep=None)
        return

    if artwork is not None and path.suffix.lower() == ".m4a":
        container = MP4(path)
        if container.tags is None:
            container.add_tags()
        for field, atom in (("title", "©nam"), ("artist", "©ART"), ("album", "©alb")):
            if field in fields and values[field]:
                container[atom] = [values[field]]
        container["covr"] = [MP4Cover(artwork.data, imageformat=MP4Cover.FORMAT_JPEG)]
        container.save()
        return

    if artwork is not None and not isinstance(audio, FLAC):
        raise TagWriteError("This audio layout does not support artwork editing.")

    if "title" in fields and title and title.upper() != "UNKNOWN":
        audio["title"] = [title]
    if "artist" in fields and artist and artist.upper() != "UNKNOWN":
        audio["artist"] = [artist]
    if "album" in fields and album and album.upper() != "UNKNOWN":
        audio["album"] = [album]

    if artwork is not None:
        retained = [picture for picture in audio.pictures if picture.type != 3]
        audio.clear_pictures()
        for picture in retained:
            audio.add_picture(picture)
        picture = Picture()
        picture.type = 3
        picture.mime = "image/jpeg"
        picture.width = artwork.width
        picture.height = artwork.height
        picture.depth = 24
        picture.data = artwork.data
        audio.add_picture(picture)

    audio.save()


def verify_written_tags(path: Path, values: dict[str, str | None]) -> None:
    values = {field: value for field, value in values.items() if field in {"title", "artist", "album"}}
    if not values:
        return
    audio = MutagenFile(path, easy=True)
    if audio is None or audio.tags is None:
        raise TagWriteError("The file could not be read after writing its metadata.")
    for field, value in values.items():
        if audio.tags.get(field) != [value]:
            raise TagWriteError(f"The written {field} did not match the approved value.")


def write_basic_tags(path: Path, *, title: str | None, artist: str | None, album: str | None) -> None:
    write_tags(path, title=title, artist=artist, album=album, fields=["title", "artist", "album"])
