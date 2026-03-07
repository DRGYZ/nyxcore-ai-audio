from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".wav",
    ".aiff",
    ".aif",
)


def is_supported_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_supported_audio_file(path))
