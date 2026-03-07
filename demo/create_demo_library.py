from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SynthTrack:
    relative_path: str
    frequency_hz: float
    duration_seconds: float
    codec: str
    bitrate: str | None
    metadata: dict[str, str]
    notes: list[str]


TRACKS: tuple[SynthTrack, ...] = (
    SynthTrack(
        relative_path="artists/signal-path/Signal Path - Blue Hour.flac",
        frequency_hz=261.63,
        duration_seconds=6.0,
        codec="flac",
        bitrate=None,
        metadata={
            "title": "Blue Hour",
            "artist": "Signal Path",
            "album": "Archive Sessions",
        },
        notes=["clean_reference", "exact_duplicate_source"],
    ),
    SynthTrack(
        relative_path="artists/signal-path/Signal Path - Night Drift Ambient.flac",
        frequency_hz=329.63,
        duration_seconds=8.0,
        codec="flac",
        bitrate=None,
        metadata={
            "title": "Night Drift Ambient",
            "artist": "Signal Path",
            "album": "Focus Studies",
        },
        notes=["likely_duplicate_source", "saved_playlist_candidate"],
    ),
    SynthTrack(
        relative_path="imports/rips/Signal Path - Night Drift Ambient.mp3",
        frequency_hz=329.63,
        duration_seconds=8.2,
        codec="mp3",
        bitrate="96k",
        metadata={
            "title": "Night Drift Ambient",
            "artist": "Signal Path",
            "album": "Focus Studies",
        },
        notes=["likely_duplicate_match", "low_bitrate", "saved_playlist_candidate"],
    ),
    SynthTrack(
        relative_path="ambient/Mira Vale - Stillness Focus Instrumental.mp3",
        frequency_hz=392.0,
        duration_seconds=7.0,
        codec="mp3",
        bitrate="192k",
        metadata={
            "title": "Stillness Focus Instrumental",
            "artist": "Mira Vale",
            "album": "Focus Studies",
        },
        notes=["saved_playlist_candidate"],
    ),
    SynthTrack(
        relative_path="imports/legacy/Track 01.mp3",
        frequency_hz=220.0,
        duration_seconds=5.0,
        codec="mp3",
        bitrate="96k",
        metadata={
            "title": "Track 01",
            "artist": "Unknown",
        },
        notes=["placeholder_metadata", "missing_album", "low_bitrate", "folder_hotspot"],
    ),
    SynthTrack(
        relative_path="imports/legacy/Untitled Tape [Official Audio].mp3",
        frequency_hz=174.61,
        duration_seconds=4.5,
        codec="mp3",
        bitrate="96k",
        metadata={},
        notes=["missing_metadata", "filename_noise", "low_bitrate", "folder_hotspot"],
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tiny NyxCore demo library with exact duplicates, likely duplicates, "
            "metadata problems, missing artwork, low-bitrate files, and saved-playlist candidates."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="demo/generated/sample-library",
        help="Destination folder for the generated demo library",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the destination first if it already exists",
    )
    return parser


def _ffmpeg_path() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise SystemExit(
            "ffmpeg is required to generate the NyxCore demo library. "
            "Install ffmpeg and make sure it is on PATH, then rerun this script."
        )
    return binary


def _ffmpeg_command(track: SynthTrack, destination: Path) -> list[str]:
    extension = destination.suffix.lower()
    command = [
        _ffmpeg_path(),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={track.frequency_hz}:duration={track.duration_seconds}",
    ]
    if extension == ".mp3":
        command.extend(["-c:a", "libmp3lame", "-b:a", track.bitrate or "192k"])
    elif extension == ".flac":
        command.extend(["-c:a", "flac"])
    else:
        raise ValueError(f"Unsupported demo track extension: {extension}")
    for key, value in sorted(track.metadata.items()):
        command.extend(["-metadata", f"{key}={value}"])
    command.append(str(destination))
    return command


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _write_manifest(output_root: Path) -> None:
    manifest = {
        "generator": "demo/create_demo_library.py",
        "tracks": [asdict(track) for track in TRACKS],
        "exact_duplicate_copy": {
            "source": "artists/signal-path/Signal Path - Blue Hour.flac",
            "copy": "imports/exact/Signal Path - Blue Hour copy.flac",
        },
        "expected_highlights": [
            "1 exact duplicate group",
            "1 likely duplicate group",
            "missing metadata and placeholder metadata findings",
            "missing artwork across the demo set",
            "low-quality audio from 96k MP3 rips",
            "saved playlist candidates for queries like 'ambient focus instrumental'",
        ],
    }
    (output_root / "nyxcore_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_readme(output_root: Path) -> None:
    lines = [
        "# NyxCore Demo Library",
        "",
        "This library is safe to mutate during local demos.",
        "",
        "Expected findings:",
        "- exact duplicate: Blue Hour plus its copied FLAC",
        "- likely duplicate: FLAC vs 96k MP3 Night Drift Ambient",
        "- missing metadata: Untitled Tape [Official Audio].mp3",
        "- weak metadata: Track 01.mp3 with placeholder tags",
        "- missing artwork: every file in this demo library",
        "- low quality: 96k MP3 files in imports/",
        "",
        "Suggested CLI walkthrough:",
        "1. python -m nyxcore.cli scan <this-folder> --out data/reports",
        "2. python -m nyxcore.cli duplicates <this-folder> --out data/reports",
        "3. python -m nyxcore.cli health <this-folder> --out data/reports",
        "4. python -m nyxcore.cli review <this-folder> --out data/reports",
        "5. python -m nyxcore.cli save-playlist <this-folder> --out data/reports --name \"Ambient Focus\" --query \"ambient focus instrumental\"",
        "",
        "If you use the web UI, point the API at this folder with NYXCORE_WEB_MUSIC_DIR.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_demo_library(output_root: Path, *, force: bool = False) -> Path:
    if output_root.exists():
        if not force:
            raise SystemExit(f"Destination already exists: {output_root}. Use --force to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    for track in TRACKS:
        destination = output_root / track.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(_ffmpeg_command(track, destination))

    exact_source = output_root / "artists/signal-path/Signal Path - Blue Hour.flac"
    exact_copy = output_root / "imports/exact/Signal Path - Blue Hour copy.flac"
    exact_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exact_source, exact_copy)

    _write_manifest(output_root)
    _write_readme(output_root)
    return output_root


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = Path(args.output).resolve()
    generated = generate_demo_library(output_root, force=args.force)
    print(f"NyxCore demo library written to: {generated}")
    print("Suggested next step:")
    print(f"  python -m nyxcore.cli review \"{generated}\" --out data/reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
