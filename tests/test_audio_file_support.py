from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nyxcore.core.audio_files import SUPPORTED_AUDIO_EXTENSIONS, iter_audio_files
from nyxcore.core.scanner import scan_music_folder
from nyxcore.normalize.parser import build_normalize_preview
from nyxcore.tagging.writer import TagWriteError, write_basic_tags

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_audio"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AudioFileSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_iter_audio_files_filters_and_sorts_supported_extensions(self) -> None:
        (self.root / "b.flac").write_text("", encoding="utf-8")
        (self.root / "a.mp3").write_text("", encoding="utf-8")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "c.M4A").write_text("", encoding="utf-8")
        (self.root / "ignore.txt").write_text("", encoding="utf-8")

        files = iter_audio_files(self.root)

        self.assertEqual([path.name for path in files], ["a.mp3", "b.flac", "c.M4A"])
        self.assertIn(".flac", SUPPORTED_AUDIO_EXTENSIONS)
        self.assertIn(".wav", SUPPORTED_AUDIO_EXTENSIONS)

    @patch("nyxcore.core.scanner._has_cover_art", side_effect=lambda path: path.suffix.lower() == ".flac")
    @patch("nyxcore.core.scanner.MutagenFile")
    def test_scanner_traverses_mixed_extensions(self, mutagen_file: MagicMock, _cover_art: MagicMock) -> None:
        for name in ("one.mp3", "two.flac", "three.wav", "skip.txt"):
            path = self.root / name
            path.write_text("", encoding="utf-8")

        def fake_mutagen(path: Path, easy: bool = True):
            if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                return None
            audio = MagicMock()
            audio.tags = {
                "title": [f"title-{path.suffix.lower()}"],
                "artist": ["artist"],
                "album": ["album"],
            }
            info = MagicMock()
            info.length = 123.456
            info.bitrate = 192000
            audio.info = info
            return audio

        mutagen_file.side_effect = fake_mutagen

        records, stats = scan_music_folder(self.root)

        self.assertEqual([Path(record.path).name for record in records], ["one.mp3", "three.wav", "two.flac"])
        self.assertEqual(stats["total_tracks"], 3)
        self.assertEqual(stats["cover_art_present"], 1)

    @patch("nyxcore.normalize.parser.MutagenFile")
    def test_normalize_preview_traverses_mixed_extensions(self, mutagen_file: MagicMock) -> None:
        for name in ("Alpha - Song.mp3", "Beta - Tune.ogg", "Gamma.aiff"):
            (self.root / name).write_text("", encoding="utf-8")

        def fake_mutagen(path: Path, easy: bool = True):
            audio = MagicMock()
            audio.tags = {
                "title": [],
                "artist": [],
                "album": [],
            }
            return audio

        mutagen_file.side_effect = fake_mutagen

        records = build_normalize_preview(self.root)

        self.assertEqual(len(records), 3)
        self.assertEqual([Path(record.path).suffix.lower() for record in records], [".mp3", ".ogg", ".aiff"])

    @patch("nyxcore.tagging.writer.MutagenFile")
    def test_generic_tag_writer_supports_non_mp3_writable_case(self, mutagen_file: MagicMock) -> None:
        target = self.root / "track.flac"
        target.write_text("", encoding="utf-8")
        class FakeAudio(dict):
            def __init__(self) -> None:
                super().__init__()
                self.tags = {}
                self.saved = False

            def save(self) -> None:
                self.saved = True

        audio = FakeAudio()
        mutagen_file.return_value = audio

        write_basic_tags(target, title="Song", artist="Artist", album="Album")

        self.assertEqual(audio["title"], ["Song"])
        self.assertEqual(audio["artist"], ["Artist"])
        self.assertEqual(audio["album"], ["Album"])
        self.assertTrue(audio.saved)

    @patch("nyxcore.tagging.writer.MutagenFile")
    def test_generic_tag_writer_fails_clearly_for_unwritable_case(self, mutagen_file: MagicMock) -> None:
        target = self.root / "track.wav"
        target.write_text("", encoding="utf-8")
        audio = MagicMock(spec=["tags"])
        audio.tags = None
        mutagen_file.return_value = audio

        with self.assertRaises(TagWriteError) as ctx:
            write_basic_tags(target, title="Song", artist="Artist", album="Album")

        self.assertIn("safe generic tag writing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
