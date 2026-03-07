from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.cli import app
from nyxcore.core.track import TrackRecord
from nyxcore.duplicates.service import analyze_duplicates

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_duplicates"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _track(path: Path, *, title: str | None, artist: str | None, album: str | None, duration: float, cover: bool) -> TrackRecord:
    stat = path.stat()
    return TrackRecord(
        path=str(path),
        file_size_bytes=stat.st_size,
        mtime_iso="",
        tags={
            "title": title,
            "artist": artist,
            "album": album,
            "albumartist": None,
            "tracknumber": None,
            "date": None,
            "genre": None,
        },
        has_cover_art=cover,
        duration_seconds=duration,
        warnings=[],
    )


class DuplicateAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_exact_duplicate_grouping_for_same_content_different_names(self) -> None:
        first = self.root / "song-a.mp3"
        second = self.root / "song-b.flac"
        content = b"same-bytes" * 512
        first.write_bytes(content)
        second.write_bytes(content)
        tracks = [
            _track(first, title="Track", artist="Artist", album="Album", duration=100.0, cover=False),
            _track(second, title="Track", artist="Artist", album="Album", duration=100.0, cover=True),
        ]

        with patch("nyxcore.duplicates.service._load_bitrate_bps", side_effect=[192000, None]):
            report = analyze_duplicates(tracks)

        self.assertEqual(report.summary.exact_group_count, 1)
        group = report.exact_duplicates[0]
        self.assertEqual(sorted(item.path for item in group.files), sorted([str(first), str(second)]))
        self.assertEqual(group.preferred.path, str(second))

    def test_same_filename_different_content_is_not_exact_duplicate(self) -> None:
        left_dir = self.root / "left"
        right_dir = self.root / "right"
        left_dir.mkdir()
        right_dir.mkdir()
        first = left_dir / "same.mp3"
        second = right_dir / "same.mp3"
        first.write_bytes(b"aaa" * 512)
        second.write_bytes(b"bbb" * 512)
        tracks = [
            _track(first, title="Track", artist="Artist", album="Album", duration=100.0, cover=False),
            _track(second, title="Track", artist="Artist", album="Album", duration=100.0, cover=False),
        ]

        with patch("nyxcore.duplicates.service._load_bitrate_bps", return_value=192000):
            report = analyze_duplicates(tracks)

        self.assertEqual(report.summary.exact_group_count, 0)

    def test_likely_duplicate_grouping_for_close_metadata_and_duration(self) -> None:
        first = self.root / "Artist - Song.mp3"
        second = self.root / "Artist - Song (Clean).m4a"
        first.write_bytes(b"a" * 1024)
        second.write_bytes(b"b" * 1400)
        tracks = [
            _track(first, title="Song", artist="Artist", album="Singles", duration=201.1, cover=False),
            _track(second, title="Song", artist="Artist", album="Singles", duration=202.0, cover=True),
        ]

        def fake_bitrate(path: Path) -> int:
            return 256000 if path.suffix.lower() == ".m4a" else 192000

        with patch("nyxcore.duplicates.service._load_bitrate_bps", side_effect=fake_bitrate):
            report = analyze_duplicates(tracks)

        self.assertEqual(report.summary.likely_group_count, 1)
        group = report.likely_duplicates[0]
        self.assertGreaterEqual(group.confidence, 0.78)
        self.assertIn("title_match_strong", group.reasons)
        self.assertEqual(group.preferred.path, str(second))

    def test_preferred_copy_prefers_higher_quality_and_richer_metadata(self) -> None:
        first = self.root / "track.mp3"
        second = self.root / "track.flac"
        content = b"same-bytes" * 1024
        first.write_bytes(content)
        second.write_bytes(content)
        sparse = _track(first, title="Track", artist="Artist", album=None, duration=180.0, cover=False)
        rich = _track(second, title="Track", artist="Artist", album="Album", duration=180.0, cover=True)
        rich.tags["genre"] = "electronic"

        with patch("nyxcore.duplicates.service._load_bitrate_bps", side_effect=[192000, None]):
            report = analyze_duplicates([sparse, rich])

        preferred = report.exact_duplicates[0].preferred
        self.assertEqual(preferred.path, str(second))
        self.assertIn("preferred_lossless_format", preferred.reasons)

    def test_report_serialization_structure_is_stable(self) -> None:
        first = self.root / "a.mp3"
        second = self.root / "b.mp3"
        content = b"dup" * 1024
        first.write_bytes(content)
        second.write_bytes(content)
        tracks = [
            _track(first, title="Track", artist="Artist", album="Album", duration=10.0, cover=False),
            _track(second, title="Track", artist="Artist", album="Album", duration=10.0, cover=False),
        ]

        with patch("nyxcore.duplicates.service._load_bitrate_bps", return_value=192000):
            report = analyze_duplicates(tracks)

        payload = report.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("exact_duplicates", payload)
        self.assertIn("likely_duplicates", payload)
        self.assertIn("summary", payload)
        self.assertIn("preferred", encoded)

    def test_cli_duplicates_smoke_writes_reviewable_report(self) -> None:
        music = self.root / "music"
        out = self.root / "out"
        music.mkdir()
        out.mkdir()
        (music / "dup-a.mp3").write_bytes(b"dup" * 512)
        (music / "dup-b.mp3").write_bytes(b"dup" * 512)

        runner = CliRunner()
        result = runner.invoke(app, ["duplicates", str(music), "--out", str(out)])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertTrue((out / "duplicates.json").exists())
        payload = json.loads((out / "duplicates.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["exact_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
