from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.cli import app
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.duplicates.service import (
    DuplicateAnalysisReport,
    DuplicateSummary,
    DuplicateTrackInfo,
    ExactDuplicateGroup,
    PreferredCopyRecommendation,
)
from nyxcore.health.service import build_health_report

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_health"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _track(
    path: Path,
    *,
    title: str | None,
    artist: str | None,
    album: str | None,
    duration: float | None,
    cover: bool,
    warnings: list[WarningCode] | None = None,
) -> TrackRecord:
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
        warnings=list(warnings or []),
    )


class HealthReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_overview_counts_mixed_formats(self) -> None:
        folder = self.root / "music"
        (folder / "sub").mkdir(parents=True)
        a = folder / "a.mp3"
        b = folder / "sub" / "b.flac"
        a.write_bytes(b"a" * 100)
        b.write_bytes(b"b" * 200)
        report = build_health_report(
            folder,
            [
                _track(a, title="A", artist="Artist", album="Album", duration=100.0, cover=False),
                _track(b, title="B", artist="Artist", album="Album", duration=100.0, cover=True),
            ],
            duplicate_report=self._empty_duplicates(2),
        )

        self.assertEqual(report.overview.total_audio_files, 2)
        self.assertEqual(report.overview.total_folders_touched, 2)
        self.assertEqual(report.overview.format_breakdown, {".flac": 1, ".mp3": 1})

    def test_missing_metadata_detection(self) -> None:
        folder = self.root / "music"
        folder.mkdir()
        a = folder / "Artist - Track 01.mp3"
        a.write_bytes(b"a" * 100)
        report = build_health_report(
            folder,
            [
                _track(
                    a,
                    title=None,
                    artist="UNKNOWN",
                    album=None,
                    duration=100.0,
                    cover=False,
                    warnings=[WarningCode.missing_title, WarningCode.missing_album],
                )
            ],
            duplicate_report=self._empty_duplicates(1),
        )

        self.assertEqual(report.metadata.missing_title.count, 1)
        self.assertEqual(report.metadata.missing_album.count, 1)
        self.assertEqual(report.metadata.placeholder_metadata.count, 1)

    def test_artwork_coverage_counting(self) -> None:
        folder = self.root / "music"
        folder.mkdir()
        a = folder / "a.mp3"
        b = folder / "b.flac"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        report = build_health_report(
            folder,
            [
                _track(a, title="A", artist="Artist", album="Album", duration=100.0, cover=True),
                _track(b, title="B", artist="Artist", album="Album", duration=100.0, cover=False),
            ],
            duplicate_report=self._empty_duplicates(2),
        )

        self.assertEqual(report.artwork.with_artwork, 1)
        self.assertEqual(report.artwork.without_artwork, 1)
        self.assertEqual(report.artwork.coverage_percent, 50.0)

    def test_duplicate_summary_integration_and_reclaimable_bytes(self) -> None:
        folder = self.root / "music"
        folder.mkdir()
        a = folder / "a.mp3"
        b = folder / "b.flac"
        a.write_bytes(b"x" * 100)
        b.write_bytes(b"x" * 100)
        report = build_health_report(
            folder,
            [
                _track(a, title="A", artist="Artist", album="Album", duration=100.0, cover=False),
                _track(b, title="A", artist="Artist", album="Album", duration=100.0, cover=True),
            ],
            duplicate_report=DuplicateAnalysisReport(
                summary=DuplicateSummary(
                    total_tracks=2,
                    exact_group_count=1,
                    exact_duplicate_file_count=2,
                    likely_group_count=0,
                    likely_duplicate_file_count=0,
                ),
                exact_duplicates=[
                    ExactDuplicateGroup(
                        group_id="exact-001",
                        content_hash="hash",
                        files=[
                            self._dup_file(a, 100),
                            self._dup_file(b, 100),
                        ],
                        preferred=PreferredCopyRecommendation(path=str(b), reasons=["preferred_lossless_format"]),
                    )
                ],
                likely_duplicates=[],
            ),
        )

        self.assertEqual(report.duplicates.exact_duplicate_groups, 1)
        self.assertEqual(report.duplicates.reclaimable_bytes_exact, 100)

    def test_priority_recommendations_are_deterministic(self) -> None:
        folder = self.root / "music"
        folder.mkdir()
        a = folder / "a.mp3"
        a.write_bytes(b"x" * 100)
        dup_report = self._empty_duplicates(1)
        dup_report.summary.exact_group_count = 2
        report = build_health_report(
            folder,
            [
                _track(
                    a,
                    title="Track 01",
                    artist=None,
                    album="Album",
                    duration=10.0,
                    cover=False,
                    warnings=[WarningCode.missing_artist, WarningCode.low_bitrate],
                )
            ],
            duplicate_report=dup_report,
        )

        self.assertEqual(report.priorities.recommended_actions[0], "Review exact duplicates first")
        self.assertEqual(report.priorities.top_issue_categories[0].category, "exact_duplicates")

    @patch("nyxcore.cli.analyze_duplicates")
    @patch("nyxcore.cli.scan_music_folder")
    def test_cli_health_smoke_writes_reports(self, scan_music_folder, analyze_duplicates) -> None:
        music = self.root / "music"
        out = self.root / "out"
        music.mkdir()
        out.mkdir()
        track_path = music / "a.mp3"
        track_path.write_bytes(b"x" * 128)
        record = _track(track_path, title="A", artist="Artist", album="Album", duration=100.0, cover=True)
        scan_music_folder.return_value = ([record], {})
        analyze_duplicates.return_value = self._empty_duplicates(1)

        runner = CliRunner()
        result = runner.invoke(app, ["health", str(music), "--out", str(out)])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertTrue((out / "health.json").exists())
        payload = json.loads((out / "health.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["overview"]["total_audio_files"], 1)

    def _empty_duplicates(self, total_tracks: int) -> DuplicateAnalysisReport:
        return DuplicateAnalysisReport(
            summary=DuplicateSummary(
                total_tracks=total_tracks,
                exact_group_count=0,
                exact_duplicate_file_count=0,
                likely_group_count=0,
                likely_duplicate_file_count=0,
            ),
            exact_duplicates=[],
            likely_duplicates=[],
        )

    def _dup_file(self, path: Path, size: int) -> DuplicateTrackInfo:
        return DuplicateTrackInfo(
            path=str(path),
            file_size_bytes=size,
            extension=path.suffix.lower(),
            duration_seconds=100.0,
            bitrate_bps=None,
            title="A",
            artist="Artist",
            album="Album",
            has_cover_art=path.suffix.lower() == ".flac",
            metadata_fields_present=3,
        )


if __name__ == "__main__":
    unittest.main()
