from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.cli import app
from nyxcore.core.track import TrackRecord
from nyxcore.incremental.service import ChangeSet, RefreshSummary
from nyxcore.playlist_query.service import build_playlist_report, parse_playlist_query

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_playlist_query"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _track(path: Path, *, title: str | None, artist: str | None, album: str | None, duration: float | None) -> TrackRecord:
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
        has_cover_art=False,
        duration_seconds=duration,
        warnings=[],
    )


class PlaylistQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_parse_common_constraints(self) -> None:
        parsed = parse_playlist_query("high energy gym tracks under 4 minutes no vocals around 128 bpm")

        self.assertEqual(parsed.max_duration_seconds, 240.0)
        self.assertEqual(parsed.instrumental_preference, "prefer_instrumental")
        self.assertEqual(parsed.bpm_min, 123.0)
        self.assertEqual(parsed.bpm_max, 133.0)
        self.assertIn("gym", parsed.moods)

    @patch("nyxcore.playlist_query.service._load_analysis_map")
    def test_deterministic_scoring_and_ranking(self, load_analysis_map) -> None:
        first = self.root / "gym-dark.mp3"
        second = self.root / "calm-focus.mp3"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        records = [
            _track(first, title="Dark Run", artist="Artist", album="Album", duration=200.0),
            _track(second, title="Soft Focus", artist="Artist", album="Album", duration=220.0),
        ]
        load_analysis_map.return_value = {
            str(first): {"tags": ["dark", "energetic"], "genre_top": "electronic", "bpm": 130.0, "energy_0_10": 8.5},
            str(second): {"tags": ["ambient", "focus"], "genre_top": "ambient", "bpm": 90.0, "energy_0_10": 2.0},
        }

        report = build_playlist_report(
            records,
            query="dark cinematic around 120 to 140 bpm",
            analysis_cache_path=Path("fake"),
            min_score=-10.0,
        )

        self.assertEqual(report.ranked_tracks[0].path, str(first))
        self.assertGreater(report.ranked_tracks[0].score, report.ranked_tracks[1].score)

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_graceful_degradation_when_fields_unavailable(self, _load_analysis_map) -> None:
        track_path = self.root / "track.mp3"
        track_path.write_bytes(b"a")
        records = [_track(track_path, title="Track", artist="Artist", album="Album", duration=180.0)]

        report = build_playlist_report(records, query="dark cinematic around 120 to 140 bpm", analysis_cache_path=Path("fake"))

        self.assertIn("bpm_requested_but_not_available_for_current_library_subset", report.unsupported_request_aspects)

    @patch("nyxcore.playlist_query.service._load_analysis_map")
    def test_negative_constraint_handling(self, load_analysis_map) -> None:
        vocal = self.root / "vocal.mp3"
        instrumental = self.root / "instrumental.mp3"
        vocal.write_bytes(b"a")
        instrumental.write_bytes(b"b")
        records = [
            _track(vocal, title="Feat Singer", artist="Artist", album="Album", duration=180.0),
            _track(instrumental, title="Instrumental Focus", artist="Artist", album="Album", duration=180.0),
        ]
        load_analysis_map.return_value = {
            str(vocal): {"tags": ["dark"], "genre_top": "electronic", "bpm": 120.0, "energy_0_10": 4.0},
            str(instrumental): {"tags": ["dark", "instrumental"], "genre_top": "ambient", "bpm": 118.0, "energy_0_10": 3.5},
        }

        report = build_playlist_report(records, query="late night chill with no vocals", analysis_cache_path=Path("fake"))

        self.assertEqual(report.ranked_tracks[0].path, str(instrumental))

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_output_serialization(self, _load_analysis_map) -> None:
        track_path = self.root / "track.mp3"
        track_path.write_bytes(b"a")
        records = [_track(track_path, title="Track", artist="Artist", album="Album", duration=180.0)]

        report = build_playlist_report(records, query="focus music")
        payload = report.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertIn("parsed_query", payload)
        self.assertIn("ranked_tracks", payload)
        self.assertIn("summary", payload)
        self.assertIn("focus", encoded)

    @patch("nyxcore.cli._load_library_records")
    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_cli_smoke_behavior(self, _load_analysis_map, load_library_records) -> None:
        music = self.root / "music"
        out = self.root / "out"
        music.mkdir()
        out.mkdir()
        track_path = music / "focus.mp3"
        track_path.write_bytes(b"a")
        load_library_records.return_value = (
            [_track(track_path, title="Focus Ambient", artist="Artist", album="Album", duration=180.0)],
            RefreshSummary(
                mode="full",
                changes=ChangeSet(
                    added_files=[str(track_path)],
                    modified_files=[],
                    removed_files=[],
                    unchanged_files=[],
                ),
                rescanned_files=1,
            ),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["playlist", str(music), "--query", "focus music", "--out", str(out)])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertTrue((out / "playlist_query.json").exists())
        payload = json.loads((out / "playlist_query.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["track_count"], 1)

    @patch("nyxcore.cli._load_library_records")
    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_cli_profile_selection_affects_output(self, _load_analysis_map, load_library_records) -> None:
        music = self.root / "music_profile"
        out = self.root / "out_profile"
        music.mkdir()
        out.mkdir()
        track_path = music / "plain.mp3"
        track_path.write_bytes(b"a")
        load_library_records.return_value = (
            [_track(track_path, title="Plain Song", artist="Artist", album="Album", duration=180.0)],
            RefreshSummary(
                mode="full",
                changes=ChangeSet(
                    added_files=[str(track_path)],
                    modified_files=[],
                    removed_files=[],
                    unchanged_files=[],
                ),
                rescanned_files=1,
            ),
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["playlist", str(music), "--query", "focus music", "--out", str(out), "--profile", "dj"],
        )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        payload = json.loads((out / "playlist_query.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["config"]["active_profile"], "dj")
        self.assertEqual(payload["summary"]["track_count"], 0)


if __name__ == "__main__":
    unittest.main()
