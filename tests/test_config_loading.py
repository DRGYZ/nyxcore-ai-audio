from __future__ import annotations

import os
import unittest
from pathlib import Path

from nyxcore.playlist_query.service import build_playlist_report
from nyxcore.config import load_config
from nyxcore.core.track import TrackRecord

TEST_ROOT = Path(__file__).resolve().parent


class ConfigLoadingTests(unittest.TestCase):
    def test_packaged_default_config_loads(self) -> None:
        config = load_config()
        self.assertEqual(config.profile, "default")
        self.assertEqual(config.judge.prompt_version, "judge_v1_heuristics")
        self.assertGreater(len(config.judge.moods), 0)
        self.assertEqual(config.review.sample_limit, 5)

    def test_packaged_default_ignores_current_working_directory(self) -> None:
        original_cwd = Path.cwd()
        os.chdir(TEST_ROOT)
        try:
            config = load_config()
        finally:
            os.chdir(original_cwd)
        self.assertEqual(config.judge.concurrency_default, 10)

    def test_custom_config_path_overrides_packaged_default(self) -> None:
        original_cwd = Path.cwd()
        config_path = TEST_ROOT / "_custom_config.yaml"
        try:
            config_path.write_text(
                "\n".join(
                    [
                        "judge:",
                        '  prompt_version: "custom_v1"',
                        "  prompts:",
                        '    system: "custom system"',
                        "    user_rules: []",
                        '  moods: ["calm"]',
                        '  genres: ["ambient"]',
                        "  concurrency_default: 3",
                    ]
                ),
                encoding="utf-8",
            )
            os.chdir(TEST_ROOT)
            try:
                config = load_config(config_path)
            finally:
                os.chdir(original_cwd)
        finally:
            config_path.unlink(missing_ok=True)
        self.assertEqual(config.judge.prompt_version, "custom_v1")
        self.assertEqual(config.judge.concurrency_default, 3)

    def test_builtin_profile_loads(self) -> None:
        config = load_config(profile="dj")
        self.assertEqual(config.profile, "dj")
        self.assertEqual(config.watch.default_interval_seconds, 2.0)
        self.assertGreater(config.playlist.bpm_match_weight, 2.0)
        self.assertGreater(config.review.likely_duplicate_base_score, 52.0)

    def test_explicit_override_beats_profile(self) -> None:
        config_path = TEST_ROOT / "_profile_override.yaml"
        try:
            config_path.write_text(
                "\n".join(
                    [
                        'profile: "dj"',
                        "playlist:",
                        "  default_min_score: -1.0",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
        finally:
            config_path.unlink(missing_ok=True)
        self.assertEqual(config.profile, "dj")
        self.assertEqual(config.playlist.default_min_score, -1.0)

    def test_invalid_config_fails_clearly(self) -> None:
        config_path = TEST_ROOT / "_invalid_config.yaml"
        try:
            config_path.write_text(
                "\n".join(
                    [
                        "watch:",
                        "  default_interval_seconds: 0.1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(Exception) as ctx:
                load_config(config_path)
        finally:
            config_path.unlink(missing_ok=True)
        self.assertIn("default_interval_seconds", str(ctx.exception))

    def test_profile_driven_playlist_behavior_differs(self) -> None:
        track_path = TEST_ROOT / "_profile_track.mp3"
        try:
            track_path.write_bytes(b"a")
            record = TrackRecord(
                path=str(track_path),
                file_size_bytes=track_path.stat().st_size,
                mtime_iso="",
                tags={
                    "title": "Plain Song",
                    "artist": "Artist",
                    "album": "Album",
                    "albumartist": None,
                    "tracknumber": None,
                    "date": None,
                    "genre": None,
                },
                has_cover_art=False,
                duration_seconds=180.0,
                warnings=[],
            )
            default_report = build_playlist_report([record], query="plain", settings=load_config().playlist)
            dj_report = build_playlist_report([record], query="plain", settings=load_config(profile="dj").playlist)
        finally:
            track_path.unlink(missing_ok=True)
        self.assertEqual(default_report.summary.track_count, 1)
        self.assertEqual(dj_report.summary.track_count, 0)


if __name__ == "__main__":
    unittest.main()
