from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.config import load_config
from nyxcore.core.track import TrackRecord
from nyxcore.incremental.service import ChangeSet, RefreshSummary
from nyxcore.saved_playlists.service import (
    create_saved_playlist_definition,
    delete_saved_playlist_definition,
    edit_saved_playlist_definition,
    export_saved_playlist_json,
    export_saved_playlist_m3u,
    load_saved_playlist_store,
    rename_saved_playlist_definition,
    read_saved_playlist_latest_result,
    refresh_saved_playlist,
    save_saved_playlist_definition,
)
from nyxcore.playlist_query.service import build_playlist_report
from nyxcore.cli import app

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_saved_playlists"


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


class SavedPlaylistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.out = self.root / "out"
        self.music.mkdir()
        self.out.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_deterministic_saved_playlist_creation(self) -> None:
        first = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        second = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )

        self.assertEqual(first.playlist_id, second.playlist_id)

    def test_listing_and_reading_saved_definitions(self) -> None:
        store_root = self.out / "saved_playlists"
        store = load_saved_playlist_store(store_root)
        definition = create_saved_playlist_definition(
            name="Night Drive",
            query="late night chill",
            profile="default",
            max_tracks=None,
            min_score=None,
        )
        store.playlists[definition.playlist_id] = definition
        save_saved_playlist_definition(store_root, store)

        loaded = load_saved_playlist_store(store_root)

        self.assertIn(definition.playlist_id, loaded.playlists)
        self.assertEqual(loaded.playlists[definition.playlist_id].name, "Night Drive")

    def test_rename_preserves_playlist_id(self) -> None:
        definition = create_saved_playlist_definition(
            name="Night Drive",
            query="late night chill",
            profile="default",
            max_tracks=None,
            min_score=None,
        )
        original_id = definition.playlist_id

        rename_saved_playlist_definition(definition, name="After Hours")

        self.assertEqual(definition.playlist_id, original_id)
        self.assertEqual(definition.name, "After Hours")

    def test_edit_updates_query_and_constraints_without_changing_id(self) -> None:
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.2,
        )
        original_id = definition.playlist_id

        edit_saved_playlist_definition(
            definition,
            query="dark focus music",
            max_tracks=25,
            min_score=0.35,
            profile="dj",
        )

        self.assertEqual(definition.playlist_id, original_id)
        self.assertEqual(definition.query, "dark focus music")
        self.assertEqual(definition.max_tracks, 25)
        self.assertEqual(definition.min_score, 0.35)
        self.assertEqual(definition.profile, "dj")

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_refresh_produces_latest_result(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        target = self.music / "focus.mp3"
        target.write_bytes(b"a")
        records = [_track(target, title="Focus Ambient", artist="Artist", album="Album", duration=180.0)]
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=records,
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(target)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )

        self.assertEqual(latest.summary["track_count"], 1)
        self.assertIsNotNone(read_saved_playlist_latest_result(store_root, definition.playlist_id))
        self.assertFalse(latest.refresh_diff["has_previous_result"])

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_saved_profile_context_is_reused(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        target = self.music / "plain.mp3"
        target.write_bytes(b"a")
        records = [_track(target, title="Plain Song", artist="Artist", album="Album", duration=180.0)]
        definition = create_saved_playlist_definition(
            name="DJ Set",
            query="focus music",
            profile="dj",
            max_tracks=None,
            min_score=None,
        )
        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=records,
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(target)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(profile="dj"),
            analysis_cache_path=Path("fake"),
        )

        self.assertEqual(latest.active_profile, "dj")
        self.assertEqual(latest.summary["track_count"], 0)

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_refresh_diff_shows_added_and_removed_tracks(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        first = self.music / "focus-a.mp3"
        second = self.music / "focus-b.mp3"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        refresh_saved_playlist(
            store_root,
            definition,
            records=[_track(first, title="Focus A", artist="Artist", album="Album", duration=180.0)],
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(first)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )

        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=[_track(second, title="Focus B", artist="Artist", album="Album", duration=120.0)],
            refresh_summary=RefreshSummary(
                mode="incremental",
                changes=ChangeSet(
                    added_files=[str(second)],
                    modified_files=[],
                    removed_files=[str(first)],
                    unchanged_files=[],
                ),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )

        self.assertTrue(latest.refresh_diff["has_previous_result"])
        self.assertEqual(latest.refresh_diff["tracks_added"], [str(second)])
        self.assertEqual(latest.refresh_diff["tracks_removed"], [str(first)])
        self.assertEqual(latest.refresh_diff["track_count_delta"], 0)
        self.assertEqual(latest.refresh_diff["estimated_duration_delta_seconds"], -60.0)

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_refresh_diff_stays_stable_when_library_root_moves(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        first_root = self.root / "library-a"
        second_root = self.root / "library-b"
        relative = Path("focus") / "moved.mp3"
        (first_root / relative.parent).mkdir(parents=True, exist_ok=True)
        (second_root / relative.parent).mkdir(parents=True, exist_ok=True)
        first = first_root / relative
        second = second_root / relative
        first.write_bytes(b"a")
        shutil.copy2(first, second)
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        refresh_saved_playlist(
            store_root,
            definition,
            records=[_track(first, title="Focus Moved", artist="Artist", album="Album", duration=180.0)],
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(first)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
            library_root=first_root,
        )

        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=[_track(second, title="Focus Moved", artist="Artist", album="Album", duration=180.0)],
            refresh_summary=RefreshSummary(
                mode="incremental",
                changes=ChangeSet(added_files=[], modified_files=[], removed_files=[], unchanged_files=[str(second)]),
                rescanned_files=0,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
            library_root=second_root,
        )

        self.assertTrue(latest.refresh_diff["has_previous_result"])
        self.assertEqual(latest.refresh_diff["tracks_added"], [])
        self.assertEqual(latest.refresh_diff["tracks_removed"], [])
        self.assertEqual(latest.refresh_diff["track_count_delta"], 0)

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_optional_export_behavior(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        target = self.music / "focus.mp3"
        target.write_bytes(b"a")
        records = [_track(target, title="Focus Ambient", artist="Artist", album="Album", duration=180.0)]
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=records,
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(target)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )

        m3u_path = export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
        json_path = export_saved_playlist_json(store_root, definition.playlist_id, latest)

        self.assertTrue(m3u_path.exists())
        self.assertTrue(json_path.exists())

    def test_delete_removes_definition_and_latest_result(self) -> None:
        store_root = self.out / "saved_playlists"
        store = load_saved_playlist_store(store_root)
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=None,
            min_score=None,
        )
        store.playlists[definition.playlist_id] = definition
        save_saved_playlist_definition(store_root, store)
        playlist_dir = store_root / "playlists" / definition.playlist_id
        playlist_dir.mkdir(parents=True, exist_ok=True)
        (playlist_dir / "latest_result.json").write_text("{}", encoding="utf-8")

        deleted = delete_saved_playlist_definition(store_root, store, definition.playlist_id)
        save_saved_playlist_definition(store_root, store)

        self.assertIsNotNone(deleted)
        self.assertNotIn(definition.playlist_id, store.playlists)
        self.assertFalse(playlist_dir.exists())

    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_graceful_when_previously_matched_tracks_disappear(self, _load_analysis_map) -> None:
        store_root = self.out / "saved_playlists"
        first = self.music / "focus.mp3"
        first.write_bytes(b"a")
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        refresh_saved_playlist(
            store_root,
            definition,
            records=[_track(first, title="Focus Ambient", artist="Artist", album="Album", duration=180.0)],
            refresh_summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(first)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )
        first.unlink()

        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=[],
            refresh_summary=RefreshSummary(
                mode="incremental",
                changes=ChangeSet(added_files=[], modified_files=[], removed_files=[str(first)], unchanged_files=[]),
                rescanned_files=0,
            ),
            app_config=load_config(),
            analysis_cache_path=Path("fake"),
        )

        self.assertEqual(latest.summary["track_count"], 0)
        self.assertEqual(latest.summary["removed_files"], 1)

    @patch("nyxcore.cli._load_library_records")
    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_cli_save_list_refresh_show_smoke(self, _load_analysis_map, load_library_records) -> None:
        target = self.music / "focus.mp3"
        target.write_bytes(b"a")
        load_library_records.return_value = (
            [_track(target, title="Focus Ambient", artist="Artist", album="Album", duration=180.0)],
            RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=[str(target)], modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=1,
            ),
        )
        runner = CliRunner()

        save_result = runner.invoke(
            app,
            ["save-playlist", str(self.music), "--out", str(self.out), "--name", "Focus Set", "--query", "focus music"],
        )
        self.assertEqual(save_result.exit_code, 0, msg=save_result.stdout)

        list_result = runner.invoke(app, ["list-playlists", "--out", str(self.out)])
        self.assertEqual(list_result.exit_code, 0, msg=list_result.stdout)
        store = load_saved_playlist_store(self.out / "saved_playlists")
        playlist_id = next(iter(store.playlists))

        refresh_result = runner.invoke(app, ["refresh-playlist", str(self.music), playlist_id, "--out", str(self.out), "--export-m3u"])
        self.assertEqual(refresh_result.exit_code, 0, msg=refresh_result.stdout)

        show_result = runner.invoke(app, ["show-playlist", playlist_id, "--out", str(self.out)])
        self.assertEqual(show_result.exit_code, 0, msg=show_result.stdout)
        self.assertTrue((self.out / "saved_playlists" / "playlists" / playlist_id / "latest_result.json").exists())
        self.assertTrue((self.out / "saved_playlists" / "playlists" / playlist_id / "latest.m3u").exists())

    @patch("nyxcore.cli._load_library_records")
    @patch("nyxcore.playlist_query.service._load_analysis_map", return_value={})
    def test_cli_edit_delete_show_list_smoke(self, _load_analysis_map, load_library_records) -> None:
        first = self.music / "focus-a.mp3"
        second = self.music / "focus-b.mp3"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        load_library_records.side_effect = [
            (
                [_track(first, title="Focus A", artist="Artist", album="Album", duration=180.0)],
                RefreshSummary(
                    mode="full",
                    changes=ChangeSet(added_files=[str(first)], modified_files=[], removed_files=[], unchanged_files=[]),
                    rescanned_files=1,
                ),
            ),
            (
                [_track(second, title="Focus B", artist="Artist", album="Album", duration=120.0)],
                RefreshSummary(
                    mode="incremental",
                    changes=ChangeSet(
                        added_files=[str(second)],
                        modified_files=[],
                        removed_files=[str(first)],
                        unchanged_files=[],
                    ),
                    rescanned_files=1,
                ),
            ),
        ]
        runner = CliRunner()

        save_result = runner.invoke(
            app,
            ["save-playlist", str(self.music), "--out", str(self.out), "--name", "Focus Set", "--query", "focus music"],
        )
        self.assertEqual(save_result.exit_code, 0, msg=save_result.stdout)

        store = load_saved_playlist_store(self.out / "saved_playlists")
        playlist_id = next(iter(store.playlists))

        rename_result = runner.invoke(
            app,
            ["rename-playlist", playlist_id, "--out", str(self.out), "--name", "Deep Focus"],
        )
        self.assertEqual(rename_result.exit_code, 0, msg=rename_result.stdout)

        edit_result = runner.invoke(
            app,
            [
                "edit-playlist",
                playlist_id,
                "--out",
                str(self.out),
                "--query",
                "dark focus music",
                "--max-tracks",
                "5",
            ],
        )
        self.assertEqual(edit_result.exit_code, 0, msg=edit_result.stdout)

        refresh_result = runner.invoke(app, ["refresh-playlist", str(self.music), playlist_id, "--out", str(self.out)])
        self.assertEqual(refresh_result.exit_code, 0, msg=refresh_result.stdout)
        self.assertIn("Tracks added", refresh_result.stdout)

        list_result = runner.invoke(app, ["list-playlists", "--out", str(self.out)])
        self.assertEqual(list_result.exit_code, 0, msg=list_result.stdout)
        self.assertIn("Deep Focus", list_result.stdout)

        show_result = runner.invoke(app, ["show-playlist", playlist_id, "--out", str(self.out)])
        self.assertEqual(show_result.exit_code, 0, msg=show_result.stdout)
        self.assertIn("Saved Playlist Refresh Diff", show_result.stdout)

        delete_result = runner.invoke(app, ["delete-playlist", playlist_id, "--out", str(self.out), "--yes"])
        self.assertEqual(delete_result.exit_code, 0, msg=delete_result.stdout)
        self.assertFalse((self.out / "saved_playlists" / "playlists" / playlist_id).exists())
        self.assertNotIn(playlist_id, load_saved_playlist_store(self.out / "saved_playlists").playlists)


if __name__ == "__main__":
    unittest.main()
