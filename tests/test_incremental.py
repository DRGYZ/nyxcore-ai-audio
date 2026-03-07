from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.cli import app
from nyxcore.incremental.service import (
    ChangeSet,
    FileSnapshot,
    IncrementalState,
    diff_file_snapshots,
    load_incremental_state,
    refresh_incremental_state,
    save_incremental_state,
    watch_incremental_state,
)

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_incremental"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class IncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.out = self.root / "out"
        self.music.mkdir()
        self.out.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_change_set_detection(self) -> None:
        previous = {
            "a.mp3": FileSnapshot(path="a.mp3", size=10, mtime_ns=1),
            "b.mp3": FileSnapshot(path="b.mp3", size=10, mtime_ns=2),
            "c.mp3": FileSnapshot(path="c.mp3", size=10, mtime_ns=3),
        }
        current = {
            "b.mp3": FileSnapshot(path="b.mp3", size=10, mtime_ns=2),
            "c.mp3": FileSnapshot(path="c.mp3", size=12, mtime_ns=4),
            "d.mp3": FileSnapshot(path="d.mp3", size=9, mtime_ns=5),
        }

        changes = diff_file_snapshots(previous, current)

        self.assertEqual(changes.added_files, ["d.mp3"])
        self.assertEqual(changes.modified_files, ["c.mp3"])
        self.assertEqual(changes.removed_files, ["a.mp3"])
        self.assertEqual(changes.unchanged_files, ["b.mp3"])

    def test_state_persistence_is_deterministic(self) -> None:
        state_path = self.out / "state.json"
        state = IncrementalState(
            root=str(self.music),
            files={
                "b.mp3": FileSnapshot(path="b.mp3", size=2, mtime_ns=2),
                "a.mp3": FileSnapshot(path="a.mp3", size=1, mtime_ns=1),
            },
            records={},
        )

        save_incremental_state(state_path, state)
        first = state_path.read_text(encoding="utf-8")
        loaded = load_incremental_state(state_path)
        save_incremental_state(state_path, loaded)
        second = state_path.read_text(encoding="utf-8")

        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(list(payload["files"].keys()), ["a.mp3", "b.mp3"])

    @patch("nyxcore.core.scanner._has_cover_art", return_value=False)
    @patch("nyxcore.core.scanner.MutagenFile", return_value=None)
    def test_incremental_refresh_reuses_unchanged_records(self, _mutagen, _cover_art) -> None:
        state_path = self.out / "state.json"
        keep = self.music / "keep.mp3"
        added = self.music / "added.flac"
        keep.write_bytes(b"keep-v1")

        first = refresh_incremental_state(self.music, state_path)
        self.assertEqual(first.summary.mode, "full")
        self.assertEqual(first.summary.rescanned_files, 1)

        keep.write_bytes(b"keep-v2-longer")
        added.write_bytes(b"added")

        second = refresh_incremental_state(self.music, state_path)

        self.assertEqual(second.summary.mode, "incremental")
        self.assertEqual(second.summary.rescanned_files, 2)
        self.assertEqual(second.summary.changes.added_files, [str(added)])
        self.assertEqual(second.summary.changes.modified_files, [str(keep)])
        self.assertEqual(second.summary.changes.unchanged_files, [])
        self.assertEqual(sorted(record.path for record in second.records), sorted([str(added), str(keep)]))

    @patch("nyxcore.core.scanner._has_cover_art", return_value=False)
    @patch("nyxcore.core.scanner.MutagenFile", return_value=None)
    def test_safe_when_file_disappears_during_processing(self, _mutagen, _cover_art) -> None:
        state_path = self.out / "state.json"
        target = self.music / "vanish.mp3"
        target.write_bytes(b"initial")
        refresh_incremental_state(self.music, state_path)
        target.write_bytes(b"changed-longer")

        def fake_scan(paths):
            for path in paths:
                Path(path).unlink(missing_ok=True)
            return []

        with patch("nyxcore.incremental.service.scan_audio_files", side_effect=fake_scan):
            refreshed = refresh_incremental_state(self.music, state_path)

        self.assertEqual(refreshed.records, [])
        self.assertEqual(refreshed.state.files, {})

    @patch("nyxcore.core.scanner._has_cover_art", return_value=False)
    @patch("nyxcore.core.scanner.MutagenFile", return_value=None)
    def test_watch_polling_logic_is_bounded(self, _mutagen, _cover_art) -> None:
        state_path = self.out / "state.json"
        (self.music / "a.mp3").write_bytes(b"a")
        sleeps: list[float] = []

        results = watch_incremental_state(
            self.music,
            state_path,
            interval_seconds=0.5,
            max_cycles=2,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(sleeps, [0.5])

    @patch("nyxcore.health.service.MutagenFile", return_value=None)
    @patch("nyxcore.core.scanner._has_cover_art", return_value=False)
    @patch("nyxcore.core.scanner.MutagenFile", return_value=None)
    def test_cli_watch_once_incremental_smoke(self, _mutagen_scan, _cover_art, _mutagen_health) -> None:
        runner = CliRunner()
        state_path = self.out / "state.json"
        first = self.music / "first.mp3"
        first.write_bytes(b"one")

        baseline = runner.invoke(
            app,
            ["watch", str(self.music), "--out", str(self.out), "--state", str(state_path), "--once"],
        )
        self.assertEqual(baseline.exit_code, 0, msg=baseline.stdout)
        self.assertTrue((self.out / "health.json").exists())
        self.assertTrue((self.out / "review.json").exists())

        second = self.music / "second.flac"
        second.write_bytes(b"two")
        first.write_bytes(b"one-modified")

        refreshed = runner.invoke(
            app,
            ["watch", str(self.music), "--out", str(self.out), "--state", str(state_path), "--once"],
        )
        self.assertEqual(refreshed.exit_code, 0, msg=refreshed.stdout)

        payload = json.loads((self.out / "health.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["refresh"]["mode"], "incremental")
        self.assertEqual(payload["refresh"]["changes"]["added_files"], [str(second)])
        self.assertEqual(payload["refresh"]["changes"]["modified_files"], [str(first)])


if __name__ == "__main__":
    unittest.main()
