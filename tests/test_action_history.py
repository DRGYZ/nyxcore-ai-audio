from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.action_plan.ledger import (
    append_operation_batch,
    load_operation_ledger,
    save_operation_ledger,
    undo_operation_batch,
)
from nyxcore.action_plan.service import apply_action_plan_report, build_action_plan_report
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
from nyxcore.review_queue.service import build_review_queue
from nyxcore.review_queue.state import ReviewStateStore

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_action_history"


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


def _dup_info(path: Path, *, size: int, cover: bool = False) -> DuplicateTrackInfo:
    return DuplicateTrackInfo(
        path=str(path),
        file_size_bytes=size,
        extension=path.suffix.lower(),
        duration_seconds=180.0,
        bitrate_bps=192000,
        title="Track",
        artist="Artist",
        album="Album",
        has_cover_art=cover,
        metadata_fields_present=3,
    )


class ActionHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.out = self.root / "out"
        self.music.mkdir()
        self.out.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _duplicate_fixture(self) -> tuple[list[TrackRecord], DuplicateAnalysisReport, object]:
        left = self.music / "dup-a.mp3"
        right = self.music / "dup-b.flac"
        left.write_bytes(b"dup" * 256)
        right.write_bytes(b"dup" * 256)
        records = [
            _track(left, title="Track", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(right, title="Track", artist="Artist", album="Album", duration=180.0, cover=True),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 1, 2, 0, 0),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-001",
                    content_hash="hash",
                    files=[_dup_info(left, size=left.stat().st_size), _dup_info(right, size=right.stat().st_size, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(right), reasons=["preferred_lossless_format"]),
                )
            ],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        exact_item = next(item for item in review_report.items if item.item_type == "exact_duplicate_group")
        return records, duplicate_report, exact_item

    def test_operation_ledger_persists_after_apply(self) -> None:
        records, _duplicate_report, exact_item = self._duplicate_fixture()
        review_report = build_review_queue(
            records,
            health_report=build_health_report(self.music, records, duplicate_report=DuplicateAnalysisReport(DuplicateSummary(2, 1, 2, 0, 0), [], [])),
            duplicate_report=DuplicateAnalysisReport(DuplicateSummary(2, 1, 2, 0, 0), [], []),
        )
        # rebuild with the actual duplicate info for the selected exact item
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 1, 2, 0, 0),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-001",
                    content_hash="hash",
                    files=[_dup_info(self.music / "dup-a.mp3", size=(self.music / "dup-a.mp3").stat().st_size), _dup_info(self.music / "dup-b.flac", size=(self.music / "dup-b.flac").stat().st_size, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(self.music / "dup-b.flac"), reasons=["preferred_lossless_format"]),
                )
            ],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        review_state = ReviewStateStore()
        results = apply_action_plan_report(plan_report, review_state=review_state)
        ledger_path = self.out / "review_history.json"
        ledger = load_operation_ledger(ledger_path)
        batch = append_operation_batch(ledger, plan_report=plan_report, results=results)
        save_operation_ledger(ledger_path, ledger)

        loaded = load_operation_ledger(ledger_path)
        self.assertEqual(len(loaded.batches), 1)
        self.assertEqual(loaded.batches[0].batch_id, batch.batch_id)

    def test_restore_quarantined_duplicate_to_original_location(self) -> None:
        records, duplicate_report, exact_item = self._duplicate_fixture()
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        review_state = ReviewStateStore()
        results = apply_action_plan_report(plan_report, review_state=review_state)
        ledger = load_operation_ledger(self.out / "review_history.json")
        batch = append_operation_batch(ledger, plan_report=plan_report, results=results)

        changed = undo_operation_batch(batch, review_state=review_state)

        self.assertTrue(any(item.undo_status == "ok" for item in changed if item.operation_type == "quarantine_move"))
        self.assertTrue((self.music / "dup-a.mp3").exists())
        self.assertEqual(review_state.items[exact_item.item_id].status, "seen")

    def test_restore_conflict_when_original_path_is_occupied_is_safe(self) -> None:
        records, duplicate_report, exact_item = self._duplicate_fixture()
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        review_state = ReviewStateStore()
        results = apply_action_plan_report(plan_report, review_state=review_state)
        ledger = load_operation_ledger(self.out / "review_history.json")
        batch = append_operation_batch(ledger, plan_report=plan_report, results=results)
        (self.music / "dup-a.mp3").write_bytes(b"occupied")

        changed = undo_operation_batch(batch, review_state=review_state)

        conflict = next(item for item in changed if item.operation_type == "quarantine_move")
        self.assertEqual(conflict.undo_status, "error")
        self.assertIn("restore destination already exists", conflict.undo_message or "")

    def test_undo_reversible_rename_operation(self) -> None:
        target = self.music / "Artist - Song [Official Video].mp3"
        target.write_bytes(b"x" * 100)
        records = [
            _track(
                target,
                title=None,
                artist=None,
                album=None,
                duration=180.0,
                cover=False,
                warnings=[WarningCode.missing_title, WarningCode.missing_artist, WarningCode.missing_album],
            )
        ]
        duplicate_report = DuplicateAnalysisReport(DuplicateSummary(1, 0, 0, 0, 0), [], [])
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        metadata_item = next(item for item in review_report.items if item.item_type == "missing_metadata")
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[metadata_item.item_id])
        rename_only = plan_report.__class__(
            created_at=plan_report.created_at,
            source_review_item_ids=plan_report.source_review_item_ids,
            plans=[plan for plan in plan_report.plans if plan.action_type == "rename_normalize_plan"],
            unsupported_items=[],
            summary=plan_report.summary,
        )
        review_state = ReviewStateStore()
        results = apply_action_plan_report(rename_only, review_state=review_state)
        ledger = load_operation_ledger(self.out / "review_history.json")
        batch = append_operation_batch(ledger, plan_report=rename_only, results=results)

        undo_operation_batch(batch, review_state=review_state)

        self.assertTrue(target.exists())

    @patch("nyxcore.action_plan.service.write_tags")
    def test_non_undoable_metadata_without_backup_is_honest(self, write_tags_mock) -> None:
        target = self.music / "Artist - Song.mp3"
        target.write_bytes(b"x" * 100)
        records = [
            _track(
                target,
                title=None,
                artist=None,
                album=None,
                duration=180.0,
                cover=False,
                warnings=[WarningCode.missing_title, WarningCode.missing_artist, WarningCode.missing_album],
            )
        ]
        duplicate_report = DuplicateAnalysisReport(DuplicateSummary(1, 0, 0, 0, 0), [], [])
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        metadata_item = next(item for item in review_report.items if item.item_type == "missing_metadata")
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[metadata_item.item_id])
        metadata_only = plan_report.__class__(
            created_at=plan_report.created_at,
            source_review_item_ids=plan_report.source_review_item_ids,
            plans=[plan for plan in plan_report.plans if plan.action_type == "metadata_fix_plan"],
            unsupported_items=[],
            summary=plan_report.summary,
        )
        review_state = ReviewStateStore()
        results = apply_action_plan_report(metadata_only, review_state=review_state)
        ledger = load_operation_ledger(self.out / "review_history.json")
        batch = append_operation_batch(ledger, plan_report=metadata_only, results=results)

        undo_operation_batch(batch, review_state=review_state)

        operation = next(item for item in batch.operations if item.operation_type == "write_metadata")
        self.assertEqual(operation.undo_status, "not_supported")

    def test_history_listing_and_detail_cli(self) -> None:
        records, duplicate_report, exact_item = self._duplicate_fixture()
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        review_state = ReviewStateStore()
        results = apply_action_plan_report(plan_report, review_state=review_state)
        ledger = load_operation_ledger(self.out / "review_history.json")
        batch = append_operation_batch(ledger, plan_report=plan_report, results=results)
        save_operation_ledger(self.out / "review_history.json", ledger)
        runner = CliRunner()

        listing = runner.invoke(app, ["history", "--out", str(self.out)])
        detail = runner.invoke(app, ["show-history", batch.batch_id, "--out", str(self.out)])

        self.assertEqual(listing.exit_code, 0, msg=listing.stdout)
        self.assertEqual(detail.exit_code, 0, msg=detail.stdout)
        self.assertIn(batch.batch_id, listing.stdout)
        self.assertIn(batch.batch_id, detail.stdout)

    def test_cli_apply_history_restore_smoke(self) -> None:
        first = self.music / "dup-a.mp3"
        second = self.music / "dup-b.flac"
        content = b"dup" * 512
        first.write_bytes(content)
        second.write_bytes(content)
        runner = CliRunner()

        review_result = runner.invoke(app, ["review", str(self.music), "--out", str(self.out)])
        self.assertEqual(review_result.exit_code, 0, msg=review_result.stdout)
        review_payload = json.loads((self.out / "review.json").read_text(encoding="utf-8"))
        exact_item_id = next(item["item_id"] for item in review_payload["items"] if item["item_type"] == "exact_duplicate_group")

        plan_result = runner.invoke(app, ["review-plan", str(self.music), "--out", str(self.out), "--item-id", exact_item_id])
        self.assertEqual(plan_result.exit_code, 0, msg=plan_result.stdout)
        apply_result = runner.invoke(app, ["apply-review-plan", str(self.out / "review_plan.json"), "--out", str(self.out)])
        self.assertEqual(apply_result.exit_code, 0, msg=apply_result.stdout)

        history_payload = json.loads((self.out / "review_history.json").read_text(encoding="utf-8"))
        batch_id = history_payload["batches"][0]["batch_id"]
        history_result = runner.invoke(app, ["history", "--out", str(self.out)])
        self.assertEqual(history_result.exit_code, 0, msg=history_result.stdout)
        restore_result = runner.invoke(app, ["restore-review-action", batch_id, "--out", str(self.out)])
        self.assertEqual(restore_result.exit_code, 0, msg=restore_result.stdout)

        state_payload = json.loads((self.out / "review_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state_payload["items"][exact_item_id]["status"], "seen")


if __name__ == "__main__":
    unittest.main()
