from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.action_plan.service import apply_action_plan_report, build_action_plan_report
from nyxcore.cli import app
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.duplicates.service import (
    DuplicateAnalysisReport,
    DuplicateSummary,
    DuplicateTrackInfo,
    ExactDuplicateGroup,
    LikelyDuplicateGroup,
    LikelyDuplicateRelationship,
    PreferredCopyRecommendation,
)
from nyxcore.health.service import build_health_report
from nyxcore.review_queue.service import build_review_queue
from nyxcore.review_queue.state import ReviewStateStore

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_action_plan"


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


class ActionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.out = self.root / "out"
        self.music.mkdir()
        self.out.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_exact_duplicate_plan_generation_is_deterministic(self) -> None:
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

        first = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        second = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])

        self.assertEqual(first.plans[0].plan_id, second.plans[0].plan_id)
        self.assertEqual(first.plans[0].action_type, "exact_duplicate_keep_plan")
        self.assertTrue(any(".nyxcore_quarantine" in (op.destination_path or "") for op in first.plans[0].proposed_operations))

    def test_metadata_and_rename_plan_generation_is_deterministic(self) -> None:
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
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        metadata_item = next(item for item in review_report.items if item.item_type == "missing_metadata")

        report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[metadata_item.item_id])
        action_types = {plan.action_type for plan in report.plans}

        self.assertIn("metadata_fix_plan", action_types)
        self.assertIn("rename_normalize_plan", action_types)

    def test_unsupported_plan_generation_returns_clear_reason(self) -> None:
        first = self.music / "Artist - Song.mp3"
        second = self.music / "Artist - Song (Alt).m4a"
        first.write_bytes(b"a" * 100)
        second.write_bytes(b"b" * 120)
        records = [
            _track(first, title="Song", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(second, title="Song", artist="Artist", album="Album", duration=181.0, cover=False),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 0, 0, 1, 2),
            exact_duplicates=[],
            likely_duplicates=[
                LikelyDuplicateGroup(
                    group_id="likely-001",
                    confidence=0.82,
                    files=[_dup_info(first, size=first.stat().st_size), _dup_info(second, size=second.stat().st_size)],
                    relationships=[
                        LikelyDuplicateRelationship(
                            paths=sorted([str(first), str(second)]),
                            confidence=0.82,
                            reasons=["title_match_strong"],
                        )
                    ],
                    preferred=PreferredCopyRecommendation(path=str(first), reasons=["highest_bitrate"]),
                    reasons=["title_match_strong"],
                )
            ],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        likely_item = next(item for item in review_report.items if item.item_type == "likely_duplicate_group")

        report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[likely_item.item_id])

        self.assertEqual(report.summary.generated_plan_count, 0)
        self.assertEqual(report.summary.unsupported_item_count, 1)
        self.assertIn("no safe action plan", report.unsupported_items[0].reason)

    @patch("nyxcore.action_plan.service.write_tags")
    def test_apply_metadata_plan_updates_review_state(self, write_tags_mock) -> None:
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
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review_report = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        metadata_item = next(item for item in review_report.items if item.item_type == "missing_metadata")
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[metadata_item.item_id])
        metadata_report = plan_report.__class__(
            created_at=plan_report.created_at,
            source_review_item_ids=plan_report.source_review_item_ids,
            plans=[plan for plan in plan_report.plans if plan.action_type == "metadata_fix_plan"],
            unsupported_items=[],
            summary=plan_report.summary,
        )
        review_state = ReviewStateStore()

        results = apply_action_plan_report(metadata_report, review_state=review_state, backup_dir=self.out / "backups")

        self.assertEqual(results[0].status, "ok")
        self.assertEqual(review_state.items[metadata_item.item_id].status, "resolved")
        self.assertTrue(write_tags_mock.called)

    def test_duplicate_apply_uses_quarantine_move_only(self) -> None:
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
        plan_report = build_action_plan_report(self.music, records, review_report, source_review_item_ids=[exact_item.item_id])
        review_state = ReviewStateStore()

        results = apply_action_plan_report(plan_report, review_state=review_state)

        self.assertEqual(results[0].status, "ok")
        self.assertFalse(left.exists())
        quarantine_target = next(
            Path(operation.destination_path)
            for operation in plan_report.plans[0].proposed_operations
            if operation.operation_type == "quarantine_move"
        )
        self.assertTrue(quarantine_target.exists())
        self.assertEqual(review_state.items[exact_item.item_id].status, "resolved")

    @patch("nyxcore.action_plan.service.write_tags")
    def test_partial_failure_keeps_review_state_honest(self, write_tags_mock) -> None:
        first = self.music / "Artist - One.mp3"
        second = self.music / "Artist - Two.mp3"
        first.write_bytes(b"a" * 100)
        second.write_bytes(b"b" * 100)
        records = [
            _track(first, title=None, artist=None, album=None, duration=180.0, cover=False, warnings=[WarningCode.missing_title, WarningCode.missing_artist, WarningCode.missing_album]),
            _track(second, title=None, artist=None, album=None, duration=180.0, cover=False, warnings=[WarningCode.missing_title, WarningCode.missing_artist, WarningCode.missing_album]),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
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
        write_tags_mock.side_effect = [None, RuntimeError("write failed")]
        review_state = ReviewStateStore()

        results = apply_action_plan_report(metadata_only, review_state=review_state)

        self.assertEqual(results[0].status, "partial_failure")
        self.assertNotIn(metadata_item.item_id, review_state.items)

    def test_cli_plan_generation_and_apply_smoke(self) -> None:
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
        self.assertTrue((self.out / "review_plan.json").exists())

        apply_result = runner.invoke(app, ["apply-review-plan", str(self.out / "review_plan.json"), "--out", str(self.out)])
        self.assertEqual(apply_result.exit_code, 0, msg=apply_result.stdout)
        state_payload = json.loads((self.out / "review_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state_payload["items"][exact_item_id]["status"], "resolved")


if __name__ == "__main__":
    unittest.main()
