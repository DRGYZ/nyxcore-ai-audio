from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nyxcore.cli import app
from nyxcore.config import ReviewConfig, load_config
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.duplicates.service import (
    DuplicateAnalysisReport,
    DuplicateSummary,
    DuplicateTrackInfo,
    ExactDuplicateGroup,
    LikelyDuplicateGroup,
    LikelyDuplicateRelationship,
    PreferredCopyRecommendation,
    analyze_duplicates,
)
from nyxcore.health.service import build_health_report
from nyxcore.review_queue.service import build_review_queue
from nyxcore.review_queue.state import (
    ReviewStateEntry,
    ReviewStateStore,
    apply_review_action,
    load_review_state,
    normalize_review_state,
    save_review_state,
)

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_review"


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


class ReviewQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.music.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_queue_construction_from_mixed_findings(self) -> None:
        dup_a = self.music / "dup-a.mp3"
        dup_b = self.music / "dup-b.flac"
        weak = self.music / "weak.mp3"
        dup_a.write_bytes(b"same-bytes" * 256)
        dup_b.write_bytes(b"same-bytes" * 256)
        weak.write_bytes(b"weak")
        records = [
            _track(dup_a, title="Track", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(dup_b, title="Track", artist="Artist", album="Album", duration=180.0, cover=True),
            _track(
                weak,
                title="Track 01",
                artist="unknown",
                album=None,
                duration=120.0,
                cover=False,
                warnings=[WarningCode.missing_album, WarningCode.low_bitrate],
            ),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(
                total_tracks=3,
                exact_group_count=1,
                exact_duplicate_file_count=2,
                likely_group_count=1,
                likely_duplicate_file_count=2,
            ),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-001",
                    content_hash="hash-a",
                    files=[_dup_info(dup_a, size=dup_a.stat().st_size), _dup_info(dup_b, size=dup_b.stat().st_size, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(dup_b), reasons=["preferred_lossless_format"]),
                )
            ],
            likely_duplicates=[
                LikelyDuplicateGroup(
                    group_id="likely-001",
                    confidence=0.83,
                    files=[_dup_info(dup_a, size=dup_a.stat().st_size), _dup_info(weak, size=weak.stat().st_size)],
                    relationships=[
                        LikelyDuplicateRelationship(
                            paths=sorted([str(dup_a), str(weak)]),
                            confidence=0.83,
                            reasons=["title_match_good", "duration_delta=1.0s"],
                        )
                    ],
                    preferred=PreferredCopyRecommendation(path=str(dup_a), reasons=["highest_bitrate"]),
                    reasons=["title_match_good", "duration_delta=1.0s"],
                )
            ],
        )
        health_report = build_health_report(
            self.music,
            records,
            settings=load_config().health,
            duplicate_report=duplicate_report,
        )

        review = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_settings=load_config().review,
            health_settings=load_config().health,
        )

        item_types = {item.item_type for item in review.items}
        self.assertIn("exact_duplicate_group", item_types)
        self.assertIn("likely_duplicate_group", item_types)
        self.assertIn("missing_metadata", item_types)
        self.assertIn("weak_or_placeholder_metadata", item_types)
        self.assertIn("artwork_missing", item_types)
        self.assertIn("low_quality_audio", item_types)

    def test_priority_ordering_is_deterministic(self) -> None:
        a = self.music / "a.mp3"
        b = self.music / "b.mp3"
        c = self.music / "folder" / "c.mp3"
        c.parent.mkdir()
        a.write_bytes(b"x" * 100)
        b.write_bytes(b"x" * 100)
        c.write_bytes(b"c" * 100)
        records = [
            _track(a, title=None, artist="Artist", album="Album", duration=180.0, cover=False, warnings=[WarningCode.missing_title]),
            _track(b, title="Track", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(c, title="Track 01", artist="unknown", album="Album", duration=180.0, cover=False, warnings=[WarningCode.low_bitrate]),
        ]
        with patch("nyxcore.duplicates.service._load_bitrate_bps", return_value=192000):
            duplicate_report = analyze_duplicates(records)
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)

        first = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        second = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)

        self.assertEqual([item.item_id for item in first.items], [item.item_id for item in second.items])

    def test_stable_item_ids_across_equivalent_reruns(self) -> None:
        first = self.music / "dup-a.mp3"
        second = self.music / "dup-b.flac"
        first.write_bytes(b"dup" * 256)
        second.write_bytes(b"dup" * 256)
        records = [
            _track(first, title="Track", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(second, title="Track", artist="Artist", album="Album", duration=180.0, cover=True),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 1, 2, 0, 0),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-001",
                    content_hash="hash-a",
                    files=[_dup_info(first, size=first.stat().st_size), _dup_info(second, size=second.stat().st_size, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(second), reasons=["preferred_lossless_format"]),
                )
            ],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)

        left = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        right = build_review_queue(list(reversed(records)), health_report=health_report, duplicate_report=duplicate_report)

        self.assertEqual([item.item_id for item in left.items], [item.item_id for item in right.items])

    def test_legacy_absolute_item_ids_are_migrated_to_current_queue_items(self) -> None:
        first = self.music / "dup-a.mp3"
        second = self.music / "dup-b.flac"
        first.write_bytes(b"dup" * 256)
        second.write_bytes(b"dup" * 256)
        records = [
            _track(first, title="Track", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(second, title="Track", artist="Artist", album="Album", duration=180.0, cover=True),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(2, 1, 2, 0, 0),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-001",
                    content_hash="hash-a",
                    files=[_dup_info(first, size=first.stat().st_size), _dup_info(second, size=second.stat().st_size, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(second), reasons=["preferred_lossless_format"]),
                )
            ],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        current_review = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        current_exact_item = next(item for item in current_review.items if item.item_type == "exact_duplicate_group")
        legacy_root = self.root / "legacy-music"
        legacy_paths = sorted([str(legacy_root / first.name), str(legacy_root / second.name)])
        legacy_payload = json.dumps(
            ["exact_duplicate_group", legacy_paths],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        legacy_item_id = f"exact_duplicate_group-{hashlib.sha1(legacy_payload.encode('utf-8')).hexdigest()[:12]}"
        state = ReviewStateStore(
            items={
                legacy_item_id: ReviewStateEntry(
                    item_id=legacy_item_id,
                    status="ignored",
                    updated_at="2026-03-01T00:00:00+00:00",
                    item_type="exact_duplicate_group",
                    summary=current_exact_item.summary,
                )
            }
        )

        review = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_state=state,
            include_ignored=True,
        )

        exact_item = next(item for item in review.items if item.item_type == "exact_duplicate_group")
        self.assertNotEqual(exact_item.item_id, legacy_item_id)
        self.assertEqual(exact_item.item_id, current_exact_item.item_id)
        self.assertEqual(exact_item.review_status, "ignored")
        self.assertIn(exact_item.item_id, state.items)
        self.assertNotIn(legacy_item_id, state.items)

    def test_large_reclaimable_exact_duplicates_rank_higher(self) -> None:
        big_a = self.music / "big-a.mp3"
        big_b = self.music / "big-b.flac"
        small_a = self.music / "small-a.mp3"
        small_b = self.music / "small-b.flac"
        for path, size in ((big_a, 1_200), (big_b, 1_200), (small_a, 120), (small_b, 120)):
            path.write_bytes(b"x" * size)
        records = [
            _track(big_a, title="Big", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(big_b, title="Big", artist="Artist", album="Album", duration=180.0, cover=True),
            _track(small_a, title="Small", artist="Artist", album="Album", duration=180.0, cover=False),
            _track(small_b, title="Small", artist="Artist", album="Album", duration=180.0, cover=True),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(
                total_tracks=4,
                exact_group_count=2,
                exact_duplicate_file_count=4,
                likely_group_count=0,
                likely_duplicate_file_count=0,
            ),
            exact_duplicates=[
                ExactDuplicateGroup(
                    group_id="exact-big",
                    content_hash="hash-big",
                    files=[_dup_info(big_a, size=1_200), _dup_info(big_b, size=1_200, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(big_b), reasons=["preferred_lossless_format"]),
                ),
                ExactDuplicateGroup(
                    group_id="exact-small",
                    content_hash="hash-small",
                    files=[_dup_info(small_a, size=120), _dup_info(small_b, size=120, cover=True)],
                    preferred=PreferredCopyRecommendation(path=str(small_b), reasons=["preferred_lossless_format"]),
                ),
            ],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        settings = ReviewConfig(reclaimable_bytes_unit=100)

        review = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_settings=settings,
        )

        exact_items = [item for item in review.items if item.item_type == "exact_duplicate_group"]
        self.assertGreater(exact_items[0].priority_score, exact_items[1].priority_score)
        self.assertEqual(exact_items[0].reclaimable_bytes, 1_200)

    def test_folder_hotspots_are_aggregated_in_issue_order(self) -> None:
        hot = self.music / "hot"
        cold = self.music / "cold"
        hot.mkdir()
        cold.mkdir()
        hot_a = hot / "a.mp3"
        hot_b = hot / "b.mp3"
        cold_a = cold / "c.mp3"
        for path in (hot_a, hot_b, cold_a):
            path.write_bytes(b"x" * 100)
        records = [
            _track(
                hot_a,
                title=None,
                artist="unknown",
                album=None,
                duration=180.0,
                cover=False,
                warnings=[WarningCode.missing_title, WarningCode.missing_album, WarningCode.low_bitrate],
            ),
            _track(
                hot_b,
                title="Track 01",
                artist="unknown",
                album="Album",
                duration=180.0,
                cover=False,
                warnings=[WarningCode.low_bitrate],
            ),
            _track(
                cold_a,
                title="Track",
                artist="Artist",
                album="Album",
                duration=180.0,
                cover=True,
                warnings=[WarningCode.low_bitrate],
            ),
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(3, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)

        hotspot_items = [item for item in review.items if item.item_type == "folder_hotspot"]
        self.assertGreaterEqual(len(hotspot_items), 2)
        self.assertEqual(hotspot_items[0].folder, "hot")
        self.assertGreater(hotspot_items[0].priority_score, hotspot_items[1].priority_score)

    def test_profile_influenced_priority_changes(self) -> None:
        target = self.music / "lofi.mp3"
        target.write_bytes(b"x" * 100)
        records = [
            _track(
                target,
                title="Track",
                artist="Artist",
                album="Album",
                duration=180.0,
                cover=False,
                warnings=[WarningCode.low_bitrate],
            )
        ]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)

        default_cfg = load_config()
        archivist_cfg = load_config(profile="archivist")
        default_report = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_settings=default_cfg.review,
            health_settings=default_cfg.health,
        )
        archivist_report = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_settings=archivist_cfg.review,
            health_settings=archivist_cfg.health,
            active_profile=archivist_cfg.profile,
        )

        default_item = next(item for item in default_report.items if item.item_type == "low_quality_audio")
        archivist_item = next(item for item in archivist_report.items if item.item_type == "low_quality_audio")
        self.assertGreater(archivist_item.priority_score, default_item.priority_score)

    def test_state_persistence_and_transitions(self) -> None:
        state_path = self.root / "review_state.json"
        state = ReviewStateStore()
        apply_review_action(state, item_ids=["item-1"], status="seen")
        apply_review_action(state, item_ids=["item-2"], status="ignored")
        apply_review_action(state, item_ids=["item-3"], status="resolved")
        save_review_state(state_path, state)
        loaded = load_review_state(state_path)

        self.assertEqual(loaded.items["item-1"].status, "seen")
        self.assertEqual(loaded.items["item-2"].status, "ignored")
        self.assertEqual(loaded.items["item-3"].status, "resolved")

    def test_snooze_expiry_returns_item_to_seen(self) -> None:
        state = ReviewStateStore(
            items={
                "item-1": ReviewStateEntry(
                    item_id="item-1",
                    status="snoozed",
                    updated_at="2026-03-01T00:00:00+00:00",
                    snooze_until="2026-03-02T00:00:00+00:00",
                )
            }
        )

        normalize_review_state(
            state,
            now=datetime.fromisoformat("2026-03-03T00:00:00+00:00"),
        )

        self.assertEqual(state.items["item-1"].status, "seen")
        self.assertIsNone(state.items["item-1"].snooze_until)

    def test_default_filtering_hides_ignored_and_resolved(self) -> None:
        target = self.music / "a.mp3"
        target.write_bytes(b"x" * 100)
        records = [_track(target, title=None, artist="Artist", album="Album", duration=180.0, cover=False, warnings=[WarningCode.missing_title])]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        full_report = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
        )
        item_id = full_report.items[0].item_id
        state = ReviewStateStore(
            items={
                item_id: ReviewStateEntry(
                    item_id=item_id,
                    status="ignored",
                    updated_at="2026-03-01T00:00:00+00:00",
                )
            }
        )

        hidden = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_state=state,
        )
        shown = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_state=state,
            include_ignored=True,
        )

        self.assertTrue(all(item.item_id != item_id for item in hidden.items))
        shown_item = next(item for item in shown.items if item.item_id == item_id)
        self.assertEqual(shown_item.review_status, "ignored")

    def test_resolved_item_reappearing_reactivates_as_seen(self) -> None:
        target = self.music / "a.mp3"
        target.write_bytes(b"x" * 100)
        records = [_track(target, title=None, artist="Artist", album="Album", duration=180.0, cover=False, warnings=[WarningCode.missing_title])]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        initial = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)
        item_id = initial.items[0].item_id
        state = ReviewStateStore(
            items={
                item_id: ReviewStateEntry(
                    item_id=item_id,
                    status="resolved",
                    updated_at="2026-03-01T00:00:00+00:00",
                )
            }
        )

        rerun = build_review_queue(
            records,
            health_report=health_report,
            duplicate_report=duplicate_report,
            review_state=state,
        )

        rerun_item = next(item for item in rerun.items if item.item_id == item_id)
        self.assertEqual(rerun_item.review_status, "seen")

    def test_report_serialization_is_stable(self) -> None:
        target = self.music / "a.mp3"
        target.write_bytes(b"x" * 100)
        records = [_track(target, title=None, artist="Artist", album="Album", duration=180.0, cover=False, warnings=[WarningCode.missing_title])]
        duplicate_report = DuplicateAnalysisReport(
            summary=DuplicateSummary(1, 0, 0, 0, 0),
            exact_duplicates=[],
            likely_duplicates=[],
        )
        health_report = build_health_report(self.music, records, duplicate_report=duplicate_report)
        review = build_review_queue(records, health_report=health_report, duplicate_report=duplicate_report)

        payload = review.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("summary", payload)
        self.assertIn("items", payload)
        self.assertIn("metadata", payload)
        self.assertIn("missing_metadata", encoded)
        self.assertIn("counts_by_state", payload["summary"])

    def test_cli_review_smoke_writes_reports(self) -> None:
        out = self.root / "out"
        out.mkdir()
        first = self.music / "dup-a.mp3"
        second = self.music / "dup-b.mp3"
        content = b"dup" * 512
        first.write_bytes(content)
        second.write_bytes(content)

        runner = CliRunner()
        result = runner.invoke(app, ["review", str(self.music), "--out", str(out), "--max-items", "3"])

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertTrue((out / "review.json").exists())
        payload = json.loads((out / "review.json").read_text(encoding="utf-8"))
        self.assertIn("items", payload)
        self.assertEqual(payload["metadata"]["generation_mode"], "full")

    def test_cli_mark_ignore_hides_item_on_next_run(self) -> None:
        out = self.root / "out"
        out.mkdir()
        first = self.music / "dup-a.mp3"
        second = self.music / "dup-b.mp3"
        content = b"dup" * 512
        first.write_bytes(content)
        second.write_bytes(content)
        runner = CliRunner()

        initial = runner.invoke(app, ["review", str(self.music), "--out", str(out)])
        self.assertEqual(initial.exit_code, 0, msg=initial.stdout)
        payload = json.loads((out / "review.json").read_text(encoding="utf-8"))
        item_id = payload["items"][0]["item_id"]

        ignored = runner.invoke(app, ["review", str(self.music), "--out", str(out), "--ignore", item_id])
        self.assertEqual(ignored.exit_code, 0, msg=ignored.stdout)

        next_run = runner.invoke(app, ["review", str(self.music), "--out", str(out)])
        self.assertEqual(next_run.exit_code, 0, msg=next_run.stdout)
        filtered_payload = json.loads((out / "review.json").read_text(encoding="utf-8"))
        self.assertTrue(all(item["item_id"] != item_id for item in filtered_payload["items"]))


if __name__ == "__main__":
    unittest.main()
