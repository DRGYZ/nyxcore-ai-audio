from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
from mutagen import File as MutagenFile

from nyxcore.action_plan.ledger import (
    OperationLedger,
    inspect_batch_recovery,
    load_operation_ledger,
    prepare_operation_batch,
    save_operation_ledger,
)
from nyxcore.action_plan.lifecycle import recover_recorded_batch, reverse_recorded_batch
from nyxcore.action_plan.lock import LibraryMutationLock, MutationLockError
from nyxcore.action_plan.service import (
    ActionPlan,
    ActionPlanOperation,
    ActionPlanReport,
    ActionPlanSummary,
    build_action_plan_report,
    execute_reviewed_action_plan,
)
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
from nyxcore.review_queue.state import ReviewStateStore, load_review_state
from nyxcore.tagging.writer import backup_file, write_tags


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_mp3(path: Path) -> Path:
    frame = bytes.fromhex("FFFB9064") + bytes(413)
    path.write_bytes(frame * 3)
    return path


def _track(path: Path, *, cover: bool = False) -> TrackRecord:
    return TrackRecord(
        path=str(path),
        file_size_bytes=path.stat().st_size,
        mtime_iso="",
        tags={
            "title": None,
            "artist": None,
            "album": None,
            "albumartist": None,
            "tracknumber": None,
            "date": None,
            "genre": None,
        },
        has_cover_art=cover,
        duration_seconds=0.08,
        warnings=[WarningCode.missing_title, WarningCode.missing_artist, WarningCode.missing_album],
    )


def _metadata_context(tmp_path: Path):
    music = tmp_path / "music"
    out = tmp_path / "out"
    music.mkdir()
    out.mkdir()
    source = _minimal_mp3(music / "Artist - Song.mp3")
    records = [_track(source)]
    duplicates = DuplicateAnalysisReport(DuplicateSummary(1, 0, 0, 0, 0), [], [])
    health = build_health_report(music, records, duplicate_report=duplicates)
    review = build_review_queue(records, health_report=health, duplicate_report=duplicates)
    item = next(entry for entry in review.items if entry.item_type == "missing_metadata")
    generated = build_action_plan_report(music, records, review, source_review_item_ids=[item.item_id])
    generated.plans = [plan for plan in generated.plans if plan.action_type == "metadata_fix_plan"]
    return music, out, source, records, review, generated, item.item_id


def _prepared_move(tmp_path: Path):
    music = tmp_path / "music"
    out = tmp_path / "out"
    music.mkdir()
    out.mkdir()
    source = music / "source.mp3"
    destination = music / ".nyxcore_quarantine" / "source.mp3"
    source.write_bytes(b"reviewed-content")
    operation = ActionPlanOperation(
        "move-1",
        "quarantine_move",
        path=str(source),
        destination_path=str(destination),
        apply_supported=True,
        expected_hash=_hash(source),
    )
    plan = ActionPlan("plan-1", ["item-1"], "exact_duplicate_keep_plan", [str(source)], [operation], 1.0, "low-risk", [], [], True)
    report = ActionPlanReport("", ["item-1"], [plan], [], ActionPlanSummary(1, 1, 0, 1))
    ledger = OperationLedger()
    batch = prepare_operation_batch(ledger, plan_report=report, planned_backup_paths={})
    ledger_path = out / "review_history.json"
    save_operation_ledger(ledger_path, ledger)
    return music, out, source, destination, ledger_path, batch


def test_real_metadata_write_backup_journal_and_reversal(tmp_path: Path) -> None:
    music, out, source, records, review, requested, item_id = _metadata_context(tmp_path)
    original = source.read_bytes()
    state = ReviewStateStore()

    _authorized, results, batch = execute_reviewed_action_plan(
        music,
        records,
        review,
        requested,
        review_state=state,
        review_state_path=out / "review_state.json",
        ledger_path=out / "review_history.json",
        workspace_root=out,
    )

    assert results[0].status == "ok"
    assert batch.lifecycle_state == "applied"
    operation = batch.operations[0]
    assert operation.journal_state == "applied"
    assert Path(operation.backup_path).read_bytes() == original
    assert MutagenFile(source, easy=True).tags["title"] == ["Song"]
    assert load_review_state(out / "review_state.json").items[item_id].status == "resolved"

    _batch, changed = reverse_recorded_batch(
        ledger_path=out / "review_history.json",
        review_state_path=out / "review_state.json",
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
    )
    assert changed[0].undo_status == "ok"
    assert source.read_bytes() == original
    assert not (music / ".nyxcore_mutation.lock").exists()


def test_journal_is_durable_before_real_metadata_mutation(tmp_path: Path, monkeypatch) -> None:
    music, out, _source, records, review, requested, _item_id = _metadata_context(tmp_path)
    ledger_path = out / "review_history.json"
    import nyxcore.action_plan.service as mutation_service

    real_write_tags = mutation_service.write_tags

    def assert_journal_then_write(path, **kwargs):
        ledger = load_operation_ledger(ledger_path)
        operation = ledger.batches[-1].operations[0]
        assert operation.journal_state == "applying"
        assert operation.expected_hash == _hash(Path(operation.backup_path))
        assert (music / ".nyxcore_mutation.lock").exists()
        return real_write_tags(path, **kwargs)

    monkeypatch.setattr(mutation_service, "write_tags", assert_journal_then_write)
    _authorized, results, batch = execute_reviewed_action_plan(
        music,
        records,
        review,
        requested,
        review_state=ReviewStateStore(),
        review_state_path=out / "review_state.json",
        ledger_path=ledger_path,
        workspace_root=out,
    )
    assert results[0].status == "ok"
    assert batch.operations[0].journal_state == "applied"


def test_metadata_reversal_refuses_later_user_edit(tmp_path: Path) -> None:
    music, out, source, records, review, requested, _item_id = _metadata_context(tmp_path)
    _authorized, _results, batch = execute_reviewed_action_plan(
        music,
        records,
        review,
        requested,
        review_state=ReviewStateStore(),
        review_state_path=out / "review_state.json",
        ledger_path=out / "review_history.json",
        workspace_root=out,
    )
    source.write_bytes(source.read_bytes() + b"user-edit")

    _batch, changed = reverse_recorded_batch(
        ledger_path=out / "review_history.json",
        review_state_path=out / "review_state.json",
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
    )
    assert changed[0].undo_status == "error"
    assert "newer content" in changed[0].undo_message
    assert source.read_bytes().endswith(b"user-edit")


def test_move_recovery_classifies_never_started(tmp_path: Path) -> None:
    _music, _out, source, destination, _ledger_path, batch = _prepared_move(tmp_path)
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "not_started"
    assert assessment.safe_actions == ["abort"]
    assert source.exists() and not destination.exists()


def test_move_recovery_finalizes_completed_unjournaled_move_and_is_repeatable(tmp_path: Path) -> None:
    music, out, source, destination, ledger_path, batch = _prepared_move(tmp_path)
    destination.parent.mkdir(parents=True)
    os.link(source, destination)
    source.unlink()
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "completed_unfinalized"

    recovered, assessments = recover_recorded_batch(
        ledger_path=ledger_path,
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
        action="finalize",
    )
    assert recovered.operations[0].journal_state == "applied"
    assert assessments[0].classification == "complete"
    repeated, repeated_assessments = recover_recorded_batch(
        ledger_path=ledger_path,
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
        action="finalize",
    )
    assert repeated.lifecycle_state == "applied"
    assert repeated_assessments[0].classification == "complete"


def test_move_recovery_finalizes_link_created_source_present(tmp_path: Path) -> None:
    music, out, source, destination, ledger_path, batch = _prepared_move(tmp_path)
    destination.parent.mkdir(parents=True)
    os.link(source, destination)
    assert inspect_batch_recovery(batch)[0].classification == "destination_linked_source_present"

    recovered, _assessments = recover_recorded_batch(
        ledger_path=ledger_path,
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
        action="finalize",
    )
    assert recovered.operations[0].journal_state == "applied"
    assert not source.exists() and destination.exists()


def test_move_recovery_refuses_unexpected_destination_content(tmp_path: Path) -> None:
    _music, _out, source, destination, _ledger_path, batch = _prepared_move(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"unexpected-user-content")
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "manual_intervention"
    assert assessment.safe_actions == []
    assert source.exists() and destination.read_bytes() == b"unexpected-user-content"


def test_move_recovery_reports_when_both_paths_are_missing(tmp_path: Path) -> None:
    _music, _out, source, destination, _ledger_path, batch = _prepared_move(tmp_path)
    source.unlink()
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "manual_intervention"
    assert "both missing" in assessment.detail
    assert not source.exists() and not destination.exists()


def test_metadata_recovery_states_are_diagnosed(tmp_path: Path) -> None:
    music, out, source, _records, _review, requested, _item_id = _metadata_context(tmp_path)
    operation = requested.plans[0].proposed_operations[0]
    backup = out / "backups" / "source.mp3"
    ledger = OperationLedger()
    batch = prepare_operation_batch(
        ledger,
        plan_report=requested,
        planned_backup_paths={operation.operation_id: str(backup)},
    )
    backup_file(source, backup.parent, destination=backup)
    assert inspect_batch_recovery(batch)[0].classification == "backup_created_original_unchanged"

    write_tags(source, title="Song", artist="Artist", album=None, fields=["title", "artist"])
    batch.operations[0].journal_state = "applying"
    batch.operations[0].applied_hash = _hash(source)
    assert inspect_batch_recovery(batch)[0].classification == "completed_unfinalized"

    source.write_bytes(source.read_bytes() + b"user-edit")
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "manual_intervention"
    assert assessment.safe_actions == []

    backup.unlink()
    assessment = inspect_batch_recovery(batch)[0]
    assert assessment.classification == "manual_intervention"
    assert "backup is missing" in assessment.detail
    assert music.exists()


def test_partial_batch_persists_applied_failed_and_not_started_operations(tmp_path: Path) -> None:
    music = tmp_path / "music"
    out = tmp_path / "out"
    music.mkdir()
    out.mkdir()
    paths = [music / name for name in ("a.mp3", "b.mp3", "c.mp3", "keep.flac")]
    for path in paths:
        path.write_bytes(b"same-content")
    records = [_track(path, cover=path.suffix == ".flac") for path in paths]
    infos = [
        DuplicateTrackInfo(str(path), path.stat().st_size, path.suffix, 1.0, 192000, "Track", "Artist", "Album", path.suffix == ".flac", 3)
        for path in paths
    ]
    duplicates = DuplicateAnalysisReport(
        DuplicateSummary(4, 1, 4, 0, 0),
        [ExactDuplicateGroup("exact-partial", "hash", infos, PreferredCopyRecommendation(str(paths[-1]), ["lossless"]))],
        [],
    )
    health = build_health_report(music, records, duplicate_report=duplicates)
    review = build_review_queue(records, health_report=health, duplicate_report=duplicates)
    item = next(entry for entry in review.items if entry.item_type == "exact_duplicate_group")
    requested = build_action_plan_report(music, records, review, source_review_item_ids=[item.item_id])
    moves = [entry for entry in requested.plans[0].proposed_operations if entry.operation_type == "quarantine_move"]
    collision = Path(moves[1].destination_path)
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"existing-user-file")

    _authorized, results, batch = execute_reviewed_action_plan(
        music,
        records,
        review,
        requested,
        review_state=ReviewStateStore(),
        review_state_path=out / "review_state.json",
        ledger_path=out / "review_history.json",
        workspace_root=out,
    )
    assert results[0].status == "partial_failure"
    assert [entry.journal_state for entry in batch.operations] == ["applied", "failed", "prepared"]
    assert batch.lifecycle_state == "partial_failure"
    assert item.item_id not in load_review_state(out / "review_state.json").items
    assert collision.read_bytes() == b"existing-user-file"
    assert not (music / ".nyxcore_mutation.lock").exists()


def test_library_lock_rejects_second_mutation_and_releases_after_error(tmp_path: Path) -> None:
    music, out, _source, records, review, requested, _item_id = _metadata_context(tmp_path)
    with LibraryMutationLock(music):
        with pytest.raises(MutationLockError, match="already held"):
            execute_reviewed_action_plan(
                music,
                records,
                review,
                requested,
                review_state=ReviewStateStore(),
                review_state_path=out / "review_state.json",
                ledger_path=out / "review_history.json",
                workspace_root=out,
            )
    assert not (music / ".nyxcore_mutation.lock").exists()

    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    _move_music, move_out, _move_source, _move_destination, ledger_path, batch = _prepared_move(recovery_root)
    with LibraryMutationLock(_move_music):
        with pytest.raises(MutationLockError, match="already held"):
            recover_recorded_batch(
                ledger_path=ledger_path,
                batch_id=batch.batch_id,
                library_root=_move_music,
                workspace_root=move_out,
                action="abort",
            )

    with pytest.raises(RuntimeError, match="handled failure"):
        with LibraryMutationLock(music):
            raise RuntimeError("handled failure")
    assert not (music / ".nyxcore_mutation.lock").exists()


def test_stale_lock_takeover_requires_exact_token_and_dead_local_owner(tmp_path: Path) -> None:
    music, out, _source, _destination, ledger_path, batch = _prepared_move(tmp_path)
    lock_path = music / ".nyxcore_mutation.lock"
    lock_path.write_text(json.dumps({
        "token": "stale-token",
        "pid": 99_999_999,
        "hostname": socket.gethostname(),
        "library_root": str(music.resolve()),
        "acquired_at": "2020-01-01T00:00:00+00:00",
    }), encoding="utf-8")

    with pytest.raises(MutationLockError, match="does not match"):
        recover_recorded_batch(
            ledger_path=ledger_path,
            batch_id=batch.batch_id,
            library_root=music,
            workspace_root=out,
            action="abort",
            stale_lock_token="wrong-token",
        )
    assert lock_path.exists()

    recovered, assessments = recover_recorded_batch(
        ledger_path=ledger_path,
        batch_id=batch.batch_id,
        library_root=music,
        workspace_root=out,
        action="abort",
        stale_lock_token="stale-token",
    )
    assert recovered.lifecycle_state == "failed"
    assert assessments[0].journal_state == "failed"
    assert not lock_path.exists()
