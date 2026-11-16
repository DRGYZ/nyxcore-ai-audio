from __future__ import annotations

import hashlib
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from mutagen import File as MutagenFile

from nyxcore.action_plan.ledger import (
    OperationLedger,
    inspect_batch_recovery,
    load_operation_ledger,
    mark_journal_operation_result,
    mark_journal_operation_started,
    prepare_operation_batch,
    save_operation_ledger,
)
from nyxcore.action_plan.lifecycle import inspect_recorded_batch, recover_recorded_batch, reverse_recorded_batch
from nyxcore.action_plan.lock import LibraryMutationLock, MutationLockError
from nyxcore.action_plan.service import (
    ActionPlan,
    ActionPlanOperation,
    ActionPlanReport,
    ActionPlanSummary,
    AppliedOperationResult,
    build_action_plan_report,
    execute_reviewed_action_plan,
)
from nyxcore.core.filesystem import move_file_no_replace, regular_file_identity
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


def _start_move(ledger_path: Path, source: Path):
    ledger = load_operation_ledger(ledger_path)
    batch = ledger.batches[0]
    mark_journal_operation_started(batch, batch.operations[0].operation_id, move_identity=regular_file_identity(source))
    save_operation_ledger(ledger_path, ledger)
    return batch


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
    music, out, source, destination, ledger_path, _batch = _prepared_move(tmp_path)
    batch = _start_move(ledger_path, source)
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
    music, out, source, destination, ledger_path, _batch = _prepared_move(tmp_path)
    batch = _start_move(ledger_path, source)
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

    mark_journal_operation_started(batch, operation.operation_id)
    write_tags(source, title="Song", artist="Artist", album=None, fields=["title", "artist"])
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


@pytest.mark.parametrize("failure", ["collision", "identical_collision", "external_edit"])
def test_partial_batch_persists_applied_failed_and_not_started_operations(
    tmp_path: Path, monkeypatch, failure: str,
) -> None:
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
    rejected_source = Path(moves[1].path)
    rejected_hash = moves[1].expected_hash
    if failure == "external_edit":
        import nyxcore.action_plan.service as service

        def move_then_edit(source, destination, **kwargs):
            move_file_no_replace(source, destination, **kwargs)
            if destination == Path(moves[0].destination_path):
                rejected_source.write_bytes(b"newer user content")

        monkeypatch.setattr(service, "move_file_no_replace", move_then_edit)
    else:
        collision.parent.mkdir(parents=True, exist_ok=True)
        collision.write_bytes(b"same-content" if failure == "identical_collision" else b"existing-user-file")

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
    rejected = batch.operations[1]
    assert rejected.expected_hash == rejected_hash
    if failure == "external_edit":
        assert not collision.exists()
        assert rejected_source.read_bytes() == b"newer user content"
        assert "source changed" in rejected.failure_reason
    else:
        before = collision.read_bytes()
        _, assessments = recover_recorded_batch(
            ledger_path=out / "review_history.json", batch_id=batch.batch_id,
            library_root=music, workspace_root=out, action="finalize",
        )
        assert assessments[1].classification == "manual_intervention"
        assert assessments[1].safe_actions == []
        assert not os.path.samefile(rejected_source, collision)
        assert collision.read_bytes() == before and rejected_source.read_bytes() == b"same-content"
    assert not (music / ".nyxcore_mutation.lock").exists()


@pytest.mark.parametrize("phase", ["prepared", "legacy-started", "started"])
def test_move_recovery_refuses_unproven_destination(tmp_path: Path, phase: str) -> None:
    music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
    ledger = load_operation_ledger(ledger_path)
    batch = ledger.batches[0]
    if phase != "prepared":
        mark_journal_operation_started(
            batch, batch.operations[0].operation_id,
            move_identity=regular_file_identity(source) if phase == "started" else None,
        )
    save_operation_ledger(ledger_path, ledger)
    destination.parent.mkdir()
    if phase == "started":
        destination.write_bytes(source.read_bytes())
    else:
        os.link(source, destination)
    for _ in range(2):
        _, [assessment] = recover_recorded_batch(
            ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out,
            action="finalize",
        )
        assert assessment.classification == "manual_intervention"
        assert assessment.safe_actions == []
        assert source.read_bytes() == destination.read_bytes() == b"reviewed-content"


@pytest.mark.parametrize("backup_state", ["missing", "corrupt", "damaged_staging_copy"])
def test_metadata_restore_preserves_live_audio_with_invalid_backup(
    tmp_path: Path, monkeypatch, backup_state: str,
) -> None:
    music, out, source, records, review, requested, _ = _metadata_context(tmp_path)
    ledger_path = out / "review_history.json"
    state_path = out / "review_state.json"
    _, _, batch = execute_reviewed_action_plan(
        music, records, review, requested, review_state=ReviewStateStore(),
        review_state_path=state_path, ledger_path=ledger_path, workspace_root=out,
    )
    edited = source.read_bytes()
    backup = Path(batch.operations[0].backup_path)
    if backup_state == "missing":
        backup.unlink()
    elif backup_state == "corrupt":
        backup.write_bytes(b"truncated")
    else:
        monkeypatch.setattr("nyxcore.action_plan.ledger.shutil.copyfileobj",
                            lambda _original, output: output.write(b"truncated staging copy"))
    _, [changed] = reverse_recorded_batch(
        ledger_path=ledger_path, review_state_path=state_path, batch_id=batch.batch_id,
        library_root=music, workspace_root=out,
    )
    assert changed.undo_status == "error"
    assert "fingerprint" in changed.undo_message
    assert source.read_bytes() == edited
    assert not list(music.glob(".nyxcore-restore-*"))


@pytest.mark.skipif(os.name == "nt", reason="directory symlinks require optional Windows privileges")
def test_default_backup_symlink_is_rejected_before_journaling(tmp_path: Path) -> None:
    music, out, source, records, review, requested, _ = _metadata_context(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (music / ".nyxcore_backups").symlink_to(outside, target_is_directory=True)
    edited = source.read_bytes()
    ledger_path = out / "review_history.json"
    with pytest.raises(ValueError, match="backup directory is outside"):
        execute_reviewed_action_plan(
            music, records, review, requested, review_state=ReviewStateStore(),
            review_state_path=out / "review_state.json", ledger_path=ledger_path, workspace_root=out,
        )
    assert not ledger_path.exists()
    assert source.read_bytes() == edited
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="directory symlinks require optional Windows privileges")
def test_history_rejects_escaped_move_destination_before_inspection_or_recovery(tmp_path: Path) -> None:
    music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
    batch = _start_move(ledger_path, source)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.link(source, outside / destination.name)
    destination.parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="history destination path is outside"):
        inspect_recorded_batch(ledger_path, batch.batch_id, library_root=music, workspace_root=out)
    with pytest.raises(ValueError, match="history destination path is outside"):
        recover_recorded_batch(
            ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out, action="finalize",
        )
    assert source.read_bytes() == (outside / destination.name).read_bytes() == b"reviewed-content"


@pytest.mark.skipif(os.name == "nt", reason="directory symlinks require optional Windows privileges")
def test_alternate_restore_checks_computed_destination(tmp_path: Path) -> None:
    from nyxcore.action_plan.ledger import _alternate_restore_target

    music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
    batch = _start_move(ledger_path, source)
    move_file_no_replace(source, destination)
    recover_recorded_batch(
        ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out, action="finalize",
    )
    source.write_bytes(b"new music at the original path")
    alternate = out / "restored"
    alternate.mkdir()
    escaped = tmp_path / "outside"
    escaped.mkdir()
    target = _alternate_restore_target(alternate, str(source))
    target.parent.symlink_to(escaped, target_is_directory=True)
    args = dict(ledger_path=ledger_path, review_state_path=out / "review_state.json", batch_id=batch.batch_id,
                library_root=music, workspace_root=out, alternate_restore_dir=alternate)
    with pytest.raises(ValueError, match="alternate restore destination is outside"):
        reverse_recorded_batch(**args)
    assert list(escaped.iterdir()) == []
    assert destination.read_bytes() == b"reviewed-content"
    assert source.read_bytes() == b"new music at the original path"
    target.parent.unlink()
    _, [changed] = reverse_recorded_batch(**args)
    assert changed.undo_status == "ok"
    assert target.read_bytes() == b"reviewed-content" and not destination.exists()
    assert source.read_bytes() == b"new music at the original path"


def test_recovery_repairs_batch_summary_after_last_result_was_journaled(tmp_path: Path) -> None:
    music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
    _start_move(ledger_path, source)
    ledger = load_operation_ledger(ledger_path)
    [batch] = ledger.batches
    move_file_no_replace(source, destination)
    mark_journal_operation_result(batch, AppliedOperationResult(
        "move-1", "quarantine_move", str(source), str(destination), "ok", "applied", applied_hash=_hash(destination),
    ))
    assert batch.lifecycle_state == "applying"
    save_operation_ledger(ledger_path, ledger)
    recovered, _ = recover_recorded_batch(
        ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out, action="finalize",
    )
    assert recovered.lifecycle_state == "applied"
    assert recovered.plan_result_statuses == {"plan-1": "ok"}
    assert recovered.completed_at is not None
    saved = ledger_path.read_bytes()
    recover_recorded_batch(
        ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out, action="finalize",
    )
    assert ledger_path.read_bytes() == saved


@pytest.mark.parametrize("operation_type", ["move", "metadata"])
def test_reversal_reconciles_after_files_changed_but_journal_save_failed(
    tmp_path: Path, monkeypatch, operation_type: str,
) -> None:
    import nyxcore.action_plan.lifecycle as lifecycle

    if operation_type == "move":
        music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
        original = source.read_bytes()
        batch = _start_move(ledger_path, source)
        move_file_no_replace(source, destination)
        recover_recorded_batch(
            ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out, action="finalize",
        )
    else:
        music, out, source, records, review, requested, _ = _metadata_context(tmp_path)
        original = source.read_bytes()
        ledger_path = out / "review_history.json"
        _, _, batch = execute_reviewed_action_plan(
            music, records, review, requested, review_state=ReviewStateStore(),
            review_state_path=out / "review_state.json", ledger_path=ledger_path, workspace_root=out,
        )

    def fail_after_restore(path, ledger):
        operation = ledger.batches[0].operations[0]
        if operation.undo_status == "ok":
            on_disk = load_operation_ledger(path).batches[0].operations[0]
            assert on_disk.reversal_intent is not None and source.read_bytes() == original
            raise OSError("injected failure saving reversal result")
        save_operation_ledger(path, ledger)

    reverse_args = dict(ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out,
                        review_state_path=out / "review_state.json")
    with monkeypatch.context() as interrupted:
        interrupted.setattr(lifecycle, "save_operation_ledger", fail_after_restore)
        with pytest.raises(OSError, match="injected failure"):
            reverse_recorded_batch(**reverse_args)
    _, [assessment] = inspect_recorded_batch(
        ledger_path, batch.batch_id, library_root=music, workspace_root=out,
    )
    assert assessment.classification == "reversal_completed_unfinalized"
    assert assessment.safe_actions == ["finalize"]
    restored_identity = regular_file_identity(source)
    if operation_type == "move":
        recovered, _ = recover_recorded_batch(
            ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out,
            action="finalize",
        )
        assert not destination.exists()
    else:
        recovered, _ = reverse_recorded_batch(**reverse_args)
    assert recovered.lifecycle_state == "reversed"
    _, [operation] = reverse_recorded_batch(**reverse_args)
    assert operation.undo_status == "ok" and operation.reversal_intent is None
    assert regular_file_identity(source) == restored_identity
    assert source.read_bytes() == original


@pytest.mark.parametrize("replace_link", [False, True])
def test_reversal_recovery_finishes_only_its_own_interrupted_link(tmp_path: Path, monkeypatch, replace_link: bool) -> None:
    music, out, source, destination, ledger_path, _ = _prepared_move(tmp_path)
    batch = _start_move(ledger_path, source)
    move_file_no_replace(source, destination)
    recovery_args = dict(ledger_path=ledger_path, batch_id=batch.batch_id, library_root=music, workspace_root=out)
    recover_recorded_batch(**recovery_args, action="finalize")

    def interrupt_after_link(current, target, **_kwargs):
        recorded = load_operation_ledger(ledger_path).batches[0].operations[0]
        assert recorded.reversal_intent is not None and Path(recorded.reversal_intent.destination_path) == target
        os.link(current, target)
        raise KeyboardInterrupt()

    with monkeypatch.context() as interrupted:
        interrupted.setattr("nyxcore.action_plan.ledger.move_file_no_replace", interrupt_after_link)
        with pytest.raises(KeyboardInterrupt):
            reverse_recorded_batch(**recovery_args, review_state_path=out / "review_state.json")
    assert os.path.samefile(source, destination)
    if replace_link:
        source.unlink()
        source.write_bytes(destination.read_bytes())
        assert not os.path.samefile(source, destination)
    _, [assessment] = inspect_recorded_batch(
        ledger_path, batch.batch_id, library_root=music, workspace_root=out,
    )
    expected = "manual_intervention" if replace_link else "reversal_destination_linked_source_present"
    assert assessment.classification == expected
    for _ in range(2):
        recovered, [assessment] = recover_recorded_batch(**recovery_args, action="finalize")
        assert source.read_bytes() == b"reviewed-content"
        if replace_link:
            assert assessment.safe_actions == []
            assert destination.read_bytes() == b"reviewed-content"
        else:
            assert recovered.lifecycle_state == "reversed"
            assert not destination.exists()


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


@pytest.fixture
def forbid_pid_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_signal(_pid: int, _signal: int) -> None:
        pytest.fail("lock acquisition must not signal a process")

    monkeypatch.setattr(os, "kill", unexpected_signal)


@pytest.mark.usefixtures("forbid_pid_signal")
@pytest.mark.parametrize(
    ("token", "malformed"),
    [
        pytest.param("stale-token", False, id="matching-dead-owner"),
        pytest.param("wrong-token", False, id="wrong-token"),
        pytest.param("stale-token", True, id="malformed-owner"),
    ],
)
def test_recovery_refuses_stale_lock_takeover(tmp_path: Path, token: str, malformed: bool) -> None:
    music, out, _source, _destination, ledger_path, batch = _prepared_move(tmp_path)
    lock_path = music / ".nyxcore_mutation.lock"
    contents = b"{" if malformed else json.dumps({
        "token": "stale-token",
        "pid": 99_999_999,
        "hostname": socket.gethostname(),
        "library_root": str(music.resolve()),
        "acquired_at": "2020-01-01T00:00:00+00:00",
    }).encode("utf-8")
    lock_path.write_bytes(contents)
    owner = LibraryMutationLock.read_owner(lock_path)
    ledger_contents = ledger_path.read_bytes()

    with pytest.raises(MutationLockError, match="Automatic stale-lock takeover is disabled"):
        recover_recorded_batch(
            ledger_path=ledger_path,
            batch_id=batch.batch_id,
            library_root=music,
            workspace_root=out,
            action="abort",
            stale_lock_token=token,
        )
    assert lock_path.read_bytes() == contents
    assert LibraryMutationLock.read_owner(lock_path) == owner
    assert ledger_path.read_bytes() == ledger_contents


@pytest.mark.usefixtures("forbid_pid_signal")
@pytest.mark.parametrize("token", ["stale-token", ""])
def test_stale_lock_token_is_refused_without_an_existing_lock(tmp_path: Path, token: str) -> None:
    lock = LibraryMutationLock(tmp_path)
    with pytest.raises(MutationLockError, match="Retry without a takeover token"):
        lock.acquire(stale_owner_token=token)
    assert not lock.path.exists()
    with lock:
        assert LibraryMutationLock.read_owner(lock.path) == lock.record
    assert not lock.path.exists()


@pytest.mark.usefixtures("forbid_pid_signal")
def test_concurrent_takeover_attempts_preserve_the_owner(tmp_path: Path) -> None:
    start = Barrier(2, timeout=5)
    with LibraryMutationLock(tmp_path) as owner:
        contents = owner.path.read_bytes()

        def attempt_takeover() -> None:
            contender = LibraryMutationLock(tmp_path)
            start.wait()
            with pytest.raises(MutationLockError, match="Automatic stale-lock takeover is disabled"):
                contender.acquire(stale_owner_token=owner.record.token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(attempt_takeover) for _ in range(2)]
            for attempt in attempts:
                attempt.result()

        assert owner.path.read_bytes() == contents
        assert LibraryMutationLock.read_owner(owner.path) == owner.record
    assert not owner.path.exists()


def test_library_lock_release_refuses_a_changed_owner_token(tmp_path: Path) -> None:
    lock = LibraryMutationLock(tmp_path).acquire()
    original = lock.path.read_bytes()
    changed = json.loads(original)
    changed["token"] = "different-owner"
    changed_contents = json.dumps(changed).encode("utf-8")
    lock.path.write_bytes(changed_contents)

    with pytest.raises(MutationLockError, match="ownership changed"):
        lock.release()
    assert lock.path.read_bytes() == changed_contents
    lock.path.write_bytes(original)
    lock.release()
    assert not lock.path.exists()
