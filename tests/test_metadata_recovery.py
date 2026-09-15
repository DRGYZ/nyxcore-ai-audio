from pathlib import Path
from unittest.mock import patch

from nyxcore.action_plan.service import ActionPlan, ActionPlanOperation, ActionPlanReport, ActionPlanSummary, apply_action_plan_report
from nyxcore.action_plan.ledger import OperationLedger, append_operation_batch, undo_operation_batch
from nyxcore.core.audio_files import iter_audio_files
from nyxcore.review_queue.state import ReviewStateStore


def _apply(tmp_path, *, rename=False):
    source = tmp_path / 'song.mp3'
    source.write_bytes(b'original recording')
    operations = [ActionPlanOperation('metadata', 'write_metadata', path=str(source), fields=['title'], values={'title': 'Song'}, apply_supported=True)]
    if rename:
        operations.append(ActionPlanOperation('rename', 'rename_file', path=str(source), destination_path=str(tmp_path / 'renamed.mp3'), apply_supported=True))
    plan = ActionPlan('plan', [], 'metadata_fix_plan', [str(source)], operations, 1.0, 'low-risk', [], [], True)
    report = ActionPlanReport('', [], [plan], [], ActionPlanSummary(1, 1, 0, 1))
    state = ReviewStateStore()
    with patch('nyxcore.action_plan.service.write_tags', side_effect=lambda path, **kwargs: path.write_bytes(b'edited tags')):
        results = apply_action_plan_report(report, review_state=state)
    assert results[0].status == 'ok'
    batch = append_operation_batch(OperationLedger(), plan_report=report, results=results)
    return source, state, batch


def test_default_backup_and_restore(tmp_path):
    source, state, batch = _apply(tmp_path)
    assert Path(batch.operations[0].backup_path).read_bytes() == b'original recording'
    assert iter_audio_files(tmp_path) == [source]
    undo_operation_batch(batch, review_state=state)
    assert source.read_bytes() == b'original recording'


def test_restore_preserves_newer_content_after_ledger_reload(tmp_path):
    source, state, batch = _apply(tmp_path)
    batch = type(batch).from_dict(batch.to_dict())
    source.write_bytes(b'newer user edit')
    changed = undo_operation_batch(batch, review_state=state)
    assert changed[0].undo_status == 'error'
    assert source.read_bytes() == b'newer user edit'


def test_metadata_then_rename_restores_in_reverse_order(tmp_path):
    source, state, batch = _apply(tmp_path, rename=True)
    assert not source.exists()
    changed = undo_operation_batch(batch, review_state=state)
    assert all(op.undo_status == 'ok' for op in changed)
    assert source.read_bytes() == b'original recording'
    assert not (tmp_path / 'renamed.mp3').exists()
