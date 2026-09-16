from __future__ import annotations

from pathlib import Path

from nyxcore.action_plan.ledger import (
    LedgerOperation,
    OperationBatch,
    RecoveryAssessment,
    apply_batch_recovery,
    find_batch,
    inspect_batch_recovery,
    load_operation_ledger,
    reverse_operation_batch,
    save_operation_ledger,
    validate_history_mutation_paths,
)
from nyxcore.action_plan.lock import LibraryMutationLock
from nyxcore.review_queue.state import load_review_state, save_review_state


def inspect_recorded_batch(
    ledger_path: Path,
    batch_id: str,
    *,
    library_root: Path,
    workspace_root: Path,
) -> tuple[OperationBatch, list[RecoveryAssessment]]:
    ledger = load_operation_ledger(ledger_path)
    batch = find_batch(ledger, batch_id)
    if batch is None:
        raise ValueError(f"History batch not found: {batch_id}")
    validate_history_mutation_paths(batch, allowed_roots=(library_root, workspace_root))
    return batch, inspect_batch_recovery(batch)


def recover_recorded_batch(
    *,
    ledger_path: Path,
    batch_id: str,
    library_root: Path,
    workspace_root: Path,
    action: str,
    stale_lock_token: str | None = None,
) -> tuple[OperationBatch, list[RecoveryAssessment]]:
    lock = LibraryMutationLock(library_root)
    lock.acquire(stale_owner_token=stale_lock_token)
    try:
        ledger = load_operation_ledger(ledger_path)
        batch = find_batch(ledger, batch_id)
        if batch is None:
            raise ValueError(f"History batch not found: {batch_id}")
        validate_history_mutation_paths(batch, allowed_roots=(library_root, workspace_root))
        assessments = apply_batch_recovery(batch, action=action)
        save_operation_ledger(ledger_path, ledger)
        return batch, assessments
    finally:
        lock.release()


def reverse_recorded_batch(
    *,
    ledger_path: Path,
    review_state_path: Path,
    batch_id: str,
    library_root: Path,
    workspace_root: Path,
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
    stale_lock_token: str | None = None,
) -> tuple[OperationBatch, list[LedgerOperation]]:
    lock = LibraryMutationLock(library_root)
    lock.acquire(stale_owner_token=stale_lock_token)
    try:
        ledger = load_operation_ledger(ledger_path)
        batch = find_batch(ledger, batch_id)
        if batch is None:
            raise ValueError(f"History batch not found: {batch_id}")
        review_state = load_review_state(review_state_path)
        changed = reverse_operation_batch(
            batch,
            review_state=review_state,
            allowed_roots=(library_root, workspace_root),
            alternate_restore_dir=alternate_restore_dir,
            target_path=target_path,
        )
        save_operation_ledger(ledger_path, ledger)
        save_review_state(review_state_path, review_state)
        return batch, changed
    finally:
        lock.release()
