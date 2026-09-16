from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nyxcore.core.atomic import atomic_write_text
from nyxcore.core.filesystem import move_file_no_replace
from nyxcore.review_queue.state import ReviewStateStore, apply_review_action

if TYPE_CHECKING:
    from nyxcore.action_plan.service import ActionPlanReport, AppliedOperationResult, AppliedPlanResult


@dataclass(slots=True)
class LedgerOperation:
    operation_id: str
    plan_id: str
    action_type: str
    source_review_item_ids: list[str]
    operation_type: str
    original_path: str | None
    current_path: str | None
    destination_path: str | None
    backup_path: str | None
    status: str
    message: str
    reversible: bool
    undo_status: str = "pending"
    undone_at: str | None = None
    undo_message: str | None = None
    expected_hash: str | None = None
    applied_hash: str | None = None
    journal_state: str = "legacy"
    prepared_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerOperation":
        return cls(
            operation_id=str(data["operation_id"]),
            plan_id=str(data["plan_id"]),
            action_type=str(data["action_type"]),
            source_review_item_ids=[str(item) for item in data.get("source_review_item_ids", [])],
            operation_type=str(data["operation_type"]),
            original_path=None if data.get("original_path") is None else str(data.get("original_path")),
            current_path=None if data.get("current_path") is None else str(data.get("current_path")),
            destination_path=None if data.get("destination_path") is None else str(data.get("destination_path")),
            backup_path=None if data.get("backup_path") is None else str(data.get("backup_path")),
            status=str(data.get("status", "")),
            message=str(data.get("message", "")),
            reversible=bool(data.get("reversible", False)),
            undo_status=str(data.get("undo_status", "pending")),
            undone_at=None if data.get("undone_at") is None else str(data.get("undone_at")),
            undo_message=None if data.get("undo_message") is None else str(data.get("undo_message")),
            expected_hash=data.get("expected_hash"),
            applied_hash=data.get("applied_hash"),
            journal_state=str(data.get("journal_state", "legacy")),
            prepared_at=None if data.get("prepared_at") is None else str(data.get("prepared_at")),
            started_at=None if data.get("started_at") is None else str(data.get("started_at")),
            completed_at=None if data.get("completed_at") is None else str(data.get("completed_at")),
            failed_at=None if data.get("failed_at") is None else str(data.get("failed_at")),
            failure_reason=None if data.get("failure_reason") is None else str(data.get("failure_reason")),
        )


@dataclass(slots=True)
class OperationBatch:
    batch_id: str
    applied_at: str
    source_plan_ids: list[str]
    source_review_item_ids: list[str]
    action_types: list[str]
    plan_result_statuses: dict[str, str]
    operations: list[LedgerOperation]
    lifecycle_state: str = "legacy"
    prepared_at: str | None = None
    completed_at: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "applied_at": self.applied_at,
            "source_plan_ids": list(self.source_plan_ids),
            "source_review_item_ids": list(self.source_review_item_ids),
            "action_types": list(self.action_types),
            "plan_result_statuses": dict(self.plan_result_statuses),
            "operations": [operation.to_dict() for operation in self.operations],
            "lifecycle_state": self.lifecycle_state,
            "prepared_at": self.prepared_at,
            "completed_at": self.completed_at,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperationBatch":
        return cls(
            batch_id=str(data["batch_id"]),
            applied_at=str(data["applied_at"]),
            source_plan_ids=[str(item) for item in data.get("source_plan_ids", [])],
            source_review_item_ids=[str(item) for item in data.get("source_review_item_ids", [])],
            action_types=[str(item) for item in data.get("action_types", [])],
            plan_result_statuses={str(key): str(value) for key, value in data.get("plan_result_statuses", {}).items()},
            operations=[LedgerOperation.from_dict(item) for item in data.get("operations", [])],
            lifecycle_state=str(data.get("lifecycle_state", "legacy")),
            prepared_at=None if data.get("prepared_at") is None else str(data.get("prepared_at")),
            completed_at=None if data.get("completed_at") is None else str(data.get("completed_at")),
            failure_reason=None if data.get("failure_reason") is None else str(data.get("failure_reason")),
        )


@dataclass(slots=True)
class OperationLedger:
    schema_version: int = 2
    batches: list[OperationBatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "batches": [batch.to_dict() for batch in self.batches],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperationLedger":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            batches=[OperationBatch.from_dict(item) for item in data.get("batches", [])],
        )


@dataclass(slots=True)
class RecoveryAssessment:
    batch_id: str
    operation_id: str
    operation_type: str
    journal_state: str
    classification: str
    safe_actions: list[str]
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_operation_ledger(path: Path) -> OperationLedger:
    if not path.exists():
        return OperationLedger()
    return OperationLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_operation_ledger(path: Path, ledger: OperationLedger) -> None:
    atomic_write_text(path, json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False))


def _batch_id(plan_ids: list[str], applied_at: str) -> str:
    payload = json.dumps([sorted(plan_ids), applied_at], separators=(",", ":"), ensure_ascii=False)
    return f"batch-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _operation_reversible(operation_type: str, backup_path: str | None) -> bool:
    if operation_type in {"quarantine_move", "rename_file"}:
        return True
    if operation_type == "write_metadata" and backup_path:
        return True
    return False


def _sanitize_path_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "restore"


def _alternate_restore_target(alternate_restore_dir: Path, original_path: str | None) -> Path:
    original = Path(original_path or "restored-file")
    digest = hashlib.sha1(str(original.parent).encode("utf-8")).hexdigest()[:10]
    folder = f"{_sanitize_path_label(original.parent.name or original.anchor or 'restore')}-{digest}"
    return alternate_restore_dir / folder / original.name


def append_operation_batch(
    ledger: OperationLedger,
    *,
    plan_report: ActionPlanReport,
    results: list[AppliedPlanResult],
    applied_at: datetime | None = None,
) -> OperationBatch:
    applied_at = applied_at or datetime.now(tz=UTC)
    applied_at_iso = applied_at.isoformat()
    plan_map = {plan.plan_id: plan for plan in plan_report.plans}
    operations: list[LedgerOperation] = []
    for result in results:
        plan = plan_map.get(result.plan_id)
        for operation in result.operation_results:
            planned_operation = None if plan is None else next(
                (item for item in plan.proposed_operations if item.operation_id == operation.operation_id),
                None,
            )
            operations.append(
                LedgerOperation(
                    operation_id=operation.operation_id,
                    plan_id=result.plan_id,
                    action_type=result.action_type,
                    source_review_item_ids=list(result.source_review_item_ids),
                    operation_type=operation.operation_type,
                    original_path=operation.path,
                    current_path=operation.destination_path if operation.status == "ok" and operation.destination_path else operation.path,
                    destination_path=operation.destination_path,
                    backup_path=operation.backup_path,
                    expected_hash=None if planned_operation is None else planned_operation.expected_hash,
                    applied_hash=operation.applied_hash,
                    status=operation.status,
                    message=operation.message,
                    reversible=_operation_reversible(operation.operation_type, operation.backup_path),
                    undo_status="pending" if _operation_reversible(operation.operation_type, operation.backup_path) else "not_supported",
                    journal_state="applied" if operation.status == "ok" else "failed",
                    completed_at=applied_at_iso if operation.status == "ok" else None,
                    failed_at=applied_at_iso if operation.status == "error" else None,
                    failure_reason=operation.message if operation.status == "error" else None,
                )
            )
    batch = OperationBatch(
        batch_id=_batch_id([result.plan_id for result in results], applied_at_iso),
        applied_at=applied_at_iso,
        source_plan_ids=[result.plan_id for result in results],
        source_review_item_ids=sorted({item_id for result in results for item_id in result.source_review_item_ids}),
        action_types=sorted({result.action_type for result in results}),
        plan_result_statuses={result.plan_id: result.status for result in results},
        operations=operations,
        lifecycle_state="applied" if all(result.status == "ok" for result in results) else "partial_failure",
        prepared_at=applied_at_iso,
        completed_at=applied_at_iso,
    )
    ledger.batches.append(batch)
    ledger.batches.sort(key=lambda item: item.applied_at)
    return batch


def prepare_operation_batch(
    ledger: OperationLedger,
    *,
    plan_report: "ActionPlanReport",
    planned_backup_paths: dict[str, str],
    prepared_at: datetime | None = None,
) -> OperationBatch:
    prepared_at = prepared_at or datetime.now(tz=UTC)
    prepared_at_iso = prepared_at.isoformat()
    operations: list[LedgerOperation] = []
    for plan in plan_report.plans:
        for operation in plan.proposed_operations:
            if not operation.apply_supported or operation.operation_type not in {
                "write_metadata", "rename_file", "quarantine_move"
            }:
                continue
            backup_path = planned_backup_paths.get(operation.operation_id)
            operations.append(LedgerOperation(
                operation_id=operation.operation_id,
                plan_id=plan.plan_id,
                action_type=plan.action_type,
                source_review_item_ids=list(plan.source_review_item_ids),
                operation_type=operation.operation_type,
                original_path=operation.path,
                current_path=operation.path,
                destination_path=operation.destination_path,
                backup_path=backup_path,
                status="prepared",
                message="durable intent recorded",
                reversible=_operation_reversible(operation.operation_type, backup_path),
                undo_status="pending",
                expected_hash=operation.expected_hash,
                journal_state="prepared",
                prepared_at=prepared_at_iso,
            ))
    batch = OperationBatch(
        batch_id=_batch_id([plan.plan_id for plan in plan_report.plans], prepared_at_iso),
        applied_at=prepared_at_iso,
        source_plan_ids=[plan.plan_id for plan in plan_report.plans],
        source_review_item_ids=sorted({
            item_id for plan in plan_report.plans for item_id in plan.source_review_item_ids
        }),
        action_types=sorted({plan.action_type for plan in plan_report.plans}),
        plan_result_statuses={plan.plan_id: "prepared" for plan in plan_report.plans},
        operations=operations,
        lifecycle_state="prepared",
        prepared_at=prepared_at_iso,
    )
    ledger.batches.append(batch)
    ledger.batches.sort(key=lambda item: item.applied_at)
    return batch


def mark_journal_operation_started(
    batch: OperationBatch,
    operation_id: str,
    *,
    expected_hash: str | None = None,
) -> None:
    operation = next(item for item in batch.operations if item.operation_id == operation_id)
    now = datetime.now(tz=UTC).isoformat()
    operation.journal_state = "applying"
    operation.status = "applying"
    operation.message = "filesystem mutation started"
    operation.started_at = now
    if expected_hash is not None:
        operation.expected_hash = expected_hash
    batch.lifecycle_state = "applying"


def mark_journal_operation_result(
    batch: OperationBatch,
    result: "AppliedOperationResult",
) -> None:
    operation = next(item for item in batch.operations if item.operation_id == result.operation_id)
    now = datetime.now(tz=UTC).isoformat()
    operation.status = result.status
    operation.message = result.message
    operation.backup_path = result.backup_path or operation.backup_path
    operation.applied_hash = result.applied_hash
    if result.status == "ok":
        operation.journal_state = "applied"
        operation.current_path = result.destination_path or result.path
        operation.completed_at = now
    else:
        operation.journal_state = "failed"
        operation.failed_at = now
        operation.failure_reason = result.message


def finalize_operation_batch(batch: OperationBatch, results: list["AppliedPlanResult"]) -> None:
    batch.plan_result_statuses = {result.plan_id: result.status for result in results}
    applied = [operation for operation in batch.operations if operation.journal_state == "applied"]
    failed = [operation for operation in batch.operations if operation.journal_state == "failed"]
    prepared = [operation for operation in batch.operations if operation.journal_state in {"prepared", "applying"}]
    if failed or prepared:
        batch.lifecycle_state = "partial_failure" if applied else "failed"
        reasons = [operation.failure_reason for operation in failed if operation.failure_reason]
        if prepared:
            reasons.append(f"{len(prepared)} operation(s) not completed")
        batch.failure_reason = "; ".join(reasons) or "batch did not complete"
    else:
        batch.lifecycle_state = "applied"
    batch.completed_at = datetime.now(tz=UTC).isoformat()


def find_batch(ledger: OperationLedger, batch_id: str) -> OperationBatch | None:
    for batch in ledger.batches:
        if batch.batch_id == batch_id:
            return batch
    return None


def _validate_bounded_path(path: Path, roots: tuple[Path, ...], *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    resolved_roots = tuple(root.resolve(strict=True) for root in roots)
    if not any(resolved == root or root in resolved.parents for root in resolved_roots):
        allowed = ", ".join(str(root) for root in resolved_roots)
        raise ValueError(f"{label} is outside the configured roots: {allowed}")
    return resolved


def validate_history_mutation_paths(
    batch: OperationBatch,
    *,
    allowed_roots: tuple[Path, ...],
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
) -> None:
    for operation in batch.operations:
        for label, value in (
            ("history original path", operation.original_path),
            ("history current path", operation.current_path),
            ("history backup path", operation.backup_path),
        ):
            if value is not None:
                _validate_bounded_path(Path(value), allowed_roots, label=label)
    if alternate_restore_dir is not None:
        _validate_bounded_path(alternate_restore_dir, allowed_roots, label="alternate restore directory")
    if target_path is not None:
        _validate_bounded_path(Path(target_path), allowed_roots, label="target path")


def _path_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assess_operation_recovery(batch: OperationBatch, operation: LedgerOperation) -> RecoveryAssessment:
    state = operation.journal_state
    if state in {"applied", "reversed"}:
        return RecoveryAssessment(
            batch.batch_id, operation.operation_id, operation.operation_type, state,
            "complete", [], "operation already has a final journal state",
        )
    source = Path(operation.original_path or "")
    source_hash = _path_hash(source)
    expected_hash = operation.expected_hash
    if operation.operation_type in {"rename_file", "quarantine_move"}:
        if not operation.original_path or not operation.destination_path or not expected_hash:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "manual_intervention", [], "move journal is missing a path or source fingerprint",
            )
        move_hash = operation.applied_hash or expected_hash
        destination = Path(operation.destination_path or "")
        destination_hash = _path_hash(destination)
        if source_hash == move_hash and destination_hash is None:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "not_started", ["abort"], "source is unchanged and destination is absent",
            )
        if source_hash is None and destination_hash == move_hash:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "completed_unfinalized", ["finalize"], "destination has expected content and source is absent",
            )
        if source_hash == move_hash and destination_hash == move_hash:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "destination_linked_source_present", ["finalize"],
                "source and destination both contain the expected content",
            )
        if destination_hash is not None and destination_hash != move_hash:
            detail = "destination contains unexpected content; no automatic recovery is safe"
        elif source_hash is None and destination_hash is None:
            detail = "source and destination are both missing"
        else:
            detail = "filesystem state does not match the recorded move intent"
        return RecoveryAssessment(
            batch.batch_id, operation.operation_id, operation.operation_type, state,
            "manual_intervention", [], detail,
        )

    if operation.operation_type == "write_metadata":
        if not operation.original_path or not expected_hash:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "manual_intervention", [], "metadata journal is missing its source path or fingerprint",
            )
        backup = Path(operation.backup_path or "")
        backup_hash = _path_hash(backup)
        if source_hash == expected_hash:
            classification = "backup_created_original_unchanged" if backup_hash == expected_hash else "not_started"
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                classification, ["abort"], "original file still matches the reviewed fingerprint",
            )
        if operation.applied_hash and source_hash == operation.applied_hash and backup_hash == expected_hash:
            return RecoveryAssessment(
                batch.batch_id, operation.operation_id, operation.operation_type, state,
                "completed_unfinalized", ["finalize"], "post-write and backup fingerprints match the journal",
            )
        if backup_hash is None:
            detail = "metadata changed but the recorded backup is missing"
        else:
            detail = "metadata differs from the reviewed source without a confirmed post-write fingerprint"
        return RecoveryAssessment(
            batch.batch_id, operation.operation_id, operation.operation_type, state,
            "manual_intervention", [], detail,
        )

    return RecoveryAssessment(
        batch.batch_id, operation.operation_id, operation.operation_type, state,
        "manual_intervention", [], "unsupported journaled operation type",
    )


def inspect_batch_recovery(batch: OperationBatch) -> list[RecoveryAssessment]:
    return [assess_operation_recovery(batch, operation) for operation in batch.operations]


def apply_batch_recovery(batch: OperationBatch, *, action: str) -> list[RecoveryAssessment]:
    if action not in {"finalize", "abort"}:
        raise ValueError("recovery action must be 'finalize' or 'abort'")
    assessments = inspect_batch_recovery(batch)
    by_id = {operation.operation_id: operation for operation in batch.operations}
    changed = False
    for assessment in assessments:
        if action not in assessment.safe_actions:
            continue
        operation = by_id[assessment.operation_id]
        now = datetime.now(tz=UTC).isoformat()
        if action == "abort":
            operation.journal_state = "failed"
            operation.status = "error"
            operation.failure_reason = "recovery confirmed mutation was not completed"
            operation.message = operation.failure_reason
            operation.failed_at = now
            changed = True
            continue
        if assessment.classification == "destination_linked_source_present":
            source = Path(operation.original_path or "")
            destination = Path(operation.destination_path or "")
            move_hash = operation.applied_hash or operation.expected_hash
            if _path_hash(source) != move_hash or _path_hash(destination) != move_hash:
                raise RuntimeError("move state changed during recovery; refusing to finalize")
            source.unlink()
        operation.journal_state = "applied"
        operation.status = "ok"
        operation.message = "recovered and finalized"
        operation.current_path = operation.destination_path or operation.original_path
        operation.completed_at = now
        operation.failure_reason = None
        changed = True
    if changed:
        applied = [operation for operation in batch.operations if operation.journal_state == "applied"]
        incomplete = [
            operation for operation in batch.operations
            if operation.journal_state in {"prepared", "applying"}
        ]
        failed = [operation for operation in batch.operations if operation.journal_state == "failed"]
        if incomplete:
            batch.lifecycle_state = "recovery_required"
        elif failed:
            batch.lifecycle_state = "partial_failure" if applied else "failed"
        else:
            batch.lifecycle_state = "applied"
        batch.completed_at = now
    return inspect_batch_recovery(batch)


def reverse_operation_batch(
    batch: OperationBatch,
    *,
    review_state: ReviewStateStore,
    allowed_roots: tuple[Path, ...],
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
    now: datetime | None = None,
) -> list[LedgerOperation]:
    validate_history_mutation_paths(
        batch,
        allowed_roots=allowed_roots,
        alternate_restore_dir=alternate_restore_dir,
        target_path=target_path,
    )
    now = now or datetime.now(tz=UTC)
    now_iso = now.isoformat()
    changed_operations: list[LedgerOperation] = []
    touched_review_item_ids: set[str] = set()

    for operation in reversed(batch.operations):
        if operation.status != "ok":
            continue
        if target_path is not None and target_path not in {operation.original_path, operation.current_path}:
            continue
        if not operation.reversible:
            operation.undo_status = "not_supported"
            operation.undo_message = "operation is not safely undoable"
            changed_operations.append(operation)
            continue
        if operation.undo_status == "ok":
            changed_operations.append(operation)
            continue
        try:
            if operation.operation_type in {"quarantine_move", "rename_file"}:
                source = Path(operation.current_path or "")
                destination = Path(operation.original_path or "")
                if not source.exists():
                    raise RuntimeError(f"current path does not exist: {source}")
                if not operation.expected_hash:
                    raise RuntimeError("Cannot verify legacy move reversal; current file left untouched")
                move_hash = operation.applied_hash or operation.expected_hash
                if _path_hash(source) != move_hash:
                    raise RuntimeError("Current file changed after move; reversal would overwrite newer content")
                if destination.exists():
                    if alternate_restore_dir is None:
                        raise RuntimeError(f"restore destination already exists: {destination}")
                    destination = _alternate_restore_target(alternate_restore_dir, operation.original_path)
                    if destination.exists():
                        raise RuntimeError(f"alternate restore destination already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                move_file_no_replace(source, destination)
                operation.current_path = str(destination)
            elif operation.operation_type == "write_metadata" and operation.backup_path:
                backup = Path(operation.backup_path)
                destination = Path(operation.original_path or "")
                if not backup.exists():
                    raise RuntimeError(f"backup path does not exist: {backup}")
                if destination.exists():
                    if not operation.applied_hash:
                        raise RuntimeError("Cannot verify legacy metadata restore; original file left untouched")
                    with destination.open("rb") as stream:
                        current_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                    if current_hash != operation.applied_hash:
                        raise RuntimeError("File changed after metadata edit; restore would overwrite newer content")
                shutil.copy2(backup, destination)
                operation.current_path = str(destination)
            else:
                raise RuntimeError("undo is not supported for this operation")
            operation.undo_status = "ok"
            operation.journal_state = "reversed"
            operation.undone_at = now_iso
            operation.undo_message = "undone"
            touched_review_item_ids.update(operation.source_review_item_ids)
        except Exception as exc:
            operation.undo_status = "error"
            operation.undone_at = now_iso
            operation.undo_message = str(exc)
        changed_operations.append(operation)

    tracked_item_ids = sorted(item_id for item_id in touched_review_item_ids if item_id in review_state.items)
    if tracked_item_ids and any(operation.undo_status == "ok" for operation in changed_operations):
        apply_review_action(review_state, item_ids=tracked_item_ids, status="seen", now=now)
    reversible_applied = [
        operation for operation in batch.operations
        if operation.status == "ok" and operation.reversible
    ]
    if reversible_applied and all(operation.undo_status == "ok" for operation in reversible_applied):
        batch.lifecycle_state = "reversed"
        batch.completed_at = now_iso
    return changed_operations


def undo_operation_batch(
    batch: OperationBatch,
    *,
    review_state: ReviewStateStore,
    allowed_roots: tuple[Path, ...],
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
    now: datetime | None = None,
) -> list[LedgerOperation]:
    """Compatibility wrapper for the canonical reversal implementation."""

    return reverse_operation_batch(
        batch,
        review_state=review_state,
        allowed_roots=allowed_roots,
        alternate_restore_dir=alternate_restore_dir,
        target_path=target_path,
        now=now,
    )
