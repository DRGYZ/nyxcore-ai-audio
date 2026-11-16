from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from nyxcore.core.atomic import atomic_write_text
from nyxcore.core.filesystem import (
    FileIdentity,
    bounded_path,
    file_snapshot,
    finish_linked_move,
    move_file_no_replace,
    path_present,
    regular_file_identity,
)
from nyxcore.review_queue.state import ReviewStateStore, apply_review_action

if TYPE_CHECKING:
    from nyxcore.action_plan.service import ActionPlanReport, AppliedOperationResult, AppliedPlanResult

def _read_identity(value: list[int] | FileIdentity | None) -> FileIdentity | None:
    if value is None:
        return None
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(part) is not int for part in value) or value[0] < 0 or value[1] <= 0):
        raise ValueError("invalid recorded file identity")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class ReversalIntent:
    destination_path: str
    source_identity: FileIdentity
    staging_path: str | None = None
    staging_identity: FileIdentity | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ReversalIntent":
        identity = _read_identity(data["source_identity"])
        if identity is None:
            raise ValueError("reversal intent is missing its source identity")
        return cls(
            destination_path=str(data["destination_path"]),
            source_identity=identity,
            staging_path=data.get("staging_path"),
            staging_identity=_read_identity(data.get("staging_identity")),
        )


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
    move_identity: FileIdentity | None = None
    reversal_intent: ReversalIntent | None = None

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
            move_identity=_read_identity(data.get("move_identity")),
            reversal_intent=(
                ReversalIntent.from_dict(data["reversal_intent"])
                if data.get("reversal_intent") is not None else None
            ),
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
    move_identity: FileIdentity | None = None,
) -> None:
    operation = next(item for item in batch.operations if item.operation_id == operation_id)
    now = datetime.now(tz=UTC).isoformat()
    operation.journal_state = "applying"
    operation.status = "applying"
    operation.message = "filesystem mutation started"
    operation.started_at = now
    if expected_hash is not None:
        operation.expected_hash = expected_hash
    operation.move_identity = move_identity
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


def batch_requires_recovery(batch: OperationBatch) -> bool:
    unfinished = {"prepared", "applying"}
    return (
        batch.lifecycle_state in unfinished | {"recovery_required"}
        or any(
            operation.journal_state in unfinished or operation.status in unfinished
            or operation.reversal_intent is not None
            for operation in batch.operations
        )
    )


def _history_file(value: str, roots: tuple[Path, ...], *, label: str) -> Path:
    path = Path(value)
    bounded_path(path, roots, label=label)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    return path


_MOVE_TYPES = {"rename_file", "quarantine_move"}


def validate_history_mutation_paths(
    batch: OperationBatch,
    *,
    allowed_roots: tuple[Path, ...],
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
) -> None:
    # In recorded workflows the first root is the locked library. Backups
    # and explicitly chosen restore locations may also be in the workspace.
    library = allowed_roots[:1]
    if alternate_restore_dir is not None:
        bounded_path(alternate_restore_dir, allowed_roots, label="alternate restore directory")
    if target_path is not None:
        bounded_path(Path(target_path), allowed_roots, label="target path")
    for operation in batch.operations:
        current_roots = allowed_roots if operation.undo_status == "ok" else library
        for label, value, roots in (
            ("history original path", operation.original_path, library),
            ("history current path", operation.current_path, current_roots),
            ("history destination path", operation.destination_path, library),
            ("history backup path", operation.backup_path, allowed_roots),
        ):
            if value is not None:
                _history_file(value, roots, label=label)
        intent = operation.reversal_intent
        if intent is not None:
            destination = _history_file(intent.destination_path, allowed_roots, label="recorded reversal destination")
            if operation.operation_type == "write_metadata" and str(destination) != operation.original_path:
                raise ValueError("metadata reversal destination must match the original path")
            if intent.staging_path is not None:
                stage = _history_file(intent.staging_path, library, label="metadata restore staging path")
                if stage.parent != destination.parent or not stage.name.startswith(".nyxcore-restore-"):
                    raise ValueError("metadata restore staging file must be beside the destination")
        elif (alternate_restore_dir is not None and operation.operation_type in _MOVE_TYPES
              and operation.original_path and path_present(Path(operation.original_path))
              and operation.status == "ok" and operation.undo_status != "ok"
              and target_path in (None, operation.original_path, operation.current_path)):
            destination = _alternate_restore_target(alternate_restore_dir, operation.original_path)
            _history_file(str(destination), allowed_roots, label="alternate restore destination")


def _move_state(
    source: Path, destination: Path, expected_hash: str, identity: FileIdentity,
) -> tuple[str, str]:
    if source.resolve() == destination.resolve():
        return "manual_intervention", "move source and destination resolve to the same path"
    expected = (identity, expected_hash)
    left, right = file_snapshot(source), file_snapshot(destination)
    if left == expected and not path_present(destination):
        return "not_started", "source matches the recorded file and destination is absent"
    if right == expected and not path_present(source):
        return "completed_unfinalized", "destination matches the recorded file and source is absent"
    if left == right == expected:
        return "destination_linked_source_present", "both names refer to the recorded regular file"
    if not path_present(source) and not path_present(destination):
        return "manual_intervention", "source and destination are both missing"
    return "manual_intervention", "move paths lack matching file identity and fingerprints"


def _reversal_state(operation: LedgerOperation, intent: ReversalIntent) -> tuple[str, str]:
    destination = Path(intent.destination_path)
    if operation.operation_type in _MOVE_TYPES and operation.current_path:
        expected = operation.applied_hash or operation.expected_hash
        if expected:
            return _move_state(Path(operation.current_path), destination, expected, intent.source_identity)
    elif (operation.operation_type == "write_metadata" and intent.staging_path
          and intent.staging_identity and operation.expected_hash and operation.applied_hash):
        stage = Path(intent.staging_path)
        restored = (intent.staging_identity, operation.expected_hash)
        live = file_snapshot(destination)
        if live == restored and not path_present(stage):
            return "completed_unfinalized", "verified staged file is now at the live path"
        if live == (intent.source_identity, operation.applied_hash) and file_snapshot(stage) == restored:
            return "not_started", "live file and verified staging file still match the reversal intent"
    return "manual_intervention", "reversal paths or fingerprints do not match the recorded intent"


def assess_operation_recovery(batch: OperationBatch, operation: LedgerOperation) -> RecoveryAssessment:
    state = operation.journal_state

    def assessment(classification: str, actions: list[str], detail: str) -> RecoveryAssessment:
        return RecoveryAssessment(
            batch.batch_id, operation.operation_id, operation.operation_type, state, classification, actions, detail,
        )

    if operation.reversal_intent is not None:
        classification, detail = _reversal_state(operation, operation.reversal_intent)
        if classification == "manual_intervention":
            return assessment(classification, [], detail)
        actions = [] if classification == "not_started" else ["finalize"]
        return assessment(f"reversal_{classification}", actions, detail)
    if state in {"applied", "reversed"} or (state == "legacy" and operation.status == "ok"):
        return assessment("complete", [], "operation already has a final journal state")
    if not operation.original_path or not operation.expected_hash:
        return assessment("manual_intervention", [], "journal is missing a source path or fingerprint")
    source = Path(operation.original_path)
    snapshot = file_snapshot(source)
    unchanged = snapshot is not None and snapshot[1] == operation.expected_hash
    abort_actions = ["abort"] if state in {"prepared", "applying"} else []

    if operation.operation_type in _MOVE_TYPES:
        if not operation.destination_path:
            return assessment("manual_intervention", [], "move journal is missing its destination")
        destination = Path(operation.destination_path)
        if unchanged and not path_present(destination):
            if operation.move_identity is None or snapshot[0] == operation.move_identity:
                return assessment("not_started", abort_actions, "source is unchanged and destination is absent")
        if not path_present(source) and not path_present(destination):
            return assessment("manual_intervention", [], "source and destination are both missing")
        if state != "applying" or not operation.started_at or operation.move_identity is None:
            return assessment("manual_intervention", [], "move journal lacks evidence that this move created the destination")
        classification, detail = _move_state(source, destination, operation.expected_hash, operation.move_identity)
        actions = ["finalize"] if classification in {"completed_unfinalized", "destination_linked_source_present"} else []
        return assessment(classification, actions, detail)

    if operation.operation_type == "write_metadata":
        backup = file_snapshot(Path(operation.backup_path)) if operation.backup_path else None
        valid_backup = backup is not None and backup[1] == operation.expected_hash
        if unchanged:
            classification = "backup_created_original_unchanged" if valid_backup else "not_started"
            return assessment(classification, abort_actions, "original file still matches the reviewed fingerprint")
        if (state == "applying" and operation.started_at and operation.applied_hash
                and snapshot is not None and snapshot[1] == operation.applied_hash and valid_backup):
            return assessment("completed_unfinalized", ["finalize"], "post-write and backup fingerprints match the journal")
        detail = ("metadata changed but the recorded backup is missing" if backup is None
                  else "metadata or backup differs without a confirmed post-write fingerprint")
        return assessment("manual_intervention", [], detail)

    return assessment("manual_intervention", [], "unsupported journaled operation type")


def inspect_batch_recovery(batch: OperationBatch) -> list[RecoveryAssessment]:
    return [assess_operation_recovery(batch, operation) for operation in batch.operations]


def _refresh_recovered_batch(batch: OperationBatch, now: str) -> None:
    if not batch.operations:
        return
    reversible = [op for op in batch.operations if op.status == "ok" and op.reversible]
    complete = {"applied", "reversed"}
    if any(op.reversal_intent or op.journal_state in {"prepared", "applying"} for op in batch.operations):
        state = "recovery_required"
    elif reversible and all(op.undo_status == "ok" for op in reversible):
        state = "reversed"
    elif any(op.journal_state == "failed" for op in batch.operations):
        state = "partial_failure" if any(op.journal_state in complete for op in batch.operations) else "failed"
    elif all(op.journal_state in complete for op in batch.operations):
        state = "applied"
    else:
        return  # Preserve summaries of old batches without journal states.

    if batch.lifecycle_state != state:
        batch.lifecycle_state = state
        batch.completed_at = None if state == "recovery_required" else now
    if state in {"applied", "reversed"}:
        batch.failure_reason = None
    for plan_id in batch.plan_result_statuses:
        operations = [op for op in batch.operations if op.plan_id == plan_id
                      and op.operation_type in _MOVE_TYPES | {"write_metadata"}]
        if operations and all(op.journal_state in complete for op in operations):
            batch.plan_result_statuses[plan_id] = "ok"
        elif any(op.journal_state == "failed" for op in operations):
            batch.plan_result_statuses[plan_id] = "partial_failure"


def _finish_reversal(operation: LedgerOperation, now: str) -> None:
    intent = operation.reversal_intent
    if intent is None:
        raise RuntimeError("missing reversal intent")
    operation.current_path = intent.destination_path
    operation.reversal_intent = None
    operation.undo_status = "ok"
    operation.journal_state = "reversed"
    operation.undone_at = now
    operation.undo_message = "undone"


def apply_batch_recovery(
    batch: OperationBatch, *, action: str, allowed_roots: tuple[Path, ...] | None = None,
) -> list[RecoveryAssessment]:
    if action not in {"finalize", "abort"}:
        raise ValueError("recovery action must be 'finalize' or 'abort'")
    if allowed_roots is not None:
        validate_history_mutation_paths(batch, allowed_roots=allowed_roots)
    now = datetime.now(tz=UTC).isoformat()
    for operation in batch.operations:
        # Assess each entry when it is reached. An earlier operation may have
        # changed a path referenced by a later entry in the same batch.
        assessment = assess_operation_recovery(batch, operation)
        if action not in assessment.safe_actions:
            continue
        if action == "abort":
            operation.journal_state = "failed"
            operation.status = "error"
            operation.failure_reason = "recovery confirmed mutation was not completed"
            operation.message = operation.failure_reason
            operation.failed_at = now
            continue

        if allowed_roots is not None:
            validate_history_mutation_paths(batch, allowed_roots=allowed_roots)
        if assessment != assess_operation_recovery(batch, operation):
            raise RuntimeError("filesystem state changed during recovery; refusing to finalize")
        intent = operation.reversal_intent
        if intent is not None:
            if assessment.classification == "reversal_destination_linked_source_present":
                finish_linked_move(
                    Path(operation.current_path), Path(intent.destination_path), intent.source_identity,
                    expected_hash=operation.applied_hash or operation.expected_hash,
                )
            _finish_reversal(operation, now)
            continue
        if assessment.classification == "destination_linked_source_present":
            finish_linked_move(
                Path(operation.original_path), Path(operation.destination_path), operation.move_identity,
                expected_hash=operation.expected_hash,
            )
        if operation.operation_type in _MOVE_TYPES:
            operation.applied_hash = operation.expected_hash
        operation.journal_state = "applied"
        operation.status = "ok"
        operation.message = "recovered and finalized"
        operation.current_path = operation.destination_path or operation.original_path
        operation.completed_at = now
        operation.failure_reason = None
    _refresh_recovered_batch(batch, now)
    return inspect_batch_recovery(batch)


def _verify_backup(operation: LedgerOperation) -> Path:
    if not operation.backup_path or not operation.expected_hash:
        raise RuntimeError("metadata restore requires a backup and its original fingerprint")
    backup = Path(operation.backup_path)
    snapshot = file_snapshot(backup)
    if snapshot is None or snapshot[1] != operation.expected_hash:
        raise RuntimeError("metadata backup is missing, invalid, or differs from its recorded fingerprint")
    return backup


def _stage_metadata_restore(backup: Path, destination: Path, expected_hash: str) -> tuple[Path, FileIdentity]:
    stage = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".nyxcore-restore-",
                                         suffix=".tmp", delete=False) as output:
            stage = Path(output.name)
            with backup.open("rb") as original:
                shutil.copyfileobj(original, output)
            output.flush()
            shutil.copystat(backup, stage)
            os.fsync(output.fileno())
        snapshot = file_snapshot(stage)
        if snapshot is None or snapshot[1] != expected_hash:
            raise RuntimeError("staged metadata restore differs from the original fingerprint")
        return stage, snapshot[0]
    except BaseException:
        if stage is not None:
            stage.unlink(missing_ok=True)
        raise


def _prepare_reversal(
    operation: LedgerOperation, allowed_roots: tuple[Path, ...], alternate_restore_dir: Path | None,
) -> ReversalIntent:
    if not operation.original_path or not operation.current_path:
        raise RuntimeError("reversal is missing its original or current path")
    source = Path(operation.current_path)
    destination = Path(operation.original_path)
    snapshot = file_snapshot(source)
    if operation.operation_type in _MOVE_TYPES:
        if not operation.expected_hash:
            raise RuntimeError("cannot verify legacy move reversal without the original fingerprint")
        expected_hash = operation.applied_hash or operation.expected_hash
        if not expected_hash or snapshot is None or snapshot[1] != expected_hash:
            raise RuntimeError("current file is missing or changed after the move")
        if operation.move_identity is not None and snapshot[0] != operation.move_identity:
            raise RuntimeError("current file identity changed after the move")
        if path_present(destination):
            if alternate_restore_dir is None:
                raise RuntimeError(f"restore destination already exists: {destination}")
            destination = _alternate_restore_target(alternate_restore_dir, operation.original_path)
        _history_file(str(destination), allowed_roots, label="restore destination")
        if path_present(destination):
            raise RuntimeError(f"restore destination already exists: {destination}")
        return ReversalIntent(str(destination), snapshot[0])

    if operation.operation_type == "write_metadata":
        if (source != destination or not operation.applied_hash or snapshot is None
                or snapshot[1] != operation.applied_hash):
            raise RuntimeError("File is missing or changed after metadata edit; restore could overwrite newer content")
        backup = _verify_backup(operation)
        stage, identity = _stage_metadata_restore(backup, destination, operation.expected_hash)
        return ReversalIntent(str(destination), snapshot[0], str(stage), identity)
    raise RuntimeError("undo is not supported for this operation")


def _continue_reversal(operation: LedgerOperation, intent: ReversalIntent) -> None:
    classification, detail = _reversal_state(operation, intent)
    if classification == "completed_unfinalized":
        return
    destination = Path(intent.destination_path)
    if operation.operation_type in _MOVE_TYPES:
        source = Path(operation.current_path)
        expected_hash = operation.applied_hash or operation.expected_hash
        if classification == "not_started":
            move_file_no_replace(source, destination, expected_identity=intent.source_identity, expected_hash=expected_hash)
            return
        if classification == "destination_linked_source_present":
            finish_linked_move(source, destination, intent.source_identity, expected_hash=expected_hash)
            return
    elif operation.operation_type == "write_metadata" and classification == "not_started":
        _verify_backup(operation)
        if _reversal_state(operation, intent)[0] != "not_started":
            raise RuntimeError("metadata reversal paths changed; refusing to replace live file")
        os.replace(intent.staging_path, destination)
        return
    raise RuntimeError(f"reversal requires manual intervention: {detail}")


def reverse_operation_batch(
    batch: OperationBatch,
    *,
    review_state: ReviewStateStore,
    allowed_roots: tuple[Path, ...],
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
    now: datetime | None = None,
    persist: Callable[[], None] | None = None,
) -> list[LedgerOperation]:
    """Reverse a batch, persisting each intent before editing library files.

    Recorded callers supply persist. The lower-level compatibility helper
    with an in-memory batch cannot provide restart recovery.
    """
    def validate_paths() -> None:
        validate_history_mutation_paths(
            batch, allowed_roots=allowed_roots,
            alternate_restore_dir=alternate_restore_dir, target_path=target_path,
        )

    validate_paths()
    now = now or datetime.now(tz=UTC)
    now_iso = now.isoformat()
    changed_operations: list[LedgerOperation] = []
    touched_review_item_ids: set[str] = set()
    failed_paths: set[Path] = set()

    for operation in reversed(batch.operations):
        if operation.status != "ok":
            continue
        if target_path is not None and target_path not in {operation.original_path, operation.current_path}:
            if operation.reversal_intent is not None:
                break
            continue
        changed_operations.append(operation)
        if not operation.reversible:
            operation.undo_status = "not_supported"
            operation.undo_message = "operation is not safely undoable"
            continue
        if operation.undo_status == "ok":
            touched_review_item_ids.update(operation.source_review_item_ids)
            continue

        validate_paths()
        paths = {Path(path).resolve()
                 for path in (operation.original_path, operation.current_path, operation.destination_path) if path}
        try:
            if paths & failed_paths:
                raise RuntimeError("a related reversal failed; retry it before undoing earlier changes")
            intent = operation.reversal_intent
            if intent is None:
                if any(op.operation_type in _MOVE_TYPES and op.undo_status == "ok"
                       and op.original_path == operation.original_path and op.current_path != op.original_path
                       for op in changed_operations if op is not operation):
                    raise RuntimeError("file was restored to an alternate location; earlier reversal needs manual review")
                intent = _prepare_reversal(operation, allowed_roots, alternate_restore_dir)
                operation.reversal_intent = intent
        except (OSError, RuntimeError) as exc:
            operation.undo_status = "error"
            operation.undone_at = now_iso
            operation.undo_message = str(exc)
            if operation.reversal_intent is not None:
                break
            failed_paths.update(paths)
            continue

        if persist is not None:
            persist()
        try:
            validate_paths()
            _continue_reversal(operation, intent)
        except (OSError, RuntimeError) as exc:
            operation.undo_status = "error"
            operation.undone_at = now_iso
            operation.undo_message = str(exc)
            break

        _finish_reversal(operation, now_iso)
        touched_review_item_ids.update(operation.source_review_item_ids)
        if persist is not None:
            persist()

    tracked = sorted(item_id for item_id in touched_review_item_ids if item_id in review_state.items)
    if tracked:
        apply_review_action(review_state, item_ids=tracked, status="seen", now=now)
    _refresh_recovered_batch(batch, now_iso)
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
