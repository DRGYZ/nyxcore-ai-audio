from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nyxcore.action_plan.service import ActionPlanReport, AppliedPlanResult
from nyxcore.review_queue.state import ReviewStateStore, apply_review_action


@dataclass(slots=True)
class LedgerOperation:
    operation_id: str
    plan_id: str
    action_type: str
    source_review_item_ids: list[str]
    operation_type: str
    original_path: str | None
    current_path: str | None
    backup_path: str | None
    status: str
    message: str
    reversible: bool
    undo_status: str = "pending"
    undone_at: str | None = None
    undo_message: str | None = None

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
            backup_path=None if data.get("backup_path") is None else str(data.get("backup_path")),
            status=str(data.get("status", "")),
            message=str(data.get("message", "")),
            reversible=bool(data.get("reversible", False)),
            undo_status=str(data.get("undo_status", "pending")),
            undone_at=None if data.get("undone_at") is None else str(data.get("undone_at")),
            undo_message=None if data.get("undo_message") is None else str(data.get("undo_message")),
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

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "applied_at": self.applied_at,
            "source_plan_ids": list(self.source_plan_ids),
            "source_review_item_ids": list(self.source_review_item_ids),
            "action_types": list(self.action_types),
            "plan_result_statuses": dict(self.plan_result_statuses),
            "operations": [operation.to_dict() for operation in self.operations],
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
        )


@dataclass(slots=True)
class OperationLedger:
    schema_version: int = 1
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


def load_operation_ledger(path: Path) -> OperationLedger:
    if not path.exists():
        return OperationLedger()
    return OperationLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_operation_ledger(path: Path, ledger: OperationLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


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
            operations.append(
                LedgerOperation(
                    operation_id=operation.operation_id,
                    plan_id=result.plan_id,
                    action_type=result.action_type,
                    source_review_item_ids=list(result.source_review_item_ids),
                    operation_type=operation.operation_type,
                    original_path=operation.path,
                    current_path=operation.destination_path if operation.status == "ok" and operation.destination_path else operation.path,
                    backup_path=operation.backup_path,
                    status=operation.status,
                    message=operation.message,
                    reversible=_operation_reversible(operation.operation_type, operation.backup_path),
                    undo_status="pending" if _operation_reversible(operation.operation_type, operation.backup_path) else "not_supported",
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
    )
    ledger.batches.append(batch)
    ledger.batches.sort(key=lambda item: item.applied_at)
    return batch


def find_batch(ledger: OperationLedger, batch_id: str) -> OperationBatch | None:
    for batch in ledger.batches:
        if batch.batch_id == batch_id:
            return batch
    return None


def undo_operation_batch(
    batch: OperationBatch,
    *,
    review_state: ReviewStateStore,
    alternate_restore_dir: Path | None = None,
    target_path: str | None = None,
    now: datetime | None = None,
) -> list[LedgerOperation]:
    now = now or datetime.now(tz=UTC)
    now_iso = now.isoformat()
    changed_operations: list[LedgerOperation] = []
    touched_review_item_ids: set[str] = set()

    for operation in batch.operations:
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
                if destination.exists():
                    if alternate_restore_dir is None:
                        raise RuntimeError(f"restore destination already exists: {destination}")
                    destination = _alternate_restore_target(alternate_restore_dir, operation.original_path)
                    if destination.exists():
                        raise RuntimeError(f"alternate restore destination already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                operation.current_path = str(destination)
            elif operation.operation_type == "write_metadata" and operation.backup_path:
                backup = Path(operation.backup_path)
                destination = Path(operation.original_path or "")
                if not backup.exists():
                    raise RuntimeError(f"backup path does not exist: {backup}")
                shutil.copy2(backup, destination)
                operation.current_path = str(destination)
            else:
                raise RuntimeError("undo is not supported for this operation")
            operation.undo_status = "ok"
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
    return changed_operations
