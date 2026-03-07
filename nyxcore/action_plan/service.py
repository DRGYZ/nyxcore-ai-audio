from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nyxcore.core.track import TrackRecord
from nyxcore.normalize.parser import NormalizePreviewRecord, build_normalize_preview_for_paths
from nyxcore.rename.rules import deterministic_cleanup
from nyxcore.rename.service import apply_rename, build_rename_result
from nyxcore.review_queue.service import ReviewQueueItem, ReviewQueueReport
from nyxcore.review_queue.state import ReviewStateStore, apply_review_action
from nyxcore.tagging.writer import TagWriteError, backup_file, write_tags

SUPPORTED_ACTION_TYPES = {
    "exact_duplicate_keep_plan",
    "metadata_fix_plan",
    "rename_normalize_plan",
    "artwork_gap_plan",
}


def _plan_id(action_type: str, source_review_item_ids: list[str]) -> str:
    payload = json.dumps([action_type, sorted(source_review_item_ids)], separators=(",", ":"), ensure_ascii=False)
    return f"{action_type}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _sanitize_path_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "external"


def _quarantine_destination(root: Path, candidate_path: Path) -> Path:
    quarantine_root = root / ".nyxcore_quarantine"
    if candidate_path.is_absolute() and (candidate_path == root or root in candidate_path.parents):
        return quarantine_root / candidate_path.parent.relative_to(root) / candidate_path.name
    digest = hashlib.sha1(str(candidate_path.parent).encode("utf-8")).hexdigest()[:10]
    parent_label = _sanitize_path_label(candidate_path.parent.name or candidate_path.anchor or "external")
    return quarantine_root / "_external" / f"{parent_label}-{digest}" / candidate_path.name


@dataclass(slots=True)
class ActionPlanOperation:
    operation_id: str
    operation_type: str
    path: str | None = None
    destination_path: str | None = None
    fields: list[str] = field(default_factory=list)
    values: dict[str, str | None] = field(default_factory=dict)
    apply_supported: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionPlanOperation":
        return cls(
            operation_id=str(data["operation_id"]),
            operation_type=str(data["operation_type"]),
            path=None if data.get("path") is None else str(data.get("path")),
            destination_path=None if data.get("destination_path") is None else str(data.get("destination_path")),
            fields=[str(item) for item in data.get("fields", [])],
            values={str(key): (None if value is None else str(value)) for key, value in data.get("values", {}).items()},
            apply_supported=bool(data.get("apply_supported", False)),
            notes=[str(item) for item in data.get("notes", [])],
        )


@dataclass(slots=True)
class ActionPlan:
    plan_id: str
    source_review_item_ids: list[str]
    action_type: str
    affected_files: list[str]
    proposed_operations: list[ActionPlanOperation]
    confidence: float
    safety_level: str
    reasons: list[str]
    notes: list[str]
    apply_supported: bool
    resolves_review_items: bool = False

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "source_review_item_ids": list(self.source_review_item_ids),
            "action_type": self.action_type,
            "affected_files": list(self.affected_files),
            "proposed_operations": [operation.to_dict() for operation in self.proposed_operations],
            "confidence": self.confidence,
            "safety_level": self.safety_level,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
            "apply_supported": self.apply_supported,
            "resolves_review_items": self.resolves_review_items,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionPlan":
        return cls(
            plan_id=str(data["plan_id"]),
            source_review_item_ids=[str(item) for item in data.get("source_review_item_ids", [])],
            action_type=str(data["action_type"]),
            affected_files=[str(item) for item in data.get("affected_files", [])],
            proposed_operations=[ActionPlanOperation.from_dict(item) for item in data.get("proposed_operations", [])],
            confidence=float(data.get("confidence", 0.0)),
            safety_level=str(data.get("safety_level", "")),
            reasons=[str(item) for item in data.get("reasons", [])],
            notes=[str(item) for item in data.get("notes", [])],
            apply_supported=bool(data.get("apply_supported", False)),
            resolves_review_items=bool(data.get("resolves_review_items", False)),
        )


@dataclass(slots=True)
class UnsupportedActionPlan:
    source_review_item_id: str
    item_type: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UnsupportedActionPlan":
        return cls(
            source_review_item_id=str(data["source_review_item_id"]),
            item_type=str(data["item_type"]),
            reason=str(data["reason"]),
        )


@dataclass(slots=True)
class ActionPlanSummary:
    requested_item_count: int
    generated_plan_count: int
    unsupported_item_count: int
    apply_supported_plan_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionPlanSummary":
        return cls(
            requested_item_count=int(data.get("requested_item_count", 0)),
            generated_plan_count=int(data.get("generated_plan_count", 0)),
            unsupported_item_count=int(data.get("unsupported_item_count", 0)),
            apply_supported_plan_count=int(data.get("apply_supported_plan_count", 0)),
        )


@dataclass(slots=True)
class ActionPlanReport:
    created_at: str
    source_review_item_ids: list[str]
    plans: list[ActionPlan]
    unsupported_items: list[UnsupportedActionPlan]
    summary: ActionPlanSummary

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "source_review_item_ids": list(self.source_review_item_ids),
            "plans": [plan.to_dict() for plan in self.plans],
            "unsupported_items": [item.to_dict() for item in self.unsupported_items],
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionPlanReport":
        return cls(
            created_at=str(data.get("created_at", "")),
            source_review_item_ids=[str(item) for item in data.get("source_review_item_ids", [])],
            plans=[ActionPlan.from_dict(item) for item in data.get("plans", [])],
            unsupported_items=[UnsupportedActionPlan.from_dict(item) for item in data.get("unsupported_items", [])],
            summary=ActionPlanSummary.from_dict(data.get("summary", {})),
        )


@dataclass(slots=True)
class AppliedOperationResult:
    operation_id: str
    operation_type: str
    path: str | None
    destination_path: str | None
    status: str
    message: str
    backup_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AppliedPlanResult:
    plan_id: str
    action_type: str
    status: str
    source_review_item_ids: list[str]
    operation_results: list[AppliedOperationResult]
    resolved_review_item_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "action_type": self.action_type,
            "status": self.status,
            "source_review_item_ids": list(self.source_review_item_ids),
            "operation_results": [item.to_dict() for item in self.operation_results],
            "resolved_review_item_ids": list(self.resolved_review_item_ids),
        }


class ActionPlanBuilder:
    def __init__(self, root: Path, records: list[TrackRecord], review_report: ReviewQueueReport) -> None:
        self.root = root
        self.records_by_path = {record.path: record for record in records}
        self.items_by_id = {item.item_id: item for item in review_report.items}

    def build(self, source_review_item_ids: list[str]) -> ActionPlanReport:
        plans: list[ActionPlan] = []
        unsupported: list[UnsupportedActionPlan] = []
        for item_id in source_review_item_ids:
            item = self.items_by_id.get(item_id)
            if item is None:
                unsupported.append(
                    UnsupportedActionPlan(
                        source_review_item_id=item_id,
                        item_type="unknown",
                        reason="review item not found in current queue output",
                    )
                )
                continue
            item_plans, item_unsupported = self._plans_for_item(item)
            plans.extend(item_plans)
            unsupported.extend(item_unsupported)
        return ActionPlanReport(
            created_at=datetime.now(tz=UTC).isoformat(),
            source_review_item_ids=list(source_review_item_ids),
            plans=plans,
            unsupported_items=unsupported,
            summary=ActionPlanSummary(
                requested_item_count=len(source_review_item_ids),
                generated_plan_count=len(plans),
                unsupported_item_count=len(unsupported),
                apply_supported_plan_count=sum(1 for plan in plans if plan.apply_supported),
            ),
        )

    def _plans_for_item(self, item: ReviewQueueItem) -> tuple[list[ActionPlan], list[UnsupportedActionPlan]]:
        if item.item_type == "exact_duplicate_group":
            return [self._build_exact_duplicate_plan(item)], []
        if item.item_type in {"missing_metadata", "weak_or_placeholder_metadata"}:
            return self._build_metadata_related_plans(item), []
        if item.item_type == "artwork_missing":
            return [self._build_artwork_gap_plan(item)], []
        return [], [
            UnsupportedActionPlan(
                source_review_item_id=item.item_id,
                item_type=item.item_type,
                reason="no safe action plan is available for this review item type in this pass",
            )
        ]

    def _build_exact_duplicate_plan(self, item: ReviewQueueItem) -> ActionPlan:
        keep_path = item.preferred_path
        candidate_paths = [path for path in item.affected_paths if path != keep_path]
        operations = [
            ActionPlanOperation(
                operation_id=f"{item.item_id}-keep",
                operation_type="keep_preferred",
                path=keep_path,
                apply_supported=False,
                notes=["preferred copy selected by duplicate analysis"],
            )
        ]
        for index, path in enumerate(candidate_paths, start=1):
            candidate_path = Path(path)
            destination = _quarantine_destination(self.root, candidate_path)
            operations.append(
                ActionPlanOperation(
                    operation_id=f"{item.item_id}-candidate-{index:03d}",
                    operation_type="quarantine_move",
                    path=path,
                    destination_path=str(destination),
                    apply_supported=True,
                    notes=["non-preferred duplicate is moved to quarantine instead of deleted"],
                )
            )
        return ActionPlan(
            plan_id=_plan_id("exact_duplicate_keep_plan", [item.item_id]),
            source_review_item_ids=[item.item_id],
            action_type="exact_duplicate_keep_plan",
            affected_files=list(item.affected_paths),
            proposed_operations=operations,
            confidence=1.0,
            safety_level="low-risk",
            reasons=["exact duplicates are confirmed by full-content hashing"],
            notes=["apply moves non-preferred files into a quarantine folder; no hard delete occurs"],
            apply_supported=True,
            resolves_review_items=True,
        )

    def _build_metadata_related_plans(self, item: ReviewQueueItem) -> list[ActionPlan]:
        paths = [Path(path) for path in item.affected_paths if Path(path).exists()]
        previews = build_normalize_preview_for_paths(paths, strategy="smart")
        metadata_operations: list[ActionPlanOperation] = []
        metadata_notes: list[str] = []
        rename_operations: list[ActionPlanOperation] = []

        for preview in previews:
            metadata_fields: list[str] = []
            values = {
                "title": preview.proposed_title,
                "artist": preview.proposed_artist,
                "album": preview.proposed_album,
            }
            for field in ("title", "artist", "album"):
                current_value = getattr(preview, f"current_{field}")
                proposed_value = values[field]
                if proposed_value and proposed_value != "UNKNOWN" and proposed_value != current_value:
                    metadata_fields.append(field)
            if metadata_fields and preview.confidence >= 0.7:
                metadata_operations.append(
                    ActionPlanOperation(
                        operation_id=f"{item.item_id}-metadata-{len(metadata_operations)+1:03d}",
                        operation_type="write_metadata",
                        path=preview.path,
                        fields=metadata_fields,
                        values={field: values[field] for field in metadata_fields},
                        apply_supported=True,
                        notes=list(preview.reasons),
                    )
                )
            elif preview.would_change:
                metadata_notes.append(f"{preview.path}: insufficient deterministic metadata confidence")

            rename_proposal = deterministic_cleanup(Path(preview.path).stem)
            rename_result = build_rename_result(Path(preview.path), rename_proposal.new_base, list(rename_proposal.rule_notes), False)
            if rename_result.changed:
                rename_operations.append(
                    ActionPlanOperation(
                        operation_id=f"{item.item_id}-rename-{len(rename_operations)+1:03d}",
                        operation_type="rename_file",
                        path=str(rename_result.old_path),
                        destination_path=str(rename_result.new_path),
                        apply_supported=True,
                        notes=list(rename_proposal.rule_notes),
                    )
                )

        plans: list[ActionPlan] = []
        if metadata_operations:
            plans.append(
                ActionPlan(
                    plan_id=_plan_id("metadata_fix_plan", [item.item_id]),
                    source_review_item_ids=[item.item_id],
                    action_type="metadata_fix_plan",
                    affected_files=sorted({operation.path for operation in metadata_operations if operation.path}),
                    proposed_operations=metadata_operations,
                    confidence=round(min(operation_count_conf(metadata_operations), 0.95), 3),
                    safety_level="low-risk",
                    reasons=["deterministic normalize preview produced non-placeholder metadata values"],
                    notes=metadata_notes or ["only confident deterministic metadata writes are included"],
                    apply_supported=True,
                    resolves_review_items=True,
                )
            )
        elif metadata_notes:
            plans.append(
                ActionPlan(
                    plan_id=_plan_id("metadata_fix_plan", [item.item_id]),
                    source_review_item_ids=[item.item_id],
                    action_type="metadata_fix_plan",
                    affected_files=list(item.affected_paths),
                    proposed_operations=[],
                    confidence=0.25,
                    safety_level="manual-review",
                    reasons=["deterministic metadata proposal was insufficiently confident"],
                    notes=metadata_notes,
                    apply_supported=False,
                    resolves_review_items=False,
                )
            )

        if rename_operations:
            plans.append(
                ActionPlan(
                    plan_id=_plan_id("rename_normalize_plan", [item.item_id]),
                    source_review_item_ids=[item.item_id],
                    action_type="rename_normalize_plan",
                    affected_files=sorted({operation.path for operation in rename_operations if operation.path}),
                    proposed_operations=rename_operations,
                    confidence=0.9,
                    safety_level="low-risk",
                    reasons=["deterministic filename cleanup produced concrete rename targets"],
                    notes=["rename apply uses the existing deterministic rename path"],
                    apply_supported=True,
                    resolves_review_items=False,
                )
            )
        return plans

    def _build_artwork_gap_plan(self, item: ReviewQueueItem) -> ActionPlan:
        return ActionPlan(
            plan_id=_plan_id("artwork_gap_plan", [item.item_id]),
            source_review_item_ids=[item.item_id],
            action_type="artwork_gap_plan",
            affected_files=list(item.affected_paths),
            proposed_operations=[],
            confidence=0.0,
            safety_level="review-only",
            reasons=["no internal artwork source or scraper is used in this pass"],
            notes=["artwork actions remain manual review only"],
            apply_supported=False,
            resolves_review_items=False,
        )


def operation_count_conf(operations: list[ActionPlanOperation]) -> float:
    return 0.75 + min(0.2, len(operations) * 0.03)


def build_action_plan_report(
    root: Path,
    records: list[TrackRecord],
    review_report: ReviewQueueReport,
    *,
    source_review_item_ids: list[str],
) -> ActionPlanReport:
    return ActionPlanBuilder(root, records, review_report).build(source_review_item_ids)


def apply_action_plan_report(
    report: ActionPlanReport,
    *,
    review_state: ReviewStateStore,
    backup_dir: Path | None = None,
) -> list[AppliedPlanResult]:
    results: list[AppliedPlanResult] = []
    now = datetime.now(tz=UTC)
    for plan in report.plans:
        if not plan.apply_supported:
            results.append(
                AppliedPlanResult(
                    plan_id=plan.plan_id,
                    action_type=plan.action_type,
                    status="skipped",
                    source_review_item_ids=plan.source_review_item_ids,
                    operation_results=[],
                    resolved_review_item_ids=[],
                )
            )
            continue

        operation_results: list[AppliedOperationResult] = []
        all_succeeded = True
        for operation in plan.proposed_operations:
            if not operation.apply_supported:
                operation_results.append(
                    AppliedOperationResult(
                        operation_id=operation.operation_id,
                        operation_type=operation.operation_type,
                        path=operation.path,
                        destination_path=operation.destination_path,
                        status="skipped",
                        message="operation is review-only",
                    )
                )
                continue
            try:
                backup_path = None
                if operation.path is not None and backup_dir is not None:
                    backup_path = str(backup_file(Path(operation.path), backup_dir))
                if operation.operation_type == "write_metadata":
                    if operation.path is None:
                        raise TagWriteError("metadata write operation is missing a source path")
                    write_tags(
                        Path(operation.path),
                        title=operation.values.get("title"),
                        artist=operation.values.get("artist"),
                        album=operation.values.get("album"),
                        fields=list(operation.fields),
                    )
                elif operation.operation_type == "rename_file":
                    if operation.path is None or operation.destination_path is None:
                        raise RuntimeError("rename operation is missing source or destination path")
                    rename_result = build_rename_result(
                        Path(operation.path),
                        Path(operation.destination_path).stem,
                        list(operation.notes),
                        False,
                    )
                    rename_result = rename_result.__class__(
                        old_path=Path(operation.path),
                        new_path=Path(operation.destination_path),
                        ts=rename_result.ts,
                        rule_notes=rename_result.rule_notes,
                        llm_used=False,
                        changed=Path(operation.path) != Path(operation.destination_path),
                    )
                    apply_rename(rename_result)
                elif operation.operation_type == "quarantine_move":
                    if operation.path is None or operation.destination_path is None:
                        raise RuntimeError("quarantine operation is missing source or destination path")
                    destination = Path(operation.destination_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise RuntimeError(f"quarantine destination already exists: {destination}")
                    shutil.move(operation.path, operation.destination_path)
                else:
                    raise RuntimeError(f"unsupported apply operation: {operation.operation_type}")
                operation_results.append(
                    AppliedOperationResult(
                        operation_id=operation.operation_id,
                        operation_type=operation.operation_type,
                        path=operation.path,
                        destination_path=operation.destination_path,
                        status="ok",
                        message="applied",
                        backup_path=backup_path,
                    )
                )
            except Exception as exc:
                all_succeeded = False
                operation_results.append(
                    AppliedOperationResult(
                        operation_id=operation.operation_id,
                        operation_type=operation.operation_type,
                        path=operation.path,
                        destination_path=operation.destination_path,
                        status="error",
                        message=str(exc),
                    )
                )
        resolved_review_item_ids: list[str] = []
        if all_succeeded and operation_results and plan.resolves_review_items:
            apply_review_action(review_state, item_ids=plan.source_review_item_ids, status="resolved", now=now)
            resolved_review_item_ids = list(plan.source_review_item_ids)
        results.append(
            AppliedPlanResult(
                plan_id=plan.plan_id,
                action_type=plan.action_type,
                status="ok" if all_succeeded else "partial_failure",
                source_review_item_ids=plan.source_review_item_ids,
                operation_results=operation_results,
                resolved_review_item_ids=resolved_review_item_ids,
            )
        )
    return results
