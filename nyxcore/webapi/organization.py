from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from nyxcore.action_plan.service import (
    MAX_AUTOMATIC_PLAN_OPERATIONS,
    ActionPlan,
    ActionPlanOperation,
    ActionPlanReport,
    ActionPlanSummary,
    execute_authorized_action_plan,
)

from nyxcore.config import HealthConfig
from nyxcore.core.track import TrackRecord
from nyxcore.core.filesystem import ORGANIZATION_ACTION, FileIdentity, check_move_path, file_snapshot, scan_stat_key
from nyxcore.rename.organization import (
    STATUS_VALUES, Proposal, Status, build_organization_preview, organization_root_stat, path_key,
)

PAGE_SIZE = 50


@dataclass(frozen=True)
class Preview:
    id: UUID
    root: Path
    root_identity: FileIdentity
    rows: list[Proposal]
    folders: list[Proposal]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(min_length=1, max_length=MAX_AUTOMATIC_PLAN_OPERATIONS)


class Confirmation(Selection):
    approval: UUID
    confirm: StrictBool


@dataclass(frozen=True)
class ApprovedMove:
    source: Path
    destination: Path
    identity: FileIdentity
    sha256: str


@dataclass(frozen=True)
class Approval:
    id: UUID
    moves: tuple[ApprovedMove, ...]


def install_organization_routes(
    app: FastAPI, *, library: Callable[[], Path], settings: Callable[[], HealthConfig],
    cached: Callable[[Path], list[TrackRecord]],
    workspace: Callable[[], Path], enabled: Callable[[], bool], invalidate: Callable[[], None],
) -> None:
    lock = Lock()
    current: Preview | None = None
    approval: Approval | None = None

    def require_apply() -> None:
        if not enabled():
            raise HTTPException(403, "Organization apply requires an explicitly configured disposable development library.")

    def get_preview(preview_id: UUID, root: Path) -> Preview:
        if current is None or current.id != preview_id or current.root != root:
            raise HTTPException(409, "This preview is no longer current. Build a new preview.")
        return current

    def root_identity(root: Path) -> FileIdentity:
        info = organization_root_stat(root)
        return info.st_dev, info.st_ino

    def validate_selection(preview: Preview, paths: list[str]) -> list[tuple[Path, Path, tuple[int, int, int, int]]]:
        if root_identity(preview.root) != preview.root_identity:
            raise ValueError("The configured library changed. Build a new preview.")
        ready = {row.path: row for row in preview.rows if row.status == "ready"}
        if len(set(paths)) != len(paths) or any(path not in ready for path in paths):
            raise ValueError("Select distinct Ready files from this preview.")
        latest = {row.path: row for row in build_organization_preview(preview.root, settings(), cached(preview.root))}
        moves = []
        for path in paths:
            row, updated = ready[path], latest.get(path)
            if (updated is None or updated.status != "ready" or row.destination != updated.destination
                    or row.source_stat != updated.source_stat or row.destination is None):
                raise ValueError(f"Proposal changed or is blocked. Rebuild the preview for {path}.")
            source, destination = Path(path), preview.root.joinpath(*row.destination)
            info = check_move_path(preview.root, source)
            if info is None or info.st_nlink != 1 or scan_stat_key(info) != row.source_stat:
                raise ValueError(f"Source changed or has multiple hard links: {source}")
            if check_move_path(preview.root, destination, missing_ok=True) is not None:
                raise ValueError(f"Destination already exists: {destination}")
            moves.append((source, destination, scan_stat_key(info)))
        if root_identity(preview.root) != preview.root_identity:
            raise ValueError("The configured library changed. Build a new preview.")
        return moves

    @app.post("/api/organization/preview")
    def create_preview() -> dict:
        nonlocal current, approval
        root = library()
        with lock:
            try:
                identity = root_identity(root)
                rows = build_organization_preview(root, settings(), cached(root))
                if root_identity(root) != identity:
                    raise ValueError("The configured library changed while building the preview.")
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(409, "The configured library could not be read. Check the folder and build again.") from exc
            folders = sorted(
                (row for row in rows if row.destination and row.status in {"ready", "unchanged"}),
                key=lambda row: path_key(row.destination or ()),
            )
            current = Preview(uuid4(), root, identity, rows, folders)
            approval = None
            counts = Counter(row.status for row in rows)
            return {"id": current.id, "root": str(root), "counts": {key: counts[key] for key in STATUS_VALUES},
                    "apply_enabled": enabled()}

    @app.get("/api/organization/preview/{preview_id}")
    def preview_page(
        preview_id: UUID, status: Status | Literal["all"] = "all",
        view: Literal["files", "folders"] = "files", offset: int = Query(0, ge=0),
    ) -> dict:
        preview = current
        if preview is None or preview.id != preview_id:
            raise HTTPException(409, "This preview is no longer current. Build a new preview.")
        rows = preview.folders if view == "folders" else preview.rows
        if status != "all":
            rows = [row for row in rows if row.status == status]
        return {"rows": [{key: value for key, value in asdict(row).items() if key != "source_stat"}
                         for row in rows[offset:offset + PAGE_SIZE]], "total": len(rows), "page_size": PAGE_SIZE}

    @app.post("/api/organization/preview/{preview_id}/review", dependencies=[Depends(require_apply)])
    def review_selection(preview_id: UUID, request: Selection) -> dict:
        nonlocal approval
        with lock:
            preview = get_preview(preview_id, library())
            approval = None
            try:
                approved = []
                for source, destination, stamp in validate_selection(preview, request.paths):
                    before = check_move_path(preview.root, source)
                    snapshot = file_snapshot(source)
                    after = check_move_path(preview.root, source)
                    if (before is None or after is None or snapshot is None or before.st_nlink != 1
                            or after.st_nlink != 1 or scan_stat_key(before) != stamp or scan_stat_key(after) != stamp
                            or snapshot[0] != (before.st_dev, before.st_ino)):
                        raise ValueError(f"Source changed while reviewing: {source}")
                    approved.append(ApprovedMove(source, destination, *snapshot))
            except (ValueError, OSError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc
            approval = Approval(uuid4(), tuple(approved))
            return {"approval": approval.id, "moves": [
                {"source": str(move.source), "destination": str(move.destination)} for move in approval.moves
            ]}

    @app.post("/api/organization/preview/{preview_id}/apply", dependencies=[Depends(require_apply)])
    def apply_selection(preview_id: UUID, request: Confirmation) -> dict:
        nonlocal current, approval
        if not request.confirm:
            raise HTTPException(422, "Confirm the exact organization moves before applying.")
        root, out = library(), workspace()
        with lock:
            preview = get_preview(preview_id, root)
            selected = approval
            if (selected is None or selected.id != request.approval
                    or request.paths != [str(move.source) for move in selected.moves]):
                raise HTTPException(409, "The approved selection is stale or changed. Review it again.")

            def authorize() -> tuple[ActionPlanReport, dict[str, FileIdentity]]:
                nonlocal approval, current
                require_apply()
                approval = None
                moves = validate_selection(preview, request.paths)
                operations = []
                identities = {}
                for index, (move, (source, destination, _)) in enumerate(zip(selected.moves, moves, strict=True)):
                    if ((move.source, move.destination) != (source, destination)
                            or file_snapshot(move.source) != (move.identity, move.sha256)):
                        raise ValueError(f"Source or destination changed since approval: {move.source}")
                    operation_id = f"{selected.id}-{index + 1}"
                    operations.append(ActionPlanOperation(
                        operation_id, "rename_file", str(move.source), str(move.destination),
                        apply_supported=True, expected_hash=move.sha256,
                    ))
                    identities[operation_id] = move.identity
                current = None
                plan = ActionPlan(
                    str(selected.id), [], ORGANIZATION_ACTION, request.paths, operations, 1.0, "reviewed",
                    ["Explicitly approved organization paths"], [], True,
                )
                report = ActionPlanReport(
                    datetime.now(UTC).isoformat(), [], [plan], [], ActionPlanSummary(0, 1, 0, 1),
                )
                return report, identities

            try:
                _, results, batch = execute_authorized_action_plan(
                    root, authorize, review_state_path=out / "review_state.json",
                    ledger_path=out / "review_history.json", workspace_root=out,
                )
            except (ValueError, OSError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc
            finally:
                if current is None:
                    invalidate()
        return {"batch_id": batch.batch_id, "operations": [operation.to_dict() for result in results
                                                            for operation in result.operation_results]}
