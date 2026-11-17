from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query

from nyxcore.config import HealthConfig
from nyxcore.core.track import TrackRecord
from nyxcore.rename.organization import STATUS_VALUES, Proposal, Status, build_organization_preview, path_key

PAGE_SIZE = 50


@dataclass(frozen=True)
class Preview:
    id: UUID
    rows: list[Proposal]
    folders: list[Proposal]


def install_organization_routes(
    app: FastAPI, *, library: Callable[[], Path], settings: Callable[[], HealthConfig],
    cached: Callable[[Path], list[TrackRecord]],
) -> None:
    lock = Lock()
    current: Preview | None = None

    @app.post("/api/organization/preview")
    def create_preview() -> dict:
        nonlocal current
        root = library()
        with lock:
            try:
                rows = build_organization_preview(root, settings(), cached(root))
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(409, "The configured library could not be read. Check the folder and build again.") from exc
            folders = sorted(
                (row for row in rows if row.destination and row.status in {"ready", "unchanged"}),
                key=lambda row: path_key(row.destination or ()),
            )
            current = Preview(uuid4(), rows, folders)
            counts = Counter(row.status for row in rows)
            return {"id": current.id, "root": str(root), "counts": {key: counts[key] for key in STATUS_VALUES}}

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
        return {"rows": rows[offset:offset + PAGE_SIZE], "total": len(rows), "page_size": PAGE_SIZE}
