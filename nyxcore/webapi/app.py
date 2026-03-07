from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from nyxcore import __version__
from nyxcore.action_plan.ledger import (
    append_operation_batch,
    find_batch,
    load_operation_ledger,
    save_operation_ledger,
    undo_operation_batch,
)
from nyxcore.action_plan.service import ActionPlanReport, apply_action_plan_report, build_action_plan_report
from nyxcore.config import NyxConfig, load_config
from nyxcore.core.scanner import scan_music_folder
from nyxcore.report_pipeline import build_duplicate_health_reports, build_duplicate_report, build_review_pipeline
from nyxcore.review_queue.state import apply_review_action, load_review_state, save_review_state
from nyxcore.saved_playlists.service import load_saved_playlist_store, read_saved_playlist_latest_result
from nyxcore.webapi.schemas import (
    ApiMetaResponse,
    ApiReportEnvelope,
    ApiStatusResponse,
    HistoryBatchSummaryResponse,
    HistoryMutationRequest,
    HistoryMutationResponse,
    HistoryOperationResponse,
    HistoryResponse,
    PlaylistsResponse,
    PlaylistSummaryResponse,
    ReviewPlanApplyRequest,
    ReviewPlanApplyResponse,
    ReviewPlanGenerateRequest,
    ReviewStateMutationRequest,
    ReviewStateMutationResponse,
)


def _default_music_path() -> Path:
    return Path(os.environ.get("NYXCORE_WEB_MUSIC_DIR", "music")).resolve()


def _default_out_path() -> Path:
    return Path(os.environ.get("NYXCORE_WEB_OUT_DIR", "data/reports")).resolve()


def _resolve_music_path(music_path: str | None) -> Path:
    return Path(music_path).resolve() if music_path else _default_music_path()


def _resolve_out_path(out_path: str | None) -> Path:
    return Path(out_path).resolve() if out_path else _default_out_path()


def _resolve_config(config_path: str | None, profile: str | None) -> NyxConfig:
    return load_config(None if config_path is None else Path(config_path), profile=profile)


def _meta(music_path: Path, out_path: Path, app_config: NyxConfig) -> ApiMetaResponse:
    return ApiMetaResponse(music_path=str(music_path), out_path=str(out_path), active_profile=app_config.profile)


def _load_records(music_path: Path):
    records, _stats = scan_music_folder(music_path)
    return records


def _build_review_dependencies(
    music_path: Path,
    out_path: Path,
    app_config: NyxConfig,
    *,
    max_items: int | None = None,
    min_priority: str | None = None,
    include_ignored: bool = False,
    include_snoozed: bool = False,
    include_resolved: bool = False,
    only_unresolved: bool = False,
):
    records = _load_records(music_path)
    review_state = load_review_state(out_path / "review_state.json")
    pipeline = build_review_pipeline(
        music_path,
        records,
        app_config=app_config,
        review_state=review_state,
        generation_mode="live",
        max_items=max_items,
        min_priority_band=min_priority,
        include_ignored=include_ignored,
        include_snoozed=include_snoozed,
        include_resolved=include_resolved,
        only_unresolved=only_unresolved,
    )
    return records, pipeline.duplicate_report, pipeline.health_report, review_state, pipeline.review_report


def create_app() -> FastAPI:
    app = FastAPI(title="NyxCore Web API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/status", response_model=ApiStatusResponse)
    def status(
        music_path: str | None = Query(default=None),
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
    ) -> ApiStatusResponse:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        store = load_saved_playlist_store(resolved_out / "saved_playlists")
        return ApiStatusResponse(
            music_path=str(resolved_music),
            out_path=str(resolved_out),
            active_profile=app_config.profile,
            review_state_exists=(resolved_out / "review_state.json").exists(),
            history_exists=(resolved_out / "review_history.json").exists(),
            saved_playlist_count=len(store.playlists),
        )

    @app.get("/api/duplicates", response_model=ApiReportEnvelope)
    def duplicates(
        music_path: str | None = Query(default=None),
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
    ) -> ApiReportEnvelope:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        records = _load_records(resolved_music)
        report = build_duplicate_report(records, app_config=app_config)
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config),
            data=report.to_dict(),
        )

    @app.get("/api/health", response_model=ApiReportEnvelope)
    def health(
        music_path: str | None = Query(default=None),
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
    ) -> ApiReportEnvelope:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        records = _load_records(resolved_music)
        report = build_duplicate_health_reports(resolved_music, records, app_config=app_config).health_report
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config),
            data=report.to_dict(),
        )

    @app.get("/api/review", response_model=ApiReportEnvelope)
    def review(
        music_path: str | None = Query(default=None),
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
        max_items: int | None = Query(default=None, ge=1),
        min_priority: str | None = Query(default=None, pattern="^(low|medium|high)$"),
        include_ignored: bool = Query(default=False),
        include_snoozed: bool = Query(default=False),
        include_resolved: bool = Query(default=False),
        only_unresolved: bool = Query(default=False),
    ) -> ApiReportEnvelope:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        _records, _duplicates_report, _health_report, _review_state, report = _build_review_dependencies(
            resolved_music,
            resolved_out,
            app_config,
            max_items=max_items,
            min_priority=min_priority,
            include_ignored=include_ignored,
            include_snoozed=include_snoozed,
            include_resolved=include_resolved,
            only_unresolved=only_unresolved,
        )
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config),
            data=report.to_dict(),
        )

    @app.post("/api/review/state", response_model=ReviewStateMutationResponse)
    def review_state_mutation(request: ReviewStateMutationRequest) -> ReviewStateMutationResponse:
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        app_config = _resolve_config(request.config_path, request.profile)
        _records, _duplicates_report, _health_report, review_state, review_report = _build_review_dependencies(
            resolved_music, resolved_out, app_config
        )
        item_type_by_id = {item.item_id: item.item_type for item in review_report.items}
        summary_by_id = {item.item_id: item.summary for item in review_report.items}
        missing = sorted(item_id for item_id in request.item_ids if item_id not in item_type_by_id)
        if missing:
            raise HTTPException(status_code=404, detail=f"Review item(s) not found: {', '.join(missing)}")
        apply_review_action(
            review_state,
            item_ids=request.item_ids,
            status=request.action,
            days=request.days,
            item_type_by_id=item_type_by_id,
            summary_by_id=summary_by_id,
        )
        review_state_path = resolved_out / "review_state.json"
        save_review_state(review_state_path, review_state)
        return ReviewStateMutationResponse(
            updated_item_ids=list(request.item_ids),
            status=request.action,
            review_state_path=str(review_state_path),
        )

    @app.post("/api/review/plan", response_model=ApiReportEnvelope)
    def generate_review_plan(request: ReviewPlanGenerateRequest) -> ApiReportEnvelope:
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        app_config = _resolve_config(request.config_path, request.profile)
        records, duplicates_report, health_report, review_state, review_report = _build_review_dependencies(
            resolved_music, resolved_out, app_config
        )
        del duplicates_report, health_report, review_state
        plan_report = build_action_plan_report(
            resolved_music,
            records,
            review_report,
            source_review_item_ids=list(request.item_ids),
        )
        return ApiReportEnvelope(meta=_meta(resolved_music, resolved_out, app_config), data=plan_report.to_dict())

    @app.post("/api/review/plan/apply", response_model=ReviewPlanApplyResponse)
    def apply_review_plan(request: ReviewPlanApplyRequest) -> ReviewPlanApplyResponse:
        resolved_out = _resolve_out_path(request.out_path)
        review_state_path = resolved_out / "review_state.json"
        review_state = load_review_state(review_state_path)
        plan_report = ActionPlanReport.from_dict(request.plan_report)
        backup_dir = None if request.backup_dir is None else Path(request.backup_dir)
        results = apply_action_plan_report(plan_report, review_state=review_state, backup_dir=backup_dir)
        save_review_state(review_state_path, review_state)
        ledger_path = resolved_out / "review_history.json"
        ledger = load_operation_ledger(ledger_path)
        batch = append_operation_batch(ledger, plan_report=plan_report, results=results)
        save_operation_ledger(ledger_path, ledger)
        resolved_review_item_ids = sorted({item_id for result in results for item_id in result.resolved_review_item_ids})
        return ReviewPlanApplyResponse(
            result_count=len(results),
            resolved_review_item_ids=resolved_review_item_ids,
            batch_id=batch.batch_id,
            results=[result.to_dict() for result in results],
        )

    @app.get("/api/playlists", response_model=PlaylistsResponse)
    def playlists(
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
        music_path: str | None = Query(default=None),
    ) -> PlaylistsResponse:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        store = load_saved_playlist_store(resolved_out / "saved_playlists")
        items: list[PlaylistSummaryResponse] = []
        for definition in sorted(store.playlists.values(), key=lambda item: item.playlist_id):
            latest = read_saved_playlist_latest_result(resolved_out / "saved_playlists", definition.playlist_id)
            items.append(
                PlaylistSummaryResponse(
                    playlist_id=definition.playlist_id,
                    name=definition.name,
                    profile=definition.profile,
                    query=definition.query,
                    last_refreshed_at=definition.last_refreshed_at,
                    track_count=0 if latest is None else int(latest.summary.get("track_count", 0)),
                    latest_summary={} if latest is None else dict(latest.summary),
                    latest_refresh_diff={} if latest is None else dict(latest.refresh_diff),
                )
            )
        return PlaylistsResponse(meta=_meta(resolved_music, resolved_out, app_config), items=items)

    @app.get("/api/history", response_model=HistoryResponse)
    def history(
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
        music_path: str | None = Query(default=None),
    ) -> HistoryResponse:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        ledger = load_operation_ledger(resolved_out / "review_history.json")
        items = [
            HistoryBatchSummaryResponse(
                batch_id=batch.batch_id,
                applied_at=batch.applied_at,
                action_types=list(batch.action_types),
                reversible=all(operation.reversible for operation in batch.operations) if batch.operations else False,
                affected_count=len(batch.operations),
                source_plan_ids=list(batch.source_plan_ids),
                source_review_item_ids=list(batch.source_review_item_ids),
                operations=[
                    HistoryOperationResponse(
                        operation_id=operation.operation_id,
                        operation_type=operation.operation_type,
                        status=operation.status,
                        reversible=operation.reversible,
                        original_path=operation.original_path,
                        current_path=operation.current_path,
                    )
                    for operation in batch.operations
                ],
            )
            for batch in sorted(ledger.batches, key=lambda item: item.applied_at, reverse=True)
        ]
        return HistoryResponse(meta=_meta(resolved_music, resolved_out, app_config), items=items)

    @app.post("/api/history/{batch_id}/restore", response_model=HistoryMutationResponse)
    def restore_history_batch(batch_id: str, request: HistoryMutationRequest) -> HistoryMutationResponse:
        resolved_out = _resolve_out_path(request.out_path)
        ledger_path = resolved_out / "review_history.json"
        review_state_path = resolved_out / "review_state.json"
        ledger = load_operation_ledger(ledger_path)
        batch = find_batch(ledger, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"History batch not found: {batch_id}")
        review_state = load_review_state(review_state_path)
        changed = undo_operation_batch(
            batch,
            review_state=review_state,
            alternate_restore_dir=None if request.alternate_restore_dir is None else Path(request.alternate_restore_dir),
            target_path=request.target_path,
        )
        save_operation_ledger(ledger_path, ledger)
        save_review_state(review_state_path, review_state)
        reactivated_review_item_ids = sorted(
            item_id for item_id, entry in review_state.items.items() if entry.status == "seen" and item_id in batch.source_review_item_ids
        )
        return HistoryMutationResponse(
            batch_id=batch_id,
            changed_operations=[operation.to_dict() for operation in changed],
            reactivated_review_item_ids=reactivated_review_item_ids,
        )

    @app.post("/api/history/{batch_id}/undo", response_model=HistoryMutationResponse)
    def undo_history_batch(batch_id: str, request: HistoryMutationRequest) -> HistoryMutationResponse:
        return restore_history_batch(batch_id, request)

    return app


app = create_app()
