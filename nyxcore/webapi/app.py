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
from nyxcore.action_plan.service import (
    ActionPlanReport,
    apply_action_plan_report,
    authorize_action_plan_selection,
    build_action_plan_report,
)
from nyxcore.config import NyxConfig, load_config
from nyxcore.core.scanner import scan_music_folder
from nyxcore.incremental.service import ChangeSet, RefreshSummary
from nyxcore.report_pipeline import build_duplicate_health_reports, build_duplicate_report, build_review_pipeline
from nyxcore.review_queue.state import apply_review_action, load_review_state, save_review_state
from nyxcore.search.service import search_tracks
from nyxcore.saved_playlists.service import (
    SavedPlaylistDefinition,
    SavedPlaylistLatestResult,
    create_saved_playlist_definition,
    export_saved_playlist_m3u,
    load_saved_playlist_store,
    read_saved_playlist_latest_result,
    refresh_saved_playlist,
    save_saved_playlist_definition,
)
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
    PlaylistCreateRequest,
    PlaylistMutationResponse,
    PlaylistRefreshRequest,
    PlaylistSummaryResponse,
    ReviewPlanApplyRequest,
    ReviewPlanApplyResponse,
    ReviewPlanGenerateRequest,
    ReviewStateMutationRequest,
    ReviewStateMutationResponse,
    SearchResponse,
    SearchTrackResponse,
)


def _default_music_path() -> Path:
    return Path(os.environ.get("NYXCORE_WEB_MUSIC_DIR", "music")).resolve()


def _default_out_path() -> Path:
    return Path(os.environ.get("NYXCORE_WEB_OUT_DIR", "data/reports")).resolve()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_configured_root(value: str | None, *, configured_root: Path, label: str) -> Path:
    if value is None:
        return configured_root
    candidate = Path(value).resolve()
    if candidate != configured_root:
        raise HTTPException(
            status_code=403,
            detail=f"{label} must match the server-configured root: {configured_root}",
        )
    return candidate


def _resolve_bounded_path(value: str, *, roots: tuple[Path, ...], label: str) -> Path:
    candidate = Path(value).resolve()
    if not any(_is_within(candidate, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise HTTPException(status_code=403, detail=f"{label} is outside the configured roots: {allowed}")
    return candidate


def _resolve_music_path(music_path: str | None) -> Path:
    return _resolve_configured_root(
        music_path,
        configured_root=_default_music_path(),
        label="music_path",
    )


def _resolve_out_path(out_path: str | None) -> Path:
    return _resolve_configured_root(
        out_path,
        configured_root=_default_out_path(),
        label="out_path",
    )


def _resolve_config(config_path: str | None, profile: str | None) -> NyxConfig:
    configured_value = os.environ.get("NYXCORE_WEB_CONFIG_PATH")
    if config_path is not None:
        if configured_value is None:
            raise HTTPException(status_code=403, detail="config_path overrides are disabled for this server")
        configured_path = Path(configured_value).resolve()
        candidate = Path(config_path).resolve()
        if candidate != configured_path:
            raise HTTPException(
                status_code=403,
                detail=f"config_path must match the server-configured file: {configured_path}",
            )
    else:
        candidate = None if configured_value is None else Path(configured_value).resolve()
    return load_config(candidate, profile=profile)


def _validate_history_batch_paths(batch, *, music_root: Path, out_root: Path) -> None:
    allowed_roots = (music_root, out_root)
    for operation in batch.operations:
        for label, value in (
            ("history original path", operation.original_path),
            ("history current path", operation.current_path),
            ("history backup path", operation.backup_path),
        ):
            if value is not None:
                _resolve_bounded_path(value, roots=allowed_roots, label=label)


def _meta(music_path: Path, out_path: Path, app_config: NyxConfig) -> ApiMetaResponse:
    return ApiMetaResponse(music_path=str(music_path), out_path=str(out_path), active_profile=app_config.profile)


def _load_records(music_path: Path):
    records, _stats = scan_music_folder(music_path)
    return records


def _full_refresh_summary(records) -> RefreshSummary:
    return RefreshSummary(
        mode="full",
        changes=ChangeSet(
            added_files=sorted(record.path for record in records),
            modified_files=[],
            removed_files=[],
            unchanged_files=[],
        ),
        rescanned_files=len(records),
    )


def _playlist_summary_response(
    definition: SavedPlaylistDefinition,
    latest: SavedPlaylistLatestResult | None,
) -> PlaylistSummaryResponse:
    return PlaylistSummaryResponse(
        playlist_id=definition.playlist_id,
        name=definition.name,
        profile=definition.profile,
        query=definition.query,
        last_refreshed_at=definition.last_refreshed_at,
        track_count=0 if latest is None else int(latest.summary.get("track_count", 0)),
        latest_summary={} if latest is None else dict(latest.summary),
        latest_refresh_diff={} if latest is None else dict(latest.refresh_diff),
        latest_tracks=[] if latest is None else list(latest.report.get("ranked_tracks", [])),
    )


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

    @app.get("/api/search", response_model=SearchResponse)
    def search(
        q: str = Query(min_length=2, max_length=200),
        limit: int = Query(default=20, ge=1, le=50),
        music_path: str | None = Query(default=None),
        out_path: str | None = Query(default=None),
        profile: str | None = Query(default=None),
        config_path: str | None = Query(default=None),
    ) -> SearchResponse:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        records = _load_records(resolved_music)
        total_matches, results = search_tracks(records, q, limit=limit)
        items = [
            SearchTrackResponse(
                path=result.record.path,
                filename=result.filename,
                title=result.record.tags.get("title"),
                artist=result.record.tags.get("artist"),
                album=result.record.tags.get("album"),
                duration_seconds=result.record.duration_seconds,
                file_size_bytes=result.record.file_size_bytes,
                has_cover_art=result.record.has_cover_art,
                warnings=[warning.value for warning in result.record.warnings],
                match_fields=list(result.match_fields),
            )
            for result in results
        ]
        return SearchResponse(
            meta=_meta(resolved_music, resolved_out, app_config),
            query=q,
            total_matches=total_matches,
            returned_count=len(items),
            items=items,
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
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        app_config = _resolve_config(request.config_path, request.profile)
        review_state_path = resolved_out / "review_state.json"
        records, duplicates_report, health_report, review_state, review_report = _build_review_dependencies(
            resolved_music, resolved_out, app_config
        )
        del duplicates_report, health_report
        requested_plan_report = ActionPlanReport.from_dict(request.plan_report)
        generated_plan_report = build_action_plan_report(
            resolved_music,
            records,
            review_report,
            source_review_item_ids=list(requested_plan_report.source_review_item_ids),
        )
        try:
            plan_report = authorize_action_plan_selection(generated_plan_report, requested_plan_report)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        backup_dir = (
            None
            if request.backup_dir is None
            else _resolve_bounded_path(
                request.backup_dir,
                roots=(resolved_out,),
                label="backup_dir",
            )
        )
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
            items.append(_playlist_summary_response(definition, latest))
        return PlaylistsResponse(meta=_meta(resolved_music, resolved_out, app_config), items=items)

    @app.post("/api/playlists", response_model=PlaylistMutationResponse)
    def create_playlist(request: PlaylistCreateRequest) -> PlaylistMutationResponse:
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        if not resolved_music.is_dir():
            raise HTTPException(status_code=400, detail=f"Music directory does not exist: {resolved_music}")
        app_config = _resolve_config(request.config_path, request.profile)
        name = request.name.strip()
        query = request.query.strip()
        if not name or not query:
            raise HTTPException(status_code=422, detail="Playlist name and query cannot be blank")
        store_root = resolved_out / "saved_playlists"
        store = load_saved_playlist_store(store_root)
        definition = create_saved_playlist_definition(
            name=name,
            query=query,
            profile=app_config.profile,
            max_tracks=request.max_tracks,
            min_score=request.min_score,
        )
        if definition.playlist_id in store.playlists:
            raise HTTPException(status_code=409, detail="An identical saved playlist already exists")
        records = _load_records(resolved_music)
        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=records,
            refresh_summary=_full_refresh_summary(records),
            app_config=app_config,
            analysis_cache_path=None,
            library_root=resolved_music,
        )
        store.playlists[definition.playlist_id] = definition
        save_saved_playlist_definition(store_root, store)
        m3u_path = (
            export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
            if request.export_m3u
            else None
        )
        return PlaylistMutationResponse(
            item=_playlist_summary_response(definition, latest),
            m3u_path=None if m3u_path is None else str(m3u_path),
        )

    @app.post("/api/playlists/{playlist_id}/refresh", response_model=PlaylistMutationResponse)
    def refresh_playlist(playlist_id: str, request: PlaylistRefreshRequest) -> PlaylistMutationResponse:
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        if not resolved_music.is_dir():
            raise HTTPException(status_code=400, detail=f"Music directory does not exist: {resolved_music}")
        store_root = resolved_out / "saved_playlists"
        store = load_saved_playlist_store(store_root)
        definition = store.playlists.get(playlist_id)
        if definition is None:
            raise HTTPException(status_code=404, detail=f"Saved playlist not found: {playlist_id}")
        app_config = _resolve_config(request.config_path, request.profile or definition.profile)
        records = _load_records(resolved_music)
        latest = refresh_saved_playlist(
            store_root,
            definition,
            records=records,
            refresh_summary=_full_refresh_summary(records),
            app_config=app_config,
            analysis_cache_path=None,
            profile_override=app_config.profile,
            library_root=resolved_music,
        )
        save_saved_playlist_definition(store_root, store)
        m3u_path = (
            export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
            if request.export_m3u
            else None
        )
        return PlaylistMutationResponse(
            item=_playlist_summary_response(definition, latest),
            m3u_path=None if m3u_path is None else str(m3u_path),
        )

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
                reversible=any(
                    operation.status == "ok" and operation.reversible and operation.undo_status != "ok"
                    for operation in batch.operations
                ),
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
                        undo_status=operation.undo_status,
                        undone_at=operation.undone_at,
                        undo_message=operation.undo_message,
                    )
                    for operation in batch.operations
                ],
            )
            for batch in sorted(ledger.batches, key=lambda item: item.applied_at, reverse=True)
        ]
        return HistoryResponse(meta=_meta(resolved_music, resolved_out, app_config), items=items)

    @app.post("/api/history/{batch_id}/restore", response_model=HistoryMutationResponse)
    def restore_history_batch(batch_id: str, request: HistoryMutationRequest) -> HistoryMutationResponse:
        resolved_music = _default_music_path()
        resolved_out = _resolve_out_path(request.out_path)
        ledger_path = resolved_out / "review_history.json"
        review_state_path = resolved_out / "review_state.json"
        ledger = load_operation_ledger(ledger_path)
        batch = find_batch(ledger, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"History batch not found: {batch_id}")
        _validate_history_batch_paths(batch, music_root=resolved_music, out_root=resolved_out)
        alternate_restore_dir = (
            None
            if request.alternate_restore_dir is None
            else _resolve_bounded_path(
                request.alternate_restore_dir,
                roots=(resolved_music, resolved_out),
                label="alternate_restore_dir",
            )
        )
        target_path = (
            None
            if request.target_path is None
            else str(
                _resolve_bounded_path(
                    request.target_path,
                    roots=(resolved_music, resolved_out),
                    label="target_path",
                )
            )
        )
        review_state = load_review_state(review_state_path)
        changed = undo_operation_batch(
            batch,
            review_state=review_state,
            alternate_restore_dir=alternate_restore_dir,
            target_path=target_path,
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
