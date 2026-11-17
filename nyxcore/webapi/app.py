from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from nyxcore import __version__
from nyxcore.action_plan.ledger import (
    OperationBatch,
    batch_requires_recovery,
    find_batch,
    load_operation_ledger,
)
from nyxcore.action_plan.service import (
    ActionPlanReport,
    build_action_plan_report,
    execute_reviewed_action_plan,
)
from nyxcore.action_plan.lifecycle import HistoryRecoveryRequiredError, reverse_recorded_batch
from nyxcore.action_plan.lock import MutationLockError
from nyxcore.config import NyxConfig, load_config
from nyxcore.core.scanner import scan_music_folder
from nyxcore.incremental.service import ChangeSet, RefreshSummary
from nyxcore.report_pipeline import build_duplicate_health_reports, build_duplicate_report, build_review_pipeline
from nyxcore.review_queue.state import apply_review_action, load_review_state, save_review_state
from nyxcore.search.service import search_tracks
from nyxcore.webapi.analysis import LibrarySnapshot, ReviewOnlyAnalysis
from nyxcore.webapi.metadata import install_metadata_routes
from nyxcore.webapi.organization import install_organization_routes
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
    LibraryReportsResponse,
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

logger = logging.getLogger(__name__)


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


def configured_config_path() -> Path | None:
    configured_value = os.environ.get("NYXCORE_WEB_CONFIG_PATH")
    return Path(configured_value).resolve() if configured_value is not None else None


def _resolve_config(config_path: str | None, profile: str | None) -> NyxConfig:
    configured_path = configured_config_path()
    if config_path is not None:
        if configured_path is None:
            raise HTTPException(status_code=403, detail="config_path overrides are disabled for this server")
        candidate = Path(config_path).resolve()
        if candidate != configured_path:
            raise HTTPException(
                status_code=403,
                detail=f"config_path must match the server-configured file: {configured_path}",
            )
    else:
        candidate = configured_path
    return load_config(candidate, profile=profile)


def _meta(
    music_path: Path, out_path: Path, app_config: NyxConfig, analysis: LibrarySnapshot | None = None,
) -> ApiMetaResponse:
    return ApiMetaResponse(
        music_path=str(music_path),
        out_path=str(out_path),
        active_profile=app_config.profile,
        mode="snapshot" if analysis else "live",
        analysis_generation=analysis.generation if analysis else None,
    )


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
    analysis: LibrarySnapshot | None = None,
    max_items: int | None = None,
    min_priority: str | None = None,
    include_ignored: bool = False,
    include_snoozed: bool = False,
    include_resolved: bool = False,
    only_unresolved: bool = False,
):
    records = analysis.records if analysis else _load_records(music_path)
    review_state = load_review_state(out_path / "review_state.json")
    pipeline = build_review_pipeline(
        music_path,
        records,
        app_config=app_config,
        review_state=review_state,
        generation_mode="snapshot" if analysis else "live",
        reports=analysis.reports if analysis else None,
        resolved_before=analysis.started_at if analysis else None,
        max_items=max_items,
        min_priority_band=min_priority,
        include_ignored=include_ignored,
        include_snoozed=include_snoozed,
        include_resolved=include_resolved,
        only_unresolved=only_unresolved,
    )
    return records, pipeline.duplicate_report, pipeline.health_report, review_state, pipeline.review_report


def create_app(
    *,
    music_root: Path | None = None,
    workspace_root: Path | None = None,
    review_only: bool = False,
) -> FastAPI:
    app = FastAPI(title="NyxCore Web API", version=__version__)
    analysis_store = ReviewOnlyAnalysis(_load_records) if review_only else None
    metadata_generation = uuid4().hex
    review_state_lock = Lock() if review_only else nullcontext()
    if not review_only:
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

    def _resolve_music_path(music_path: str | None) -> Path:
        return _resolve_configured_root(
            music_path, configured_root=music_root or _default_music_path(), label="music_path",
        )

    def _resolve_out_path(out_path: str | None) -> Path:
        return _resolve_configured_root(
            out_path, configured_root=workspace_root or _default_out_path(), label="out_path",
        )

    def require_audio_mutations() -> None:
        if review_only:
            raise HTTPException(
                status_code=403,
                detail="This NyxCore server is in review-only mode. Applying plans, restoring and undoing are disabled.",
            )

    def load_analysis(music: Path, config: NyxConfig, *, refresh: bool = False) -> LibrarySnapshot | None:
        if analysis_store is None:
            return None
        try:
            return analysis_store.get(music, config, refresh=refresh)
        except Exception as exc:
            logger.exception("Could not analyze the configured library")
            raise HTTPException(
                status_code=503, detail="Library analysis failed. Check the library folder and try Refresh Library again.",
            ) from exc

    def metadata_library() -> Path:
        if music_root is None and not os.environ.get("NYXCORE_WEB_MUSIC_DIR"):
            raise HTTPException(409, "Set NYXCORE_WEB_MUSIC_DIR to the library you want to inspect.")
        return _resolve_music_path(None).resolve()

    def metadata_workspace() -> Path:
        out = _resolve_out_path(None).resolve()
        library = metadata_library()
        if _is_within(out, library) or _is_within(library, out):
            raise HTTPException(409, "Choose a separate metadata workspace outside the configured library.")
        return out

    def current_metadata_generation() -> str:
        if review_only:
            analysis = load_analysis(metadata_library(), _resolve_config(None, None))
            assert analysis is not None
            return analysis.generation
        return metadata_generation

    def invalidate_metadata_generation() -> None:
        nonlocal metadata_generation
        metadata_generation = uuid4().hex

    def metadata_records():
        root = metadata_library()
        analysis = load_analysis(root, _resolve_config(None, None))
        return analysis.records if analysis else _load_records(root)

    install_metadata_routes(
        app, library=metadata_library, workspace=metadata_workspace,
        settings=lambda: _resolve_config(None, None).health, generation=current_metadata_generation,
        records=metadata_records, require_mutations=require_audio_mutations,
        invalidate=invalidate_metadata_generation, review_only=review_only,
    )

    def organization_library() -> Path:
        configured = music_root or os.environ.get("NYXCORE_WEB_MUSIC_DIR")
        if not configured:
            raise HTTPException(409, "Configure a library folder before building an organization preview.")
        return Path(configured).absolute()

    def cached_organization_records(root: Path):
        analysis = analysis_store.peek(root, _resolve_config(None, None)) if analysis_store else None
        return analysis.records if analysis else []

    install_organization_routes(app, library=organization_library, settings=lambda: _resolve_config(None, None).health,
                                cached=cached_organization_records)

    if review_only:
        @app.get("/api/library", response_model=LibraryReportsResponse)
        def library_reports(
            music_path: str | None = Query(default=None),
            out_path: str | None = Query(default=None),
            profile: str | None = Query(default=None),
            config_path: str | None = Query(default=None),
        ) -> LibraryReportsResponse:
            music = _resolve_music_path(music_path)
            out = _resolve_out_path(out_path)
            config = _resolve_config(config_path, profile)
            analysis = load_analysis(music, config)
            assert analysis is not None
            _, duplicates, health, _, review = _build_review_dependencies(
                music, out, config, analysis=analysis, include_snoozed=True, include_resolved=True,
            )
            return LibraryReportsResponse(
                meta=_meta(music, out, config, analysis),
                analyzed_at=analysis.analyzed_at,
                health=health.to_dict(),
                duplicates=duplicates.to_dict(),
                review=review.to_dict(),
            )

        @app.post("/api/library/refresh", response_model=ApiMetaResponse)
        def refresh_library(
            music_path: str | None = Query(default=None),
            out_path: str | None = Query(default=None),
            profile: str | None = Query(default=None),
            config_path: str | None = Query(default=None),
        ) -> ApiMetaResponse:
            music = _resolve_music_path(music_path)
            out = _resolve_out_path(out_path)
            config = _resolve_config(config_path, profile)
            return _meta(music, out, config, load_analysis(music, config, refresh=True))

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
            review_only=review_only,
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
        analysis = load_analysis(resolved_music, app_config)
        report = analysis.reports.duplicate_report if analysis else build_duplicate_report(
            _load_records(resolved_music), app_config=app_config,
        )
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config, analysis),
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
        analysis = load_analysis(resolved_music, app_config)
        records = analysis.records if analysis else _load_records(resolved_music)
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
            meta=_meta(resolved_music, resolved_out, app_config, analysis),
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
        refresh: bool = Query(default=False),
    ) -> ApiReportEnvelope:
        resolved_music = _resolve_music_path(music_path)
        resolved_out = _resolve_out_path(out_path)
        app_config = _resolve_config(config_path, profile)
        analysis = load_analysis(resolved_music, app_config)
        report = analysis.reports.health_report if analysis else build_duplicate_health_reports(
            resolved_music, _load_records(resolved_music), app_config=app_config,
        ).health_report
        if refresh and not review_only:
            invalidate_metadata_generation()
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config, analysis),
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
        analysis = load_analysis(resolved_music, app_config)
        _records, _duplicates_report, _health_report, _review_state, report = _build_review_dependencies(
            resolved_music,
            resolved_out,
            app_config,
            analysis=analysis,
            max_items=max_items,
            min_priority=min_priority,
            include_ignored=include_ignored,
            include_snoozed=include_snoozed,
            include_resolved=include_resolved,
            only_unresolved=only_unresolved,
        )
        return ApiReportEnvelope(
            meta=_meta(resolved_music, resolved_out, app_config, analysis),
            data=report.to_dict(),
        )

    @app.post("/api/review/state", response_model=ReviewStateMutationResponse)
    def review_state_mutation(request: ReviewStateMutationRequest) -> ReviewStateMutationResponse:
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        app_config = _resolve_config(request.config_path, request.profile)
        analysis = load_analysis(resolved_music, app_config)
        with review_state_lock:
            _records, _duplicates_report, _health_report, review_state, review_report = _build_review_dependencies(
                resolved_music, resolved_out, app_config,
                analysis=analysis,
                include_resolved=analysis is not None,
                include_ignored=analysis is not None,
                include_snoozed=analysis is not None,
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
        analysis = load_analysis(resolved_music, app_config)
        records, duplicates_report, health_report, review_state, review_report = _build_review_dependencies(
            resolved_music, resolved_out, app_config, analysis=analysis,
        )
        del duplicates_report, health_report, review_state
        plan_report = build_action_plan_report(
            resolved_music,
            records,
            review_report,
            source_review_item_ids=list(request.item_ids),
        )
        return ApiReportEnvelope(meta=_meta(resolved_music, resolved_out, app_config, analysis), data=plan_report.to_dict())

    @app.post("/api/review/plan/apply", response_model=ReviewPlanApplyResponse)
    def apply_review_plan(request: ReviewPlanApplyRequest) -> ReviewPlanApplyResponse:
        require_audio_mutations()
        resolved_music = _resolve_music_path(request.music_path)
        resolved_out = _resolve_out_path(request.out_path)
        app_config = _resolve_config(request.config_path, request.profile)
        review_state_path = resolved_out / "review_state.json"
        records, duplicates_report, health_report, review_state, review_report = _build_review_dependencies(
            resolved_music, resolved_out, app_config
        )
        del duplicates_report, health_report
        requested_plan_report = ActionPlanReport.from_dict(request.plan_report)
        backup_dir = (
            None
            if request.backup_dir is None
            else _resolve_bounded_path(
                request.backup_dir,
                roots=(resolved_out,),
                label="backup_dir",
            )
        )
        try:
            plan_report, results, batch = execute_reviewed_action_plan(
                resolved_music,
                records,
                review_report,
                requested_plan_report,
                review_state=review_state,
                review_state_path=review_state_path,
                ledger_path=resolved_out / "review_history.json",
                backup_dir=backup_dir,
                workspace_root=resolved_out,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        resolved_review_item_ids = sorted({item_id for result in results for item_id in result.resolved_review_item_ids})
        invalidate_metadata_generation()
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
            _history_batch_summary(batch)
            for batch in sorted(ledger.batches, key=lambda item: item.applied_at, reverse=True)
        ]
        return HistoryResponse(meta=_meta(resolved_music, resolved_out, app_config), items=items)

    @app.post("/api/history/{batch_id}/restore", response_model=HistoryMutationResponse)
    def restore_history_batch(batch_id: str, request: HistoryMutationRequest) -> HistoryMutationResponse:
        require_audio_mutations()
        resolved_music = _resolve_music_path(None)
        resolved_out = _resolve_out_path(request.out_path)
        ledger_path = resolved_out / "review_history.json"
        review_state_path = resolved_out / "review_state.json"
        ledger = load_operation_ledger(ledger_path)
        batch = find_batch(ledger, batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"History batch not found: {batch_id}")
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
        try:
            batch, changed = reverse_recorded_batch(
                ledger_path=ledger_path,
                review_state_path=review_state_path,
                batch_id=batch_id,
                library_root=resolved_music,
                workspace_root=resolved_out,
                alternate_restore_dir=alternate_restore_dir,
                target_path=target_path,
                stale_lock_token=request.stale_lock_token,
                require_settled=True,
            )
        except (MutationLockError, HistoryRecoveryRequiredError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if changed:
            invalidate_metadata_generation()
        review_state = load_review_state(review_state_path)
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


def _history_batch_summary(batch: OperationBatch) -> HistoryBatchSummaryResponse:
    recovery_required = batch_requires_recovery(batch)
    return HistoryBatchSummaryResponse(
        batch_id=batch.batch_id,
        applied_at=batch.applied_at,
        lifecycle_state=batch.lifecycle_state,
        recovery_required=recovery_required,
        action_types=batch.action_types,
        reversible=(
            not recovery_required
            and batch.lifecycle_state in {"applied", "partial_failure", "legacy"}
            and any(
                operation.status == "ok"
                and operation.journal_state in {"applied", "legacy"}
                and operation.reversible
                and operation.undo_status in {"pending", "error"}
                for operation in batch.operations
            )
        ),
        affected_count=len(batch.operations),
        source_plan_ids=batch.source_plan_ids,
        source_review_item_ids=batch.source_review_item_ids,
        operations=[
            HistoryOperationResponse(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                status=operation.status,
                journal_state=operation.journal_state,
                reversal_pending=operation.reversal_intent is not None,
                reversible=operation.reversible,
                original_path=operation.original_path,
                current_path=operation.current_path,
                undo_status=operation.undo_status,
                undone_at=operation.undone_at,
                undo_message=operation.undo_message,
                failure_reason=operation.failure_reason,
            )
            for operation in batch.operations
        ],
    )


app = create_app()
