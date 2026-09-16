from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ApiStatus = Literal["ok"]
ApiMode = Literal["live"]
ReviewStateAction = Literal["seen", "snoozed", "ignored", "resolved"]
OperationExecutionStatus = Literal["ok", "error", "skipped"]
PlanExecutionStatus = Literal["ok", "partial_failure", "skipped"]
UndoExecutionStatus = Literal["pending", "ok", "error", "not_supported"]


class ApiStatusResponse(BaseModel):
    status: ApiStatus = "ok"
    service: str = "nyxcore-webapi"
    music_path: str
    out_path: str
    active_profile: str
    review_state_exists: bool
    history_exists: bool
    saved_playlist_count: int


class ApiMetaResponse(BaseModel):
    music_path: str
    out_path: str
    active_profile: str
    mode: ApiMode = "live"


class ApiReportEnvelope(BaseModel):
    meta: ApiMetaResponse
    data: dict[str, Any] = Field(default_factory=dict)


class SearchTrackResponse(BaseModel):
    path: str
    filename: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_seconds: float | None = None
    file_size_bytes: int
    has_cover_art: bool
    warnings: list[str] = Field(default_factory=list)
    match_fields: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    meta: ApiMetaResponse
    query: str
    total_matches: int
    returned_count: int
    items: list[SearchTrackResponse] = Field(default_factory=list)


class PlaylistSummaryResponse(BaseModel):
    playlist_id: str
    name: str
    profile: str
    query: str
    last_refreshed_at: str | None = None
    track_count: int = 0
    latest_summary: dict[str, Any] = Field(default_factory=dict)
    latest_refresh_diff: dict[str, Any] = Field(default_factory=dict)
    latest_tracks: list[dict[str, Any]] = Field(default_factory=list)


class PlaylistsResponse(BaseModel):
    meta: ApiMetaResponse
    items: list[PlaylistSummaryResponse] = Field(default_factory=list)


class PlaylistCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=500)
    max_tracks: int | None = Field(default=None, ge=1, le=500)
    min_score: float | None = None
    export_m3u: bool = True
    music_path: str | None = None
    out_path: str | None = None
    profile: str | None = None
    config_path: str | None = None


class PlaylistRefreshRequest(BaseModel):
    export_m3u: bool = True
    music_path: str | None = None
    out_path: str | None = None
    profile: str | None = None
    config_path: str | None = None


class PlaylistMutationResponse(BaseModel):
    item: PlaylistSummaryResponse
    m3u_path: str | None = None


class HistoryOperationResponse(BaseModel):
    operation_id: str
    operation_type: str
    status: OperationExecutionStatus
    reversible: bool
    original_path: str | None = None
    current_path: str | None = None
    undo_status: UndoExecutionStatus = "pending"
    undone_at: str | None = None
    undo_message: str | None = None


class HistoryBatchSummaryResponse(BaseModel):
    batch_id: str
    applied_at: str
    action_types: list[str] = Field(default_factory=list)
    reversible: bool
    affected_count: int
    source_plan_ids: list[str] = Field(default_factory=list)
    source_review_item_ids: list[str] = Field(default_factory=list)
    operations: list[HistoryOperationResponse] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    meta: ApiMetaResponse
    items: list[HistoryBatchSummaryResponse] = Field(default_factory=list)


class ReviewStateMutationRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1)
    action: ReviewStateAction
    days: int | None = None
    music_path: str | None = None
    out_path: str | None = None
    profile: str | None = None
    config_path: str | None = None


class ReviewStateMutationResponse(BaseModel):
    updated_item_ids: list[str] = Field(default_factory=list)
    status: ReviewStateAction
    review_state_path: str


class ReviewPlanGenerateRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1)
    music_path: str | None = None
    out_path: str | None = None
    profile: str | None = None
    config_path: str | None = None


class ReviewPlanApplyRequest(BaseModel):
    plan_report: dict[str, Any]
    music_path: str | None = None
    out_path: str | None = None
    profile: str | None = None
    config_path: str | None = None
    backup_dir: str | None = None


class ReviewPlanOperationResultResponse(BaseModel):
    operation_id: str
    operation_type: str
    path: str | None = None
    destination_path: str | None = None
    status: OperationExecutionStatus
    message: str
    backup_path: str | None = None


class ReviewPlanResultResponse(BaseModel):
    plan_id: str
    action_type: str
    status: PlanExecutionStatus
    source_review_item_ids: list[str] = Field(default_factory=list)
    resolved_review_item_ids: list[str] = Field(default_factory=list)
    operation_results: list[ReviewPlanOperationResultResponse] = Field(default_factory=list)


class ReviewPlanApplyResponse(BaseModel):
    result_count: int
    resolved_review_item_ids: list[str] = Field(default_factory=list)
    batch_id: str | None = None
    results: list[ReviewPlanResultResponse] = Field(default_factory=list)


class HistoryMutationRequest(BaseModel):
    out_path: str | None = None
    alternate_restore_dir: str | None = None
    target_path: str | None = None
    stale_lock_token: str | None = None


class HistoryMutationOperationResponse(BaseModel):
    operation_id: str
    plan_id: str
    action_type: str
    source_review_item_ids: list[str] = Field(default_factory=list)
    operation_type: str
    original_path: str | None = None
    current_path: str | None = None
    backup_path: str | None = None
    status: OperationExecutionStatus
    message: str
    reversible: bool
    undo_status: UndoExecutionStatus
    undone_at: str | None = None
    undo_message: str | None = None


class HistoryMutationResponse(BaseModel):
    batch_id: str
    changed_operations: list[HistoryMutationOperationResponse] = Field(default_factory=list)
    reactivated_review_item_ids: list[str] = Field(default_factory=list)
