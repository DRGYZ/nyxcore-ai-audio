export type PriorityBand = "high" | "medium" | "low";
export type ReviewStatus = "new" | "seen" | "snoozed" | "ignored" | "resolved";
export type ReviewStateAction = Exclude<ReviewStatus, "new">;
export type ReportMode = "live" | "mock";
export type OperationExecutionStatus = "ok" | "error" | "skipped";
export type PlanExecutionStatus = "ok" | "partial_failure" | "skipped";
export type UndoExecutionStatus = "pending" | "ok" | "error" | "not_supported";
export type HealthBitrateBucketKey = "unknown" | "<128k" | "128k-191k" | "192k-255k" | ">=256k";

export interface ApiMeta {
  music_path: string;
  out_path: string;
  active_profile: string;
  mode: ReportMode;
}

export interface StatusResponse {
  status: "ok";
  service: "nyxcore-webapi";
  music_path: string;
  out_path: string;
  active_profile: string;
  review_state_exists: boolean;
  history_exists: boolean;
  saved_playlist_count: number;
}

export interface ReportEnvelope<T> {
  meta: ApiMeta;
  data: T;
}

export interface ReviewItem {
  item_id: string;
  item_type: string;
  priority_band: PriorityBand;
  priority_score: number;
  summary: string;
  reason_summary: string;
  review_status: ReviewStatus;
  state_updated_at?: string | null;
  snooze_until?: string | null;
  affected_paths?: string[];
  sample_paths?: string[];
  folder?: string | null;
  preferred_path?: string | null;
  reclaimable_bytes?: number | null;
  confidence?: number | null;
  file_count?: number;
  details?: Record<string, unknown>;
}

export interface ReviewReport {
  summary: {
    total_items: number;
    counts_by_type: Record<string, number>;
    counts_by_priority_band: Record<string, number>;
    counts_by_state: Record<string, number>;
    total_files_referenced: number;
  };
  metadata: {
    active_profile: string;
    generation_mode: string;
  };
  items: ReviewItem[];
}

export interface DuplicatePreferred {
  path: string;
  reasons: string[];
}

export interface DuplicateFileInfo {
  path: string;
  file_size_bytes: number;
  extension?: string;
  duration_seconds?: number | null;
  bitrate_bps?: number | null;
  title?: string | null;
  artist?: string | null;
  album?: string | null;
  has_cover_art?: boolean;
  metadata_fields_present?: number;
}

export interface DuplicateGroup {
  group_id: string;
  preferred: DuplicatePreferred;
  files: DuplicateFileInfo[];
  content_hash?: string;
  confidence?: number;
  reclaimable_bytes?: number;
  reasons?: string[];
  relationships?: Array<{ paths: string[]; confidence: number; reasons: string[] }>;
}

export interface DuplicateReport {
  summary: {
    total_tracks: number;
    exact_group_count: number;
    likely_group_count: number;
    exact_duplicate_file_count: number;
    likely_duplicate_file_count: number;
  };
  exact_duplicates: DuplicateGroup[];
  likely_duplicates: DuplicateGroup[];
}

export interface HealthReport {
  overview: {
    total_audio_files: number;
    total_folders_touched: number;
    format_breakdown: Record<string, number>;
    total_library_size_bytes: number;
    duplicate_exact_groups: number;
    duplicate_likely_groups: number;
  };
  metadata: {
    missing_title: { count: number };
    missing_artist: { count: number };
    missing_album: { count: number };
    placeholder_metadata: { count: number };
    suspicious_title_artist_swaps: { count: number };
  };
  artwork: {
    coverage_percent: number;
    with_artwork: number;
    without_artwork: number;
  };
  quality: {
    bitrate_buckets: Record<HealthBitrateBucketKey, number>;
    low_bitrate_files: { count: number };
    lossless_files: number;
    lossy_files: number;
    duration_outliers: { count: number };
    unreadable_or_unparseable_files: { count: number };
  };
  naming: {
    noisy_filenames: { count: number };
    long_filenames: { count: number };
    filename_metadata_disagreement: { count: number };
    high_issue_folders: Array<{ folder: string; issue_count: number }>;
  };
  duplicates: {
    exact_duplicate_groups: number;
    likely_duplicate_groups: number;
    files_in_exact_duplicates: number;
    files_in_likely_duplicates: number;
    total_files_in_duplicates: number;
    reclaimable_bytes_exact: number;
  };
  priorities: {
    top_issue_categories?: Array<{ category: string; count: number; action: string }>;
    top_problematic_folders?: Array<{ folder: string; issue_count: number }>;
    recommended_actions: string[];
  };
}

export interface PlaylistSummary {
  playlist_id: string;
  name: string;
  profile: string;
  query: string;
  last_refreshed_at: string | null;
  track_count: number;
  latest_summary: Record<string, unknown>;
  latest_refresh_diff: Record<string, unknown>;
}

export interface PlaylistsResponse {
  meta: ApiMeta;
  items: PlaylistSummary[];
}

export interface HistoryBatch {
  batch_id: string;
  applied_at: string;
  action_types: string[];
  reversible: boolean;
  affected_count: number;
  source_plan_ids: string[];
  source_review_item_ids: string[];
  operations: Array<{
    operation_id: string;
    operation_type: string;
    status: OperationExecutionStatus;
    reversible: boolean;
    original_path?: string | null;
    current_path?: string | null;
  }>;
}

export interface HistoryResponse {
  meta: ApiMeta;
  items: HistoryBatch[];
}

export interface ReviewStateMutationResponse {
  updated_item_ids: string[];
  status: ReviewStateAction;
  review_state_path: string;
}

export interface ActionPlanOperation {
  operation_id: string;
  operation_type: string;
  path?: string | null;
  destination_path?: string | null;
  fields: string[];
  values: Record<string, string | null>;
  apply_supported: boolean;
  notes: string[];
}

export interface ActionPlan {
  plan_id: string;
  source_review_item_ids: string[];
  action_type: string;
  affected_files: string[];
  proposed_operations: ActionPlanOperation[];
  confidence: number;
  safety_level: string;
  reasons: string[];
  notes: string[];
  apply_supported: boolean;
  resolves_review_items: boolean;
}

export interface ActionPlanReport {
  created_at: string;
  source_review_item_ids: string[];
  plans: ActionPlan[];
  unsupported_items: Array<{
    source_review_item_id: string;
    item_type: string;
    reason: string;
  }>;
  summary: {
    requested_item_count: number;
    generated_plan_count: number;
    unsupported_item_count: number;
    apply_supported_plan_count: number;
  };
}

export interface ReviewPlanApplyResponse {
  result_count: number;
  resolved_review_item_ids: string[];
  batch_id?: string | null;
  results: Array<{
    plan_id: string;
    action_type: string;
    status: PlanExecutionStatus;
    source_review_item_ids: string[];
    resolved_review_item_ids: string[];
    operation_results: Array<{
      operation_id: string;
      operation_type: string;
      path?: string | null;
      destination_path?: string | null;
      status: OperationExecutionStatus;
      message: string;
      backup_path?: string | null;
    }>;
  }>;
}

export interface HistoryMutationResponse {
  batch_id: string;
  changed_operations: Array<{
    operation_id: string;
    plan_id: string;
    action_type: string;
    source_review_item_ids: string[];
    operation_type: string;
    status: OperationExecutionStatus;
    reversible: boolean;
    original_path?: string | null;
    current_path?: string | null;
    backup_path?: string | null;
    message: string;
    undo_status: UndoExecutionStatus;
    undone_at?: string | null;
    undo_message?: string | null;
  }>;
  reactivated_review_item_ids: string[];
}
