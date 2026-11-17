import type {
  ActionPlanReport,
  ApiMeta,
  DuplicateReport,
  HealthReport,
  HistoryMutationResponse,
  HistoryResponse,
  LibraryReports,
  PlaylistMutationResponse,
  PlaylistsResponse,
  ReportEnvelope,
  ReviewStateAction,
  ReviewPlanApplyResponse,
  ReviewReport,
  ReviewStateMutationResponse,
  SearchResponse,
  StatusResponse,
} from "./types";

const API_BASE_URL = import.meta.env.PROD
  ? "/api"
  : (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api");

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

async function getJson<T>(path: string): Promise<T> {
  return readJson<T>(await fetch(`${API_BASE_URL}${path}`));
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return readJson<T>(await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }));
}

export function fetchLibraryReports() {
  return getJson<LibraryReports>("/library");
}

export function refreshLibrary() {
  return postJson<ApiMeta>("/library/refresh", {});
}

export function fetchStatus() {
  return getJson<StatusResponse>("/status");
}

export function fetchHealth() {
  return getJson<ReportEnvelope<HealthReport>>("/health");
}

export function fetchReview() {
  return getJson<ReportEnvelope<ReviewReport>>("/review?include_snoozed=true");
}

export function fetchDuplicates() {
  return getJson<ReportEnvelope<DuplicateReport>>("/duplicates");
}

export function fetchArchiveSearch(query: string, limit = 50) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return getJson<SearchResponse>(`/search?${params.toString()}`);
}

export function fetchPlaylists() {
  return getJson<PlaylistsResponse>("/playlists");
}

export function createPlaylist(payload: {
  name: string;
  query: string;
  max_tracks?: number;
  min_score?: number;
  export_m3u?: boolean;
}) {
  return postJson<PlaylistMutationResponse>("/playlists", payload);
}

export function refreshPlaylist(playlistId: string, payload: { export_m3u?: boolean } = {}) {
  return postJson<PlaylistMutationResponse>(`/playlists/${playlistId}/refresh`, payload);
}

export function fetchHistory() {
  return getJson<HistoryResponse>("/history");
}

export function mutateReviewState(payload: { item_ids: string[]; action: ReviewStateAction; days?: number }) {
  return postJson<ReviewStateMutationResponse>("/review/state", payload);
}

export function generateReviewPlan(payload: { item_ids: string[] }) {
  return postJson<ReportEnvelope<ActionPlanReport>>("/review/plan", payload);
}

export function applyReviewPlan(payload: { plan_report: ActionPlanReport }) {
  return postJson<ReviewPlanApplyResponse>("/review/plan/apply", payload);
}

export function restoreHistoryBatch(batchId: string, payload: { target_path?: string; alternate_restore_dir?: string } = {}) {
  return postJson<HistoryMutationResponse>(`/history/${batchId}/restore`, payload);
}

export function undoHistoryBatch(batchId: string, payload: { target_path?: string; alternate_restore_dir?: string } = {}) {
  return postJson<HistoryMutationResponse>(`/history/${batchId}/undo`, payload);
}
