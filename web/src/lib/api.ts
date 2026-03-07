import type {
  ActionPlanReport,
  DuplicateReport,
  HealthReport,
  HistoryMutationResponse,
  HistoryResponse,
  PlaylistsResponse,
  ReportEnvelope,
  ReviewStateAction,
  ReviewPlanApplyResponse,
  ReviewReport,
  ReviewStateMutationResponse,
  StatusResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function fetchStatus() {
  return getJson<StatusResponse>("/status");
}

export function fetchHealth() {
  return getJson<ReportEnvelope<HealthReport>>("/health");
}

export function fetchReview() {
  return getJson<ReportEnvelope<ReviewReport>>("/review");
}

export function fetchDuplicates() {
  return getJson<ReportEnvelope<DuplicateReport>>("/duplicates");
}

export function fetchPlaylists() {
  return getJson<PlaylistsResponse>("/playlists");
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
