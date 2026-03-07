import type {
  HealthBitrateBucketKey,
  OperationExecutionStatus,
  PlanExecutionStatus,
  UndoExecutionStatus,
} from "./types";

export type StatusTone = "neutral" | "success" | "warning" | "danger";

export const HEALTH_BITRATE_BUCKET_ORDER: HealthBitrateBucketKey[] = [
  "unknown",
  "<128k",
  "128k-191k",
  "192k-255k",
  ">=256k",
];

export const HEALTH_BITRATE_BUCKET_LABELS: Record<HealthBitrateBucketKey, string> = {
  unknown: "Unknown",
  "<128k": "<128k",
  "128k-191k": "128-191k",
  "192k-255k": "192-255k",
  ">=256k": ">=256k",
};

export function formatPlanExecutionStatus(status: PlanExecutionStatus) {
  return status === "partial_failure" ? "partial failure" : status;
}

export function planExecutionTone(status: PlanExecutionStatus): StatusTone {
  if (status === "ok") return "success";
  if (status === "partial_failure") return "warning";
  return "neutral";
}

export function formatOperationExecutionStatus(status: OperationExecutionStatus) {
  return status;
}

export function operationExecutionTone(status: OperationExecutionStatus): StatusTone {
  if (status === "ok") return "success";
  if (status === "error") return "danger";
  return "neutral";
}

export function formatUndoExecutionStatus(status: UndoExecutionStatus) {
  return status === "not_supported" ? "not supported" : status;
}

export function undoExecutionTone(status: UndoExecutionStatus): StatusTone {
  if (status === "ok") return "success";
  if (status === "error") return "danger";
  if (status === "not_supported") return "warning";
  return "neutral";
}

export function isSuccessfulOperationStatus(status: OperationExecutionStatus) {
  return status === "ok";
}

export function isSuccessfulUndoStatus(status: UndoExecutionStatus) {
  return status === "ok";
}
