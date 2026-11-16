import type {
  HealthBitrateBucketKey,
  HistoryBatch,
  HistoryLifecycleState,
  HistoryOperationStatus,
  OperationExecutionStatus,
  PlanExecutionStatus,
  UndoExecutionStatus,
} from "./types";

export type StatusTone = "neutral" | "success" | "warning" | "danger";

const HISTORY_BATCH_STATES: Record<HistoryLifecycleState, { label: string; tone: StatusTone }> = {
  legacy: { label: "Legacy", tone: "neutral" },
  prepared: { label: "Prepared", tone: "warning" },
  applying: { label: "Applying", tone: "warning" },
  applied: { label: "Completed", tone: "success" },
  failed: { label: "Failed", tone: "danger" },
  partial_failure: { label: "Partial Failure", tone: "warning" },
  recovery_required: { label: "Recovery Required", tone: "warning" },
  reversed: { label: "Reversed", tone: "neutral" },
};

export const HISTORY_OPERATION_LABELS: Record<HistoryOperationStatus, string> = {
  prepared: "Prepared",
  applying: "Applying",
  ok: "Applied",
  error: "Failed",
  skipped: "Skipped",
};

export function historyBatchState(batch: HistoryBatch) {
  return HISTORY_BATCH_STATES[batch.recovery_required ? "recovery_required" : batch.lifecycle_state ?? "legacy"];
}

export function canReverseHistoryBatch(batch: HistoryBatch) {
  return batch.reversible && !batch.recovery_required;
}

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

export function operationExecutionTone(status: HistoryOperationStatus): StatusTone {
  if (status === "ok") return "success";
  if (status === "error") return "danger";
  if (status === "prepared" || status === "applying") return "warning";
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
