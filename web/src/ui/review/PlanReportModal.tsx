import {
  formatOperationExecutionStatus,
  formatPlanExecutionStatus,
  isSuccessfulOperationStatus,
  planExecutionTone,
} from "../../lib/contracts";
import type { ActionPlan, ActionPlanOperation, ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";
import { ActionBanner, Button, Chip, Modal, Panel, PathBlock } from "../components";

function renderOperationPath(operation: ActionPlanOperation) {
  if (operation.path && operation.destination_path) {
    return `${operation.path} -> ${operation.destination_path}`;
  }
  if (operation.path) {
    return operation.path;
  }
  const fields = operation.fields.map((field) => `${field}: ${operation.values[field] ?? "unset"}`);
  return fields.length > 0 ? fields.join(" | ") : "review-only";
}

function PlanCard({ plan }: { plan: ActionPlan }) {
  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-display text-lg font-bold text-slate-100">{plan.action_type}</p>
          <p className="mt-1 text-xs text-slate-500">
            Safety: {plan.safety_level} • Confidence: {(plan.confidence * 100).toFixed(0)}% • {plan.affected_files.length} files
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Chip tone={plan.apply_supported ? "primary" : "warning"}>{plan.apply_supported ? "apply-capable" : "review-only"}</Chip>
          <Chip tone="violet">{plan.proposed_operations.length} ops</Chip>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Affected Files</p>
          <div className="space-y-2">
            {plan.affected_files.map((file) => (
              <PathBlock key={file} value={file} />
            ))}
          </div>
        </div>
        <div>
          <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Proposed Operations</p>
          <div className="space-y-2">
            {plan.proposed_operations.map((operation) => (
              <div key={operation.operation_id} className="rounded-lg border border-border-dark bg-background-dark/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold text-slate-200">{operation.operation_type}</p>
                  <Chip tone={operation.apply_supported ? "success" : "warning"}>{operation.apply_supported ? "apply" : "review"}</Chip>
                </div>
                <PathBlock value={renderOperationPath(operation)} />
                {operation.notes.length > 0 ? <p className="mt-2 text-[11px] text-amber-400">{operation.notes.join(" ")}</p> : null}
              </div>
            ))}
          </div>
        </div>
      </div>

      {plan.reasons.length > 0 ? <p className="mt-4 text-xs text-slate-400">Reasons: {plan.reasons.join(" • ")}</p> : null}
      {plan.notes.length > 0 ? <p className="mt-2 text-xs text-slate-400">Notes: {plan.notes.join(" ")}</p> : null}
    </Panel>
  );
}

export function PlanReportModal({
  report,
  sourceItemId,
  applyPending,
  usingMock,
  onClose,
  onApply,
}: {
  report: ActionPlanReport | null;
  sourceItemId?: string;
  applyPending: boolean;
  usingMock: boolean;
  onClose: () => void;
  onApply: () => void;
}) {
  const applyCapable = (report?.plans ?? []).some((plan) => plan.apply_supported);

  return (
    <Modal
      open={!!report}
      title="Generated Action Plan"
      subtitle={sourceItemId}
      onClose={() => {
        if (!applyPending) {
          onClose();
        }
      }}
      footer={
        <>
          <Button tone="ghost" onClick={onClose} disabled={applyPending}>
            Close
          </Button>
          <Button tone="primary" disabled={!applyCapable || usingMock || applyPending} onClick={onApply}>
            {applyCapable ? "Review Apply Options" : "Review-Only"}
          </Button>
        </>
      }
    >
      {report ? (
        <div className="space-y-6">
          {report.unsupported_items.length > 0 ? (
            <ActionBanner tone="info" message={report.unsupported_items.map((item) => `${item.source_review_item_id}: ${item.reason}`).join(" ")} />
          ) : null}
          {report.plans.map((plan) => (
            <PlanCard key={plan.plan_id} plan={plan} />
          ))}
        </div>
      ) : null}
    </Modal>
  );
}

export function ApplyResultPanel({
  result,
}: {
  result: ReviewPlanApplyResponse;
}) {
  const operationResults = result.results.flatMap((entry) => entry.operation_results);
  const successCount = operationResults.filter((entry) => isSuccessfulOperationStatus(entry.status)).length;
  const failureCount = operationResults.filter((entry) => entry.status === "error").length;
  const skippedCount = operationResults.filter((entry) => entry.status === "skipped").length;

  return (
    <Panel className="border-emerald-500/20 bg-emerald-500/5 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.24em] text-emerald-400">Apply Complete</p>
          <h3 className="mt-1 font-display text-xl font-bold text-slate-100">
            {result.result_count} plan result{result.result_count === 1 ? "" : "s"} processed
          </h3>
          <p className="mt-2 text-sm text-slate-300">
            {successCount} operations succeeded
            {failureCount > 0 ? `, ${failureCount} reported errors` : ""}
            {skippedCount > 0 ? `${failureCount > 0 ? "," : ""} ${skippedCount} skipped` : ""}
            .
          </p>
        </div>
        {result.batch_id ? <Chip tone="success">history batch {result.batch_id}</Chip> : null}
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {result.results.map((entry) => (
          <div key={entry.plan_id} className="rounded-xl border border-emerald-500/10 bg-background-dark/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-100">{entry.action_type}</p>
              <Chip tone={planExecutionTone(entry.status)}>{formatPlanExecutionStatus(entry.status)}</Chip>
            </div>
            <p className="mt-2 text-xs text-slate-500">{entry.source_review_item_ids.join(", ")}</p>
            <div className="mt-3 space-y-2">
              {entry.operation_results.map((operation) => (
                <div key={operation.operation_id} className="rounded-lg bg-background-dark px-3 py-2">
                  <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">{operation.operation_type}</p>
                  <PathBlock
                    value={[operation.path, operation.destination_path].filter(Boolean).join(" -> ") || operation.operation_id}
                    tone={operation.status === "ok" ? "success" : operation.status === "error" ? "danger" : "default"}
                  />
                  <p className={`mt-1 text-[11px] ${operation.status === "ok" ? "text-emerald-400" : operation.status === "error" ? "text-rose-400" : "text-slate-400"}`}>
                    {formatOperationExecutionStatus(operation.status)}: {operation.message}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
