import { useMemo, useState } from "react";
import {
  actionOperationSelectionKey,
  FALLBACK_MAX_SELECTED_OPERATIONS,
  isActionOperationSelectable,
  isSelectableActionOperation,
} from "../../lib/action-plan";
import {
  formatOperationExecutionStatus,
  formatPlanExecutionStatus,
  isSuccessfulOperationStatus,
  planExecutionTone,
} from "../../lib/contracts";
import type { ActionPlan, ActionPlanOperation, ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";
import { ActionBanner, Button, Chip, Modal, Panel, PathBlock } from "../components";

const OPERATION_PAGE_SIZE = 20;

function renderOperationPath(operation: ActionPlanOperation) {
  if (operation.path && operation.destination_path) {
    return `${operation.path} -> ${operation.destination_path}`;
  }
  if (operation.path) {
    const fields = operation.fields.map((field) => `${field}: ${operation.values[field] ?? "unset"}`);
    return fields.length > 0 ? `${operation.path} | ${fields.join(" | ")}` : operation.path;
  }
  const fields = operation.fields.map((field) => `${field}: ${operation.values[field] ?? "unset"}`);
  return fields.length > 0 ? fields.join(" | ") : "review-only";
}

function PlanCard({
  plan,
  selectedTotal,
  selectedOperationIds,
  maximum,
  disabled,
  onToggleOperation,
}: {
  plan: ActionPlan;
  selectedTotal: number;
  selectedOperationIds: ReadonlySet<string>;
  maximum: number;
  disabled: boolean;
  onToggleOperation: (planId: string, operationId: string, checked: boolean) => void;
}) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const availableInPlan = plan.proposed_operations.filter((operation) =>
    isActionOperationSelectable(plan, operation)
  ).length;
  const selectedInPlan = plan.proposed_operations.filter(
    (operation) =>
      isActionOperationSelectable(plan, operation)
      && selectedOperationIds.has(actionOperationSelectionKey(plan.plan_id, operation.operation_id)),
  ).length;
  const filteredOperations = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    if (!query) return plan.proposed_operations;
    return plan.proposed_operations.filter((operation) =>
      [
        operation.operation_type,
        operation.path,
        operation.destination_path,
        operation.fields.join(" "),
        Object.values(operation.values).join(" "),
      ]
        .filter(Boolean)
        .join(" ")
        .toLocaleLowerCase()
        .includes(query),
    );
  }, [filter, plan.proposed_operations]);
  const pageCount = Math.max(1, Math.ceil(filteredOperations.length / OPERATION_PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visibleOperations = filteredOperations.slice(
    safePage * OPERATION_PAGE_SIZE,
    (safePage + 1) * OPERATION_PAGE_SIZE,
  );
  const displayedFiles = plan.affected_files.slice(0, 8);

  return (
    <Panel className="border border-border bg-surface-low p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <p className="font-display text-base font-bold uppercase tracking-wider text-primary">{plan.action_type}</p>
          <p className="mt-1 font-mono text-xs text-primary-subtle">
            Safety: {plan.safety_level} • Confidence: {(plan.confidence * 100).toFixed(0)}% • {plan.affected_files.length} files
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Chip tone={selectedInPlan > 0 ? "primary" : availableInPlan > 0 ? "warning" : "neutral"}>
            {selectedInPlan > 0 ? `${selectedInPlan} selected` : availableInPlan > 0 ? `${availableInPlan} available` : "review-only"}
          </Chip>
          <Chip tone="neutral">{plan.proposed_operations.length} ops</Chip>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          <p className="mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Affected Files</p>
          <div className="space-y-1.5">
            {displayedFiles.map((file) => (
              <PathBlock key={file} value={file} />
            ))}
            {plan.affected_files.length > displayedFiles.length ? (
              <p className="font-mono text-[10px] text-primary-subtle">
                {plan.affected_files.length - displayedFiles.length} more files are listed with the operations.
              </p>
            ) : null}
          </div>
        </div>
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Proposed Operations</p>
            <span className="font-mono text-[10px] text-primary-subtle">
              {selectedTotal}/{maximum} selected overall
            </span>
          </div>
          {plan.proposed_operations.length > OPERATION_PAGE_SIZE ? (
            <input
              type="search"
              value={filter}
              onChange={(event) => {
                setFilter(event.target.value);
                setPage(0);
              }}
              placeholder="Filter by path, type, field, or value"
              className="mb-3 w-full border border-border bg-surface px-3 py-1.5 font-mono text-xs text-primary outline-none focus:border-accent"
            />
          ) : null}
          <div className="space-y-1.5">
            {visibleOperations.map((operation) => {
              const selectable = isActionOperationSelectable(plan, operation);
              const checked = selectable && selectedOperationIds.has(
                actionOperationSelectionKey(plan.plan_id, operation.operation_id),
              );
              return (
                <label key={operation.operation_id} className="block border border-border bg-surface p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-3">
                      {selectable ? (
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={disabled || (!checked && selectedTotal >= maximum)}
                          onChange={(event) => onToggleOperation(plan.plan_id, operation.operation_id, event.target.checked)}
                          className="size-4 shrink-0 accent-accent"
                        />
                      ) : null}
                      <p className="truncate font-mono text-xs font-semibold text-primary">{operation.operation_type}</p>
                    </div>
                    <Chip tone={checked ? "success" : selectable ? "warning" : "neutral"}>
                      {checked ? "selected" : selectable ? "available" : "review-only"}
                    </Chip>
                  </div>
                  <div className="mt-2">
                    <PathBlock value={renderOperationPath(operation)} />
                  </div>
                  {operation.notes.length > 0 ? <p className="mt-1.5 font-mono text-[10px] text-amber-400">{operation.notes.join(" ")}</p> : null}
                </label>
              );
            })}
            {visibleOperations.length === 0 ? (
              <p className="border border-border bg-surface-low px-3 py-6 text-center font-mono text-xs text-primary-subtle">
                No operations match this filter.
              </p>
            ) : null}
          </div>
          {pageCount > 1 ? (
            <div className="mt-3 flex items-center justify-between gap-3">
              <Button tone="ghost" className="px-3 py-1 text-xs" disabled={safePage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                Previous
              </Button>
              <span className="font-mono text-xs text-primary-subtle">Page {safePage + 1} / {pageCount}</span>
              <Button tone="ghost" className="px-3 py-1 text-xs" disabled={safePage >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>
                Next
              </Button>
            </div>
          ) : null}
        </div>
      </div>

      {plan.reasons.length > 0 ? <p className="mt-4 font-mono text-xs text-primary-muted">Reasons: {plan.reasons.join(" • ")}</p> : null}
      {plan.notes.length > 0 ? <p className="mt-1 font-mono text-xs text-primary-muted">Notes: {plan.notes.join(" ")}</p> : null}
    </Panel>
  );
}

export function PlanReportModal({
  report,
  sourceItemId,
  applyPending,
  usingMock,
  selectedOperationIds,
  onToggleOperation,
  onClose,
  onApply,
}: {
  report: ActionPlanReport | null;
  sourceItemId?: string;
  applyPending: boolean;
  usingMock: boolean;
  selectedOperationIds: ReadonlySet<string>;
  onToggleOperation: (planId: string, operationId: string, checked: boolean) => void;
  onClose: () => void;
  onApply: () => void;
}) {
  const selectedTotal = selectedOperationIds.size;
  const maximum = report?.summary.max_automatic_operations ?? FALLBACK_MAX_SELECTED_OPERATIONS;
  const applyCapable = selectedTotal > 0;

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
            {applyCapable ? `Apply ${selectedTotal} Selected` : "Select Operations"}
          </Button>
        </>
      }
    >
      {report ? (
        <div className="space-y-6">
          {report.unsupported_items.length > 0 ? (
            <ActionBanner tone="info" message={report.unsupported_items.map((item) => `${item.source_review_item_id}: ${item.reason}`).join(" ")} />
          ) : null}
          {selectedTotal >= maximum ? (
            <ActionBanner tone="info" message={`Selection limit reached. Apply these ${maximum} operations before choosing another batch.`} />
          ) : null}
          {report.plans.map((plan) => (
            <PlanCard
              key={plan.plan_id}
              plan={plan}
              selectedTotal={selectedTotal}
              selectedOperationIds={selectedOperationIds}
              maximum={maximum}
              disabled={usingMock || applyPending}
              onToggleOperation={onToggleOperation}
            />
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
    <Panel className="border border-emerald-500/30 bg-surface-low p-5">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-3">
        <div>
          <p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-400">Apply Complete</p>
          <h3 className="mt-1 font-display text-base font-bold uppercase tracking-wider text-primary">
            {result.result_count} plan result{result.result_count === 1 ? "" : "s"} processed
          </h3>
          <p className="mt-1 font-mono text-xs text-primary-muted">
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
          <div key={entry.plan_id} className="border border-border bg-surface p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="font-mono text-xs font-semibold text-primary">{entry.action_type}</p>
              <Chip tone={planExecutionTone(entry.status)}>{formatPlanExecutionStatus(entry.status)}</Chip>
            </div>
            <p className="mt-1 font-mono text-[10px] text-primary-subtle">{entry.source_review_item_ids.join(", ")}</p>
            <div className="mt-3 space-y-1.5">
              {entry.operation_results.map((operation) => (
                <div key={operation.operation_id} className="border border-border bg-surface-low p-2.5">
                  <p className="font-mono text-[10px] font-bold uppercase tracking-wider text-primary-subtle">{operation.operation_type}</p>
                  <div className="mt-1">
                    <PathBlock
                      value={[operation.path, operation.destination_path].filter(Boolean).join(" -> ") || operation.operation_id}
                      tone={operation.status === "ok" ? "success" : operation.status === "error" ? "danger" : "default"}
                    />
                  </div>
                  <p className={`mt-1 font-mono text-[10px] ${operation.status === "ok" ? "text-emerald-400" : operation.status === "error" ? "text-rose-400" : "text-primary-muted"}`}>
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
