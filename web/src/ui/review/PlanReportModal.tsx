import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  actionOperationSelectionKey,
  FALLBACK_MAX_SELECTED_OPERATIONS,
  isActionOperationSelectable,
  isSelectableActionOperation,
} from "../../lib/action-plan";
import { formatPlanExecutionStatus, planExecutionTone } from "../../lib/contracts";
import { libraryRelativePath, pathFilename } from "../../lib/library-path";
import type { ActionPlan, ActionPlanOperation, ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";
import { ActionBanner, Button, Chip, Modal, Panel, PathBlock } from "../components";

const OPERATION_PAGE_SIZE = 20;

const ACTION_LABELS: Record<string, string> = {
  quarantine_move: "Quarantine duplicate",
  write_metadata: "Write metadata",
  rename_file: "Rename file",
  keep_preferred: "Keep preferred copy",
  plan_safety_check: "Safety check",
};

const ACTION_GROUP_LABELS: Record<string, string> = {
  quarantine_move: "Duplicate moves",
  write_metadata: "Metadata edits",
  rename_file: "Renames",
};

function actionLabel(type: string) {
  return ACTION_LABELS[type] ?? type.replace(/_/g, " ");
}

function OperationPath({ path, root, filenameOnly = false }: { path?: string | null; root?: string; filenameOnly?: boolean }) {
  if (!path) return <span className="text-primary-muted">Path unavailable</span>;
  const relative = libraryRelativePath(path, root);
  const visible = filenameOnly && relative !== null ? pathFilename(path) : relative ?? path;

  return (
    <div className="min-w-0 space-y-1">
      {filenameOnly && relative === null ? <p className="text-xs text-primary">{pathFilename(path)}</p> : null}
      <PathBlock value={visible} />
      {visible !== path ? (
        <details className="text-[11px] text-primary-muted">
          <summary className="cursor-pointer text-accent focus-visible:outline-accent">Full path</summary>
          <PathBlock className="mt-1 select-all" value={path} />
        </details>
      ) : (
        <span className="text-[11px] text-primary-subtle">Full path</span>
      )}
    </div>
  );
}

function preferredCopy(plan: ActionPlan) {
  const retained = plan.proposed_operations.filter((operation) => operation.operation_type === "keep_preferred" && operation.path);
  return retained.length === 1 ? retained[0].path : undefined;
}

function OperationDetails({
  operation, keeper, root, applies,
}: { operation: ActionPlanOperation; keeper?: string | null; root?: string; applies: boolean }) {
  const entries: Array<{ label: string; path?: string | null; filenameOnly?: boolean }> = [];
  if (operation.operation_type === "quarantine_move") {
    if (keeper && keeper !== operation.path && keeper !== operation.destination_path) entries.push({ label: "Keep", path: keeper });
    entries.push({ label: "Move from", path: operation.path }, { label: "Move to", path: operation.destination_path });
  } else if (operation.operation_type === "rename_file") {
    entries.push(
      { label: "Current filename", path: operation.path, filenameOnly: true },
      { label: "Proposed filename", path: operation.destination_path, filenameOnly: true },
    );
  } else if (operation.path) {
    entries.push({ label: "File", path: operation.path });
  }

  return (
    <div className="mt-2 min-w-0 space-y-2">
      {entries.length ? (
        <dl className="space-y-2">
          {entries.map(({ label, path, filenameOnly }) => (
            <div key={label} className="grid min-w-0 gap-1 sm:grid-cols-[7.5rem_minmax(0,1fr)] sm:items-start">
              <dt className="pt-1 text-xs font-semibold text-primary-muted">{label}</dt>
              <dd className="min-w-0"><OperationPath path={path} root={root} filenameOnly={filenameOnly} /></dd>
            </div>
          ))}
        </dl>
      ) : null}
      {operation.operation_type === "write_metadata" && operation.fields.length ? (
        <dl className="grid min-w-0 gap-1 rounded-[3px] bg-surface-low p-2 text-xs">
          {operation.fields.map((field) => (
            <div key={field} className="grid gap-1 sm:grid-cols-[7.5rem_minmax(0,1fr)]">
              <dt className="font-semibold capitalize text-primary-muted">{field.replace(/_/g, " ")}</dt>
              <dd className="text-primary [overflow-wrap:anywhere]">{operation.values[field] ?? "Unset"}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {operation.operation_type === "quarantine_move" && applies ? (
        <p className="text-xs text-primary-muted"><strong>Effect</strong> Relocates the file to quarantine and preserves it there for possible restore.</p>
      ) : null}
    </div>
  );
}

function PlanCard({
  plan, root, selectedTotal, selectedOperationIds, maximum, disabled, onToggleOperation,
}: {
  plan: ActionPlan;
  root?: string;
  selectedTotal: number;
  selectedOperationIds: ReadonlySet<string>;
  maximum: number;
  disabled: boolean;
  onToggleOperation: (planId: string, operationId: string, checked: boolean) => void;
}) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const keeper = preferredCopy(plan);
  const filteredOperations = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    // Preferred copies already appear as Keep on each move that has a grounded keeper.
    const hasNamedKeeper = keeper && plan.proposed_operations.some(
      (op) => op.operation_type === "quarantine_move" && keeper !== op.path && keeper !== op.destination_path,
    );
    return plan.proposed_operations.filter((operation) =>
      !(hasNamedKeeper && operation.operation_type === "keep_preferred")
      && (!query || [
        actionLabel(operation.operation_type), operation.path, operation.destination_path,
        ...operation.fields, ...Object.values(operation.values),
      ].join(" ").toLocaleLowerCase().includes(query)),
    );
  }, [filter, keeper, plan]);
  const pageCount = Math.max(1, Math.ceil(filteredOperations.length / OPERATION_PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visible = filteredOperations.slice(safePage * OPERATION_PAGE_SIZE, (safePage + 1) * OPERATION_PAGE_SIZE);

  return (
    <Panel className="min-w-0 border border-white/[0.07] bg-surface-low/60 p-3 sm:p-4">
      <div className="mb-3 border-b border-white/[0.06] pb-2">
        <h4 className="font-display text-sm font-semibold capitalize text-primary">{plan.action_type.replace(/_/g, " ")}</h4>
        {plan.reasons.length ? <p className="mt-1 text-xs text-primary-muted">{plan.reasons.join(" · ")}</p> : null}
      </div>
      {plan.proposed_operations.length > OPERATION_PAGE_SIZE ? (
        <input
          type="search"
          aria-label="Filter proposed operations"
          value={filter}
          onChange={(event) => { setFilter(event.target.value); setPage(0); }}
          placeholder="Filter proposed operations"
          className="mb-3 w-full rounded-[3px] border border-white/[0.08] bg-surface px-3 py-2 text-xs text-primary"
        />
      ) : null}
      <div className="space-y-2">
        {visible.map((operation) => {
          const selectable = isActionOperationSelectable(plan, operation);
          const checked = selectable && selectedOperationIds.has(actionOperationSelectionKey(plan.plan_id, operation.operation_id));
          const displayPath = operation.path && (libraryRelativePath(operation.path, root) ?? pathFilename(operation.path));
          return (
            <div key={operation.operation_id} className="min-w-0 rounded-[3px] border border-white/[0.07] bg-surface p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label className="flex items-center gap-2 text-xs font-semibold text-primary">
                  {selectable ? (
                    <input
                      type="checkbox"
                      aria-label={["Select", actionLabel(operation.operation_type).toLowerCase(), displayPath && "for " + displayPath].filter(Boolean).join(" ")}
                      checked={checked}
                      disabled={disabled || (!checked && selectedTotal >= maximum)}
                      onChange={(event) => onToggleOperation(plan.plan_id, operation.operation_id, event.target.checked)}
                      className="size-4 shrink-0 accent-accent"
                    />
                  ) : null}
                  {actionLabel(operation.operation_type)}
                </label>
                {checked ? <Chip tone="success">Selected</Chip> : null}
                {!selectable ? <Chip tone="neutral">Review only</Chip> : null}
              </div>
              <OperationDetails operation={operation} keeper={keeper} root={root} applies={selectable} />
              {!selectable ? <p className="mt-2 text-xs text-primary-muted">This operation cannot be applied.</p> : null}
              {operation.notes.length ? <p className="mt-2 text-xs text-primary-subtle [overflow-wrap:anywhere]">{operation.notes.join(" ")}</p> : null}
            </div>
          );
        })}
        {!visible.length ? <p className="py-4 text-center text-xs text-primary-muted">No operations match this filter.</p> : null}
      </div>
      {plan.notes.length ? <p className="mt-2 text-xs text-primary-muted">{plan.notes.join(" ")}</p> : null}
      {pageCount > 1 ? (
        <div className="mt-3 flex items-center justify-between gap-3">
          <Button tone="ghost" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>Previous</Button>
          <span className="text-xs text-primary-muted">Page {safePage + 1} / {pageCount}</span>
          <Button tone="ghost" disabled={safePage === pageCount - 1} onClick={() => setPage(safePage + 1)}>Next</Button>
        </div>
      ) : null}
    </Panel>
  );
}

export function PlanReportModal({
  report, selection, root, confirming, applyPending, applyError, selectedOperationIds,
  onToggleOperation, onClose, onContinue, onCancel, onApply,
}: {
  report: ActionPlanReport | null;
  selection: ActionPlanReport | null;
  root?: string;
  confirming: boolean;
  applyPending: boolean;
  applyError: string | null;
  selectedOperationIds: ReadonlySet<string>;
  onToggleOperation: (planId: string, operationId: string, checked: boolean) => void;
  onClose: () => void;
  onContinue: () => void;
  onCancel: () => void;
  onApply: () => void;
}) {
  const selectedTotal = selectedOperationIds.size;
  const maximum = report?.summary.max_automatic_operations ?? FALLBACK_MAX_SELECTED_OPERATIONS;
  const selectedChanges = selection?.plans.flatMap((plan) => {
    const keeper = preferredCopy(plan);
    return plan.proposed_operations.filter(isSelectableActionOperation).map((operation) => ({ planId: plan.plan_id, keeper, operation }));
  }) ?? [];
  const groups = new Map<string, number>();
  for (const { operation } of selectedChanges) {
    groups.set(operation.operation_type, (groups.get(operation.operation_type) ?? 0) + 1);
  }

  return (
    <Modal
      open={!!report}
      title={confirming ? "Confirm Plan Apply" : "Generated Action Plan"}
      subtitle={confirming ? selectedChanges.length + (selectedChanges.length === 1 ? " selected change" : " selected changes") : "Preview leaves audio files unchanged"}
      onClose={confirming ? onCancel : onClose}
      dismissible={!applyPending}
      footer={confirming ? (
        <>
          <Button tone="ghost" disabled={applyPending} onClick={onCancel}>Cancel</Button>
          <Button tone="primary" disabled={applyPending || !selectedChanges.length} onClick={onApply}>
            {applyPending ? "Applying…" : "Apply Plan"}
          </Button>
        </>
      ) : (
        <>
          <Button tone="ghost" onClick={onClose}>Close</Button>
          <Button tone="primary" disabled={!selectedChanges.length} onClick={onContinue}>Review Selected Changes</Button>
        </>
      )}
    >
      {confirming ? (
        <div className="min-w-0 space-y-3">
          {applyError ? <ActionBanner tone="error" message={applyError} /> : null}
          <p className="text-sm text-primary-muted">Apply changes the files listed below. Unchecked operations are excluded.</p>
          <div className="flex flex-wrap gap-2">
            {[...groups].map(([type, count]) => <Chip key={type} tone="default">{ACTION_GROUP_LABELS[type]} · {count}</Chip>)}
          </div>
          {selectedChanges.map(({ planId, keeper, operation }) => (
            <Panel key={actionOperationSelectionKey(planId, operation.operation_id)} className="min-w-0 bg-surface-low/60 p-3">
              <h4 className="text-sm font-semibold text-primary">{actionLabel(operation.operation_type)}</h4>
              <OperationDetails operation={operation} keeper={keeper} root={root} applies />
            </Panel>
          ))}
        </div>
      ) : report ? (
        <div className="min-w-0 space-y-3">
          <p className="text-xs text-primary-muted">{selectedTotal} / {maximum} selected. Choose the changes you want to review.</p>
          {report.unsupported_items.map((item) => (
            <ActionBanner key={item.source_review_item_id} message={"Review only. " + item.reason} />
          ))}
          {report.plans.map((plan) => (
            <PlanCard key={plan.plan_id} plan={plan} root={root} selectedTotal={selectedTotal}
              selectedOperationIds={selectedOperationIds} maximum={maximum} disabled={applyPending} onToggleOperation={onToggleOperation} />
          ))}
        </div>
      ) : null}
    </Modal>
  );
}

export function ApplyResultPanel({
  result, selection, root,
}: { result: ReviewPlanApplyResponse; selection: ActionPlanReport; root?: string }) {
  const resultsByKey = new Map(result.results.flatMap((plan) => plan.operation_results.map(
    (operation) => [actionOperationSelectionKey(plan.plan_id, operation.operation_id), operation.status] as const,
  )));
  const selectedStates = selection.plans.flatMap((plan) =>
    plan.proposed_operations.filter(isSelectableActionOperation).map(
      (operation) => resultsByKey.get(actionOperationSelectionKey(plan.plan_id, operation.operation_id)),
    ),
  );
  const applied = selectedStates.filter((state) => state === "ok").length;
  const failed = selectedStates.filter((state) => state === "error").length;
  const notApplied = selectedStates.length - applied - failed;
  const allApplied = applied > 0 && applied === selectedStates.length;
  const title = allApplied ? "All selected changes applied" : applied > 0 ? "Partially applied" : "No changes applied";
  return (
    <Panel className="min-w-0 space-y-4 p-4">
      <ActionBanner
        tone={allApplied ? "success" : "error"}
        message={title + ". " + applied + " applied · " + failed + " failed · " + notApplied + " not applied."}
        action={result.batch_id ? (
          <Link className="font-semibold text-accent underline-offset-4 hover:underline focus-visible:outline-accent"
            to={"/history?batch=" + encodeURIComponent(result.batch_id)}>Inspect History Batch</Link>
        ) : undefined}
      />
      <div className="grid min-w-0 gap-3 lg:grid-cols-2">
        {result.results.map((entry) => (
          <div key={entry.plan_id} className="min-w-0 rounded-[3px] border border-white/[0.06] bg-surface p-3.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h4 className="text-xs font-semibold capitalize text-primary">{entry.action_type.replace(/_/g, " ")}</h4>
              <Chip tone={planExecutionTone(entry.status)}>{formatPlanExecutionStatus(entry.status)}</Chip>
            </div>
            <div className="mt-3 space-y-2">
              {entry.operation_results.map((operation) => (
                <div key={operation.operation_id} className="space-y-1 rounded-[3px] bg-surface-low p-2.5">
                  <p className="text-xs font-semibold text-primary">{actionLabel(operation.operation_type)}</p>
                  {operation.path ? <OperationPath path={operation.path} root={root} /> : null}
                  {operation.destination_path ? <OperationPath path={operation.destination_path} root={root} /> : null}
                  <p className={operation.status === "error" ? "text-xs text-rose-400 [overflow-wrap:anywhere]" : "text-xs text-primary-muted [overflow-wrap:anywhere]"}>
                    <strong>{operation.status === "ok" ? "Applied" : operation.status === "error" ? "Failed" : "Skipped"}</strong> · {operation.message}
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
