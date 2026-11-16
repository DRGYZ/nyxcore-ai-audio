import {
  HISTORY_OPERATION_LABELS,
  canReverseHistoryBatch,
  formatUndoExecutionStatus,
  historyBatchState,
  isSuccessfulUndoStatus,
  operationExecutionTone,
  undoExecutionTone,
} from "../../lib/contracts";
import type { ApiMeta, HistoryBatch, HistoryMutationResponse } from "../../lib/types";
import { Button, Chip, Drawer, Panel, PathBlock, formatDate } from "../components";

function operationPath(batchOperation: HistoryBatch["operations"][number]) {
  if (batchOperation.original_path && batchOperation.current_path && batchOperation.original_path !== batchOperation.current_path) {
    return `${batchOperation.original_path} -> ${batchOperation.current_path}`;
  }
  return batchOperation.current_path ?? batchOperation.original_path ?? batchOperation.operation_id;
}

function RecoveryInstructions({ batchId, meta }: { batchId: string; meta: ApiMeta }) {
  const windows = /^(?:[a-z]:[\\/]|\\\\)/i.test(meta.music_path);
  const quote = (value: string) => "'" + (windows
    ? value.replace(/['‘’]/g, "$&$&")
    : value.replace(/'/g, "'\"'\"'")) + "'";
  const command = `python -m nyxcore.cli recover-review-action ${quote(batchId)} --action inspect --music ${quote(meta.music_path)} --out ${quote(meta.out_path)}`;

  return (
    <Panel className="border border-amber-500/30 bg-amber-500/10 p-3.5">
      <h4 className="font-sans text-sm font-semibold text-amber-300">Recovery inspection required</h4>
      <p className="mt-1.5 text-xs leading-relaxed text-primary-muted">
        Inspect this batch before further mutation. Inspection reads the journal and checks the files. Some states require manual intervention.
      </p>
      <p className="mt-2 text-xs text-primary-muted">
        Run in {windows ? "PowerShell" : "the library's shell"} with the NyxCore Python environment active.
      </p>
      <pre className="mt-2 select-all whitespace-pre-wrap rounded-[3px] bg-surface p-2.5 font-mono text-xs text-primary [overflow-wrap:anywhere]"><code>{command}</code></pre>
    </Panel>
  );
}

export function HistoryDetailPanel({
  batch,
  meta,
  mutationResult,
  usingMock,
  busy,
  onReverse,
}: {
  batch?: HistoryBatch;
  meta: ApiMeta;
  mutationResult?: HistoryMutationResponse | null;
  usingMock?: boolean;
  busy: boolean;
  onReverse: () => void;
}) {
  const state = batch ? historyBatchState(batch) : null;
  const changedById = new Map((mutationResult?.changed_operations ?? []).map((operation) => [operation.operation_id, operation]));
  const mutationSummary = mutationResult
    ? {
        successful: mutationResult.changed_operations.filter((operation) => isSuccessfulUndoStatus(operation.undo_status)).length,
        unsupported: mutationResult.changed_operations.filter((operation) => operation.undo_status === "not_supported").length,
        errors: mutationResult.changed_operations.filter((operation) => operation.undo_status === "error").length,
      }
    : null;

  return (
    <Drawer
      title="Batch Details"
      subtitle={batch ? `BATCH: ${batch.batch_id}` : "No batch selected"}
      footer={
        <Button tone="primary" className="w-full" disabled={usingMock || busy || !batch || !canReverseHistoryBatch(batch)} onClick={onReverse}>
          Reverse Batch
        </Button>
      }
    >
      {!batch ? (
        <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-4 py-8 text-center font-editorial text-xs italic text-primary-subtle">
          Select a history batch to inspect operation steps, reversibility, and restore outcomes.
        </div>
      ) : (
        <>
          <Panel className="p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="font-sans text-xs font-semibold text-primary">{batch.action_types.join(", ")}</p>
                <p className="mt-0.5 font-mono text-[11px] text-primary-subtle">{formatDate(batch.applied_at)}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {state ? <Chip tone={state.tone}>{state.label}</Chip> : null}
                <Chip tone="default">{batch.affected_count} {batch.affected_count === 1 ? "operation" : "operations"}</Chip>
              </div>
            </div>
            <div className="mt-3.5 grid grid-cols-2 gap-2.5">
              <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-2.5">
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Source Plans</p>
                <p className="mt-1 font-display text-lg font-bold tracking-tight text-primary">{batch.source_plan_ids.length}</p>
              </div>
              <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-2.5">
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Review Items</p>
                <p className="mt-1 font-display text-lg font-bold tracking-tight text-accent">{batch.source_review_item_ids.length}</p>
              </div>
            </div>
          </Panel>

          {batch.recovery_required ? <RecoveryInstructions batchId={batch.batch_id} meta={meta} /> : null}

          {mutationResult ? (
            <Panel className="border border-accent/30 bg-accent/[0.04] p-3.5">
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-accent">Latest Reversal Outcome</p>
              <p className="mt-1.5 font-sans text-xs text-primary-muted">
                {mutationSummary?.successful ?? 0} successful,
                {" "}
                {mutationSummary?.unsupported ?? 0} not supported,
                {" "}
                {mutationSummary?.errors ?? 0} errors.
              </p>
            </Panel>
          ) : null}

          <div>
            <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Operation Steps</p>
            <div className="space-y-2">
              {batch.operations.map((operation) => {
                const changed = changedById.get(operation.operation_id);
                const undoStatus = changed?.undo_status ?? operation.undo_status;
                const undoMessage = changed?.undo_message ?? operation.undo_message;
                const journalProgress = operation.journal_state === "prepared" || operation.journal_state === "applying"
                  ? HISTORY_OPERATION_LABELS[operation.journal_state]
                  : null;
                return (
                  <div key={operation.operation_id} className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="font-sans text-xs font-semibold capitalize text-primary">{operation.operation_type.replace(/_/g, " ")}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {journalProgress ? <Chip tone="warning">Journal state: {journalProgress}</Chip> : null}
                        {operation.status !== operation.journal_state || !journalProgress ? (
                          <Chip tone={operationExecutionTone(operation.status)}>
                            {HISTORY_OPERATION_LABELS[operation.status]}
                          </Chip>
                        ) : null}
                        {operation.reversal_pending ? <Chip tone="warning">Reversal pending inspection</Chip> : null}
                        {undoStatus !== "pending" ? (
                          <Chip tone={undoExecutionTone(undoStatus)}>
                            {undoStatus === "ok" ? "Reversed" : `Reversal: ${formatUndoExecutionStatus(undoStatus)}`}
                          </Chip>
                        ) : null}
                      </div>
                    </div>
                    <div className="mt-2">
                      <PathBlock value={operationPath(operation)} />
                    </div>
                    {operation.failure_reason ? (
                      <p className="mt-2 break-words font-sans text-xs text-rose-400">{operation.failure_reason}</p>
                    ) : null}
                    {undoMessage ? (
                      <p className={`mt-2 break-words font-mono text-xs ${undoStatus === "error" ? "text-rose-400" : "text-primary-muted"}`}>
                        {undoMessage}
                      </p>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </Drawer>
  );
}
