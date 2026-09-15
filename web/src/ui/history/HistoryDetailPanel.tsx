import {
  formatOperationExecutionStatus,
  formatUndoExecutionStatus,
  isSuccessfulUndoStatus,
  operationExecutionTone,
  undoExecutionTone,
} from "../../lib/contracts";
import type { HistoryBatch, HistoryMutationResponse } from "../../lib/types";
import { Button, Chip, Drawer, Panel, PathBlock, formatDate } from "../components";

function operationPath(batchOperation: HistoryBatch["operations"][number]) {
  if (batchOperation.original_path && batchOperation.current_path && batchOperation.original_path !== batchOperation.current_path) {
    return `${batchOperation.original_path} -> ${batchOperation.current_path}`;
  }
  return batchOperation.current_path ?? batchOperation.original_path ?? batchOperation.operation_id;
}

export function HistoryDetailPanel({
  batch,
  mutationResult,
  usingMock,
  busy,
  onReverse,
}: {
  batch?: HistoryBatch;
  mutationResult?: HistoryMutationResponse | null;
  usingMock?: boolean;
  busy: boolean;
  onReverse: () => void;
}) {
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
        <Button tone="primary" className="w-full" disabled={usingMock || busy || !batch?.reversible} onClick={onReverse}>
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
                <Chip tone={batch.reversible ? "success" : "warning"}>
                  {batch.reversible ? "Reversible" : "Non-reversible"}
                </Chip>
                <Chip tone="default">{batch.affected_count} files</Chip>
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
                return (
                  <div key={operation.operation_id} className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="font-sans text-xs font-semibold text-primary">{operation.operation_type}</p>
                      <div className="flex flex-wrap gap-1.5">
                        <Chip tone={operation.reversible ? "success" : "warning"}>
                          {operation.reversible ? "Reversible" : "Fixed"}
                        </Chip>
                        <Chip tone={changed ? undoExecutionTone(changed.undo_status) : operationExecutionTone(operation.status)}>
                          {changed ? formatUndoExecutionStatus(changed.undo_status) : formatOperationExecutionStatus(operation.status)}
                        </Chip>
                      </div>
                    </div>
                    <div className="mt-2">
                      <PathBlock value={operationPath(operation)} />
                    </div>
                    {changed?.undo_message ? (
                      <p className={`mt-2 font-mono text-xs ${changed.undo_status === "error" ? "text-rose-400" : "text-primary-muted"}`}>
                        {changed.undo_message}
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
