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
  onUndo,
  onRestore,
}: {
  batch?: HistoryBatch;
  mutationResult?: HistoryMutationResponse | null;
  usingMock: boolean;
  busy: boolean;
  onUndo: () => void;
  onRestore: () => void;
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
        <>
          <Button tone="ghost" className="w-full" disabled={usingMock || busy || !batch?.reversible} onClick={onUndo}>
            Undo Batch
          </Button>
          <Button tone="primary" className="w-full" disabled={usingMock || busy || !batch?.reversible} onClick={onRestore}>
            Restore Batch
          </Button>
        </>
      }
    >
      {!batch ? (
        <div className="rounded-xl border border-dashed border-border-dark bg-background-dark/50 px-4 py-8 text-sm text-slate-500">
          Select a history batch to inspect operation steps, reversibility, and restore outcomes.
        </div>
      ) : (
        <>
          <Panel className="p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="font-display text-lg font-bold text-slate-100">{batch.action_types.join(", ")}</p>
                <p className="mt-1 text-xs text-slate-500">{formatDate(batch.applied_at)}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Chip tone={batch.reversible ? "primary" : "warning"}>{batch.reversible ? "reversible" : "non-reversible"}</Chip>
                <Chip tone="violet">{batch.affected_count} files</Chip>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Panel className="p-3">
                <p className="text-[10px] text-slate-500">Source Plans</p>
                <p className="mt-1 text-lg font-bold text-slate-100">{batch.source_plan_ids.length}</p>
              </Panel>
              <Panel className="p-3">
                <p className="text-[10px] text-slate-500">Review Items</p>
                <p className="mt-1 text-lg font-bold text-primary">{batch.source_review_item_ids.length}</p>
              </Panel>
            </div>
          </Panel>

          <Panel className="border-primary/10 bg-background-dark/40 p-4">
            <p className="text-xs text-slate-400">
              Restore and Undo currently call the same safe reversal path. Separate labels are preserved for command and API compatibility.
            </p>
          </Panel>

          {mutationResult ? (
            <Panel className="border-primary/20 bg-primary/5 p-4">
              <p className="text-xs font-bold uppercase tracking-[0.24em] text-primary">Latest Restore / Undo Outcome</p>
              <p className="mt-2 text-sm text-slate-300">
                {mutationSummary?.successful ?? 0} successful,
                {" "}
                {mutationSummary?.unsupported ?? 0} not supported,
                {" "}
                {mutationSummary?.errors ?? 0} errors.
              </p>
            </Panel>
          ) : null}

          <div>
            <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Operation Steps</p>
            <div className="space-y-2">
              {batch.operations.map((operation) => {
                const changed = changedById.get(operation.operation_id);
                return (
                  <div key={operation.operation_id} className="rounded-xl border border-border-dark bg-background-dark/70 px-4 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-slate-100">{operation.operation_type}</p>
                        <div className="mt-2">
                          <PathBlock value={operationPath(operation)} />
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Chip tone={operation.reversible ? "primary" : "warning"}>{operation.reversible ? "reversible" : "fixed"}</Chip>
                        <Chip tone={changed ? undoExecutionTone(changed.undo_status) : operationExecutionTone(operation.status)}>
                          {changed ? formatUndoExecutionStatus(changed.undo_status) : formatOperationExecutionStatus(operation.status)}
                        </Chip>
                      </div>
                    </div>
                    {changed?.undo_message ? <p className={`mt-2 text-xs ${changed.undo_status === "error" ? "text-rose-400" : "text-slate-400"}`}>{changed.undo_message}</p> : null}
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
