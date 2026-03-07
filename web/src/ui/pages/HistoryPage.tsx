import { useState } from "react";
import { isSuccessfulUndoStatus } from "../../lib/contracts";
import type { HistoryMutationResponse } from "../../lib/types";
import { useHistoryQuery, useRestoreHistoryMutation, useUndoHistoryMutation } from "../../lib/hooks";
import { mockHistoryResponse } from "../../lib/mock-data";
import { resolveQueryData, toQueryNoticeState } from "../../lib/query-state";
import { useUrlBackedSelection } from "../../lib/url-selection";
import { ActionBanner, Button, Chip, EmptyState, Modal, PageHeader, PageQueryStateNotice, Panel, formatDate } from "../components";
import { HistoryDetailPanel } from "../history/HistoryDetailPanel";
import { SplitScreen } from "../shell";

export function HistoryPage() {
  const historyQuery = useHistoryQuery();
  const restoreMutation = useRestoreHistoryMutation();
  const undoMutation = useUndoHistoryMutation();
  const historyState = resolveQueryData(historyQuery, mockHistoryResponse);
  const response = historyState.data;
  const usingMock = historyState.usingMock;
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const [confirm, setConfirm] = useState<{ type: "restore" | "undo"; batchId: string } | null>(null);
  const [mutationResult, setMutationResult] = useState<HistoryMutationResponse | null>(null);
  const { selected, selectById } = useUrlBackedSelection({
    items: response.items,
    param: "batch",
    idKey: "batch_id",
  });

  async function handleHistoryMutation(kind: "restore" | "undo", batchId: string) {
    if (usingMock) return;
    try {
      const responsePayload =
        kind === "restore"
          ? await restoreMutation.mutateAsync({ batchId })
          : await undoMutation.mutateAsync({ batchId });
      setMutationResult(responsePayload);
      const successful = responsePayload.changed_operations.filter((item) => isSuccessfulUndoStatus(item.undo_status)).length;
      const unsupported = responsePayload.changed_operations.filter((item) => item.undo_status === "not_supported").length;
      const failed = responsePayload.changed_operations.filter((item) => item.undo_status === "error");
      const actionLabel = kind === "restore" ? "Restore" : "Undo";
      setBanner({
        tone: failed.length > 0 ? "error" : unsupported > 0 ? "info" : "success",
        message:
          failed.length > 0
            ? `${actionLabel} completed for ${batchId}. ${successful} successful, ${unsupported} not supported, ${failed.length} errors. ${failed.map((item) => item.undo_message ?? "operation failed").join(" ")}`
            : `${actionLabel} completed for ${batchId}. ${successful} successful, ${unsupported} not supported.`,
      });
      setConfirm(null);
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to mutate history batch." });
    }
  }

  const busy = restoreMutation.isPending || undoMutation.isPending;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="System Integrity Secure"
        title="Operation Audit Trail"
        description="Inspect reversible plan batches, confirm restore or undo explicitly, and keep review state honest when changes bring issues back."
      />
      <PageQueryStateNotice
        {...toQueryNoticeState(historyState)}
        fallbackMessage="Mock fallback is active. Restore and undo are disabled until the live API is available."
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}

      <SplitScreen
        main={
          <div className="space-y-6">
            <div className="flex overflow-x-auto border-b border-primary/20">
              {["All Activity", "Quarantine Moves", "Metadata Writes", "Restore Outcomes"].map((label, index) => (
                <button
                  key={label}
                  type="button"
                  className={`px-6 py-4 text-sm ${index === 0 ? "border-b-2 border-primary font-bold text-primary" : "font-medium text-slate-500 hover:text-slate-100"}`}
                >
                  {label}
                </button>
              ))}
            </div>

            {response.items.length === 0 ? (
              <EmptyState
                title="No history batches recorded yet"
                description="Applied review plans will appear here once a reversible or tracked action runs through the NyxCore ledger."
              />
            ) : (
              <Panel className="overflow-hidden">
                <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left">
                  <thead className="border-b border-primary/20 bg-primary/10">
                    <tr className="text-xs font-bold uppercase tracking-[0.22em] text-slate-500">
                      <th className="px-6 py-4">Timestamp</th>
                      <th className="px-6 py-4">Operation</th>
                      <th className="px-6 py-4">Integrity</th>
                      <th className="px-6 py-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {response.items.map((item) => {
                      const active = item.batch_id === selected?.batch_id;
                      return (
                        <tr
                          key={item.batch_id}
                          className={`cursor-pointer border-b border-primary/10 transition-colors ${active ? "border-l-4 border-l-primary bg-primary/10" : "hover:bg-primary/5"}`}
                          onClick={() => selectById(item.batch_id)}
                        >
                          <td className="px-6 py-5">
                            <div className="flex flex-col">
                              <span className="text-sm font-medium">{formatDate(item.applied_at)}</span>
                              <span className="text-xs text-slate-500">{item.batch_id}</span>
                            </div>
                          </td>
                          <td className="px-6 py-5">
                              <div className="flex items-center gap-3">
                                <div className="flex size-8 items-center justify-center rounded bg-primary/20 text-primary">
                                  <span className="material-symbols-outlined text-lg">database</span>
                                </div>
                              <div className="min-w-0">
                                <span className="block truncate text-sm font-semibold text-slate-100">{item.action_types.join(", ")}</span>
                                <p className="mt-1 text-xs text-slate-500">{item.affected_count} affected files</p>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-5">
                            <Chip tone={item.reversible ? "primary" : "warning"}>{item.reversible ? "reversible" : "mixed"}</Chip>
                          </td>
                          <td className="px-6 py-5 text-right">
                            <div className="flex justify-end gap-2">
                              <Button
                                tone="ghost"
                                className="px-3 py-1 text-xs"
                                disabled={usingMock || !item.reversible}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setConfirm({ type: "undo", batchId: item.batch_id });
                                }}
                              >
                                Undo
                              </Button>
                              <Button
                                tone="primary"
                                className="px-3 py-1 text-xs"
                                disabled={usingMock || !item.reversible}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setConfirm({ type: "restore", batchId: item.batch_id });
                                }}
                              >
                                Restore
                              </Button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                </div>
              </Panel>
            )}
          </div>
        }
        side={
          <div className="space-y-6">
            <HistoryDetailPanel
              batch={selected}
              mutationResult={mutationResult && mutationResult.batch_id === selected?.batch_id ? mutationResult : null}
              usingMock={usingMock}
              busy={busy}
              onUndo={() => selected && setConfirm({ type: "undo", batchId: selected.batch_id })}
              onRestore={() => selected && setConfirm({ type: "restore", batchId: selected.batch_id })}
            />
            <Panel className="p-6">
              <h3 className="mb-4 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-lg text-primary">analytics</span>
                Session Statistics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-slate-500">Total Ops</span>
                  <p className="text-xl font-bold">{response.items.reduce((sum, item) => sum + item.affected_count, 0)}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500">Undoable</span>
                  <p className="text-xl font-bold text-primary">{response.items.filter((item) => item.reversible).length}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500">Reactivated Review Items</span>
                  <p className="text-xl font-bold text-amber-400">{mutationResult?.reactivated_review_item_ids.length ?? 0}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500">Active Batches</span>
                  <p className="text-xl font-bold">{response.items.length}</p>
                </div>
              </div>
            </Panel>
          </div>
        }
      />

      <Modal
        open={!!confirm}
        title={confirm?.type === "restore" ? "Confirm Restore" : "Confirm Undo"}
        subtitle={confirm?.batchId}
        onClose={() => setConfirm(null)}
        footer={
          <>
            <Button tone="ghost" onClick={() => setConfirm(null)} disabled={busy}>
              Cancel
            </Button>
            <Button
              tone="primary"
              disabled={busy || !confirm}
              onClick={() => {
                if (confirm) {
                  void handleHistoryMutation(confirm.type, confirm.batchId);
                }
              }}
            >
              {busy ? "Processing..." : confirm?.type === "restore" ? "Restore Batch" : "Undo Batch"}
            </Button>
          </>
        }
      >
        <div className="space-y-4 text-sm text-slate-300">
          <p>This action uses the persisted operation ledger and recorded original paths for the selected batch.</p>
          <p>If a restore target is occupied or a file has moved outside NyxCore, the API will fail safely and report the conflict instead of overwriting anything.</p>
        </div>
      </Modal>
    </div>
  );
}
