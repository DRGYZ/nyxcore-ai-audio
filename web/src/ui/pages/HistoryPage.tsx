import { useMemo, useState } from "react";
import { canReverseHistoryBatch, historyBatchState, isSuccessfulUndoStatus } from "../../lib/contracts";
import type { HistoryMutationResponse } from "../../lib/types";
import { useHistoryQuery, useReverseHistoryMutation, useStatusQuery } from "../../lib/hooks";
import { useUrlBackedSelection } from "../../lib/url-selection";
import { ActionBanner, Button, Chip, EmptyState, Icon, Modal, PageHeader, Panel, formatDate } from "../components";
import { ApiUnavailableState } from "../feedback";
import { HistoryDetailPanel } from "../history/HistoryDetailPanel";
import { SplitScreen } from "../shell";

type HistoryFilter = "all" | "quarantine" | "metadata" | "restore";

const HISTORY_FILTERS: Array<{ value: HistoryFilter; label: string }> = [
  { value: "all", label: "All Activity" },
  { value: "quarantine", label: "Quarantine Moves" },
  { value: "metadata", label: "Metadata Writes" },
  { value: "restore", label: "Restore Outcomes" },
];

export function HistoryPage() {
  const historyQuery = useHistoryQuery();
  const statusQuery = useStatusQuery();
  const reviewOnly = statusQuery.data?.review_only === true;
  const reverseMutation = useReverseHistoryMutation();
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const [confirmBatchId, setConfirmBatchId] = useState<string | null>(null);
  const [mutationResult, setMutationResult] = useState<HistoryMutationResponse | null>(null);
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>("all");

  const response = historyQuery.data;
  const items = useMemo(() => response?.items ?? [], [response?.items]);
  const filteredHistory = useMemo(
    () => items.filter((item) => {
      if (historyFilter === "all") return true;
      if (historyFilter === "quarantine") {
        return item.operations.some((operation) => operation.operation_type === "quarantine_move");
      }
      if (historyFilter === "metadata") {
        return item.operations.some((operation) => operation.operation_type === "write_metadata");
      }
      return item.operations.some((operation) => operation.undo_status !== "pending");
    }),
    [historyFilter, items],
  );
  const { selected, selectById } = useUrlBackedSelection({
    items: filteredHistory,
    param: "batch",
    idKey: "batch_id",
    syncUrl: historyQuery.isSuccess && !historyQuery.isFetching,
  });

  function refreshHistory() {
    void historyQuery.refetch();
    void statusQuery.refetch();
  }

  const header = (
    <PageHeader
      eyebrow="Ledger Operations"
      title="Operation History"
      description="Inspect recorded batches, recovery states, and reversal outcomes."
      actions={response && !historyQuery.isError ? (
        <Button tone="secondary" disabled={historyQuery.isFetching} onClick={refreshHistory}>Refresh History</Button>
      ) : undefined}
    />
  );

  if (historyQuery.isError || !response) {
    const showError = historyQuery.isError && !statusQuery.isPending && !statusQuery.isFetching;
    return (
      <div className="space-y-6">
        {header}
        {showError ? (
          statusQuery.isError ? (
            <ApiUnavailableState contextLabel="History" />
          ) : (
            <ActionBanner
              tone="error"
              message="History could not be loaded. The local API is connected. Retry to reload the operation ledger."
              action={<Button tone="secondary" onClick={refreshHistory}>Retry History</Button>}
            />
          )
        ) : (
          <Panel className="p-8 text-center text-sm text-primary-muted">
            Loading operation history from local API…
          </Panel>
        )}
      </div>
    );
  }

  async function handleReverse(batchId: string) {
    if (reviewOnly) return;
    try {
      const responsePayload = await reverseMutation.mutateAsync({ batchId });
      setMutationResult(responsePayload);
      const successful = responsePayload.changed_operations.filter((item) => isSuccessfulUndoStatus(item.undo_status)).length;
      const unsupported = responsePayload.changed_operations.filter((item) => item.undo_status === "not_supported").length;
      const failed = responsePayload.changed_operations.filter((item) => item.undo_status === "error");
      setBanner({
        tone: failed.length > 0 ? "error" : unsupported > 0 ? "info" : "success",
        message:
          failed.length > 0
            ? `Reversal completed for ${batchId}. ${successful} successful, ${unsupported} not supported, ${failed.length} errors. ${failed.map((item) => item.undo_message ?? "operation failed").join(" ")}`
            : `Reversal completed for ${batchId}. ${successful} successful, ${unsupported} not supported.`,
      });
      setConfirmBatchId(null);
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to reverse history batch." });
      setConfirmBatchId(null);
      refreshHistory();
    }
  }

  const busy = reverseMutation.isPending || historyQuery.isFetching;
  const confirmBatch = items.find((item) => item.batch_id === confirmBatchId);

  return (
    <div className="space-y-6">
      {header}
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}

      <SplitScreen
        main={
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 border-b border-white/[0.07] pb-3">
              {HISTORY_FILTERS.map(({ value, label }) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={historyFilter === value}
                  onClick={() => setHistoryFilter(value)}
                  className={`rounded-[3px] px-3.5 py-1.5 font-sans text-xs font-medium transition-colors ${
                    historyFilter === value
                      ? "bg-white/[0.08] text-accent shadow-sm ring-1 ring-accent/30"
                      : "text-primary-subtle hover:bg-white/[0.03] hover:text-primary"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {filteredHistory.length === 0 ? (
              <EmptyState
                title={response.items.length === 0 ? "No history batches recorded yet" : "No batches match this filter"}
                description={response.items.length === 0
                  ? "Applied review plans will appear here once a reversible or tracked action runs through the NyxCore ledger."
                  : "Choose another activity type to inspect the operation ledger."}
              />
            ) : (
              <Panel className="overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-left sm:min-w-[680px]">
                    <thead className="border-b border-white/[0.07] bg-surface-low/80">
                      <tr className="font-sans text-[11px] font-medium text-primary-subtle">
                        <th className="px-2.5 py-3 sm:px-5">Timestamp & ID</th>
                        <th className="hidden px-5 py-3 sm:table-cell">Operation Details</th>
                        <th className="px-2.5 py-3 sm:px-5">State</th>
                        <th className="px-2.5 py-3 text-right sm:px-5">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.05]">
                      {filteredHistory.map((item) => {
                        const active = item.batch_id === selected?.batch_id;
                        const state = historyBatchState(item);
                        return (
                          <tr
                            key={item.batch_id}
                            className={`transition-colors ${
                              active
                                ? "border-l-2 border-l-accent bg-surface-raised"
                                : "hover:bg-surface-raised/50"
                            }`}
                          >
                            <td className="px-2.5 py-3.5 sm:px-5">
                              <button
                                type="button"
                                id={`history-batch-trigger-${item.batch_id}`}
                                onClick={() => selectById(item.batch_id)}
                                aria-label={`Inspect batch ${item.batch_id}, recorded ${formatDate(item.applied_at)}`}
                                className="flex flex-col text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent rounded-[2px] group"
                              >
                                <span className="font-sans text-xs font-medium text-primary group-hover:text-accent transition-colors">{formatDate(item.applied_at)}</span>
                                <span className="[overflow-wrap:anywhere] font-mono text-[10px] text-primary-subtle">{item.batch_id}</span>
                              </button>
                            </td>
                            <td className="hidden px-5 py-3.5 sm:table-cell">
                              <div className="flex items-center gap-2.5">
                                <div className="flex size-7 shrink-0 items-center justify-center rounded-[2px] border border-white/[0.08] bg-surface text-accent">
                                  <Icon name="database" className="text-sm" />
                                </div>
                                <div className="min-w-0">
                                  <span className="block truncate font-sans text-xs font-medium text-primary">{item.action_types.join(", ")}</span>
                                  <p className="mt-0.5 font-sans text-[11px] text-primary-subtle">
                                    {item.affected_count} {item.affected_count === 1 ? "operation" : "operations"}
                                  </p>
                                </div>
                              </div>
                            </td>
                            <td className="px-2.5 py-3.5 sm:px-5">
                              <Chip tone={state.tone}>{state.label}</Chip>
                            </td>
                            <td className="px-2.5 py-3.5 text-right sm:px-5">
                              <Button
                                tone="secondary"
                                id={`reverse-button-${item.batch_id}`}
                                className="px-2 py-1 text-xs sm:px-3"
                                disabled={reviewOnly || !canReverseHistoryBatch(item) || busy}
                                onClick={() => {
                                  setConfirmBatchId(item.batch_id);
                                }}
                              >
                                Reverse
                              </Button>
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
              meta={response.meta}
              mutationResult={mutationResult && mutationResult.batch_id === selected?.batch_id ? mutationResult : null}
              reversalDisabled={busy || reviewOnly}
              onReverse={() => selected && canReverseHistoryBatch(selected) && setConfirmBatchId(selected.batch_id)}
            />
            <Panel className="p-5">
              <h3 className="mb-4 flex items-center gap-2 border-b border-white/[0.07] pb-3 font-display text-sm font-semibold tracking-wide text-primary">
                <Icon name="analytics" className="text-base text-accent" />
                Session Statistics
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Total Ops</span>
                  <p className="mt-1 font-display text-xl font-bold tracking-tight text-primary">
                    {response.items.reduce((sum, item) => sum + item.affected_count, 0)}
                  </p>
                </div>
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Reversible</span>
                  <p className="mt-1 font-display text-xl font-bold tracking-tight text-accent">
                    {response.items.filter(canReverseHistoryBatch).length}
                  </p>
                </div>
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Reactivated</span>
                  <p className="mt-1 font-display text-xl font-bold tracking-tight text-amber-400">
                    {mutationResult?.reactivated_review_item_ids.length ?? 0}
                  </p>
                </div>
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Batches</span>
                  <p className="mt-1 font-display text-xl font-bold tracking-tight text-primary">
                    {response.items.length}
                  </p>
                </div>
              </div>
            </Panel>
          </div>
        }
      />

      <Modal
        open={!!confirmBatch}
        title="Confirm Reversal"
        subtitle={confirmBatchId ?? undefined}
        onClose={() => setConfirmBatchId(null)}
        footer={
          <>
            <Button tone="ghost" onClick={() => setConfirmBatchId(null)} disabled={busy}>
              Cancel
            </Button>
            <Button
              tone="primary"
            disabled={reviewOnly || busy || !confirmBatch || !canReverseHistoryBatch(confirmBatch)}
              onClick={() => {
                if (!busy && confirmBatch && canReverseHistoryBatch(confirmBatch)) {
                  void handleReverse(confirmBatch.batch_id);
                }
              }}
            >
              {reverseMutation.isPending ? "Reversing..." : "Reverse Batch"}
            </Button>
          </>
        }
      >
        <div className="space-y-3 text-xs leading-relaxed text-slate-300">
          <p>This action uses the persisted operation ledger and recorded original paths to safely reverse the selected batch.</p>
          <p>If a target path is occupied or a file has moved outside NyxCore, the operation halts safely and reports the conflict without overwriting.</p>
        </div>
      </Modal>
    </div>
  );
}
