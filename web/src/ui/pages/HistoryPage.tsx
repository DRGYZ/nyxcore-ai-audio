import { useMemo, useState } from "react";
import { isSuccessfulUndoStatus } from "../../lib/contracts";
import type { HistoryMutationResponse } from "../../lib/types";
import { useHistoryQuery, useReverseHistoryMutation } from "../../lib/hooks";
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
  });

  if (historyQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Ledger Operations"
          title="Operation History"
          description="Inspect applied batches, track ledger mutations, and safely reverse changes back to original file locations."
        />
        <ApiUnavailableState contextLabel="History" />
      </div>
    );
  }

  if (historyQuery.isLoading || !response) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Ledger Operations"
          title="Operation History"
          description="Inspect applied batches, track ledger mutations, and safely reverse changes back to original file locations."
        />
        <Panel className="p-8 text-center text-sm text-primary-muted">
          Loading operation history from local API…
        </Panel>
      </div>
    );
  }

  async function handleReverse(batchId: string) {
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
    }
  }

  const busy = reverseMutation.isPending;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Ledger Operations"
        title="Operation History"
        description="Inspect applied batches, track ledger mutations, and safely reverse changes back to original file locations."
      />
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
                  <table className="w-full min-w-[680px] text-left">
                    <thead className="border-b border-white/[0.07] bg-surface-low/80">
                      <tr className="font-sans text-[11px] font-medium text-primary-subtle">
                        <th className="px-5 py-3">Timestamp & ID</th>
                        <th className="px-5 py-3">Operation Details</th>
                        <th className="px-5 py-3">Reversibility</th>
                        <th className="px-5 py-3 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.05]">
                      {filteredHistory.map((item) => {
                        const active = item.batch_id === selected?.batch_id;
                        return (
                          <tr
                            key={item.batch_id}
                            className={`transition-colors ${
                              active
                                ? "border-l-2 border-l-accent bg-surface-raised"
                                : "hover:bg-surface-raised/50"
                            }`}
                          >
                            <td className="px-5 py-3.5">
                              <button
                                type="button"
                                id={`history-batch-trigger-${item.batch_id}`}
                                onClick={() => selectById(item.batch_id)}
                                aria-label={`Inspect batch ${item.batch_id}, applied ${formatDate(item.applied_at)}`}
                                className="flex flex-col text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent rounded-[2px] group"
                              >
                                <span className="font-sans text-xs font-medium text-primary group-hover:text-accent transition-colors">{formatDate(item.applied_at)}</span>
                                <span className="font-mono text-[10px] text-primary-subtle">{item.batch_id}</span>
                              </button>
                            </td>
                            <td className="px-5 py-3.5">
                              <div className="flex items-center gap-2.5">
                                <div className="flex size-7 shrink-0 items-center justify-center rounded-[2px] border border-white/[0.08] bg-surface text-accent">
                                  <Icon name="database" className="text-sm" />
                                </div>
                                <div className="min-w-0">
                                  <span className="block truncate font-sans text-xs font-medium text-primary">{item.action_types.join(", ")}</span>
                                  <p className="mt-0.5 font-sans text-[11px] text-primary-subtle">{item.affected_count} affected files</p>
                                </div>
                              </div>
                            </td>
                            <td className="px-5 py-3.5">
                              <Chip tone={item.reversible ? "success" : "warning"}>
                                {item.reversible ? "Reversible" : "Mixed"}
                              </Chip>
                            </td>
                            <td className="px-5 py-3.5 text-right">
                              <Button
                                tone="secondary"
                                id={`reverse-button-${item.batch_id}`}
                                className="px-3 py-1 text-xs"
                                disabled={!item.reversible || busy}
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
              mutationResult={mutationResult && mutationResult.batch_id === selected?.batch_id ? mutationResult : null}
              busy={busy}
              onReverse={() => selected && setConfirmBatchId(selected.batch_id)}
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
                    {response.items.filter((item) => item.reversible).length}
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
        open={!!confirmBatchId}
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
              disabled={busy || !confirmBatchId}
              onClick={() => {
                if (confirmBatchId) {
                  void handleReverse(confirmBatchId);
                }
              }}
            >
              {busy ? "Reversing..." : "Reverse Batch"}
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
