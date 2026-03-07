import { useDuplicatesQuery } from "../../lib/hooks";
import { mockDuplicateReport } from "../../lib/mock-data";
import { resolveReportQueryData, toQueryNoticeState } from "../../lib/query-state";
import { Button, EmptyState, PageHeader, PageQueryStateNotice, Panel, PathBlock, formatBytes, formatNumber } from "../components";

export function DuplicatesPage() {
  const duplicatesQuery = useDuplicatesQuery();
  const duplicatesState = resolveReportQueryData(duplicatesQuery, mockDuplicateReport);
  const report = duplicatesState.data;
  const reclaimable = report.exact_duplicates.reduce(
    (sum, group) => sum + (group.reclaimable_bytes ?? group.files.slice(1).reduce((acc, item) => acc + item.file_size_bytes, 0)),
    0,
  );

  return (
    <div className="space-y-8">
      <PageHeader
        title="Duplicate Analysis"
        description="Exact duplicates and likely duplicate clusters, with preferred-copy recommendations and reclaimable space summaries."
      />
      <PageQueryStateNotice {...toQueryNoticeState(duplicatesState)} />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <Panel className="p-6">
          <p className="text-sm font-medium text-slate-400">Total Reclaimable</p>
          <div className="mt-2 flex items-baseline gap-2">
            <h3 className="text-3xl font-bold text-primary">{formatBytes(reclaimable)}</h3>
            <span className="text-sm font-bold text-emerald-400">review only</span>
          </div>
        </Panel>
        <Panel className="p-6">
          <p className="text-sm font-medium text-slate-400">Duplicate Groups</p>
          <h3 className="mt-2 text-3xl font-bold text-slate-100">{formatNumber(report.summary.exact_group_count + report.summary.likely_group_count)}</h3>
        </Panel>
        <Panel className="border-l-4 border-l-secondary p-6">
          <p className="text-sm font-medium text-slate-400">System Health Impact</p>
          <div className="mt-2 flex items-center gap-2">
            <span className="material-symbols-outlined text-secondary">verified_user</span>
            <h3 className="text-xl font-bold text-slate-100">Optimized</h3>
          </div>
        </Panel>
      </div>
      <div className="flex gap-8 border-b border-primary/10">
        <button type="button" className="flex items-center gap-2 border-b-2 border-primary px-2 pb-4 font-bold text-primary">
          <span className="material-symbols-outlined text-sm">copy_all</span>
          Exact Duplicates
        </button>
        <button type="button" className="flex items-center gap-2 border-b-2 border-transparent px-2 pb-4 font-medium text-slate-400">
          <span className="material-symbols-outlined text-sm">difference</span>
          Likely Duplicates
        </button>
      </div>
      <div className="space-y-6">
        {report.exact_duplicates.length === 0 && report.likely_duplicates.length === 0 ? (
          <EmptyState
            title="No duplicate groups found"
            description="When exact or likely duplicate clusters are detected, preferred-copy recommendations and reclaimable space will appear here."
          />
        ) : null}
        {report.exact_duplicates.map((group) => {
          const reclaim = group.reclaimable_bytes ?? group.files.slice(1).reduce((sum, item) => sum + item.file_size_bytes, 0);
          return (
            <Panel key={group.group_id} className="overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-primary/10 bg-primary/5 p-4">
                <div className="flex items-center gap-4">
                  <div className="flex size-12 items-center justify-center rounded-lg bg-secondary/20 text-secondary">
                    <span className="material-symbols-outlined">movie</span>
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-100">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                    <p className="text-xs text-slate-400">{group.files.length} occurrences found across the library</p>
                  </div>
                </div>
                <div className="flex gap-6">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Confidence</p>
                    <p className="text-xs font-bold text-primary">100%</p>
                  </div>
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Potential Saving</p>
                    <p className="text-sm font-bold text-emerald-400">{formatBytes(reclaim)}</p>
                  </div>
                </div>
              </div>
              <div className="space-y-3 p-4">
                <div className="flex items-center justify-between rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                  <div>
                    <p className="text-xs font-bold uppercase tracking-[0.24em] text-emerald-400">Preferred Copy</p>
                    <div className="mt-2">
                      <PathBlock value={group.preferred.path} tone="success" />
                    </div>
                  </div>
                  <p className="max-w-xs text-right text-xs font-bold text-slate-400">{group.preferred.reasons.join(" • ")}</p>
                </div>
                <div className="space-y-2 pl-4">
                  {group.files
                    .filter((file) => file.path !== group.preferred.path)
                    .map((file) => (
                      <div key={file.path} className="flex items-center justify-between gap-3 rounded bg-background-dark/50 p-2">
                        <div className="flex min-w-0 items-center gap-3">
                          <span className="material-symbols-outlined text-sm text-slate-500">delete_sweep</span>
                          <div className="min-w-0 flex-1">
                            <PathBlock value={file.path} />
                          </div>
                        </div>
                        <Button tone="ghost" className="shrink-0 px-2 py-1 text-[10px]" disabled>Preserve Instead</Button>
                      </div>
                    ))}
                </div>
              </div>
            </Panel>
          );
        })}
        {report.likely_duplicates.map((group) => (
          <Panel key={group.group_id} className="overflow-hidden border-l-2 border-l-secondary/50 opacity-95">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-secondary/10 bg-secondary/5 p-4">
              <div>
                <h4 className="font-bold text-slate-100">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                <p className="text-xs text-slate-400">Similar content detected with metadata mismatch</p>
              </div>
              <div className="flex gap-6">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Confidence</p>
                  <p className="text-xs font-bold text-secondary">{((group.confidence ?? 0) * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">Preferred Copy</p>
                  <p className="text-xs font-bold text-slate-300">{group.preferred.path.split(/[\\/]/).pop()}</p>
                </div>
              </div>
            </div>
            <div className="p-4">
              <p className="break-words text-xs text-slate-400">{(group.reasons ?? []).join(", ")}</p>
            </div>
          </Panel>
        ))}
      </div>
      <Panel className="overflow-hidden bg-gradient-to-br from-secondary/20 via-background-dark to-primary/10 p-8">
        <div className="flex flex-col items-center justify-between gap-8 md:flex-row">
          <div>
            <h2 className="font-display text-2xl font-bold text-white">Ready to reclaim your space?</h2>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              NyxCore can generate a review-first action plan from these groups, but this UI pass stays read-only.
            </p>
          </div>
          <div className="flex flex-col gap-4 sm:flex-row">
            <Button tone="ghost" disabled>Review Manual</Button>
            <Button tone="primary" disabled>Plan in Review Inbox</Button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
