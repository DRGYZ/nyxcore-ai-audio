import { useDuplicatesQuery, useHealthQuery, useHistoryQuery, useReviewQuery } from "../../lib/hooks";
import { mockDuplicateReport, mockHealthReport, mockHistoryResponse, mockReviewReport } from "../../lib/mock-data";
import {
  mergeQueryNoticeStates,
  resolveQueryData,
  resolveReportQueryData,
  toQueryNoticeState,
} from "../../lib/query-state";
import {
  Button,
  DataTable,
  EmptyState,
  MetricCard,
  PageHeader,
  Panel,
  PageQueryStateNotice,
  ProgressBar,
  formatNumber,
} from "../components";

export function DashboardPage() {
  const healthQuery = useHealthQuery();
  const reviewQuery = useReviewQuery();
  const duplicatesQuery = useDuplicatesQuery();
  const historyQuery = useHistoryQuery();

  const healthState = resolveReportQueryData(healthQuery, mockHealthReport);
  const reviewState = resolveReportQueryData(reviewQuery, mockReviewReport);
  const duplicatesState = resolveReportQueryData(duplicatesQuery, mockDuplicateReport);
  const historyState = resolveQueryData(historyQuery, mockHistoryResponse);
  const health = healthState.data;
  const review = reviewState.data;
  const duplicates = duplicatesState.data;
  const history = historyState.data;
  const queryNoticeState = mergeQueryNoticeStates(
    toQueryNoticeState(healthState),
    toQueryNoticeState(reviewState),
    toQueryNoticeState(duplicatesState),
    toQueryNoticeState(historyState),
  );
  const losslessRatio = health.overview.total_audio_files === 0 ? 0 : (health.quality.lossless_files / health.overview.total_audio_files) * 100;
  const highBitrateRatio = health.overview.total_audio_files === 0 ? 0 : (health.quality.bitrate_buckets[">=256k"] / health.overview.total_audio_files) * 100;

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="System Diagnostics / V2.4.0 Stable"
        title="Mission Control"
        description="Read-only operational view over NyxCore health, review priorities, duplicate pressure, and recent action history."
        actions={
          <>
            <Button tone="ghost" disabled>Read-Only View</Button>
            <Button tone="primary" disabled>Scan Controls Stay in CLI</Button>
          </>
        }
      />
      <PageQueryStateNotice {...queryNoticeState} />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Total Files"
          value={formatNumber(health.overview.total_audio_files)}
          icon="audio_file"
          meta={<p className="text-sm font-bold text-emerald-400">+0.2% from last refresh</p>}
        />
        <MetricCard
          label="Exact Duplicates"
          value={formatNumber(duplicates.summary.exact_group_count)}
          icon="content_copy"
          meta={<p className="text-sm font-bold text-amber-400">Likely groups: {duplicates.summary.likely_group_count}</p>}
        />
        <MetricCard
          label="Missing Metadata"
          value={formatNumber(health.metadata.missing_artist.count + health.metadata.missing_title.count + health.metadata.missing_album.count)}
          icon="label_off"
          meta={<p className="text-sm font-bold text-primary">Placeholder tags: {formatNumber(health.metadata.placeholder_metadata.count)}</p>}
        />
        <MetricCard
          label="Low Quality"
          value={formatNumber(health.quality.low_bitrate_files.count)}
          icon="high_quality"
          meta={<p className="text-sm font-bold text-rose-400">Unreadable: {formatNumber(health.quality.unreadable_or_unparseable_files?.count ?? 0)}</p>}
        />
      </div>
      <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
        <div className="space-y-8 xl:col-span-2">
          <Panel className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-display text-xl font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">priority_high</span>
                Priority Review
              </h2>
              <Button tone="secondary" className="px-3 py-2 text-xs">View All Issues</Button>
            </div>
            <div className="space-y-3">
              {review.items.length === 0 ? (
                <EmptyState
                  title="No review findings"
                  description="Once NyxCore detects duplicate, metadata, or quality issues, the highest-priority findings will surface here."
                />
              ) : (
                review.items.slice(0, 4).map((item) => (
                  <div key={item.item_id} className="flex items-center justify-between gap-4 rounded-xl border border-primary/10 bg-primary/5 px-4 py-4 transition-colors hover:border-primary/30">
                    <div className="flex min-w-0 items-center gap-4">
                      <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary">
                        <span className="material-symbols-outlined">warning</span>
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-bold text-slate-200">{item.summary}</p>
                        <p className="truncate font-mono text-[11px] text-slate-500">{item.reason_summary}</p>
                      </div>
                    </div>
                    <span className="shrink-0 rounded bg-primary/10 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.18em] text-primary">{item.priority_band}</span>
                  </div>
                ))
              )}
            </div>
          </Panel>
          <Panel className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-primary/10 px-6 py-5">
              <h2 className="flex items-center gap-2 font-display text-xl font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">history</span>
                Recent History
              </h2>
            </div>
            <div className="px-4 py-4">
              <DataTable
                dense
                headers={["Action Type", "Timestamp", "Affected", "Status"]}
                rows={history.items.slice(0, 4).map((batch) => [
                  batch.action_types.join(", "),
                  batch.applied_at,
                  formatNumber(batch.affected_count),
                  <span className={`rounded px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] ${batch.reversible ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"}`}>
                    {batch.reversible ? "reversible" : "mixed"}
                  </span>,
                ])}
              />
            </div>
          </Panel>
        </div>
        <div className="space-y-8">
          <Panel className="p-8">
            <h2 className="mb-4 flex items-center gap-2 font-display text-xl font-bold text-slate-100">
              <span className="material-symbols-outlined text-primary">health_metrics</span>
              Health Summary
            </h2>
            <div className="flex flex-col items-center">
              <div className="relative grid size-48 place-items-center rounded-full border border-primary/20 bg-[radial-gradient(circle_at_center,_rgba(37,226,244,0.16),transparent_60%)]">
                <div className="absolute inset-4 rounded-full border border-primary/10" />
                <div className="text-center">
                  <p className="font-display text-5xl font-bold text-slate-100">{health.artwork.coverage_percent.toFixed(0)}%</p>
                  <p className="mt-1 text-[10px] font-bold uppercase tracking-[0.28em] text-slate-500">Overall</p>
                </div>
              </div>
              <div className="mt-8 w-full space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-400">Artwork Coverage</span>
                    <span className="font-bold text-primary">{health.artwork.coverage_percent.toFixed(1)}%</span>
                  </div>
                  <ProgressBar value={health.artwork.coverage_percent} />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-400">Metadata Health</span>
                    <span className="font-bold text-slate-200">{Math.max(0, 100 - (health.metadata.placeholder_metadata.count / 100)).toFixed(0)}%</span>
                  </div>
                  <ProgressBar value={Math.max(0, 100 - health.metadata.placeholder_metadata.count / 100)} tone="violet" />
                </div>
              </div>
            </div>
          </Panel>
          <Panel className="p-6">
            <h3 className="mb-4 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
              <span className="material-symbols-outlined text-primary">info</span>
              Storage Distribution
            </h3>
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex justify-between text-xs font-bold text-slate-400">
                  <span>Lossless Files</span>
                  <span>{formatNumber(health.quality.lossless_files)}</span>
                </div>
                <ProgressBar value={losslessRatio} />
              </div>
              <div>
                <div className="mb-2 flex justify-between text-xs font-bold text-slate-400">
                  <span>High Bitrate Lossy</span>
                  <span>{formatNumber(health.quality.bitrate_buckets[">=256k"])}</span>
                </div>
                <ProgressBar value={highBitrateRatio} tone="violet" />
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
