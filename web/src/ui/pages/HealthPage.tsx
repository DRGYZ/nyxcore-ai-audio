import { HEALTH_BITRATE_BUCKET_LABELS, HEALTH_BITRATE_BUCKET_ORDER } from "../../lib/contracts";
import { useHealthQuery } from "../../lib/hooks";
import { mockHealthReport } from "../../lib/mock-data";
import { resolveReportQueryData, toQueryNoticeState } from "../../lib/query-state";
import { Button, EmptyState, PageHeader, PageQueryStateNotice, Panel, ProgressBar, formatNumber } from "../components";

export function HealthPage() {
  const healthQuery = useHealthQuery();
  const healthState = resolveReportQueryData(healthQuery, mockHealthReport);
  const report = healthState.data;
  const bucketValues = HEALTH_BITRATE_BUCKET_ORDER.map((key) => report.quality.bitrate_buckets[key]);
  const maxBucket = bucketValues.length > 0 ? Math.max(1, ...bucketValues) : 1;
  const losslessRatio = report.overview.total_audio_files === 0
    ? 0
    : (report.quality.lossless_files / report.overview.total_audio_files) * 100;
  const withoutArtworkRatio = report.overview.total_audio_files === 0
    ? 0
    : (report.artwork.without_artwork / report.overview.total_audio_files) * 100;
  const topIssueCategories = report.priorities.top_issue_categories ?? [];
  const topFolders = report.priorities.top_problematic_folders ?? [];

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="System Diagnostics"
        title="Technical Audio Audit"
        actions={
          <>
            <Button tone="ghost" disabled>Export in CLI</Button>
            <Button tone="primary" disabled>Re-scan in CLI</Button>
          </>
        }
      />
      <PageQueryStateNotice {...toQueryNoticeState(healthState)} />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Audio Files</p>
            <span className="rounded bg-primary/10 px-2 py-0.5 text-[10px] font-bold text-primary">LIVE</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">{formatNumber(report.overview.total_audio_files)}</p>
          <p className="mt-2 text-xs font-medium text-slate-500">{formatNumber(report.overview.total_folders_touched)} folders touched</p>
        </Panel>
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Artwork Coverage</p>
            <span className="text-[10px] font-bold text-slate-500">{formatNumber(report.artwork.with_artwork)} files with artwork</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">{report.artwork.coverage_percent.toFixed(1)}%</p>
          <p className="mt-2 text-xs font-medium text-slate-500">{formatNumber(report.artwork.without_artwork)} files still missing artwork</p>
        </Panel>
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Lossless Ratio</p>
            <span className="text-[10px] font-bold text-slate-500">{formatNumber(report.quality.lossy_files)} lossy files</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">{losslessRatio.toFixed(1)}%</p>
          <p className="mt-2 text-xs font-medium text-slate-500">{formatNumber(report.quality.lossless_files)} tracks</p>
        </Panel>
        <Panel className="border border-rose-500/20 p-5 shadow-[0_0_20px_-5px_rgba(244,63,94,0.2)]">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Critical Errors</p>
            <span className="rounded bg-rose-500/10 px-2 py-0.5 text-[10px] font-bold text-rose-500">ACTION REQ</span>
          </div>
          <p className="text-3xl font-bold text-rose-500">{formatNumber(report.quality.unreadable_or_unparseable_files?.count ?? 0)}</p>
          <p className="mt-2 text-xs font-medium text-slate-500">Corrupt containers detected</p>
        </Panel>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Panel className="p-6">
            <div className="mb-6 flex items-center justify-between">
              <h3 className="flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">equalizer</span>
                Bitrate Distribution
              </h3>
              <div className="flex gap-2 text-[10px] font-bold text-slate-400">
                <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-primary/40" /> LOSSY</span>
                <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-secondary" /> HIGH BITRATE</span>
              </div>
            </div>
            <div className="flex h-56 items-end justify-between gap-3">
              {HEALTH_BITRATE_BUCKET_ORDER.map((key) => {
                const value = report.quality.bitrate_buckets[key];
                const barTone = key === ">=256k" ? "bg-secondary" : key === "unknown" ? "bg-amber-500/60" : "bg-primary/50";
                return (
                  <div key={key} className="flex flex-1 flex-col items-center gap-2">
                    <div
                      className={`w-full rounded-t ${barTone}`}
                      style={{ height: `${Math.max(12, (value / maxBucket) * 100)}%` }}
                    />
                    <span className="text-[10px] font-bold text-slate-500">{HEALTH_BITRATE_BUCKET_LABELS[key]}</span>
                  </div>
                );
              })}
            </div>
          </Panel>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Panel className="p-6">
              <h3 className="mb-6 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-secondary">grid_view</span>
                Metadata Issues
              </h3>
              <div className="space-y-3">
                {[
                  ["Missing Title", report.metadata.missing_title.count],
                  ["Missing Artist", report.metadata.missing_artist.count],
                  ["Missing Album", report.metadata.missing_album.count],
                  ["Placeholder Metadata", report.metadata.placeholder_metadata.count],
                  ["Suspicious Swaps", report.metadata.suspicious_title_artist_swaps.count],
                ].map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3">
                    <span className="text-sm text-slate-300">{label}</span>
                    <span className="text-sm font-bold text-primary">{formatNumber(Number(count))}</span>
                  </div>
                ))}
              </div>
              {topFolders.length > 0 ? (
                <p className="mt-4 text-xs text-slate-500">
                  Most concentrated folder: <span className="text-primary">{topFolders[0].folder}</span>
                </p>
              ) : null}
            </Panel>
            <Panel className="p-6">
              <h3 className="mb-6 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">image</span>
                Artwork Coverage Breakdown
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-400">With Artwork</span>
                    <span className="font-bold text-primary">{formatNumber(report.artwork.with_artwork)}</span>
                  </div>
                  <ProgressBar value={report.artwork.coverage_percent} />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-400">Without Artwork</span>
                    <span className="font-bold text-slate-200">{formatNumber(report.artwork.without_artwork)}</span>
                  </div>
                  <ProgressBar value={withoutArtworkRatio} tone="warning" />
                </div>
              </div>
            </Panel>
          </div>
        </div>
        <Panel className="flex flex-col">
          <div className="border-b border-border-dark p-6">
            <h3 className="flex items-center gap-2 font-display text-lg font-bold text-slate-100">
              <span className="material-symbols-outlined text-amber-500">auto_fix_high</span>
              What to Fix First
            </h3>
            <p className="mt-1 text-xs text-slate-500">Current report recommendations based on issue counts and priority rules.</p>
          </div>
          <div className="space-y-4 p-4">
            {report.priorities.recommended_actions.length === 0 ? (
              <EmptyState
                title="No priority recommendations"
                description="Health recommendations will appear here once NyxCore finds metadata, artwork, or quality issues worth fixing first."
              />
            ) : (
              (topIssueCategories.length > 0 ? topIssueCategories.map((item) => item.action) : report.priorities.recommended_actions).map((action, index) => (
                <div
                  key={action}
                  className={`rounded-lg border-l-4 bg-background-dark p-4 transition-colors hover:bg-border-dark ${
                    index === 0 ? "border-rose-500" : index === 1 ? "border-amber-500" : "border-primary"
                  }`}
                >
                  <div className="mb-2 flex justify-between gap-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      index === 0 ? "bg-rose-500/10 text-rose-500" : index === 1 ? "bg-amber-500/10 text-amber-500" : "bg-primary/10 text-primary"
                    }`}>
                      {topIssueCategories[index]?.category ? topIssueCategories[index].category.replace(/_/g, " ") : index === 0 ? "primary" : index === 1 ? "secondary" : "follow-up"}
                    </span>
                    {topIssueCategories[index]?.count !== undefined ? (
                      <span className="shrink-0 text-[10px] text-slate-500">{formatNumber(topIssueCategories[index].count)} items</span>
                    ) : null}
                  </div>
                  <p className="break-words text-sm font-medium text-slate-200">{action}</p>
                </div>
              ))
            )}
            <Button tone="secondary" className="w-full justify-center py-2.5 text-xs" disabled>
              Autorepair Not Available in UI
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
