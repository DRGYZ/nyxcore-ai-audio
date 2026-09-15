import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { HEALTH_BITRATE_BUCKET_LABELS, HEALTH_BITRATE_BUCKET_ORDER } from "../../lib/contracts";
import { useHealthQuery } from "../../lib/hooks";
import { ActionBanner, Button, EmptyState, Icon, PageHeader, Panel, ProgressBar, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";

export function HealthPage() {
  const navigate = useNavigate();
  const healthQuery = useHealthQuery();
  const [banner, setBanner] = useState<{ tone: "success" | "error"; message: string } | null>(null);

  if (healthQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Library Health"
          title="Library Health"
          description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        />
        <ApiUnavailableState contextLabel="Library Health" />
      </div>
    );
  }

  if (healthQuery.isLoading || !healthQuery.data) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Library Health"
          title="Library Health"
          description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        />
        <Panel className="p-8 text-center text-sm text-primary-muted">
          Loading library health audit from local API…
        </Panel>
      </div>
    );
  }

  const report = healthQuery.data.data;
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

  async function handleRefresh() {
    setBanner(null);
    const result = await healthQuery.refetch();
    setBanner(
      result.error
        ? { tone: "error", message: "Health refresh failed. Local API may be offline." }
        : { tone: "success", message: "Health audit refreshed from the current library." },
    );
  }

  function handleExport() {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `nyxcore-health-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setBanner({ tone: "success", message: "Health report exported as JSON." });
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Library Health"
        title="Library Health"
        description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        actions={
          <>
            <Button tone="ghost" onClick={handleExport}>Export JSON</Button>
            <Button tone="primary" onClick={() => void handleRefresh()} disabled={healthQuery.isFetching}>
              {healthQuery.isFetching ? "Refreshing…" : "Refresh Audit"}
            </Button>
          </>
        }
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <Panel className="p-6">
          <div className="mb-3 flex justify-between">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Audio Files</p>
            <span className="border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider text-accent">LIVE</span>
          </div>
          <p className="font-display text-3xl font-bold tracking-tight text-primary">{formatNumber(report.overview.total_audio_files)}</p>
          <p className="mt-2 font-mono text-xs text-primary-subtle">{formatNumber(report.overview.total_folders_touched)} folders touched</p>
        </Panel>
        <Panel className="p-6">
          <div className="mb-3 flex justify-between">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Artwork Coverage</p>
            <span className="font-mono text-[10px] text-primary-subtle">{formatNumber(report.artwork.with_artwork)} tracks</span>
          </div>
          <p className="font-display text-3xl font-bold tracking-tight text-primary">{report.artwork.coverage_percent.toFixed(1)}%</p>
          <p className="mt-2 font-mono text-xs text-primary-subtle">{formatNumber(report.artwork.without_artwork)} missing artwork</p>
        </Panel>
        <Panel className="p-6">
          <div className="mb-3 flex justify-between">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Lossless Ratio</p>
            <span className="font-mono text-[10px] text-primary-subtle">{formatNumber(report.quality.lossy_files)} lossy</span>
          </div>
          <p className="font-display text-3xl font-bold tracking-tight text-primary">{losslessRatio.toFixed(1)}%</p>
          <p className="mt-2 font-mono text-xs text-primary-subtle">{formatNumber(report.quality.lossless_files)} tracks</p>
        </Panel>
        <Panel className="border border-rose-500/30 p-6">
          <div className="mb-3 flex justify-between">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Unreadable Files</p>
            <span className="border border-rose-500/40 bg-rose-500/10 px-2 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider text-rose-400">Needs Attention</span>
          </div>
          <p className="font-display text-3xl font-bold tracking-tight text-rose-400">{formatNumber(report.quality.unreadable_or_unparseable_files?.count ?? 0)}</p>
          <p className="mt-2 font-mono text-xs text-primary-subtle">Unreadable metadata or unparseable headers</p>
        </Panel>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Panel className="p-6">
            <div className="mb-6 flex items-center justify-between border-b border-border pb-4">
              <h3 className="flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary">
                <Icon name="equalizer" className="text-base text-accent" />
                Bitrate Distribution
              </h3>
              <div className="flex gap-4 font-mono text-[10px] text-primary-subtle">
                <span className="flex items-center gap-1.5"><span className="size-2 bg-accent/40" /> LOSSY</span>
                <span className="flex items-center gap-1.5"><span className="size-2 bg-accent" /> HIGH BITRATE</span>
              </div>
            </div>
            <div className="flex h-52 items-end justify-between gap-3">
              {HEALTH_BITRATE_BUCKET_ORDER.map((key) => {
                const value = report.quality.bitrate_buckets[key];
                const barTone = key === ">=256k" ? "bg-accent" : key === "unknown" ? "bg-amber-400/60" : "bg-accent/40";
                return (
                  <div key={key} className="flex flex-1 flex-col items-center gap-2">
                    <div
                      className={`w-full ${barTone}`}
                      style={{ height: `${Math.max(8, (value / maxBucket) * 100)}%` }}
                    />
                    <span className="font-mono text-[10px] text-primary-subtle">{HEALTH_BITRATE_BUCKET_LABELS[key]}</span>
                  </div>
                );
              })}
            </div>
          </Panel>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Panel className="p-6">
              <h3 className="mb-4 flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary border-b border-border pb-3">
                <Icon name="grid_view" className="text-base text-accent" />
                Metadata Issues
              </h3>
              <div className="space-y-2">
                {[
                  ["Missing Title", report.metadata.missing_title.count],
                  ["Missing Artist", report.metadata.missing_artist.count],
                  ["Missing Album", report.metadata.missing_album.count],
                  ["Placeholder Metadata", report.metadata.placeholder_metadata.count],
                  ["Suspicious Swaps", report.metadata.suspicious_title_artist_swaps.count],
                ].map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between border border-border bg-surface-low px-3.5 py-2.5 font-mono text-xs">
                    <span className="text-primary-muted">{label}</span>
                    <span className="font-bold text-primary">{formatNumber(Number(count))}</span>
                  </div>
                ))}
              </div>
              {topFolders.length > 0 ? (
                <p className="mt-4 font-mono text-[11px] text-primary-subtle">
                  Concentrated in: <span className="text-accent">{topFolders[0].folder}</span>
                </p>
              ) : null}
            </Panel>
            <Panel className="p-6">
              <h3 className="mb-4 flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary border-b border-border pb-3">
                <Icon name="image" className="text-base text-accent" />
                Artwork Coverage
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between font-mono text-xs">
                    <span className="text-primary-muted">With Artwork</span>
                    <span className="font-bold text-accent">{formatNumber(report.artwork.with_artwork)}</span>
                  </div>
                  <ProgressBar value={report.artwork.coverage_percent} />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between font-mono text-xs">
                    <span className="text-primary-muted">Without Artwork</span>
                    <span className="font-bold text-primary">{formatNumber(report.artwork.without_artwork)}</span>
                  </div>
                  <ProgressBar value={withoutArtworkRatio} tone="warning" />
                </div>
              </div>
            </Panel>
          </div>
        </div>
        <Panel className="flex flex-col">
          <div className="border-b border-border bg-surface-low p-6">
            <h3 className="flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary">
              <Icon name="auto_fix_high" className="text-base text-accent" />
              What to Fix First
            </h3>
            <p className="mt-1 font-mono text-xs text-primary-subtle">Prioritized recommendations based on issue severity and count.</p>
          </div>
          <div className="space-y-3 p-4">
            {report.priorities.recommended_actions.length === 0 ? (
              <EmptyState
                title="No priority recommendations"
                description="Health recommendations will appear here once NyxCore finds metadata, artwork, or quality issues worth fixing first."
              />
            ) : (
              (topIssueCategories.length > 0 ? topIssueCategories.map((item) => item.action) : report.priorities.recommended_actions).map((action, index) => (
                <div
                  key={action}
                  className={`border border-border bg-surface-low p-3.5 transition-colors hover:border-border-bright ${
                    index === 0 ? "border-l-2 border-l-rose-500" : index === 1 ? "border-l-2 border-l-amber-500" : "border-l-2 border-l-accent"
                  }`}
                >
                  <div className="mb-2 flex justify-between gap-3">
                    <span className={`border px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider ${
                      index === 0 ? "border-rose-500/40 bg-rose-500/10 text-rose-400" : index === 1 ? "border-amber-500/40 bg-amber-500/10 text-amber-400" : "border-accent/40 bg-accent/10 text-accent"
                    }`}>
                      {topIssueCategories[index]?.category ? topIssueCategories[index].category.replace(/_/g, " ") : index === 0 ? "primary" : index === 1 ? "secondary" : "follow-up"}
                    </span>
                    {topIssueCategories[index]?.count !== undefined ? (
                      <span className="shrink-0 font-mono text-[10px] text-primary-subtle">{formatNumber(topIssueCategories[index].count)} items</span>
                    ) : null}
                  </div>
                  <p className="break-words font-mono text-xs text-primary">{action}</p>
                </div>
              ))
            )}
            <Button tone="secondary" className="w-full justify-center py-2 text-xs" onClick={() => navigate("/review")}>
              Review Fixes Safely
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
