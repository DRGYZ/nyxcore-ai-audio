import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { HEALTH_BITRATE_BUCKET_LABELS, HEALTH_BITRATE_BUCKET_ORDER } from "../../lib/contracts";
import { useHealthQuery } from "../../lib/hooks";
import { ActionBanner, Button, Chip, EmptyState, Icon, MetricCard, PageHeader, Panel, ProgressBar, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";

export function HealthPage() {
  const navigate = useNavigate();
  const healthQuery = useHealthQuery();
  const [banner, setBanner] = useState<{ tone: "success" | "error"; message: string } | null>(null);

  if (healthQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Library Diagnostics"
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
          eyebrow="Library Diagnostics"
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
        eyebrow="Library Diagnostics"
        title="Library Health"
        description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        actions={
          <>
            <Button tone="ghost" onClick={handleExport}>
              <Icon name="download" className="text-sm" />
              Export JSON
            </Button>
            <Button tone="secondary" onClick={() => void handleRefresh()} disabled={healthQuery.isFetching}>
              <Icon name="refresh" className={`text-sm ${healthQuery.isFetching ? "animate-spin" : ""}`} />
              {healthQuery.isFetching ? "Refreshing…" : "Refresh Audit"}
            </Button>
          </>
        }
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Audio Files"
          value={formatNumber(report.overview.total_audio_files)}
          icon="library_music"
          accent={<Chip tone="primary">Live</Chip>}
          meta={<p className="font-sans text-[11px] text-primary-subtle">{formatNumber(report.overview.total_folders_touched)} folders scanned</p>}
        />
        <MetricCard
          label="Artwork Coverage"
          value={`${report.artwork.coverage_percent.toFixed(1)}%`}
          icon="image"
          meta={
            <p className="font-sans text-[11px] text-primary-subtle">
              <span className="font-mono text-accent">{formatNumber(report.artwork.with_artwork)}</span> embedded • <span className="font-mono text-primary-subtle">{formatNumber(report.artwork.without_artwork)}</span> missing
            </p>
          }
        />
        <MetricCard
          label="Lossless Ratio"
          value={`${losslessRatio.toFixed(1)}%`}
          icon="graphic_eq"
          meta={
            <p className="font-sans text-[11px] text-primary-subtle">
              <span className="font-mono text-primary">{formatNumber(report.quality.lossless_files)}</span> lossless • <span className="font-mono text-primary-subtle">{formatNumber(report.quality.lossy_files)}</span> lossy
            </p>
          }
        />
        <MetricCard
          label="Unreadable Files"
          value={formatNumber(report.quality.unreadable_or_unparseable_files?.count ?? 0)}
          icon="broken_image"
          accent={
            (report.quality.unreadable_or_unparseable_files?.count ?? 0) > 0 ? (
              <Chip tone="danger">Attention</Chip>
            ) : undefined
          }
          meta={<p className="font-sans text-[11px] text-primary-subtle">Unparseable headers or corrupted files</p>}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Panel className="p-5">
            <div className="mb-6 flex items-center justify-between border-b border-white/[0.07] pb-4">
              <div>
                <h3 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-primary">
                  <Icon name="equalizer" className="text-base text-accent" />
                  Bitrate Distribution
                </h3>
                <p className="mt-0.5 font-editorial text-xs italic text-primary-subtle">
                  Fidelity composition across all indexed audio streams
                </p>
              </div>
              <div className="flex gap-4 font-sans text-xs text-primary-subtle">
                <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-accent/40" /> Lossy</span>
                <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-accent" /> High Fidelity</span>
              </div>
            </div>
            <div className="flex h-48 items-end justify-between gap-1.5 px-1 sm:gap-3 sm:px-2">
              {HEALTH_BITRATE_BUCKET_ORDER.map((key) => {
                const value = report.quality.bitrate_buckets[key];
                const barTone = key === ">=256k" ? "bg-accent" : key === "unknown" ? "bg-amber-400/60" : "bg-accent/40";
                const label = HEALTH_BITRATE_BUCKET_LABELS[key];
                return (
                  <div key={key} className="flex min-w-0 flex-1 flex-col items-center gap-2">
                    <div
                      role="img"
                      aria-label={`${label}: ${formatNumber(value)} files`}
                      title={`${label}: ${formatNumber(value)} files`}
                      className={`w-full rounded-t-[2px] transition-all duration-300 ${barTone}`}
                      style={{ height: `${Math.max(6, (value / maxBucket) * 100)}%` }}
                    />
                    <span className="w-full truncate text-center font-mono text-[9px] text-primary-subtle sm:text-[10px]">{label}</span>
                  </div>
                );
              })}
            </div>
          </Panel>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Panel className="p-5">
              <h3 className="mb-4 flex items-center gap-2 border-b border-white/[0.07] pb-3 font-display text-sm font-semibold tracking-wide text-primary">
                <Icon name="grid_view" className="text-base text-accent" />
                Metadata Issues
              </h3>
              <div className="divide-y divide-white/[0.05] rounded-[3px] border border-white/[0.06] bg-surface-low/50">
                {[
                  ["Missing Title", report.metadata.missing_title.count],
                  ["Missing Artist", report.metadata.missing_artist.count],
                  ["Missing Album", report.metadata.missing_album.count],
                  ["Placeholder Metadata", report.metadata.placeholder_metadata.count],
                  ["Suspicious Swaps", report.metadata.suspicious_title_artist_swaps.count],
                ].map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between px-3.5 py-2.5 font-sans text-xs">
                    <span className="text-primary-subtle">{label}</span>
                    <span className="font-mono font-medium text-primary">{formatNumber(Number(count))}</span>
                  </div>
                ))}
              </div>
              {topFolders.length > 0 ? (
                <p className="mt-3 font-sans text-[11px] text-primary-subtle">
                  Concentrated in: <span className="font-mono text-accent">{topFolders[0].folder}</span>
                </p>
              ) : null}
            </Panel>

            <Panel className="p-5">
              <h3 className="mb-4 flex items-center gap-2 border-b border-white/[0.07] pb-3 font-display text-sm font-semibold tracking-wide text-primary">
                <Icon name="image" className="text-base text-accent" />
                Artwork Coverage
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between font-sans text-xs">
                    <span className="text-primary-subtle">Embedded Artwork</span>
                    <span className="font-mono font-semibold text-accent">{formatNumber(report.artwork.with_artwork)}</span>
                  </div>
                  <ProgressBar value={report.artwork.coverage_percent} />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between font-sans text-xs">
                    <span className="text-primary-subtle">Missing Cover Art</span>
                    <span className="font-mono font-semibold text-primary">{formatNumber(report.artwork.without_artwork)}</span>
                  </div>
                  <ProgressBar value={withoutArtworkRatio} tone="warning" />
                </div>
              </div>
            </Panel>
          </div>
        </div>

        <Panel className="flex flex-col p-5">
          <div className="mb-4 border-b border-white/[0.07] pb-3.5">
            <h3 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-primary">
              <Icon name="auto_fix_high" className="text-base text-accent" />
              What to Fix First
            </h3>
            <p className="mt-1 font-editorial text-xs italic text-primary-subtle">
              Prioritized recommendations based on issue severity and count
            </p>
          </div>
          <div className="flex-1 space-y-3">
            {report.priorities.recommended_actions.length === 0 ? (
              <EmptyState
                title="No priority recommendations"
                description="Health recommendations will appear here once NyxCore finds metadata, artwork, or quality issues worth fixing first."
              />
            ) : (
              (topIssueCategories.length > 0 ? topIssueCategories.map((item) => item.action) : report.priorities.recommended_actions).map((action, index) => (
                <div
                  key={action}
                  className={`rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3.5 transition-colors hover:border-white/[0.12] ${
                    index === 0 ? "border-l-2 border-l-rose-500" : index === 1 ? "border-l-2 border-l-amber-500" : "border-l-2 border-l-accent"
                  }`}
                >
                  <div className="mb-2 flex justify-between gap-3">
                    <Chip tone={index === 0 ? "danger" : index === 1 ? "warning" : "primary"}>
                      {topIssueCategories[index]?.category ? topIssueCategories[index].category.replace(/_/g, " ") : index === 0 ? "primary" : index === 1 ? "secondary" : "follow-up"}
                    </Chip>
                    {topIssueCategories[index]?.count !== undefined ? (
                      <span className="shrink-0 font-mono text-[10px] text-primary-subtle">{formatNumber(topIssueCategories[index].count)} items</span>
                    ) : null}
                  </div>
                  <p className="break-words font-sans text-xs text-primary">{action}</p>
                </div>
              ))
            )}
          </div>
          <div className="mt-4 border-t border-white/[0.06] pt-3.5">
            <Button tone="secondary" className="w-full justify-center py-2 text-xs" onClick={() => navigate("/review")}>
              Review Fixes in Inbox →
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
