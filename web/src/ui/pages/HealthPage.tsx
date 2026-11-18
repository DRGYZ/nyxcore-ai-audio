import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { HEALTH_BITRATE_BUCKET_ORDER, HEALTH_BITRATE_BUCKET_LABELS } from "../../lib/contracts";
import { useHealthQuery, useRefreshHealthAudit, useStatusQuery } from "../../lib/hooks";
import { ISSUE_LABELS, type MetadataIssue } from "../../lib/metadata";
import { ActionBanner, Button, Chip, EmptyState, Icon, PageHeader, Panel, ProgressBar, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";
import { MetadataDraftLink } from "../metadata/MetadataEditor";
import { MetadataIssueList } from "../metadata/MetadataIssueList";

export function HealthPage({ desktop = false }: { desktop?: boolean }) {
  const navigate = useNavigate();
  const healthQuery = useHealthQuery();
  const refreshAudit = useRefreshHealthAudit();
  const reviewOnly = useStatusQuery().data?.review_only === true;
  const [banner, setBanner] = useState<{ tone: "success" | "error"; message: string } | null>(null);
  const [issue, setIssue] = useState<MetadataIssue | null>(null);

  if (healthQuery.isError && !reviewOnly) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Collection Diagnostics"
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
          eyebrow="Collection Diagnostics"
          title="Library Health"
          description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        />
        {!reviewOnly ? (
          <Panel className="p-8 text-center text-sm text-primary-muted">Loading library health audit from local API…</Panel>
        ) : null}
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
  const metadataIssues: [MetadataIssue, number][] = [
    ["missing_title", report.metadata.missing_title.count],
    ["missing_artist", report.metadata.missing_artist.count],
    ["missing_album", report.metadata.missing_album.count],
    ["placeholder_metadata", report.metadata.placeholder_metadata.count],
    ["suspicious_swaps", report.metadata.suspicious_title_artist_swaps.count],
  ];

  async function handleRefresh() {
    setBanner(null);
    try {
      await refreshAudit.mutateAsync();
      setBanner({ tone: "success", message: "Health audit refreshed from the current library." });
    } catch {
      setBanner({ tone: "error", message: "Health refresh failed. Local API may be offline." });
    }
  }

  function handleExport() {
    if (desktop) return;
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
    <div className="space-y-6">
      <PageHeader
        eyebrow="Collection Diagnostics"
        title="Library Health"
        description="Inspect format quality, metadata completeness, artwork coverage, and unreadable files across your music library."
        actions={
          <>
            <div className="space-y-1">
              <Button tone="ghost" onClick={handleExport} disabled={desktop} aria-describedby={desktop ? "health-export-note" : undefined}>
                <Icon name="download" className="text-sm" />
                Export JSON
              </Button>
              {desktop && <p id="health-export-note" className="text-xs text-primary-subtle">Export is available in the browser.</p>}
            </div>
            {!reviewOnly ? (
              <Button tone="secondary" onClick={() => void handleRefresh()} disabled={refreshAudit.isPending}>
                <Icon name="refresh" className={`text-sm ${refreshAudit.isPending ? "animate-spin" : ""}`} />
                {refreshAudit.isPending ? "Refreshing…" : "Refresh Audit"}
              </Button>
            ) : null}
          </>
        }
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}

      <Panel className="overflow-hidden">
        <div className="grid grid-cols-1 divide-y divide-white/[0.07] sm:grid-cols-2 sm:divide-y-0 xl:grid-cols-4 xl:divide-x">
          <div className="p-4 sm:p-5">
            <div className="flex items-center justify-between gap-2">
              <span className="font-sans text-xs font-medium text-primary-muted">Audio Files</span>
              <Chip tone="primary">{reviewOnly ? "Last analysis" : "Live"}</Chip>
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{formatNumber(report.overview.total_audio_files)}</p>
            <p className="mt-1.5 text-xs text-primary-subtle">{formatNumber(report.overview.total_folders_touched)} folders scanned</p>
          </div>
          <div className="p-4 sm:p-5 sm:border-l sm:border-white/[0.07] xl:border-l-0">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Artwork Coverage</span>
              <Icon name="image" className="text-[17px] text-accent" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{report.artwork.coverage_percent.toFixed(1)}%</p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              <span className="font-mono text-accent">{formatNumber(report.artwork.with_artwork)}</span> embedded ·{" "}
              <span className="font-mono">{formatNumber(report.artwork.without_artwork)}</span> missing
            </p>
          </div>
          <div className="p-4 sm:p-5 sm:border-t sm:border-white/[0.07] xl:border-t-0">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Lossless Ratio</span>
              <Icon name="graphic_eq" className="text-[17px] text-[#7cd4aa]" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{losslessRatio.toFixed(1)}%</p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              <span className="font-mono text-primary">{formatNumber(report.quality.lossless_files)}</span> lossless ·{" "}
              <span className="font-mono">{formatNumber(report.quality.lossy_files)}</span> lossy
            </p>
          </div>
          <div className="p-4 sm:p-5 sm:border-l sm:border-t sm:border-white/[0.07] xl:border-l-0 xl:border-t-0">
            <div className="flex items-center justify-between gap-2">
              <span className="font-sans text-xs font-medium text-primary-muted">Unreadable Files</span>
              {(report.quality.unreadable_or_unparseable_files?.count ?? 0) > 0 ? <Chip tone="danger">Attention</Chip> : <Icon name="check_circle" className="text-[17px] text-[#7cd4aa]" />}
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{formatNumber(report.quality.unreadable_or_unparseable_files?.count ?? 0)}</p>
            <p className="mt-1.5 text-xs text-primary-subtle">Unparseable headers or corrupted files</p>
          </div>
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Panel className="p-5 sm:p-6">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-4">
              <div>
                <h3 className="flex items-center gap-2 font-display text-base font-semibold text-primary">
                  <Icon name="equalizer" className="text-[18px] text-accent" />
                  Bitrate Distribution
                </h3>
                <p className="mt-0.5 text-xs text-primary-subtle">
                  Audio fidelity composition across all indexed tracks
                </p>
              </div>
              <div className="flex gap-4 font-sans text-xs text-primary-subtle">
                <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-accent/45" /> Lossy</span>
                <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-accent" /> High Fidelity</span>
              </div>
            </div>
            <div className="flex h-48 items-end justify-between gap-2 px-1 sm:gap-3 sm:px-2">
              {HEALTH_BITRATE_BUCKET_ORDER.map((key) => {
                const value = report.quality.bitrate_buckets[key];
                const barTone = key === ">=256k" ? "bg-accent" : key === "unknown" ? "bg-[#e9b44c]/65" : "bg-accent/55";
                const label = HEALTH_BITRATE_BUCKET_LABELS[key];
                const pct = value === 0 ? 4 : Math.max(12, Math.round((value / maxBucket) * 100));
                return (
                  <div key={key} className="flex h-full min-w-0 flex-1 flex-col items-center gap-2">
                    <span className="font-mono text-[11px] font-medium text-primary-muted">{formatNumber(value)}</span>
                    <div className="flex w-full flex-1 items-end rounded-t-md bg-white/[0.03] px-1 pt-1">
                      <div
                        role="img"
                        aria-label={`${label}: ${formatNumber(value)} files`}
                        title={`${label}: ${formatNumber(value)} files`}
                        className={`w-full rounded-t-md transition-all duration-300 ${barTone}`}
                        style={{ height: `${pct}%` }}
                      />
                    </div>
                    <span className="w-full truncate text-center font-mono text-[10px] text-primary-subtle">{label}</span>
                  </div>
                );
              })}
            </div>
          </Panel>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Panel className="p-5">
              <div className="mb-3.5 flex items-center justify-between border-b border-white/[0.08] pb-3">
                <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-primary">
                  <Icon name="sell" className="text-base text-accent" />
                  Metadata Issues
                </h3>
                <MetadataDraftLink />
              </div>
              <div className="divide-y divide-white/[0.06] overflow-hidden rounded-xl border border-white/[0.08] bg-surface-low/65">
                {metadataIssues.map(([category, count]) => (
                  <button
                    type="button"
                    key={category}
                    aria-expanded={issue === category}
                    disabled={!count}
                    onClick={() => setIssue(issue === category ? null : category)}
                    className={`flex w-full items-center justify-between gap-2 px-3.5 py-2.5 text-left text-xs transition-colors focus-visible:outline-accent disabled:text-primary-subtle ${
                      issue === category ? "bg-accent/15 text-primary font-medium" : "text-primary hover:bg-surface-mid"
                    }`}
                  >
                    <span>{ISSUE_LABELS[category]}</span>
                    <span className="font-mono font-semibold text-accent">{formatNumber(count)}</span>
                  </button>
                ))}
              </div>
              {issue && <MetadataIssueList key={issue} issue={issue} />}
              {topFolders.length > 0 ? (
                <p className="mt-3 text-xs text-primary-subtle">
                  Concentrated in: <span className="font-mono text-accent">{topFolders[0].folder}</span>
                </p>
              ) : null}
            </Panel>

            <Panel className="p-5">
              <h3 className="mb-4 flex items-center gap-2 border-b border-white/[0.08] pb-3 font-display text-sm font-semibold text-primary">
                <Icon name="image" className="text-base text-accent" />
                Artwork Coverage
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between font-sans text-xs">
                    <span className="text-primary-muted">Embedded Artwork</span>
                    <span className="font-mono font-semibold text-[#7cd4aa]">{formatNumber(report.artwork.with_artwork)}</span>
                  </div>
                  <ProgressBar value={report.artwork.coverage_percent} tone="violet" />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between font-sans text-xs">
                    <span className="text-primary-muted">Missing Cover Art</span>
                    <span className="font-mono font-semibold text-[#f2c86b]">{formatNumber(report.artwork.without_artwork)}</span>
                  </div>
                  <ProgressBar value={withoutArtworkRatio} tone="warning" />
                </div>
              </div>
            </Panel>
          </div>
        </div>

        <Panel className="flex flex-col p-5">
          <div className="mb-4 border-b border-white/[0.08] pb-3.5">
            <h3 className="flex items-center gap-2 font-display text-base font-semibold text-primary">
              <Icon name="auto_fix_high" className="text-[18px] text-accent" />
              What to Fix First
            </h3>
            <p className="mt-1 text-xs text-primary-subtle">
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
                  className={`rounded-xl border border-white/[0.08] bg-surface-low/75 p-3.5 transition-colors hover:border-white/[0.15] ${
                    index === 0 ? "border-l-4 border-l-[#e06d53]" : index === 1 ? "border-l-4 border-l-[#e9b44c]" : "border-l-4 border-l-accent"
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <Chip tone={index === 0 ? "danger" : index === 1 ? "warning" : "primary"}>
                      {topIssueCategories[index]?.category ? topIssueCategories[index].category.replace(/_/g, " ") : index === 0 ? "primary" : index === 1 ? "secondary" : "follow-up"}
                    </Chip>
                    {topIssueCategories[index]?.count !== undefined ? (
                      <span className="shrink-0 font-mono text-[11px] text-primary-subtle">{formatNumber(topIssueCategories[index].count)} items</span>
                    ) : null}
                  </div>
                  <p className="break-words font-sans text-xs leading-relaxed text-primary">{action}</p>
                </div>
              ))
            )}
          </div>
          <div className="mt-5 border-t border-white/[0.08] pt-4">
            <Button tone="secondary" className="w-full justify-center" onClick={() => navigate("/review")}>
              Review Fixes in Inbox →
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
