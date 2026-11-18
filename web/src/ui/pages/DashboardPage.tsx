import { Link, useNavigate } from "react-router-dom";
import {
  useCheckConnection,
  useDuplicatesQuery,
  useHealthQuery,
  useHistoryQuery,
  useLibraryAnalysis,
  useReviewQuery,
  useStatusQuery,
} from "../../lib/hooks";
import {
  ActionBanner,
  ApiUnavailableState,
  Button,
  Chip,
  DataTable,
  EmptyState,
  Icon,
  PageHeader,
  Panel,
  PathBlock,
  ProgressBar,
  SleeveArt,
  formatBytes,
  formatDate,
  formatNumber,
} from "../components";

function findingIcon(itemType: string) {
  if (itemType.includes("duplicate")) return "library_music";
  if (itemType.includes("artwork") || itemType.includes("cover")) return "album";
  if (itemType.includes("metadata")) return "sell";
  if (itemType.includes("quality") || itemType.includes("bitrate")) return "graphic_eq";
  return "folder_open";
}

export function DashboardPage({ desktop = false }: { desktop?: boolean }) {
  const navigate = useNavigate();
  const statusQuery = useStatusQuery();
  const healthQuery = useHealthQuery();
  const reviewQuery = useReviewQuery();
  const duplicatesQuery = useDuplicatesQuery();
  const historyQuery = useHistoryQuery();
  const analysis = useLibraryAnalysis().data;
  const { checkConnection, checking } = useCheckConnection();
  const status = statusQuery.data;
  const reviewOnly = status?.review_only === true;

  const refreshing =
    healthQuery.isFetching ||
    reviewQuery.isFetching ||
    duplicatesQuery.isFetching ||
    historyQuery.isFetching ||
    checking;

  if (statusQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Collection Home"
          title="Overview"
          description="Browse your local music library, review curation notes, and inspect proposed changes safely."
        />
        <ApiUnavailableState contextLabel="Overview" />
      </div>
    );
  }

  const health = reviewOnly ? analysis?.health : healthQuery.data?.data;
  const review = reviewOnly ? analysis?.review : reviewQuery.data?.data;
  const duplicates = reviewOnly ? analysis?.duplicates : duplicatesQuery.data?.data;
  const formatEntries = Object.entries(health?.overview.format_breakdown ?? {}).sort((a, b) => b[1] - a[1]);
  const firstFinding = !reviewOnly && reviewQuery.isError ? undefined : review?.items.find(
    (item) => item.review_status === "new" || item.review_status === "seen",
  );

  const failedSections = [
    healthQuery.isError && !reviewOnly && "library health",
    reviewQuery.isError && !reviewOnly && "review findings",
    duplicatesQuery.isError && !reviewOnly && "duplicates",
    historyQuery.isError && "recent operations",
  ].filter(Boolean);

  const viewHeader = (
    <>
      <PageHeader
        eyebrow="Collection Home"
        title="Overview"
        description="Inspect your personal music collection, track down duplicates, and prepare metadata improvements without altering audio."
        actions={status && !reviewOnly ? (
          <Button tone="secondary" onClick={() => void checkConnection()} disabled={refreshing}>
            <Icon name="refresh" className={`text-base ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh Overview"}
          </Button>
        ) : undefined}
      />

      {status ? (
        <Panel className="overflow-hidden">
          <div className="grid gap-6 p-5 sm:p-6 lg:grid-cols-[auto_minmax(0,1fr)] lg:items-center">
            <div aria-hidden="true" className="hidden sm:flex items-center pr-4">
              <div className="relative flex items-center">
                <SleeveArt
                  title="Secondary"
                  artist={status.music_path}
                  format={formatEntries[1]?.[0]?.replace(/^\./, "").toUpperCase() || "AUDIO"}
                  size="lg"
                  showMonogram={false}
                  className="-rotate-6 opacity-80"
                />
                <SleeveArt
                  title={status.music_path}
                  artist={status.active_profile}
                  format={formatEntries[0]?.[0]?.replace(/^\./, "").toUpperCase() || "LOCAL"}
                  hasArtwork={health ? health.artwork.with_artwork > 0 : false}
                  size="lg"
                  showVinyl
                  showMonogram={false}
                  className="-ml-8 z-10 shadow-soft"
                />
              </div>
            </div>

            <div className="min-w-0 space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2.5">
                  <h2 className="font-display text-lg font-semibold text-primary">Active library</h2>
                  {health ? (
                    <Chip tone="accent">
                      {formatNumber(health.overview.total_audio_files)} tracks · {formatBytes(health.overview.total_library_size_bytes)}
                    </Chip>
                  ) : null}
                </div>
                <span className="font-sans text-xs text-primary-subtle">
                  Profile: <strong className="font-medium text-primary-muted">{status.active_profile}</strong>
                </span>
              </div>

              <PathBlock value={status.music_path} />

              <p className="text-xs leading-relaxed text-primary-muted">
                Overview reads the library configured in your local backend. Browsing, reviewing findings, and generating plan previews leave audio files unchanged.
                {!status.review_only ? " Applying a selected plan can change files. Use a disposable library to test Apply." : null}
              </p>

              <div className="flex flex-wrap items-center gap-3 pt-0.5">
                {firstFinding ? (
                  <Link
                    className="inline-flex items-center gap-2 rounded-lg border border-accent bg-accent px-3.5 py-2 font-sans text-xs font-semibold text-[#14110e] transition-colors hover:bg-[#e8ae6d] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    to={`/review?item=${encodeURIComponent(firstFinding.item_id)}`}
                  >
                    <Icon name="tune" className="text-[16px]" />
                    Inspect the first finding
                  </Link>
                ) : null}
                <Link
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-surface-mid px-3.5 py-2 font-sans text-xs font-medium text-primary transition-colors hover:border-accent/50 hover:bg-surface-high focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  to="/review"
                >
                  Open Review Inbox
                </Link>
                {!desktop && <details className="w-full pt-1 text-xs text-primary-muted sm:w-auto">
                  <summary className="cursor-pointer font-medium text-primary-muted hover:text-accent focus-visible:outline-accent">
                    Inspect a different library
                  </summary>
                  <div className="mt-2 rounded-lg border border-white/[0.08] bg-surface-low p-3 text-xs leading-relaxed text-primary-muted">
                    {status.review_only ? (
                      <p>Press <kbd>Ctrl+C</kbd> in the launcher terminal. Restart with <code>--music</code> set to the desired library and <code>--out</code> set to a separate workspace, then open the printed URL.</p>
                    ) : (
                      <p>Stop the backend, set <code>NYXCORE_WEB_MUSIC_DIR</code> to the desired library and <code>NYXCORE_WEB_OUT_DIR</code> to its reports folder, then restart the backend and refresh Overview.</p>
                    )}
                  </div>
                </details>}
              </div>
            </div>
          </div>
        </Panel>
      ) : null}
    </>
  );

  if (!status || failedSections.length || !health || !review || !duplicates || !historyQuery.data) {
    return (
      <div className="space-y-6">
        {viewHeader}
        {status && failedSections.length ? (
          <ActionBanner
            tone="error"
            message={`The local API is connected. Overview could not load ${failedSections.join(", ")}. ${reviewOnly ? "Open History to retry." : "Use Refresh Overview to try again."}`}
          />
        ) : !reviewOnly ? (
          <Panel className="p-8 text-center text-sm text-primary-muted">Loading library data from local API…</Panel>
        ) : null}
      </div>
    );
  }

  if (health.overview.total_audio_files === 0) {
    return (
      <div className="space-y-6">
        {viewHeader}
        <EmptyState title="No supported audio found" description={
          desktop
            ? "Add audio to this folder and refresh, or use Change Library to open another folder."
            : "Add audio to this folder, then refresh the library."
        } />
      </div>
    );
  }

  const history = historyQuery.data;
  const actionableReviewItems = review.items.filter(
    (item) => item.review_status === "new" || item.review_status === "seen",
  );
  const losslessRatio =
    health.overview.total_audio_files === 0
      ? 0
      : (health.quality.lossless_files / health.overview.total_audio_files) * 100;
  const unreadableCount = health.quality.unreadable_or_unparseable_files?.count ?? 0;
  const missingTagsCount =
    health.metadata.missing_artist.count +
    health.metadata.missing_title.count +
    health.metadata.missing_album.count;

  return (
    <div className="space-y-6">
      {viewHeader}

      {/* Unified Collection Ledger Strip */}
      <Panel className="overflow-hidden">
        <div className="grid grid-cols-1 divide-y divide-white/[0.07] sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4 lg:divide-x">
          <div className="p-4 sm:p-5">
            <div className="flex items-center justify-between text-primary-subtle">
              <span className="font-sans text-xs font-medium text-primary-muted">Collection Tracks</span>
              <Icon name="library_music" className="text-[17px] text-accent/80" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold tracking-tight text-primary">
              {formatNumber(health.overview.total_audio_files)}
            </p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              {formatNumber(health.overview.total_folders_touched)} folders · {formatBytes(health.overview.total_library_size_bytes)}
            </p>
          </div>

          <div className="p-4 sm:p-5 sm:border-l sm:border-white/[0.07] lg:border-l-0">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Needs Review</span>
              {actionableReviewItems.length > 0 ? <Chip tone="danger">Actionable</Chip> : <Chip tone="success">Clear</Chip>}
            </div>
            <p className="mt-2 font-display text-2xl font-semibold tracking-tight text-primary">
              {formatNumber(actionableReviewItems.length)}
            </p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              {formatNumber(review.items.length)} total findings recorded
            </p>
          </div>

          <div className="p-4 sm:p-5 sm:border-t sm:border-white/[0.07] lg:border-t-0">
            <div className="flex items-center justify-between text-primary-subtle">
              <span className="font-sans text-xs font-medium text-primary-muted">Duplicate Sets</span>
              <Icon name="content_copy" className="text-[17px] text-accent/80" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold tracking-tight text-primary">
              {formatNumber(duplicates.summary.exact_group_count)}
            </p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              Exact: <strong className="font-semibold text-primary">{duplicates.summary.exact_group_count}</strong> · Likely:{" "}
              <strong className="font-semibold text-[#f2c86b]">{duplicates.summary.likely_group_count}</strong>
            </p>
          </div>

          <div className="p-4 sm:p-5 sm:border-l sm:border-t sm:border-white/[0.07] lg:border-l-0 lg:border-t-0">
            <div className="flex items-center justify-between text-primary-subtle">
              <span className="font-sans text-xs font-medium text-primary-muted">Tag & Artwork Gaps</span>
              <Icon name="sell" className="text-[17px] text-accent/80" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold tracking-tight text-primary">
              {formatNumber(missingTagsCount + unreadableCount)}
            </p>
            <p className="mt-1.5 text-xs text-primary-subtle">
              Artwork: {health.artwork.coverage_percent.toFixed(0)}% · Unreadable: {unreadableCount}
            </p>
          </div>
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <div className="space-y-6 xl:col-span-7">
          <Panel className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] px-5 py-4">
              <div>
                <h2 className="flex items-center gap-2 font-display text-base font-semibold text-primary">
                  <Icon name="inbox" className="text-[18px] text-accent" />
                  Priority Review Stream
                </h2>
                <p className="mt-0.5 text-xs text-primary-subtle">
                  Items worth checking first to clean up duplicates, missing tags, and artwork.
                </p>
              </div>
              <Button tone="secondary" onClick={() => navigate("/review")}>
                Open Inbox →
              </Button>
            </div>
            <div className="p-3 sm:p-4">
              {actionableReviewItems.length === 0 ? (
                <EmptyState
                  title={review.items.length === 0 ? "No review findings" : "No active review findings"}
                  description={
                    review.items.length === 0
                      ? "No findings were returned for the active library. Check the folder above and refresh after adding files."
                      : "All current findings are snoozed or resolved until the next library scan."
                  }
                />
              ) : (
                <div className="divide-y divide-white/[0.06]">
                  {actionableReviewItems.slice(0, 5).map((item) => (
                    <Link
                      key={item.item_id}
                      to={`/review?item=${encodeURIComponent(item.item_id)}`}
                      className="group flex items-center justify-between gap-4 rounded-lg px-3 py-3.5 transition-colors hover:bg-surface-mid/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    >
                      <div className="flex min-w-0 items-start gap-3.5">
                        <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg border border-white/[0.08] bg-surface-low text-accent">
                          <Icon name={findingIcon(item.item_type)} className="text-[18px]" />
                        </span>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <Chip
                              tone={item.priority_band === "high" ? "danger" : item.priority_band === "medium" ? "warning" : "default"}
                              className="shrink-0 capitalize"
                            >
                              {item.priority_band}
                            </Chip>
                            <p className="truncate font-sans text-[13px] font-medium text-primary transition-colors group-hover:text-accent">
                              {item.summary}
                            </p>
                          </div>
                          <p className="mt-1 truncate text-xs text-primary-subtle">
                            {item.reason_summary}
                          </p>
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        <span className="hidden font-mono text-xs text-primary-subtle sm:inline">
                          {item.priority_score}
                        </span>
                        <span className="inline-flex items-center gap-1 font-sans text-xs font-medium text-primary-muted transition-colors group-hover:text-accent">
                          Inspect <span aria-hidden="true">→</span>
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </Panel>

          <Panel className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] px-5 py-4">
              <div>
                <h2 className="flex items-center gap-2 font-display text-base font-semibold text-primary">
                  <Icon name="history" className="text-[18px] text-accent" />
                  Recent Operations
                </h2>
                <p className="mt-0.5 text-xs text-primary-subtle">
                  Recorded edits and moves saved with reversal safety
                </p>
              </div>
              <Button tone="secondary" onClick={() => navigate("/history")}>
                View History
              </Button>
            </div>
            <div className="p-4">
              {history.items.length === 0 ? (
                <EmptyState
                  title="No operations recorded yet"
                  description="Applied review plans will appear here in the operation ledger."
                />
              ) : (
                <DataTable
                  dense
                  headers={["Operation", "Applied At", "Affected Files", "Reversibility"]}
                  rows={history.items.slice(0, 4).map((batch) => [
                    batch.action_types.join(", "),
                    formatDate(batch.applied_at),
                    formatNumber(batch.affected_count),
                    <Chip
                      key={batch.batch_id}
                      tone={batch.reversible ? "success" : "warning"}
                    >
                      {batch.reversible ? "Reversible" : "Mixed"}
                    </Chip>,
                  ])}
                />
              )}
            </div>
          </Panel>
        </div>

        <div className="space-y-6 xl:col-span-5">
          <Panel className="p-5">
            <div className="mb-4 flex items-center justify-between border-b border-white/[0.08] pb-3.5">
              <h2 className="flex items-center gap-2 font-display text-base font-semibold text-primary">
                <Icon name="graphic_eq" className="text-[18px] text-accent" />
                Library Health
              </h2>
              <Button tone="ghost" onClick={() => navigate("/health")}>
                Full Audit
              </Button>
            </div>

            {formatEntries.length > 0 ? (
              <div className="mb-5">
                <p className="mb-2 font-sans text-xs font-medium text-primary-subtle">Audio Formats</p>
                <div className="flex flex-wrap gap-2">
                  {formatEntries.map(([ext, count]) => (
                    <span
                      key={ext}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-surface-low px-2.5 py-1 text-xs text-primary"
                    >
                      <span className="font-mono font-semibold uppercase text-accent">{ext.replace(/^\./, "")}</span>
                      <span className="text-primary-subtle">{formatNumber(count)}</span>
                    </span>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="space-y-4">
              <div>
                <div className="mb-1.5 flex items-center justify-between font-sans text-xs">
                  <span className="text-primary-muted">Artwork Coverage</span>
                  <span className="font-mono font-semibold text-accent">{health.artwork.coverage_percent.toFixed(1)}%</span>
                </div>
                <ProgressBar value={health.artwork.coverage_percent} />
              </div>

              <div>
                <div className="mb-1.5 flex items-center justify-between font-sans text-xs">
                  <span className="text-primary-muted">Lossless Files Share</span>
                  <span className="font-mono font-semibold text-[#7cd4aa]">{losslessRatio.toFixed(1)}%</span>
                </div>
                <ProgressBar value={losslessRatio} tone="violet" />
              </div>

              <div className="mt-4 space-y-2.5 border-t border-white/[0.07] pt-4 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-primary-muted">Missing Artist / Title / Album</span>
                  <span className="font-mono font-medium text-primary">{formatNumber(missingTagsCount)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-primary-muted">Unreadable / Unparseable</span>
                  <span className={`font-mono ${unreadableCount > 0 ? "font-semibold text-[#f59f8a]" : "font-medium text-primary"}`}>
                    {formatNumber(unreadableCount)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-primary-muted">Placeholder Metadata</span>
                  <span className="font-mono font-medium text-primary">
                    {formatNumber(health.metadata.placeholder_metadata.count)}
                  </span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel className="p-5">
            <h3 className="mb-3.5 font-display text-sm font-semibold text-primary">
              Explore Collection Tools
            </h3>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-1">
              {[
                { to: "/search", icon: "search", label: "Archive Search", meta: "Lookup tracks" },
                { to: "/duplicates", icon: "library_music", label: "Duplicates", meta: `${duplicates.summary.exact_group_count} exact · ${duplicates.summary.likely_group_count} likely` },
                { to: "/organization", icon: "folder_open", label: "Library Organization", meta: "Preview folders" },
                ...(desktop ? [{ to: "/metadata", icon: "sell", label: "Staged Metadata", meta: "Review tag edits" }] : []),
                { to: "/review", icon: "inbox", label: "Review Inbox", meta: `${actionableReviewItems.length} open` },
                { to: "/history", icon: "history", label: "Operation History", meta: `${history.items.length} batches` },
              ].map((entry) => (
                <Link
                  key={entry.to}
                  to={entry.to}
                  className="flex items-center justify-between rounded-lg border border-white/[0.07] bg-surface-low/80 px-3.5 py-2.5 text-xs text-primary transition-colors hover:border-accent/40 hover:bg-surface-mid"
                >
                  <div className="flex items-center gap-2.5">
                    <Icon name={entry.icon} className="text-[17px] text-accent" />
                    <span className="font-medium">{entry.label}</span>
                  </div>
                  <span className="font-sans text-[11px] text-primary-subtle">{entry.meta}</span>
                </Link>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
