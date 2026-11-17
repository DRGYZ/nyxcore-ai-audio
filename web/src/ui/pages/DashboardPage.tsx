import { Link, useNavigate } from "react-router-dom";
import {
  useCheckConnection,
  useDuplicatesQuery,
  useHealthQuery,
  useHistoryQuery,
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
  MetricCard,
  PageHeader,
  Panel,
  PathBlock,
  ProgressBar,
  formatDate,
  formatNumber,
} from "../components";

export function DashboardPage() {
  const navigate = useNavigate();
  const statusQuery = useStatusQuery();
  const healthQuery = useHealthQuery();
  const reviewQuery = useReviewQuery();
  const duplicatesQuery = useDuplicatesQuery();
  const historyQuery = useHistoryQuery();
  const { checkConnection, checking } = useCheckConnection();
  const status = statusQuery.data;

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
          eyebrow="Library Overview"
          title="Overview"
          description="Local music library review priorities, duplicate findings, and recent operations."
        />
        <ApiUnavailableState contextLabel="Overview" />
      </div>
    );
  }

  const firstFinding = reviewQuery.isError ? undefined : reviewQuery.data?.data.items.find(
    (item) => item.review_status === "new" || item.review_status === "seen",
  );
  const viewHeader = (
    <>
      <PageHeader
        eyebrow="Library Overview"
        title="Overview"
        description="Inspect your local library, review findings, and preview proposed changes."
        actions={status ? (
          <Button tone="secondary" onClick={() => void checkConnection()} disabled={refreshing}>
            <Icon name="refresh" className={`text-base ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh Overview"}
          </Button>
        ) : undefined}
      />
      {status ? (
        <Panel className="min-w-0 space-y-3 p-4 sm:p-5">
          <h2 className="font-display text-base font-semibold text-primary">Active library</h2>
          <PathBlock value={status.music_path} />
          <p className="text-xs leading-relaxed text-primary-muted">
            Overview reads the library configured in your local backend. Browsing, reviewing findings, and generating plan previews leave audio files unchanged.
            {!status.review_only ? " Applying a selected plan can change files. Use a disposable library to test Apply." : null}
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
            {firstFinding ? (
              <Link className="font-semibold text-accent underline-offset-4 hover:underline focus-visible:outline-accent" to={`/review?item=${encodeURIComponent(firstFinding.item_id)}`}>Inspect the first finding</Link>
            ) : null}
            <Link className="text-primary underline-offset-4 hover:underline focus-visible:outline-accent" to="/review">Open Review Inbox</Link>
          </div>
          <details className="border-t border-white/[0.07] pt-2 text-xs text-primary-muted">
            <summary className="cursor-pointer text-primary focus-visible:outline-accent">Inspect a different library</summary>
            {status.review_only ? (
              <p className="mt-2">Press <kbd>Ctrl+C</kbd> in the launcher terminal. Restart with <code>--music</code> set to the desired library and <code>--out</code> set to a separate workspace, then open the printed URL.</p>
            ) : (
              <p className="mt-2">Stop the backend, set <code>NYXCORE_WEB_MUSIC_DIR</code> to the desired library and <code>NYXCORE_WEB_OUT_DIR</code> to its reports folder, then restart the backend and refresh Overview.</p>
            )}
          </details>
        </Panel>
      ) : null}
    </>
  );
  const failedSections = [
    healthQuery.isError && "library health",
    reviewQuery.isError && "review findings",
    duplicatesQuery.isError && "duplicates",
    historyQuery.isError && "recent operations",
  ].filter(Boolean);
  if (!status || failedSections.length || !healthQuery.data || !reviewQuery.data || !duplicatesQuery.data || !historyQuery.data) {
    return (
      <div className="space-y-6">
        {viewHeader}
        {status && failedSections.length ? (
          <ActionBanner tone="error" message={`The local API is connected. Overview could not load ${failedSections.join(", ")}. Use Refresh Overview to try again.`} />
        ) : (
          <Panel className="p-8 text-center text-sm text-slate-400">Loading library data from local API…</Panel>
        )}
      </div>
    );
  }

  const health = healthQuery.data.data;
  const review = reviewQuery.data.data;
  const duplicates = duplicatesQuery.data.data;
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
    <div className="space-y-8">
      {viewHeader}

      {/* 4 Focused Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Library Files"
          value={formatNumber(health.overview.total_audio_files)}
          icon="audio_file"
          meta={
            <p className="truncate font-mono text-[11px] text-primary-subtle">
              {formatNumber(health.overview.total_folders_touched)} folders scanned
            </p>
          }
        />
        <MetricCard
          label="Needs Review"
          value={formatNumber(actionableReviewItems.length)}
          icon="inbox"
          accent={
            actionableReviewItems.length > 0 ? (
              <Chip tone="danger">Actionable</Chip>
            ) : undefined
          }
          meta={<p className="font-mono text-[11px] text-primary-subtle">{review.items.length} total findings recorded</p>}
        />
        <MetricCard
          label="Duplicate Sets"
          value={formatNumber(duplicates.summary.exact_group_count)}
          icon="copy_all"
          meta={
            <p className="font-mono text-[11px] text-primary-subtle">
              Exact: <span className="font-semibold text-primary">{duplicates.summary.exact_group_count}</span> • Likely: <span className="font-semibold text-amber-400">{duplicates.summary.likely_group_count}</span>
            </p>
          }
        />
        <MetricCard
          label="Health Issues"
          value={formatNumber(missingTagsCount + unreadableCount)}
          icon="health_and_safety"
          meta={
            <p className="font-mono text-[11px] text-primary-subtle">
              Artwork: {health.artwork.coverage_percent.toFixed(0)}% • Unreadable: {unreadableCount}
            </p>
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Left 2 Cols: Priority Review + Recent History */}
        <div className="space-y-6 xl:col-span-2">
          <Panel className="p-5">
            <div className="mb-4 flex items-center justify-between border-b border-white/[0.07] pb-4">
              <div>
                <h2 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-primary">
                  <Icon name="inbox" className="text-base text-accent" />
                  Priority Review Stream
                </h2>
                <p className="mt-0.5 font-editorial text-xs italic text-primary-subtle">
                  High-leverage issues requiring operator decision
                </p>
              </div>
              <Button tone="secondary" className="px-3 py-1.5 text-xs" onClick={() => navigate("/review")}>
                Open Inbox →
              </Button>
            </div>
            <div>
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
                <div className="divide-y divide-white/[0.05]">
                  {actionableReviewItems.slice(0, 4).map((item) => (
                    <Link
                      key={item.item_id}
                      to={`/review?item=${encodeURIComponent(item.item_id)}`}
                      className="group -mx-2 flex items-center justify-between gap-4 rounded-[3px] px-3 py-3 transition-colors hover:bg-surface-raised/80 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <Chip
                          tone={item.priority_band === "high" ? "danger" : item.priority_band === "medium" ? "warning" : "default"}
                          className="shrink-0"
                        >
                          {item.priority_band}
                        </Chip>
                        <div className="min-w-0">
                          <p className="truncate font-sans text-xs font-medium text-primary transition-colors group-hover:text-accent">
                            {item.summary}
                          </p>
                          <p className="truncate font-editorial text-xs italic text-primary-subtle">
                            {item.reason_summary}
                          </p>
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-4">
                        <span className="font-mono text-xs text-primary-subtle">Score {item.priority_score}</span>
                        <span className="font-sans text-xs text-primary-subtle transition-colors group-hover:text-accent">
                          Inspect →
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </Panel>

          <Panel className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/[0.07] px-5 py-4">
              <div>
                <h2 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-primary">
                  <Icon name="history" className="text-base text-accent" />
                  Recent Operations
                </h2>
                <p className="mt-0.5 font-editorial text-xs italic text-primary-subtle">
                  Batches committed to the local ledger
                </p>
              </div>
              <Button tone="secondary" className="px-3 py-1.5 text-xs" onClick={() => navigate("/history")}>
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

        {/* Right Col: Health Breakdown & Quick Inspection */}
        <div className="space-y-6">
          <Panel className="p-5">
            <div className="mb-4 flex items-center justify-between border-b border-white/[0.07] pb-3.5">
              <h2 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-primary">
                <Icon name="health_and_safety" className="text-base text-accent" />
                Library Health
              </h2>
              <Button tone="ghost" className="px-2.5 py-1 text-xs" onClick={() => navigate("/health")}>
                Full Audit
              </Button>
            </div>
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between font-sans text-xs">
                  <span className="text-primary-subtle">Artwork Coverage</span>
                  <span className="font-semibold text-accent">{health.artwork.coverage_percent.toFixed(1)}%</span>
                </div>
                <ProgressBar value={health.artwork.coverage_percent} />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between font-sans text-xs">
                  <span className="text-primary-subtle">Lossless Files Share</span>
                  <span className="font-semibold text-primary">{losslessRatio.toFixed(1)}%</span>
                </div>
                <ProgressBar value={losslessRatio} tone="violet" />
              </div>

              <div className="mt-4 space-y-2 border-t border-white/[0.06] pt-3.5">
                <div className="flex justify-between font-sans text-xs">
                  <span className="text-primary-subtle">Missing Artist / Title / Album</span>
                  <span className="font-mono text-primary">{formatNumber(missingTagsCount)}</span>
                </div>
                <div className="flex justify-between font-sans text-xs">
                  <span className="text-primary-subtle">Unreadable / Unparseable</span>
                  <span className={`font-mono ${unreadableCount > 0 ? "text-rose-400 font-semibold" : "text-primary"}`}>
                    {formatNumber(unreadableCount)}
                  </span>
                </div>
                <div className="flex justify-between font-sans text-xs">
                  <span className="text-primary-subtle">Placeholder Metadata</span>
                  <span className="font-mono text-primary">
                    {formatNumber(health.metadata.placeholder_metadata.count)}
                  </span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel className="p-5">
            <h3 className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">
              Inspect Next
            </h3>
            <div className="space-y-1.5">
              <Link
                to="/review"
                className="flex items-center justify-between rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-3.5 py-2.5 font-sans text-xs text-primary-muted transition-colors hover:border-accent/40 hover:bg-surface-mid hover:text-accent"
              >
                <div className="flex items-center gap-2.5">
                  <Icon name="inbox" className="text-sm text-accent" />
                  <span>Review Inbox</span>
                </div>
                <span className="font-mono font-semibold text-accent">{actionableReviewItems.length}</span>
              </Link>
              <Link
                to="/duplicates"
                className="flex items-center justify-between rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-3.5 py-2.5 font-sans text-xs text-primary-muted transition-colors hover:border-accent/40 hover:bg-surface-mid hover:text-accent"
              >
                <div className="flex items-center gap-2.5">
                  <Icon name="copy_all" className="text-sm text-accent" />
                  <span>Duplicates</span>
                </div>
                <span className="font-mono text-primary-subtle">{duplicates.summary.exact_group_count} exact</span>
              </Link>
              <Link
                to="/health"
                className="flex items-center justify-between rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-3.5 py-2.5 font-sans text-xs text-primary-muted transition-colors hover:border-accent/40 hover:bg-surface-mid hover:text-accent"
              >
                <div className="flex items-center gap-2.5">
                  <Icon name="health_and_safety" className="text-sm text-accent" />
                  <span>Library Health</span>
                </div>
                <span className="font-mono text-primary-subtle">{missingTagsCount} tags</span>
              </Link>
              <Link
                to="/history"
                className="flex items-center justify-between rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-3.5 py-2.5 font-sans text-xs text-primary-muted transition-colors hover:border-accent/40 hover:bg-surface-mid hover:text-accent"
              >
                <div className="flex items-center gap-2.5">
                  <Icon name="history" className="text-sm text-accent" />
                  <span>Operation History</span>
                </div>
                <span className="font-mono text-primary-subtle">{history.items.length} batches</span>
              </Link>
              <Link
                to="/search"
                className="flex items-center justify-between rounded-[3px] border border-white/[0.06] bg-surface-low/60 px-3.5 py-2.5 font-sans text-xs text-primary-muted transition-colors hover:border-accent/40 hover:bg-surface-mid hover:text-accent"
              >
                <div className="flex items-center gap-2.5">
                  <Icon name="manage_search" className="text-sm text-accent" />
                  <span>Archive Search</span>
                </div>
                <span className="font-mono text-primary-subtle">Lookup</span>
              </Link>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
