import { useState } from "react";
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
  const [refreshNotice, setRefreshNotice] = useState<{ tone: "success" | "error"; message: string } | null>(null);

  const isUnavailable =
    healthQuery.isError ||
    reviewQuery.isError ||
    duplicatesQuery.isError ||
    historyQuery.isError ||
    statusQuery.isError;

  const isLoading =
    !isUnavailable &&
    (healthQuery.isLoading ||
      reviewQuery.isLoading ||
      duplicatesQuery.isLoading ||
      historyQuery.isLoading);

  const refreshing =
    healthQuery.isFetching ||
    reviewQuery.isFetching ||
    duplicatesQuery.isFetching ||
    historyQuery.isFetching ||
    checking;

  async function handleRefresh() {
    setRefreshNotice(null);
    try {
      await checkConnection();
      setRefreshNotice({ tone: "success", message: "Library overview refreshed from local API." });
    } catch {
      setRefreshNotice({ tone: "error", message: "Refresh failed. Local API may be offline." });
    }
  }

  if (isUnavailable) {
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

  if (isLoading || !healthQuery.data || !reviewQuery.data || !duplicatesQuery.data || !historyQuery.data) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Library Overview"
          title="Overview"
          description="Local music library review priorities, duplicate findings, and recent operations."
        />
        <Panel className="p-8 text-center text-sm text-slate-400">
          Loading library data from local API…
        </Panel>
      </div>
    );
  }

  const status = statusQuery.data;
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
      <PageHeader
        eyebrow="Library Overview"
        title="Overview"
        description="Local music library overview, review priorities, duplicate findings, and recent operations."
        actions={
          <Button tone="secondary" onClick={() => void handleRefresh()} disabled={refreshing}>
            <Icon name="refresh" className={`text-base ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh Overview"}
          </Button>
        }
      />

      {refreshNotice ? <ActionBanner tone={refreshNotice.tone} message={refreshNotice.message} /> : null}

      {/* 4 Focused Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Library Files"
          value={formatNumber(health.overview.total_audio_files)}
          icon="audio_file"
          meta={
            <p className="truncate font-mono text-[11px] text-primary-subtle">
              {status?.music_path ? status.music_path : `${formatNumber(health.overview.total_folders_touched)} folders scanned`}
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
                      ? "Run a scan with `python -m nyxcore.cli review <music-dir> --out data/reports` to populate findings."
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
