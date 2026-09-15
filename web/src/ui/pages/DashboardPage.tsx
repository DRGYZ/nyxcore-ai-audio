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
          <Button tone="primary" onClick={() => void handleRefresh()} disabled={refreshing}>
            <Icon name="refresh" className={`text-base ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh Overview"}
          </Button>
        }
      />

      {refreshNotice ? <ActionBanner tone={refreshNotice.tone} message={refreshNotice.message} /> : null}

      {/* 4 Focused Metrics */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Library Files"
          value={formatNumber(health.overview.total_audio_files)}
          icon="audio_file"
          meta={
            <p className="truncate font-mono text-xs text-primary-muted">
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
              <span className="border border-rose-500/40 bg-rose-500/10 px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-rose-400">
                Actionable
              </span>
            ) : undefined
          }
          meta={<p className="font-mono text-xs text-primary-muted">{review.items.length} total findings recorded</p>}
        />
        <MetricCard
          label="Duplicate Findings"
          value={formatNumber(duplicates.summary.exact_group_count)}
          icon="copy_all"
          meta={
            <p className="font-mono text-xs text-primary-muted">
              Exact: <span className="font-bold text-primary">{duplicates.summary.exact_group_count}</span> • Likely: <span className="font-bold text-amber-400">{duplicates.summary.likely_group_count}</span>
            </p>
          }
        />
        <MetricCard
          label="Health Findings"
          value={formatNumber(missingTagsCount + unreadableCount)}
          icon="health_and_safety"
          meta={
            <p className="font-mono text-xs text-primary-muted">
              Artwork: {health.artwork.coverage_percent.toFixed(0)}% • Unreadable: {unreadableCount}
            </p>
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
        {/* Left 2 Cols: Priority Review + Recent History */}
        <div className="space-y-8 xl:col-span-2">
          <Panel className="p-6">
            <div className="mb-4 flex items-center justify-between border-b border-border pb-4">
              <h2 className="flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary">
                <Icon name="inbox" className="text-base text-accent" />
                Priority Review
              </h2>
              <Button tone="secondary" className="px-3 py-1.5 text-xs" onClick={() => navigate("/review")}>
                Open Review Inbox
              </Button>
            </div>
            <div className="space-y-2">
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
                actionableReviewItems.slice(0, 4).map((item) => (
                  <div
                    key={item.item_id}
                    onClick={() => navigate(`/review?item=${encodeURIComponent(item.item_id)}`)}
                    className="flex cursor-pointer items-center justify-between gap-4 border border-border bg-surface-low px-4 py-3 transition-colors hover:border-border-bright"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <div className="flex size-8 shrink-0 items-center justify-center border border-border bg-surface text-accent">
                        <Icon name="priority_high" className="text-sm" />
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-xs font-semibold text-primary">{item.summary}</p>
                        <p className="truncate font-mono text-[10px] text-primary-subtle">{item.reason_summary}</p>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-3">
                      <span className="font-mono text-xs text-primary-muted">Score: {item.priority_score}</span>
                      <span className="border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-accent">
                        {item.priority_band}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Panel>

          <Panel className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-border px-6 py-4">
              <h2 className="flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary">
                <Icon name="history" className="text-base text-accent" />
                Recent Operations
              </h2>
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
                    <span
                      key={batch.batch_id}
                      className={`border px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider ${
                        batch.reversible ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400" : "border-amber-500/40 bg-amber-500/10 text-amber-400"
                      }`}
                    >
                      {batch.reversible ? "reversible" : "mixed"}
                    </span>,
                  ])}
                />
              )}
            </div>
          </Panel>
        </div>

        {/* Right Col: Health Breakdown & Quick Inspection */}
        <div className="space-y-8">
          <Panel className="p-6">
            <div className="mb-4 flex items-center justify-between border-b border-border pb-4">
              <h2 className="flex items-center gap-2 font-display text-sm font-bold uppercase tracking-wider text-primary">
                <Icon name="health_and_safety" className="text-base text-accent" />
                Library Health
              </h2>
              <Button tone="ghost" className="px-2.5 py-1 text-xs" onClick={() => navigate("/health")}>
                Full Audit
              </Button>
            </div>
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between font-mono text-xs">
                  <span className="text-primary-muted">Artwork Coverage</span>
                  <span className="font-bold text-accent">{health.artwork.coverage_percent.toFixed(1)}%</span>
                </div>
                <ProgressBar value={health.artwork.coverage_percent} />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between font-mono text-xs">
                  <span className="text-primary-muted">Lossless Files Share</span>
                  <span className="font-bold text-primary">{losslessRatio.toFixed(1)}%</span>
                </div>
                <ProgressBar value={losslessRatio} tone="violet" />
              </div>

              <div className="mt-4 border-t border-border pt-4 space-y-2.5">
                <div className="flex justify-between font-mono text-xs">
                  <span className="text-primary-muted">Missing Artist / Title / Album</span>
                  <span className="font-bold text-primary">{formatNumber(missingTagsCount)}</span>
                </div>
                <div className="flex justify-between font-mono text-xs">
                  <span className="text-primary-muted">Unreadable / Unparseable</span>
                  <span className={`font-bold ${unreadableCount > 0 ? "text-rose-400" : "text-primary"}`}>
                    {formatNumber(unreadableCount)}
                  </span>
                </div>
                <div className="flex justify-between font-mono text-xs">
                  <span className="text-primary-muted">Placeholder Metadata</span>
                  <span className="font-bold text-primary">
                    {formatNumber(health.metadata.placeholder_metadata.count)}
                  </span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel className="p-6">
            <h3 className="mb-3 font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">
              Inspect Next
            </h3>
            <div className="space-y-1.5">
              <Link
                to="/review"
                className="flex items-center justify-between border border-border bg-surface-low px-4 py-2.5 font-mono text-xs text-primary-muted transition-colors hover:border-accent hover:text-accent"
              >
                <div className="flex items-center gap-3">
                  <Icon name="inbox" className="text-accent text-sm" />
                  <span>Review Inbox</span>
                </div>
                <span className="font-bold text-accent">{actionableReviewItems.length}</span>
              </Link>
              <Link
                to="/duplicates"
                className="flex items-center justify-between border border-border bg-surface-low px-4 py-2.5 font-mono text-xs text-primary-muted transition-colors hover:border-accent hover:text-accent"
              >
                <div className="flex items-center gap-3">
                  <Icon name="copy_all" className="text-accent text-sm" />
                  <span>Duplicates</span>
                </div>
                <span className="text-primary-subtle">{duplicates.summary.exact_group_count} exact</span>
              </Link>
              <Link
                to="/health"
                className="flex items-center justify-between border border-border bg-surface-low px-4 py-2.5 font-mono text-xs text-primary-muted transition-colors hover:border-accent hover:text-accent"
              >
                <div className="flex items-center gap-3">
                  <Icon name="health_and_safety" className="text-accent text-sm" />
                  <span>Library Health</span>
                </div>
                <span className="text-primary-subtle">{missingTagsCount} tags</span>
              </Link>
              <Link
                to="/history"
                className="flex items-center justify-between border border-border bg-surface-low px-4 py-2.5 font-mono text-xs text-primary-muted transition-colors hover:border-accent hover:text-accent"
              >
                <div className="flex items-center gap-3">
                  <Icon name="history" className="text-accent text-sm" />
                  <span>Operation History</span>
                </div>
                <span className="text-primary-subtle">{history.items.length} batches</span>
              </Link>
              <Link
                to="/search"
                className="flex items-center justify-between border border-border bg-surface-low px-4 py-2.5 font-mono text-xs text-primary-muted transition-colors hover:border-accent hover:text-accent"
              >
                <div className="flex items-center gap-3">
                  <Icon name="manage_search" className="text-accent text-sm" />
                  <span>Archive Search</span>
                </div>
                <span className="text-primary-subtle">Lookup</span>
              </Link>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
