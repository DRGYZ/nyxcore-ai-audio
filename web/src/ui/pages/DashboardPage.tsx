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
            <p className="truncate text-xs text-slate-400">
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
              <span className="rounded bg-rose-500/10 px-2 py-0.5 text-xs font-bold text-rose-400">
                Actionable
              </span>
            ) : undefined
          }
          meta={<p className="text-xs text-slate-400">{review.items.length} total findings recorded</p>}
        />
        <MetricCard
          label="Duplicate Findings"
          value={formatNumber(duplicates.summary.exact_group_count)}
          icon="copy_all"
          meta={
            <p className="text-xs text-slate-400">
              Exact: <span className="font-bold text-slate-200">{duplicates.summary.exact_group_count}</span> • Likely: <span className="font-bold text-amber-400">{duplicates.summary.likely_group_count}</span>
            </p>
          }
        />
        <MetricCard
          label="Health Findings"
          value={formatNumber(missingTagsCount + unreadableCount)}
          icon="health_and_safety"
          meta={
            <p className="text-xs text-slate-400">
              Artwork: {health.artwork.coverage_percent.toFixed(0)}% • Unreadable: {unreadableCount}
            </p>
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
        {/* Left 2 Cols: Priority Review + Recent History */}
        <div className="space-y-8 xl:col-span-2">
          <Panel className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-display text-xl font-bold text-slate-100">
                <Icon name="inbox" className="text-primary" />
                Priority Review
              </h2>
              <Button tone="secondary" className="px-3 py-2 text-xs" onClick={() => navigate("/review")}>
                Open Review Inbox
              </Button>
            </div>
            <div className="space-y-3">
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
                    className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-primary/10 bg-primary/5 px-4 py-4 transition-colors hover:border-primary/30"
                  >
                    <div className="flex min-w-0 items-center gap-4">
                      <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary">
                        <Icon name="priority_high" />
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-bold text-slate-200">{item.summary}</p>
                        <p className="truncate font-mono text-[11px] text-slate-500">{item.reason_summary}</p>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-3">
                      <span className="font-mono text-xs text-slate-400">Score: {item.priority_score}</span>
                      <span className="rounded bg-primary/10 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.18em] text-primary">
                        {item.priority_band}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Panel>

          <Panel className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-primary/10 px-6 py-5">
              <h2 className="flex items-center gap-2 font-display text-xl font-bold text-slate-100">
                <Icon name="history" className="text-primary" />
                Recent Operations
              </h2>
              <Button tone="secondary" className="px-3 py-2 text-xs" onClick={() => navigate("/history")}>
                View History
              </Button>
            </div>
            <div className="px-4 py-4">
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
                      className={`rounded px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] ${
                        batch.reversible ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"
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
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <Icon name="health_and_safety" className="text-primary" />
                Library Health
              </h2>
              <Button tone="ghost" className="px-2.5 py-1 text-xs" onClick={() => navigate("/health")}>
                Full Audit
              </Button>
            </div>
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between text-xs">
                  <span className="text-slate-400">Artwork Coverage</span>
                  <span className="font-bold text-primary">{health.artwork.coverage_percent.toFixed(1)}%</span>
                </div>
                <ProgressBar value={health.artwork.coverage_percent} />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between text-xs">
                  <span className="text-slate-400">Lossless Files Share</span>
                  <span className="font-bold text-slate-200">{losslessRatio.toFixed(1)}%</span>
                </div>
                <ProgressBar value={losslessRatio} tone="violet" />
              </div>

              <div className="mt-4 border-t border-border-dark pt-4 space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Missing Artist / Title / Album</span>
                  <span className="font-mono font-bold text-slate-200">{formatNumber(missingTagsCount)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Unreadable / Unparseable</span>
                  <span className={`font-mono font-bold ${unreadableCount > 0 ? "text-rose-400" : "text-slate-200"}`}>
                    {formatNumber(unreadableCount)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Placeholder Metadata</span>
                  <span className="font-mono font-bold text-slate-200">
                    {formatNumber(health.metadata.placeholder_metadata.count)}
                  </span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel className="p-6">
            <h3 className="mb-3 font-display text-sm font-bold uppercase tracking-[0.2em] text-slate-400">
              Inspect Next
            </h3>
            <div className="space-y-2">
              <Link
                to="/review"
                className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:text-primary"
              >
                <div className="flex items-center gap-3">
                  <Icon name="inbox" className="text-primary" />
                  <span>Review Inbox</span>
                </div>
                <span className="text-xs font-bold text-primary">{actionableReviewItems.length}</span>
              </Link>
              <Link
                to="/duplicates"
                className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:text-primary"
              >
                <div className="flex items-center gap-3">
                  <Icon name="copy_all" className="text-primary" />
                  <span>Duplicates</span>
                </div>
                <span className="text-xs text-slate-400">{duplicates.summary.exact_group_count} exact</span>
              </Link>
              <Link
                to="/health"
                className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:text-primary"
              >
                <div className="flex items-center gap-3">
                  <Icon name="health_and_safety" className="text-primary" />
                  <span>Library Health</span>
                </div>
                <span className="text-xs text-slate-400">{missingTagsCount} tags</span>
              </Link>
              <Link
                to="/history"
                className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:text-primary"
              >
                <div className="flex items-center gap-3">
                  <Icon name="history" className="text-primary" />
                  <span>Operation History</span>
                </div>
                <span className="text-xs text-slate-400">{history.items.length} batches</span>
              </Link>
              <Link
                to="/search"
                className="flex items-center justify-between rounded-lg border border-border-dark bg-background-dark/50 px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:text-primary"
              >
                <div className="flex items-center gap-3">
                  <Icon name="manage_search" className="text-primary" />
                  <span>Archive Search</span>
                </div>
                <span className="text-xs text-slate-500">Lookup</span>
              </Link>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
