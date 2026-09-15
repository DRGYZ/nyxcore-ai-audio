import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDuplicatesQuery, useReviewQuery } from "../../lib/hooks";
import type { DuplicateGroup, ReviewItem } from "../../lib/types";
import { Button, Chip, EmptyState, Icon, MetricCard, PageHeader, Panel, PathBlock, formatBytes, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";

export function DuplicatesPage() {
  const navigate = useNavigate();
  const duplicatesQuery = useDuplicatesQuery();
  const reviewQuery = useReviewQuery();
  const [activeTab, setActiveTab] = useState<"exact" | "likely">("exact");

  if (duplicatesQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Duplicate Detection"
          title="Duplicates"
          description="Exact duplicates and likely duplicate clusters, with preferred-copy recommendations and reclaimable space summaries."
        />
        <ApiUnavailableState contextLabel="Duplicates" />
      </div>
    );
  }

  if (duplicatesQuery.isLoading || !duplicatesQuery.data) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Duplicate Detection"
          title="Duplicates"
          description="Exact duplicates and likely duplicate clusters, with preferred-copy recommendations and reclaimable space summaries."
        />
        <Panel className="p-8 text-center text-sm text-slate-400">
          Loading duplicate findings from local API…
        </Panel>
      </div>
    );
  }

  const report = duplicatesQuery.data.data;
  const reviewItems = reviewQuery.data?.data?.items ?? [];

  const reclaimable = report.exact_duplicates.reduce(
    (sum, group) => sum + (group.reclaimable_bytes ?? group.files.slice(1).reduce((acc, item) => acc + item.file_size_bytes, 0)),
    0,
  );
  const activeGroups = activeTab === "exact" ? report.exact_duplicates : report.likely_duplicates;
  const totalGroups = report.summary.exact_group_count + report.summary.likely_group_count;

  function findReviewItem(group: DuplicateGroup, itemType: "exact_duplicate_group" | "likely_duplicate_group"): ReviewItem | undefined {
    const direct = reviewItems.find(
      (item) => item.item_type === itemType && item.details?.source_group_id === group.group_id,
    );
    if (direct) return direct;
    const groupPaths = new Set(group.files.map((file) => file.path.toLocaleLowerCase().replace(/\\/g, "/")));
    return reviewItems.find(
      (item) =>
        item.item_type === itemType
        && (item.affected_paths ?? []).some((path) => groupPaths.has(path.toLocaleLowerCase().replace(/\\/g, "/"))),
    );
  }

  function openReview(group: DuplicateGroup, itemType: "exact_duplicate_group" | "likely_duplicate_group") {
    const item = findReviewItem(group, itemType);
    const params = new URLSearchParams({ type: itemType });
    if (item) params.set("item", item.item_id);
    navigate(`/review?${params.toString()}`);
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Duplicate Detection"
        title="Duplicates"
        description="Exact duplicates and likely duplicate clusters, with preferred-copy recommendations and reclaimable space summaries."
        actions={
          <Button
            tone="primary"
            onClick={() => {
              const firstExact = report.exact_duplicates[0];
              if (firstExact) openReview(firstExact, "exact_duplicate_group");
            }}
            disabled={report.exact_duplicates.length === 0}
          >
            Plan in Review Inbox →
          </Button>
        }
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <MetricCard
          label="Total Reclaimable"
          value={formatBytes(reclaimable)}
          icon="data_saver_on"
          accent={<Chip tone="success">Review Only</Chip>}
          meta={<p className="font-sans text-[11px] text-primary-subtle">Space reclaimable via non-destructive quarantine</p>}
        />
        <MetricCard
          label="Duplicate Sets"
          value={formatNumber(totalGroups)}
          icon="copy_all"
          meta={
            <p className="font-sans text-[11px] text-primary-subtle">
              Exact: <span className="font-mono font-semibold text-primary">{report.summary.exact_group_count}</span> • Likely: <span className="font-mono font-semibold text-amber-400">{report.summary.likely_group_count}</span>
            </p>
          }
        />
        <MetricCard
          label="Triage Status"
          value={totalGroups > 0 ? "Action Required" : "Clean"}
          icon="checklist"
          accent={
            <Chip tone={totalGroups > 0 ? "warning" : "success"}>
              {totalGroups > 0 ? `${totalGroups} Unresolved` : "Optimal"}
            </Chip>
          }
          meta={<p className="font-sans text-[11px] text-primary-subtle">Operator confirmation required before execution</p>}
        />
      </div>

      <div className="flex gap-2 border-b border-white/[0.07] pb-3">
        <button
          type="button"
          onClick={() => setActiveTab("exact")}
          className={`flex items-center gap-2 rounded-[3px] px-3.5 py-1.5 font-sans text-xs font-medium transition-colors ${
            activeTab === "exact"
              ? "bg-white/[0.08] text-accent shadow-sm"
              : "text-primary-subtle hover:bg-white/[0.03] hover:text-primary"
          }`}
        >
          <Icon name="copy_all" className="text-sm" />
          <span>Exact Duplicates</span>
          <span className="rounded-[2px] bg-white/[0.06] px-1.5 py-0.2 font-mono text-[10px] text-primary-muted">
            {report.summary.exact_group_count}
          </span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("likely")}
          className={`flex items-center gap-2 rounded-[3px] px-3.5 py-1.5 font-sans text-xs font-medium transition-colors ${
            activeTab === "likely"
              ? "bg-white/[0.08] text-accent shadow-sm"
              : "text-primary-subtle hover:bg-white/[0.03] hover:text-primary"
          }`}
        >
          <Icon name="difference" className="text-sm" />
          <span>Likely Duplicates</span>
          <span className="rounded-[2px] bg-white/[0.06] px-1.5 py-0.2 font-mono text-[10px] text-primary-muted">
            {report.summary.likely_group_count}
          </span>
        </button>
      </div>

      <div className="space-y-4">
        {activeGroups.length === 0 ? (
          <EmptyState
            title={`No ${activeTab} duplicate groups found`}
            description={`When ${activeTab} duplicate clusters are detected, preferred-copy recommendations and supporting evidence will appear here.`}
          />
        ) : null}

        {activeTab === "exact" ? report.exact_duplicates.map((group) => {
          const reclaim = group.reclaimable_bytes ?? group.files.slice(1).reduce((sum, item) => sum + item.file_size_bytes, 0);
          return (
            <Panel key={group.group_id} className="overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.07] bg-surface-low/80 px-5 py-3.5">
                <div className="flex items-center gap-3">
                  <div className="flex size-7 items-center justify-center rounded-[2px] border border-white/[0.08] bg-surface text-accent">
                    <Icon name="audio_file" className="text-sm" />
                  </div>
                  <div>
                    <h4 className="font-sans text-xs font-semibold text-primary">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                    <p className="font-editorial text-xs italic text-primary-subtle">{group.files.length} redundant copies identified across library</p>
                  </div>
                </div>
                <div className="flex items-center gap-5">
                  <div className="text-right">
                    <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Confidence</p>
                    <p className="font-mono text-xs font-medium text-accent">100% (Bit-Exact)</p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Reclaimable</p>
                    <p className="font-mono text-xs font-medium text-emerald-400">{formatBytes(reclaim)}</p>
                  </div>
                  <Button tone="secondary" className="px-3 py-1 text-xs" onClick={() => openReview(group, "exact_duplicate_group")}>
                    Triage →
                  </Button>
                </div>
              </div>

              <div className="space-y-3 p-5">
                <div className="rounded-[3px] border border-emerald-500/20 bg-emerald-500/[0.03] p-3.5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Chip tone="success">Preferred Copy</Chip>
                      <span className="font-editorial text-xs italic text-primary-subtle">
                        {group.preferred.reasons.join(" • ")}
                      </span>
                    </div>
                  </div>
                  <div className="mt-2.5">
                    <PathBlock value={group.preferred.path} tone="success" />
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">
                    Redundant Copies ({group.files.length - 1})
                  </p>
                  <div className="divide-y divide-white/[0.05] rounded-[3px] border border-white/[0.06] bg-surface-low/50">
                    {group.files
                      .filter((file) => file.path !== group.preferred.path)
                      .map((file) => (
                        <div key={file.path} className="flex items-center justify-between gap-3 px-3 py-2.5">
                          <div className="flex min-w-0 flex-1 items-center gap-2.5">
                            <Icon name="delete_sweep" className="text-sm text-primary-subtle/70" />
                            <div className="min-w-0 flex-1">
                              <PathBlock value={file.path} />
                            </div>
                          </div>
                          <Button tone="ghost" className="shrink-0 px-2 py-1 text-[11px]" onClick={() => openReview(group, "exact_duplicate_group")}>
                            Review Choice
                          </Button>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </Panel>
          );
        }) : null}

        {activeTab === "likely" ? report.likely_duplicates.map((group) => (
          <Panel key={group.group_id} className="overflow-hidden border-l-2 border-l-accent/70">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.07] bg-surface-low/80 px-5 py-3.5">
              <div>
                <h4 className="font-sans text-xs font-semibold text-primary">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                <p className="font-editorial text-xs italic text-primary-subtle">Similar content detected with metadata or tag variance</p>
              </div>
              <div className="flex items-center gap-5">
                <div className="text-right">
                  <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Match Score</p>
                  <p className="font-mono text-xs font-medium text-accent">{((group.confidence ?? 0) * 100).toFixed(0)}%</p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Preferred</p>
                  <p className="font-mono text-xs font-medium text-primary">{group.preferred.path.split(/[\\/]/).pop()}</p>
                </div>
                <Button tone="secondary" className="px-3 py-1 text-xs" onClick={() => openReview(group, "likely_duplicate_group")}>
                  Inspect →
                </Button>
              </div>
            </div>
            <div className="p-5">
              <p className="font-editorial text-xs italic text-primary-subtle">{(group.reasons ?? []).join(", ")}</p>
            </div>
          </Panel>
        )) : null}
      </div>

      <Panel className="border border-white/[0.08] bg-surface-low p-6">
        <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
          <div>
            <h2 className="font-display text-base font-semibold tracking-wide text-primary">Reclaim Library Space</h2>
            <p className="mt-1 max-w-2xl font-sans text-xs text-primary-muted">
              Exact duplicates generate quarantine-first plans in the Review Inbox. Likely matches remain manual-review only to safeguard audio fidelity.
            </p>
          </div>
          <div className="flex flex-col gap-2.5 sm:flex-row">
            <Button tone="secondary" onClick={() => setActiveTab("likely")} disabled={report.likely_duplicates.length === 0}>
              Inspect Likely Matches
            </Button>
            <Button
              tone="primary"
              onClick={() => {
                const firstExact = report.exact_duplicates[0];
                if (firstExact) openReview(firstExact, "exact_duplicate_group");
              }}
              disabled={report.exact_duplicates.length === 0}
            >
              Plan in Review Inbox →
            </Button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
