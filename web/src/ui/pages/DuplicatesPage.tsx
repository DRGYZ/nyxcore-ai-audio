import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDuplicatesQuery, useReviewQuery } from "../../lib/hooks";
import type { DuplicateGroup, ReviewItem } from "../../lib/types";
import { Button, EmptyState, Icon, PageHeader, Panel, PathBlock, formatBytes, formatNumber } from "../components";
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
      />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <Panel className="p-6">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Total Reclaimable</p>
          <div className="mt-3 flex items-baseline gap-2">
            <h3 className="font-display text-3xl font-bold tracking-tight text-primary">{formatBytes(reclaimable)}</h3>
            <span className="font-mono text-xs font-semibold text-emerald-400">review only</span>
          </div>
        </Panel>
        <Panel className="p-6">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Duplicate Groups</p>
          <h3 className="mt-3 font-display text-3xl font-bold tracking-tight text-primary">{formatNumber(report.summary.exact_group_count + report.summary.likely_group_count)}</h3>
        </Panel>
        <Panel className="p-6">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">Duplicate Findings Status</p>
          <div className="mt-3 flex items-center gap-2">
            <span className="size-2 bg-accent" />
            <h3 className="font-display text-2xl font-bold tracking-tight text-primary">{totalGroups > 0 ? "Needs Review" : "Clean"}</h3>
          </div>
        </Panel>
      </div>
      <div className="flex gap-4 border-b border-border">
        <button
          type="button"
          onClick={() => setActiveTab("exact")}
          className={`flex items-center gap-2 border-b-2 px-3 pb-3 font-mono text-xs uppercase tracking-wider ${
            activeTab === "exact" ? "border-accent text-accent font-semibold" : "border-transparent text-primary-muted hover:text-primary"
          }`}
        >
          <Icon name="copy_all" className="text-sm" />
          Exact Duplicates ({report.summary.exact_group_count})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("likely")}
          className={`flex items-center gap-2 border-b-2 px-3 pb-3 font-mono text-xs uppercase tracking-wider ${
            activeTab === "likely" ? "border-accent text-accent font-semibold" : "border-transparent text-primary-muted hover:text-primary"
          }`}
        >
          <Icon name="difference" className="text-sm" />
          Likely Duplicates ({report.summary.likely_group_count})
        </button>
      </div>
      <div className="space-y-6">
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
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-surface-low p-4">
                <div className="flex items-center gap-3">
                  <div className="flex size-8 items-center justify-center border border-border bg-surface text-accent">
                    <Icon name="audio_file" className="text-sm" />
                  </div>
                  <div>
                    <h4 className="font-mono text-xs font-bold text-primary">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                    <p className="font-mono text-[10px] text-primary-subtle">{group.files.length} occurrences found across the library</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Confidence</p>
                    <p className="font-mono text-xs font-bold text-accent">100%</p>
                  </div>
                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Potential Saving</p>
                    <p className="font-mono text-xs font-bold text-emerald-400">{formatBytes(reclaim)}</p>
                  </div>
                  <Button tone="secondary" className="px-3 py-1.5 text-xs" onClick={() => openReview(group, "exact_duplicate_group")}>
                    Open Review
                  </Button>
                </div>
              </div>
              <div className="space-y-3 p-4">
                <div className="flex items-center justify-between border border-emerald-500/30 bg-surface-low p-3.5">
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-400">Preferred Copy</p>
                    <div className="mt-2">
                      <PathBlock value={group.preferred.path} tone="success" />
                    </div>
                  </div>
                  <p className="max-w-xs pl-4 text-right font-mono text-[11px] text-primary-muted">{group.preferred.reasons.join(" • ")}</p>
                </div>
                <div className="space-y-2">
                  {group.files
                    .filter((file) => file.path !== group.preferred.path)
                    .map((file) => (
                      <div key={file.path} className="flex items-center justify-between gap-3 border border-border bg-surface-low p-2.5">
                        <div className="flex min-w-0 items-center gap-3">
                          <Icon name="delete_sweep" className="text-sm text-primary-subtle" />
                          <div className="min-w-0 flex-1">
                            <PathBlock value={file.path} />
                          </div>
                        </div>
                        <Button tone="ghost" className="shrink-0 px-2 py-1 text-[10px]" onClick={() => openReview(group, "exact_duplicate_group")}>Review Choice</Button>
                      </div>
                    ))}
                </div>
              </div>
            </Panel>
          );
        }) : null}
        {activeTab === "likely" ? report.likely_duplicates.map((group) => (
          <Panel key={group.group_id} className="overflow-hidden border-l-2 border-l-accent">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-surface-low p-4">
              <div>
                <h4 className="font-mono text-xs font-bold text-primary">{group.files[0]?.path.split(/[\\/]/).pop()}</h4>
                <p className="font-mono text-[10px] text-primary-subtle">Similar content detected with metadata mismatch</p>
              </div>
              <div className="flex items-center gap-6">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Confidence</p>
                  <p className="font-mono text-xs font-bold text-accent">{((group.confidence ?? 0) * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Preferred Copy</p>
                  <p className="font-mono text-xs font-bold text-primary">{group.preferred.path.split(/[\\/]/).pop()}</p>
                </div>
                <Button tone="secondary" className="px-3 py-1.5 text-xs" onClick={() => openReview(group, "likely_duplicate_group")}>
                  Open Review
                </Button>
              </div>
            </div>
            <div className="p-4">
              <p className="break-words font-mono text-xs text-primary-muted">{(group.reasons ?? []).join(", ")}</p>
            </div>
          </Panel>
        )) : null}
      </div>
      <Panel className="border border-border bg-surface-low p-6">
        <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
          <div>
            <h2 className="font-display text-lg font-bold uppercase tracking-wider text-primary">Reclaim Library Space</h2>
            <p className="mt-1.5 max-w-2xl font-mono text-xs text-primary-muted">
              Exact duplicates can generate quarantine-first plans in the Review Inbox. Likely matches remain manual-review only.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Button tone="secondary" onClick={() => setActiveTab("likely")} disabled={report.likely_duplicates.length === 0}>Inspect Likely Matches</Button>
            <Button
              tone="primary"
              onClick={() => {
                const firstExact = report.exact_duplicates[0];
                if (firstExact) openReview(firstExact, "exact_duplicate_group");
              }}
              disabled={report.exact_duplicates.length === 0}
            >
              Plan in Review Inbox
            </Button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
