import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { matchesDuplicateFinding, sameDuplicateLibrary } from "../../lib/duplicate-evidence";
import { useDuplicatesQuery, useReviewQuery } from "../../lib/hooks";
import { Chip, EmptyState, Icon, MetricCard, PageHeader, Panel, formatBytes, formatNumber } from "../components";
import { DuplicateComparison } from "../duplicates/DuplicateComparison";
import { ApiUnavailableState } from "../feedback";

const tabs = [
  { kind: "exact", label: "Exact Duplicates", icon: "copy_all" },
  { kind: "likely", label: "Likely Duplicates", icon: "difference" },
] as const;

export function DuplicatesPage() {
  const navigate = useNavigate();
  const duplicatesQuery = useDuplicatesQuery();
  const reviewQuery = useReviewQuery();
  const [activeTab, setActiveTab] = useState<"exact" | "likely">("exact");
  const header = <PageHeader eyebrow="Duplicate Detection" title="Duplicates" description="Compare copies, inspect match evidence, and decide what to keep." />;

  if (duplicatesQuery.isError || !duplicatesQuery.data) {
    return (
      <div className="space-y-6">
        {header}
        {duplicatesQuery.isError ? <ApiUnavailableState contextLabel="Duplicates" /> : (
          <Panel className="p-8 text-center text-sm text-primary-muted">Loading duplicate findings from local API…</Panel>
        )}
      </div>
    );
  }

  const report = duplicatesQuery.data.data;
  const root = duplicatesQuery.data.meta.music_path;
  const review = reviewQuery.data;
  const reviewItems = !reviewQuery.isError && review && sameDuplicateLibrary(duplicatesQuery.data.meta, review.meta)
    ? review.data.items : [];
  const additionalBytes = report.exact_duplicates.reduce((sum, group) => {
    if (!group.files.some((file) => file.path === group.preferred.path)) return sum;
    return sum + group.files.reduce((bytes, file) => bytes + (file.path === group.preferred.path ? 0 : file.file_size_bytes), 0);
  }, 0);
  const groups = activeTab === "exact" ? report.exact_duplicates : report.likely_duplicates;
  const itemType = activeTab === "exact" ? "exact_duplicate_group" : "likely_duplicate_group";

  return (
    <div className="space-y-6">
      {header}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricCard label="Exact sets" value={formatNumber(report.summary.exact_group_count)} icon="copy_all"
          meta={<p className="text-xs text-primary-muted">Identical bytes verified by SHA-256</p>} />
        <MetricCard label="Likely sets" value={formatNumber(report.summary.likely_group_count)} icon="difference"
          meta={<p className="text-xs text-primary-muted">Manual review only</p>} />
        <div className="col-span-2 sm:col-span-1">
          <MetricCard label="Size of additional exact copies" value={formatBytes(additionalBytes)} icon="data_saver_on"
            meta={<p className="text-xs text-primary-muted">Sum of file sizes. Quarantined copies still occupy disk space.</p>} />
        </div>
      </div>
      <div className="flex flex-wrap gap-2 border-b border-white/[0.07] pb-3" role="group" aria-label="Duplicate type">
        {tabs.map(({ kind, label, icon }) => (
          <button key={kind} type="button" aria-pressed={activeTab === kind} onClick={() => setActiveTab(kind)}
            className={"flex items-center gap-2 rounded-[3px] px-3 py-1.5 text-xs font-medium focus-visible:outline-accent "
              + (activeTab === kind ? "bg-white/[0.08] text-accent ring-1 ring-accent/30" : "text-primary-muted hover:text-primary")}>
            <Icon name={icon} className="text-sm" />
            {label}
            <Chip>{kind === "exact" ? report.summary.exact_group_count : report.summary.likely_group_count}</Chip>
          </button>
        ))}
      </div>
      <div className="space-y-4">
        {groups.length ? groups.map((group) => {
          const matches = reviewItems.filter((candidate) => matchesDuplicateFinding(group, candidate, activeTab));
          const item = matches.length === 1 ? matches[0] : undefined;
          return <DuplicateComparison key={activeTab + group.group_id} group={group} kind={activeTab} root={root}
            matchedFinding={!!item} onReview={() => {
              const params = new URLSearchParams({ type: itemType });
              if (item) params.set("item", item.item_id);
              navigate("/review?" + params.toString());
            }} />;
        }) : <EmptyState title={"No " + activeTab + " duplicate groups found"} description="Related files and match evidence appear here when the backend detects a group." />}
      </div>
    </div>
  );
}
