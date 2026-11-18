import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { matchesDuplicateFinding, sameDuplicateLibrary } from "../../lib/duplicate-evidence";
import { useDuplicatesQuery, useReviewQuery, useStatusQuery } from "../../lib/hooks";
import { Chip, EmptyState, Icon, PageHeader, Panel, formatBytes, formatNumber } from "../components";
import { DuplicateComparison } from "../duplicates/DuplicateComparison";
import { ApiUnavailableState } from "../feedback";
import { ListPages } from "../ListPages";

const GROUPS_PER_PAGE = 24;
const tabs = [
  { kind: "exact", label: "Exact Duplicates", icon: "library_music" },
  { kind: "likely", label: "Likely Duplicates", icon: "difference" },
] as const;

export function DuplicatesPage() {
  const navigate = useNavigate();
  const duplicatesQuery = useDuplicatesQuery();
  const reviewOnly = useStatusQuery().data?.review_only === true;
  const reviewQuery = useReviewQuery();
  const [activeTab, setActiveTab] = useState<"exact" | "likely">("exact");
  const [requestedPage, setRequestedPage] = useState(0);
  const header = (
    <PageHeader
      eyebrow="Recording Comparison"
      title="Duplicates"
      description="Compare matching recordings side by side, inspect tag and bitrate evidence, and confirm which copy belongs in your library."
    />
  );

  if (!duplicatesQuery.data || (duplicatesQuery.isError && !reviewOnly)) {
    return (
      <div className="space-y-6">
        {header}
        {!reviewOnly && (duplicatesQuery.isError ? <ApiUnavailableState contextLabel="Duplicates" /> : (
          <Panel className="p-8 text-center text-sm text-primary-muted">Loading duplicate findings from local API…</Panel>
        ))}
      </div>
    );
  }

  const report = duplicatesQuery.data.data;
  const root = duplicatesQuery.data.meta.music_path;
  const review = reviewQuery.data;
  const reviewItems = (reviewOnly || !reviewQuery.isError) && review && sameDuplicateLibrary(duplicatesQuery.data.meta, review.meta)
    ? review.data.items : [];
  const additionalBytes = report.exact_duplicates.reduce((sum, group) => {
    if (!group.files.some((file) => file.path === group.preferred.path)) return sum;
    return sum + group.files.reduce((bytes, file) => bytes + (file.path === group.preferred.path ? 0 : file.file_size_bytes), 0);
  }, 0);
  const groups = activeTab === "exact" ? report.exact_duplicates : report.likely_duplicates;
  const page = Math.min(requestedPage, Math.max(0, Math.ceil(groups.length / GROUPS_PER_PAGE) - 1));
  const first = page * GROUPS_PER_PAGE;
  const visibleGroups = groups.slice(first, first + GROUPS_PER_PAGE);
  const itemType = activeTab === "exact" ? "exact_duplicate_group" : "likely_duplicate_group";

  return (
    <div className="space-y-6">
      {header}

      <Panel className="overflow-hidden">
        <div className="grid grid-cols-1 divide-y divide-white/[0.07] sm:grid-cols-3 sm:divide-x sm:divide-y-0">
          <div className="p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Exact sets</span>
              <Icon name="verified" className="text-[17px] text-[#7cd4aa]" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{formatNumber(report.summary.exact_group_count)}</p>
            <p className="mt-1 text-xs text-primary-subtle">Identical bytes verified by SHA-256</p>
          </div>
          <div className="p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Likely sets</span>
              <Icon name="difference" className="text-[17px] text-[#f2c86b]" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{formatNumber(report.summary.likely_group_count)}</p>
            <p className="mt-1 text-xs text-primary-subtle">Manual review only</p>
          </div>
          <div className="p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs font-medium text-primary-muted">Size of additional exact copies</span>
              <Icon name="data_saver_on" className="text-[17px] text-accent" />
            </div>
            <p className="mt-2 font-display text-2xl font-semibold text-primary">{formatBytes(additionalBytes)}</p>
            <p className="mt-1 text-xs text-primary-subtle">Sum of file sizes. Quarantined copies still occupy disk space.</p>
          </div>
        </div>
      </Panel>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div
          className="inline-flex flex-wrap gap-1.5 rounded-xl border border-white/[0.08] bg-surface p-1.5"
          role="group"
          aria-label="Duplicate type"
        >
          {tabs.map(({ kind, label, icon }) => {
            const active = activeTab === kind;
            return (
              <button
                key={kind}
                type="button"
                aria-pressed={active}
                onClick={() => {
                  setActiveTab(kind);
                  setRequestedPage(0);
                }}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 font-sans text-xs font-medium transition-colors focus-visible:outline-accent ${
                  active
                    ? "bg-accent/15 text-primary shadow-[inset_0_1px_0_rgba(255,248,235,0.08)]"
                    : "text-primary-muted hover:bg-white/[0.04] hover:text-primary"
                }`}
              >
                <Icon name={icon} className={`text-[16px] ${active ? "text-accent" : "text-primary-subtle"}`} />
                <span>{label}</span>
                <Chip tone={active ? "accent" : "neutral"}>
                  {kind === "exact" ? report.summary.exact_group_count : report.summary.likely_group_count}
                </Chip>
              </button>
            );
          })}
        </div>
      </div>

      <ListPages label="duplicate groups" count={groups.length} page={page} pageSize={GROUPS_PER_PAGE} onPageChange={setRequestedPage} />

      <div className="space-y-5">
        {visibleGroups.length ? visibleGroups.map((group) => {
          const matches = reviewItems.filter((candidate) => matchesDuplicateFinding(group, candidate, activeTab));
          const item = matches.length === 1 ? matches[0] : undefined;
          return (
            <DuplicateComparison
              key={activeTab + group.group_id}
              group={group}
              kind={activeTab}
              root={root}
              matchedFinding={!!item}
              onReview={() => {
                const params = new URLSearchParams({ type: itemType });
                if (item) params.set("item", item.item_id);
                navigate("/review?" + params.toString());
              }}
            />
          );
        }) : (
          <EmptyState
            title={"No " + activeTab + " duplicate groups found"}
            description="Related files and match evidence appear here when the backend detects a group."
          />
        )}
      </div>
    </div>
  );
}
