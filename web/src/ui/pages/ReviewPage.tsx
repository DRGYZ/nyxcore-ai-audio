import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  buildSelectedActionPlanReport,
  setActionOperationSelected,
} from "../../lib/action-plan";
import { duplicateKind, findDuplicateGroup, planMatchesDuplicateEvidence, sameDuplicateLibrary, type DuplicateEvidence, type DuplicateKind } from "../../lib/duplicate-evidence";
import {
  useApplyReviewPlanMutation,
  useCheckConnection,
  useDuplicatesQuery,
  useGenerateReviewPlanMutation,
  useReviewQuery,
  useReviewStateMutation,
  useStatusQuery,
} from "../../lib/hooks";
import { reviewPriorityTone, reviewStatusLabel, reviewStatusTone } from "../../lib/review-presenter";
import type { ActionPlanReport, ApiMeta, ReviewItem, ReviewPlanApplyResponse } from "../../lib/types";
import { useUrlBackedSelection } from "../../lib/url-selection";
import {
  ActionBanner,
  ApiUnavailableState,
  Button,
  Chip,
  EmptyState,
  Icon,
  PageHeader,
  Panel,
  ProgressBar,
} from "../components";
import { ReviewDetailPanel } from "../review/ReviewDetailPanel";
import { ApplyResultPanel, PlanReportModal } from "../review/PlanReportModal";
import { ListPages } from "../ListPages";
import { SplitScreen } from "../shell";

const REVIEW_STATUS_FILTERS = [
  { value: "all", label: "all" },
  { value: "new", label: "new" },
  { value: "seen", label: "seen" },
  { value: "snoozed", label: "snoozed" },
  { value: "resolved", label: "resolved until refresh" },
];

const FINDINGS_PER_PAGE = 50;

type PlanPreview = {
  report: ActionPlanReport;
  root?: string;
  generation?: string | null;
  duplicate?: { kind: DuplicateKind; itemId: string; meta?: ApiMeta };
};

export function ReviewPage() {
  const [searchParams] = useSearchParams();
  const requestedItemType = searchParams.get("type");
  const reviewQuery = useReviewQuery();
  const reviewOnly = useStatusQuery().data?.review_only === true;
  const description = "Inspect library findings, save triage decisions, and preview proposed file changes.";
  const reviewMutation = useReviewStateMutation();
  const planMutation = useGenerateReviewPlanMutation();
  const applyPlanMutation = useApplyReviewPlanMutation();
  const { checkConnection, checking } = useCheckConnection();
  const [priority, setPriority] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [itemType, setItemType] = useState<string>(() => requestedItemType ?? "all");
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const [plan, setPlan] = useState<PlanPreview | null>(null);
  const [selectedOperationIds, setSelectedOperationIds] = useState<Set<string>>(new Set());
  const [confirmApply, setConfirmApply] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<{ result: ReviewPlanApplyResponse; selection: ActionPlanReport; root?: string } | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(
    () => searchParams.has("item") && window.innerWidth < 1280,
  );

  const planReport = plan?.report ?? null;
  const selection = useMemo(
    () => planReport ? buildSelectedActionPlanReport(planReport, selectedOperationIds) : null,
    [planReport, selectedOperationIds],
  );
  useEffect(() => {
    if (applyResult) resultsRef.current?.focus();
  }, [applyResult]);

  useEffect(() => {
    if (requestedItemType) setItemType(requestedItemType);
  }, [requestedItemType]);

  const usingMock = false;
  const report = reviewQuery.data?.data;
  const items = useMemo(() => report?.items ?? [], [report?.items]);

  const filtered = useMemo(
    () =>
      items.filter((item) => {
        if (priority !== "all" && item.priority_band !== priority) return false;
        if (status !== "all" && item.review_status !== status) return false;
        if (itemType !== "all" && item.item_type !== itemType) return false;
        return true;
      }),
    [itemType, priority, items, status],
  );

  const { selected, selectById } = useUrlBackedSelection({
    items,
    fallbackItems: filtered,
    param: "item",
    idKey: "item_id",
    syncUrl: reviewQuery.isSuccess && !reviewQuery.isFetching,
  });
  const selectedIndex = filtered.findIndex((item) => item.item_id === selected?.item_id);
  const page = Math.max(0, Math.floor(selectedIndex / FINDINGS_PER_PAGE));
  const first = page * FINDINGS_PER_PAGE;
  const visibleFindings = filtered.slice(first, first + FINDINGS_PER_PAGE);

  const reviewMeta = reviewQuery.data?.meta;
  useEffect(() => {
    if (plan?.generation && reviewMeta?.analysis_generation && plan.generation !== reviewMeta.analysis_generation) {
      setPlan(null);
      setConfirmApply(false);
      setSelectedOperationIds(new Set());
      setBanner({ tone: "info", message: "The library was refreshed. Generate a new preview from the latest findings." });
    }
  }, [plan?.generation, reviewMeta?.analysis_generation]);
  const selectedKind = duplicateKind(selected?.item_type);
  const duplicateQuery = useDuplicatesQuery((reviewOnly || !reviewQuery.isError) && Boolean(selectedKind || plan?.duplicate));

  function evidenceFor(item: ReviewItem | undefined, kind: DuplicateKind): DuplicateEvidence {
    if (!duplicateQuery.isError && (
      duplicateQuery.isPending || (!reviewOnly && (duplicateQuery.isFetching || reviewQuery.isFetching))
    )) {
      return { kind, status: "loading" };
    }
    const duplicates = duplicateQuery.data;
    if (item && (reviewOnly || !duplicateQuery.isError) && duplicates && sameDuplicateLibrary(reviewMeta, duplicates.meta)) {
      const group = findDuplicateGroup(kind, item, duplicates.data);
      if (group) return { kind, status: "matched", group, root: duplicates.meta.music_path };
    }
    return { kind, status: "unavailable" };
  }

  const inspectorEvidence = selectedKind ? evidenceFor(selected, selectedKind) : undefined;
  const planOrigin = plan?.duplicate;
  let previewEvidence: DuplicateEvidence | undefined;
  if (plan && planOrigin) {
    const { kind, itemId, meta } = planOrigin;
    const originItem = items.find((item) => item.item_id === itemId);
    previewEvidence = sameDuplicateLibrary(reviewMeta, meta)
      ? evidenceFor(originItem, kind) : { kind, status: "unavailable" };
    if (previewEvidence.status === "matched" && !planMatchesDuplicateEvidence(plan.report, itemId, previewEvidence.group, kind)) {
      previewEvidence = { kind, status: "unavailable" };
    }
  }

  function handleSelectFinding(itemId: string) {
    selectById(itemId);
    if (typeof window !== "undefined" && window.innerWidth < 1280) {
      setMobileInspectorOpen(true);
    }
  }

  function handleCloseMobileInspector() {
    setMobileInspectorOpen(false);
    if (selected) {
      const trigger = document.getElementById(`finding-row-trigger-${selected.item_id}`);
      trigger?.focus();
    }
  }

  useEffect(() => {
    if (!mobileInspectorOpen || planReport) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        handleCloseMobileInspector();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mobileInspectorOpen, planReport, selected]);

  const outcome = applyResult ? (
    <div ref={resultsRef} tabIndex={-1} aria-label="Apply results">
      <ApplyResultPanel {...applyResult} />
    </div>
  ) : null;

  if (reviewQuery.isError && !reviewOnly) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Review Findings"
          title="Review Inbox"
          description={description}
        />
        {outcome}
        <ApiUnavailableState contextLabel="Review Inbox" />
      </div>
    );
  }

  if (reviewQuery.isLoading || !report) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Review Findings"
          title="Review Inbox"
          description={description}
        />
        {outcome}
        {!reviewOnly ? (
          <Panel className="p-8 text-center text-sm text-slate-400">Loading review findings from local API…</Panel>
        ) : null}
      </div>
    );
  }

  async function handleReviewAction(action: "seen" | "ignored" | "snoozed" | "resolved") {
    if (!selected) return;
    try {
      setApplyResult(null);
      await reviewMutation.mutateAsync({ item_ids: [selected.item_id], action, days: action === "snoozed" ? 7 : undefined });
      setBanner({
        tone: "success",
        message: action === "resolved"
          ? `Resolved ${selected.item_id} until the next library refresh. If the finding remains, it will return as Seen; no audio file was changed.`
          : `Updated ${selected.item_id} to ${action}.`,
      });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to update review state." });
    }
  }

  async function handleRefreshInbox() {
    setBanner(null);
    try {
      await checkConnection();
      setBanner({ tone: "success", message: "Inbox refreshed from local API." });
    } catch {
      setBanner({ tone: "error", message: "Inbox refresh failed. Local API may be offline." });
    }
  }

  async function handleGeneratePlan() {
    if (!selected) return;
    try {
      const response = await planMutation.mutateAsync({ item_ids: [selected.item_id] });
      setApplyResult(null);
      setBanner(null);
      setPlan({
        report: response.data,
        root: response.meta?.music_path ?? reviewMeta?.music_path,
        generation: response.meta?.analysis_generation,
        duplicate: selectedKind ? { kind: selectedKind, itemId: selected.item_id, meta: response.meta } : undefined,
      });
      setSelectedOperationIds(new Set());
      setApplyError(null);
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to generate review plan." });
    }
  }

  async function handleApplyPlan() {
    if (reviewOnly || !confirmApply || !selection?.plans.length || applyPlanMutation.isPending) return;
    setApplyError(null);
    try {
      const response = await applyPlanMutation.mutateAsync({
        plan_report: selection,
      });
      setConfirmApply(false);
      setPlan(null);
      setSelectedOperationIds(new Set());
      setMobileInspectorOpen(false);
      setBanner(null);
      setApplyResult({ result: response, selection, root: plan?.root });
    } catch (error) {
      setApplyError(error instanceof Error ? error.message : "Unable to confirm the outcome. Check History before trying again.");
    }
  }

  const busy = reviewMutation.isPending || planMutation.isPending || applyPlanMutation.isPending || checking;

  const highCount = items.filter((item) => item.priority_band === "high").length;
  const mediumCount = items.filter((item) => item.priority_band === "medium").length;
  const lowCount = items.filter((item) => item.priority_band === "low").length;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={`AUDIT STAGE 01 — ${report.items.length} FINDINGS RECORDED`}
        title={
          <span>
            Review Inbox <span className="font-editorial text-xl font-normal italic text-primary-subtle">precision triage</span>
          </span>
        }
        description={description}
        actions={!reviewOnly ? (
          <Button tone="secondary" onClick={() => void handleRefreshInbox()} disabled={reviewQuery.isFetching || busy}>
            <Icon name="refresh" className={`text-base ${reviewQuery.isFetching || checking ? "animate-spin" : ""}`} />
            {reviewQuery.isFetching || checking ? "Refreshing…" : "Refresh Inbox"}
          </Button>
        ) : undefined}
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}
      {outcome}

      <SplitScreen
        main={
          <div className="space-y-4">
            <Panel className="flex flex-wrap items-center gap-3.5 px-4 py-2.5">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Priority</span>
                <div className="flex flex-wrap gap-1">
                  <button type="button" aria-pressed={priority === "all"} onClick={() => setPriority("all")}>
                    <Chip tone="neutral" active={priority === "all"}>
                      All
                    </Chip>
                  </button>
                  <button type="button" aria-pressed={priority === "high"} onClick={() => setPriority("high")}>
                    <Chip tone="danger" active={priority === "high"}>
                      High ({highCount})
                    </Chip>
                  </button>
                  <button type="button" aria-pressed={priority === "medium"} onClick={() => setPriority("medium")}>
                    <Chip tone="warning" active={priority === "medium"}>
                      Med ({mediumCount})
                    </Chip>
                  </button>
                  <button type="button" aria-pressed={priority === "low"} onClick={() => setPriority("low")}>
                    <Chip tone="neutral" active={priority === "low"}>
                      Low ({lowCount})
                    </Chip>
                  </button>
                </div>
              </div>
              <div className="hidden h-3.5 w-px bg-white/[0.08] lg:block" />
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Status</span>
                <div className="flex flex-wrap gap-1">
                  {REVIEW_STATUS_FILTERS.map(({ value, label }) => (
                    <button key={value} type="button" aria-pressed={status === value} onClick={() => setStatus(value)}>
                      <Chip
                        tone={value === "new" ? "primary" : value === "snoozed" ? "warning" : value === "resolved" ? "success" : "neutral"}
                        active={status === value}
                      >
                        {label}
                      </Chip>
                    </button>
                  ))}
                </div>
              </div>
              <div className="hidden h-3.5 w-px bg-white/[0.08] lg:block" />
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Type</span>
                <select
                  aria-label="Filter findings by type"
                  className="rounded-[3px] border border-white/[0.08] bg-surface px-2 py-1 font-sans text-xs text-primary outline-none focus:border-accent/50"
                  value={itemType}
                  onChange={(event) => setItemType(event.target.value)}
                >
                  <option value="all">All Types</option>
                  {[...new Set(report.items.map((item) => item.item_type))].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div className="ml-auto font-mono text-[10px] text-primary-subtle">
                {filtered.length} / {report.items.length} findings
              </div>
            </Panel>

            <ListPages
              label="findings"
              count={filtered.length}
              page={page}
              pageSize={FINDINGS_PER_PAGE}
              onPageChange={(nextPage) => selectById(filtered[nextPage * FINDINGS_PER_PAGE].item_id)}
            />
            {filtered.length === 0 ? (
              <EmptyState
                title="No review items match the current filters"
                description="Try widening priority, status, or item type filters to bring items back into scope."
                action={
                  <Button
                    tone="secondary"
                    onClick={() => {
                      setPriority("all");
                      setStatus("all");
                      setItemType("all");
                    }}
                  >
                    Reset Filters
                  </Button>
                }
              />
            ) : (
              <Panel className="overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[720px] text-left">
                    <thead className="border-b border-white/[0.07] bg-surface-low/80">
                      <tr className="font-sans text-[11px] font-medium text-primary-subtle">
                        <th className="px-4 py-2.5 w-24">Priority</th>
                        <th className="px-4 py-2.5">Finding Descriptor</th>
                        <th className="px-4 py-2.5 w-24">Score</th>
                        <th className="px-4 py-2.5 w-28">State</th>
                        <th className="px-4 py-2.5 text-right w-12" />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {visibleFindings.map((item) => {
                        const active = selected?.item_id === item.item_id;
                        const kind = duplicateKind(item.item_type);
                        const reason = kind === "exact" ? "SHA-256 byte match"
                          : kind === "likely" ? "Heuristic match · Manual review" : item.reason_summary;
                        return (
                          <tr
                            key={item.item_id}
                            className={`transition-colors ${
                              active
                                ? "border-l-2 border-l-accent bg-surface-high"
                                : "border-l-2 border-l-transparent hover:bg-white/[0.02]"
                            }`}
                          >
                            <td className="px-4 py-3">
                              <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                            </td>
                            <td className="px-4 py-3">
                              <button
                                type="button"
                                id={`finding-row-trigger-${item.item_id}`}
                                onClick={() => handleSelectFinding(item.item_id)}
                                aria-label={`Inspect finding: ${item.summary}, Priority ${item.priority_band}, Score ${item.priority_score.toFixed(0)}`}
                                className="group flex w-full flex-col text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent rounded-[2px]"
                              >
                                <div className="flex items-center gap-2">
                                  <span className={`font-sans text-xs transition-colors group-hover:text-accent ${active ? "font-medium text-primary" : "text-primary/90"}`}>
                                    {item.summary}
                                  </span>
                                  <span className="rounded-[2px] border border-white/[0.05] bg-white/[0.03] px-1.5 py-0.5 font-mono text-[9px] text-primary-subtle/80">
                                    {item.item_type}
                                  </span>
                                </div>
                                {reason ? (
                                  <p className="mt-0.5 truncate font-editorial text-[11px] italic text-primary-subtle">
                                    {reason}
                                  </p>
                                ) : null}
                              </button>
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-xs font-semibold text-primary">{item.priority_score.toFixed(0)}</span>
                                <div className="hidden w-12 sm:block">
                                  <ProgressBar value={item.priority_score} />
                                </div>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
                            </td>
                            <td className="px-4 py-3 text-right">
                              <button
                                type="button"
                                tabIndex={-1}
                                aria-label={`Open details for ${item.summary}`}
                                onClick={() => handleSelectFinding(item.item_id)}
                                className={`inline-flex p-0.5 rounded-[2px] transition-colors ${active ? "text-accent" : "text-primary-subtle/60 hover:text-primary"}`}
                              >
                                <Icon name="chevron_right" className="text-base" />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center justify-between border-t border-white/[0.05] bg-surface-low/40 px-4 py-2 font-mono text-[10px] text-primary-subtle">
                  <span>{report.items.length} total findings</span>
                  <span>Sorted by Priority Score (Descending)</span>
                </div>
              </Panel>
            )}
          </div>
        }
        side={
          !mobileInspectorOpen ? (
            <div className="hidden xl:block">
              <ReviewDetailPanel
                item={selected}
                duplicateEvidence={inspectorEvidence}
                usingMock={usingMock}
                busy={busy}
                onGeneratePlan={() => void handleGeneratePlan()}
                onMarkSeen={() => void handleReviewAction("seen")}
                onIgnore={() => void handleReviewAction("ignored")}
                onSnooze={() => void handleReviewAction("snoozed")}
                onResolve={() => void handleReviewAction("resolved")}
              />
            </div>
          ) : null
        }
      />

      {/* Responsive Inspector Overlay for < xl (tablet & mobile) */}
      {mobileInspectorOpen && selected ? (
        <div
          className="fixed inset-0 z-50 flex justify-end bg-black/80 backdrop-blur-sm xl:hidden"
          role="dialog"
          aria-modal="true"
          aria-label={`Finding Inspector: ${selected.summary}`}
          onMouseDown={handleCloseMobileInspector}
        >
          <div
            className="flex h-full w-full max-w-lg flex-col border-l border-white/[0.1] bg-surface shadow-2xl overflow-hidden"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/[0.07] bg-surface-low/80 px-4 py-3">
              <button
                type="button"
                onClick={handleCloseMobileInspector}
                className="flex items-center gap-1.5 rounded-[2px] px-2 py-1 font-sans text-xs text-primary-muted transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
                aria-label="Back to finding list"
              >
                <Icon name="arrow_back" className="text-base" />
                <span>Back to Findings</span>
              </button>
              <Chip tone={reviewPriorityTone(selected.priority_band)}>{selected.priority_band}</Chip>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <ReviewDetailPanel
                item={selected}
                duplicateEvidence={inspectorEvidence}
                usingMock={usingMock}
                busy={busy}
                onGeneratePlan={() => void handleGeneratePlan()}
                onMarkSeen={() => void handleReviewAction("seen")}
                onIgnore={() => void handleReviewAction("ignored")}
                onSnooze={() => void handleReviewAction("snoozed")}
                onResolve={() => void handleReviewAction("resolved")}
                onClose={handleCloseMobileInspector}
              />
            </div>
          </div>
        </div>
      ) : null}

      <PlanReportModal
        report={planReport}
        duplicateEvidence={previewEvidence}
        selection={selection}
        root={plan?.root}
        confirming={confirmApply}
        reviewOnly={reviewOnly}
        applyPending={applyPlanMutation.isPending}
        applyError={applyError}
        selectedOperationIds={selectedOperationIds}
        onToggleOperation={(planId, operationId, checked) => {
          if (!planReport) return;
          setSelectedOperationIds((current) =>
            setActionOperationSelected(current, planReport, planId, operationId, checked),
          );
        }}
        onClose={() => {
          setPlan(null);
          setSelectedOperationIds(new Set());
        }}
        onContinue={() => {
          setApplyError(null);
          setConfirmApply(true);
        }}
        onCancel={() => setConfirmApply(false)}
        onApply={() => void handleApplyPlan()}
      />
    </div>
  );
}
