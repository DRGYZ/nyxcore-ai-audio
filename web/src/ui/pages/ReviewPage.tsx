import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  buildSelectedActionPlanReport,
  setActionOperationSelected,
} from "../../lib/action-plan";
import {
  useApplyReviewPlanMutation,
  useCheckConnection,
  useGenerateReviewPlanMutation,
  useReviewQuery,
  useReviewStateMutation,
} from "../../lib/hooks";
import { reviewPriorityTone, reviewStatusLabel, reviewStatusTone } from "../../lib/review-presenter";
import type { ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";
import { useUrlBackedSelection } from "../../lib/url-selection";
import {
  ActionBanner,
  ApiUnavailableState,
  Button,
  Chip,
  EmptyState,
  Icon,
  Modal,
  PageHeader,
  Panel,
  ProgressBar,
} from "../components";
import { ReviewDetailPanel } from "../review/ReviewDetailPanel";
import { ApplyResultPanel, PlanReportModal } from "../review/PlanReportModal";
import { SplitScreen } from "../shell";

const REVIEW_STATUS_FILTERS = [
  { value: "all", label: "all" },
  { value: "new", label: "new" },
  { value: "seen", label: "seen" },
  { value: "snoozed", label: "snoozed" },
  { value: "resolved", label: "resolved until refresh" },
];

export function ReviewPage() {
  const [searchParams] = useSearchParams();
  const requestedItemType = searchParams.get("type");
  const reviewQuery = useReviewQuery();
  const reviewMutation = useReviewStateMutation();
  const planMutation = useGenerateReviewPlanMutation();
  const applyPlanMutation = useApplyReviewPlanMutation();
  const { checkConnection, checking } = useCheckConnection();
  const [priority, setPriority] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [itemType, setItemType] = useState<string>(() => requestedItemType ?? "all");
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const [planReport, setPlanReport] = useState<ActionPlanReport | null>(null);
  const [selectedOperationIds, setSelectedOperationIds] = useState<Set<string>>(new Set());
  const [confirmApply, setConfirmApply] = useState(false);
  const [applyResult, setApplyResult] = useState<ReviewPlanApplyResponse | null>(null);

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
  });

  if (reviewQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Review Findings"
          title="Review Inbox"
          description="Inspect library findings, triage issues, generate explicit action plans, and review proposed changes before applying."
        />
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
          description="Inspect library findings, triage issues, generate explicit action plans, and review proposed changes before applying."
        />
        <Panel className="p-8 text-center text-sm text-slate-400">
          Loading review findings from local API…
        </Panel>
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
      setPlanReport(response.data);
      setSelectedOperationIds(new Set());
      setBanner({ tone: "success", message: `Generated ${response.data.summary.generated_plan_count} plan(s) for ${selected.item_id}.` });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to generate review plan." });
    }
  }

  async function handleApplyPlan() {
    if (!planReport) return;
    try {
      const response = await applyPlanMutation.mutateAsync({
        plan_report: buildSelectedActionPlanReport(planReport, selectedOperationIds),
      });
      setConfirmApply(false);
      setPlanReport(null);
      setSelectedOperationIds(new Set());
      setApplyResult(response);
      setBanner({
        tone: "success",
        message: `Applied ${response.result_count} plan result(s)${response.batch_id ? ` in history batch ${response.batch_id}` : ""}.`,
      });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to apply plan." });
    }
  }

  const applyCapable = selectedOperationIds.size > 0;
  const selectedOperations = selectedOperationIds.size;
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
        description="Inspect library findings, triage issues, generate explicit action plans, and review proposed changes before applying."
        actions={
          <Button tone="secondary" onClick={() => void handleRefreshInbox()} disabled={reviewQuery.isFetching || busy}>
            <Icon name="refresh" className={`text-base ${reviewQuery.isFetching || checking ? "animate-spin" : ""}`} />
            {reviewQuery.isFetching || checking ? "Refreshing…" : "Refresh Inbox"}
          </Button>
        }
      />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}
      {applyResult ? (
        <ActionBanner
          tone="success"
          message={`Apply completed. ${applyResult.result_count} plan result(s) processed.`}
          action={
            applyResult.batch_id ? (
              <Link to={`/history?batch=${encodeURIComponent(applyResult.batch_id)}`}>
                <Button tone="secondary">Inspect History Batch</Button>
              </Link>
            ) : undefined
          }
        />
      ) : null}
      {applyResult ? <ApplyResultPanel result={applyResult} /> : null}

      <SplitScreen
        main={
          <div className="space-y-4">
            <Panel className="flex flex-wrap items-center gap-3.5 px-4 py-2.5">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Priority</span>
                <div className="flex flex-wrap gap-1">
                  <button type="button" onClick={() => setPriority("all")}>
                    <Chip tone="neutral" active={priority === "all"}>
                      All
                    </Chip>
                  </button>
                  <button type="button" onClick={() => setPriority("high")}>
                    <Chip tone="danger" active={priority === "high"}>
                      High ({highCount})
                    </Chip>
                  </button>
                  <button type="button" onClick={() => setPriority("medium")}>
                    <Chip tone="warning" active={priority === "medium"}>
                      Med ({mediumCount})
                    </Chip>
                  </button>
                  <button type="button" onClick={() => setPriority("low")}>
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
                    <button key={value} type="button" onClick={() => setStatus(value)}>
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
                      {filtered.map((item) => {
                        const active = selected?.item_id === item.item_id;
                        return (
                          <tr
                            key={item.item_id}
                            onClick={() => selectById(item.item_id)}
                            className={`cursor-pointer transition-colors ${
                              active
                                ? "border-l-2 border-l-accent bg-surface-high"
                                : "border-l-2 border-l-transparent hover:bg-white/[0.02]"
                            }`}
                          >
                            <td className="px-4 py-3">
                              <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                            </td>
                            <td className="px-4 py-3">
                              <div className="min-w-0 max-w-xl">
                                <div className="flex items-center gap-2">
                                  <span className={`font-sans text-xs ${active ? "text-primary font-medium" : "text-primary/90"}`}>
                                    {item.summary}
                                  </span>
                                  <span className="font-mono text-[9px] text-primary-subtle/80 rounded-[2px] bg-white/[0.03] px-1.5 py-0.5 border border-white/[0.05]">
                                    {item.item_type}
                                  </span>
                                </div>
                                {item.reason_summary ? (
                                  <p className="mt-0.5 truncate font-editorial text-[11px] italic text-primary-subtle">
                                    {item.reason_summary}
                                  </p>
                                ) : null}
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-xs font-semibold text-primary">{item.priority_score.toFixed(0)}</span>
                                <div className="w-12 hidden sm:block">
                                  <ProgressBar value={item.priority_score} />
                                </div>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
                            </td>
                            <td className="px-4 py-3 text-right">
                              <span className={`inline-flex p-0.5 ${active ? "text-accent" : "text-primary-subtle/60"}`}>
                                <Icon name="chevron_right" className="text-base" />
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center justify-between border-t border-white/[0.05] bg-surface-low/40 px-4 py-2 font-mono text-[10px] text-primary-subtle">
                  <span>{filtered.length} visible / {report.items.length} total findings</span>
                  <span>Sorted by Priority Score (Descending)</span>
                </div>
              </Panel>
            )}
          </div>
        }
        side={
          <ReviewDetailPanel
            item={selected}
            usingMock={usingMock}
            busy={busy}
            onGeneratePlan={() => void handleGeneratePlan()}
            onMarkSeen={() => void handleReviewAction("seen")}
            onIgnore={() => void handleReviewAction("ignored")}
            onSnooze={() => void handleReviewAction("snoozed")}
            onResolve={() => void handleReviewAction("resolved")}
          />
        }
      />

      <PlanReportModal
        report={planReport}
        sourceItemId={selected?.item_id}
        applyPending={applyPlanMutation.isPending}
        usingMock={usingMock}
        selectedOperationIds={selectedOperationIds}
        onToggleOperation={(planId, operationId, checked) => {
          if (!planReport) return;
          setSelectedOperationIds((current) =>
            setActionOperationSelected(current, planReport, planId, operationId, checked),
          );
        }}
        onClose={() => {
          setPlanReport(null);
          setSelectedOperationIds(new Set());
          setConfirmApply(false);
        }}
        onApply={() => setConfirmApply(true)}
      />

      <Modal
        open={confirmApply}
        title="Confirm Plan Apply"
        subtitle={`${selectedOperations} selected operation${selectedOperations === 1 ? "" : "s"}`}
        onClose={() => setConfirmApply(false)}
        footer={
          <>
            <Button tone="ghost" onClick={() => setConfirmApply(false)} disabled={applyPlanMutation.isPending}>
              Cancel
            </Button>
            <Button tone="primary" onClick={() => void handleApplyPlan()} disabled={applyPlanMutation.isPending || !applyCapable}>
              {applyPlanMutation.isPending ? "Applying..." : "Apply Plan"}
            </Button>
          </>
        }
      >
        <div className="space-y-4 text-sm text-slate-300">
          <p>The checked operations may rename files, update deterministic metadata, or move duplicate candidates into quarantine.</p>
          <p>Partial selections remain in the Review Inbox until every proposed mutation has been completed.</p>
        </div>
      </Modal>
    </div>
  );
}
