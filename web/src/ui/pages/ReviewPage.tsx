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

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Review Findings"
        title="Review Inbox"
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
            <Panel className="flex flex-wrap items-center gap-4 px-5 py-3">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Priority</span>
                <div className="flex flex-wrap gap-1.5">
                  {["all", "high", "medium", "low"].map((value) => (
                    <button key={value} type="button" onClick={() => setPriority(value)}>
                      <Chip tone={value === "high" ? "danger" : value === "medium" ? "warning" : "neutral"} active={priority === value}>
                        {value}
                      </Chip>
                    </button>
                  ))}
                </div>
              </div>
              <div className="hidden h-4 w-px bg-border lg:block" />
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Status</span>
                <div className="flex flex-wrap gap-1.5">
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
              <div className="hidden h-4 w-px bg-border lg:block" />
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Type</span>
                <select
                  className="border border-border bg-surface px-2.5 py-1 font-mono text-xs uppercase tracking-wider text-primary outline-none focus:border-accent"
                  value={itemType}
                  onChange={(event) => setItemType(event.target.value)}
                >
                  <option value="all">All</option>
                  {[...new Set(report.items.map((item) => item.item_type))].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div className="ml-auto font-mono text-[10px] text-primary-subtle">
                {filtered.length} visible / {report.items.length} total
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
                  <table className="w-full min-w-[760px] text-left">
                    <thead className="border-b border-border bg-surface-low">
                      <tr className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">
                        <th className="px-4 py-3">Priority</th>
                        <th className="px-4 py-3">Item Type</th>
                        <th className="px-4 py-3">Score</th>
                        <th className="px-4 py-3">Summary</th>
                        <th className="px-4 py-3">State</th>
                        <th className="px-4 py-3 text-right">Inspect</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {filtered.map((item) => {
                        const active = selected?.item_id === item.item_id;
                        return (
                          <tr
                            key={item.item_id}
                            onClick={() => selectById(item.item_id)}
                            className={`cursor-pointer transition-colors ${
                              active ? "border-l-2 border-l-accent bg-accent/5" : "hover:bg-surface-low"
                            }`}
                          >
                            <td className="px-4 py-3.5">
                              <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                            </td>
                            <td className="px-4 py-3.5 font-mono text-xs text-primary-muted">{item.item_type}</td>
                            <td className="px-4 py-3.5">
                              <div className="flex items-center gap-2">
                                <div className="w-16">
                                  <ProgressBar value={item.priority_score} />
                                </div>
                                <span className="font-mono text-xs font-bold text-primary">{item.priority_score}</span>
                              </div>
                            </td>
                            <td className="px-4 py-3.5">
                              <span className={`block max-w-[28rem] truncate font-mono text-xs ${active ? "text-primary font-semibold" : "text-primary-muted"}`}>{item.summary}</span>
                            </td>
                            <td className="px-4 py-3.5">
                              <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
                            </td>
                            <td className="px-4 py-3.5 text-right">
                              <span className={`inline-flex p-1 ${active ? "text-accent" : "text-primary-subtle"}`}>
                                <Icon name="chevron_right" className="text-base" />
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
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
