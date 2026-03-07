import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  useApplyReviewPlanMutation,
  useGenerateReviewPlanMutation,
  useReviewQuery,
  useReviewStateMutation,
} from "../../lib/hooks";
import { mockReviewReport } from "../../lib/mock-data";
import { resolveReportQueryData, toQueryNoticeState } from "../../lib/query-state";
import { reviewPriorityTone, reviewStatusTone } from "../../lib/review-presenter";
import type { ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";
import { useUrlBackedSelection } from "../../lib/url-selection";
import {
  ActionBanner,
  Button,
  Chip,
  EmptyState,
  Modal,
  PageQueryStateNotice,
  PageHeader,
  Panel,
  ProgressBar,
} from "../components";
import { ReviewDetailPanel } from "../review/ReviewDetailPanel";
import { ApplyResultPanel, PlanReportModal } from "../review/PlanReportModal";
import { SplitScreen } from "../shell";

export function ReviewPage() {
  const reviewQuery = useReviewQuery();
  const reviewMutation = useReviewStateMutation();
  const planMutation = useGenerateReviewPlanMutation();
  const applyPlanMutation = useApplyReviewPlanMutation();
  const reviewState = resolveReportQueryData(reviewQuery, mockReviewReport);
  const report = reviewState.data;
  const usingMock = reviewState.usingMock;
  const [priority, setPriority] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [itemType, setItemType] = useState<string>("all");
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const [planReport, setPlanReport] = useState<ActionPlanReport | null>(null);
  const [confirmApply, setConfirmApply] = useState(false);
  const [applyResult, setApplyResult] = useState<ReviewPlanApplyResponse | null>(null);

  const filtered = useMemo(
    () =>
      report.items.filter((item) => {
        if (priority !== "all" && item.priority_band !== priority) return false;
        if (status !== "all" && item.review_status !== status) return false;
        if (itemType !== "all" && item.item_type !== itemType) return false;
        return true;
      }),
    [itemType, priority, report.items, status],
  );

  const { selected, selectById } = useUrlBackedSelection({
    items: report.items,
    fallbackItems: filtered,
    param: "item",
    idKey: "item_id",
  });

  async function handleReviewAction(action: "seen" | "ignored" | "snoozed" | "resolved") {
    if (!selected || usingMock) return;
    try {
      setApplyResult(null);
      await reviewMutation.mutateAsync({ item_ids: [selected.item_id], action, days: action === "snoozed" ? 7 : undefined });
      setBanner({ tone: "success", message: `Updated ${selected.item_id} to ${action}.` });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to update review state." });
    }
  }

  async function handleGeneratePlan() {
    if (!selected || usingMock) return;
    try {
      const response = await planMutation.mutateAsync({ item_ids: [selected.item_id] });
      setApplyResult(null);
      setPlanReport(response.data);
      setBanner({ tone: "success", message: `Generated ${response.data.summary.generated_plan_count} plan(s) for ${selected.item_id}.` });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to generate review plan." });
    }
  }

  async function handleApplyPlan() {
    if (!planReport || usingMock) return;
    try {
      const response = await applyPlanMutation.mutateAsync({ plan_report: planReport });
      setConfirmApply(false);
      setPlanReport(null);
      setApplyResult(response);
      setBanner({
        tone: "success",
        message: `Applied ${response.result_count} plan result(s)${response.batch_id ? ` in history batch ${response.batch_id}` : ""}.`,
      });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to apply plan." });
    }
  }

  const applyCapable = (planReport?.plans ?? []).some((plan) => plan.apply_supported);
  const busy = reviewMutation.isPending || planMutation.isPending || applyPlanMutation.isPending;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Review Inbox"
        description="Review findings, move them through triage state, inspect explicit plans, and hand off applied changes into operation history without leaving the command-center workflow."
      />
      <PageQueryStateNotice
        {...toQueryNoticeState(reviewState)}
        fallbackMessage="Mock fallback is active. Mutation actions are disabled until the live API is available."
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
            <Panel className="flex flex-wrap items-center gap-4 px-6 py-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-[0.24em] text-slate-500">Priority</span>
                <div className="flex flex-wrap gap-2">
                  {["all", "high", "medium", "low"].map((value) => (
                    <button key={value} type="button" onClick={() => setPriority(value)}>
                      <Chip tone={value === "high" ? "danger" : value === "medium" ? "warning" : "neutral"} active={priority === value}>
                        {value}
                      </Chip>
                    </button>
                  ))}
                </div>
              </div>
              <div className="hidden h-4 w-px bg-border-dark lg:block" />
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-[0.24em] text-slate-500">Status</span>
                <div className="flex flex-wrap gap-2">
                  {["all", "new", "seen", "snoozed", "resolved"].map((value) => (
                    <button key={value} type="button" onClick={() => setStatus(value)}>
                      <Chip
                        tone={value === "new" ? "primary" : value === "snoozed" ? "warning" : value === "resolved" ? "success" : "neutral"}
                        active={status === value}
                      >
                        {value}
                      </Chip>
                    </button>
                  ))}
                </div>
              </div>
              <div className="hidden h-4 w-px bg-border-dark lg:block" />
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-[0.24em] text-slate-500">Type</span>
                <select
                  className="rounded-full border border-border-dark bg-background-dark px-3 py-1.5 text-xs font-bold uppercase tracking-[0.18em] text-slate-400 outline-none focus:border-primary"
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
              <div className="ml-auto text-xs font-mono text-slate-500">
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
              <Panel className="overflow-hidden px-4 py-4">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[760px] border-separate border-spacing-y-2 text-left">
                    <thead>
                      <tr className="text-[11px] font-bold uppercase tracking-[0.24em] text-slate-500">
                        <th className="px-4 pb-3">Priority</th>
                        <th className="px-4 pb-3">Item Type</th>
                        <th className="px-4 pb-3">Score</th>
                        <th className="px-4 pb-3">Summary</th>
                        <th className="px-4 pb-3">State</th>
                        <th className="px-4 pb-3 text-right">Inspect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((item) => {
                        const active = selected?.item_id === item.item_id;
                        return (
                          <tr
                            key={item.item_id}
                            onClick={() => selectById(item.item_id)}
                            className={`cursor-pointer transition-all ${active ? "bg-primary/5 ring-1 ring-primary/20" : "border border-border-dark bg-surface-dark hover:border-primary/30"}`}
                          >
                            <td className="rounded-l-2xl px-4 py-4">
                              <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                            </td>
                            <td className="px-4 py-4 text-sm font-medium text-slate-300">{item.item_type}</td>
                            <td className="px-4 py-4">
                              <div className="flex items-center gap-2">
                                <div className="w-16">
                                  <ProgressBar value={item.priority_score} />
                                </div>
                                <span className="font-mono text-sm font-bold text-primary">{item.priority_score}</span>
                              </div>
                            </td>
                            <td className="px-4 py-4">
                              <span className={`block max-w-[28rem] truncate font-mono text-xs ${active ? "text-primary" : "text-slate-400"}`}>{item.summary}</span>
                            </td>
                            <td className="px-4 py-4">
                              <Chip tone={reviewStatusTone(item.review_status)}>{item.review_status}</Chip>
                            </td>
                            <td className="rounded-r-2xl px-4 py-4 text-right">
                              <button type="button" className={`rounded-lg p-1.5 ${active ? "bg-primary/20 text-primary" : "text-slate-500 hover:bg-primary/20 hover:text-primary"}`}>
                                <span className="material-symbols-outlined text-lg">{active ? "chevron_right" : "open_in_new"}</span>
                              </button>
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
        onClose={() => {
          setPlanReport(null);
          setConfirmApply(false);
        }}
        onApply={() => setConfirmApply(true)}
      />

      <Modal
        open={confirmApply}
        title="Confirm Plan Apply"
        subtitle="Only low-risk, apply-capable operations will run."
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
          <p>Applying this plan may rename files, update deterministic metadata, or move duplicate candidates into quarantine based on the selected operations.</p>
          <p>No plan is auto-applied from generation alone, and review-only operations will be skipped.</p>
        </div>
      </Modal>
    </div>
  );
}
