import { reviewPriorityTone, reviewStatusLabel, reviewStatusTone } from "../../lib/review-presenter";
import type { ReviewItem } from "../../lib/types";
import { Button, Chip, Drawer, LabeledValue, Panel, PathBlock, ProgressBar, formatBytes } from "../components";

const AFFECTED_PATH_PREVIEW_LIMIT = 20;

export function ReviewDetailPanel({
  item,
  usingMock,
  busy,
  onGeneratePlan,
  onMarkSeen,
  onIgnore,
  onSnooze,
  onResolve,
}: {
  item?: ReviewItem;
  usingMock: boolean;
  busy: boolean;
  onGeneratePlan: () => void;
  onMarkSeen: () => void;
  onIgnore: () => void;
  onSnooze: () => void;
  onResolve: () => void;
}) {
  const affectedPaths = item?.affected_paths ?? item?.sample_paths ?? [];
  const displayedAffectedPaths = affectedPaths.slice(0, AFFECTED_PATH_PREVIEW_LIMIT);
  const remainingAffectedPathCount = affectedPaths.length - displayedAffectedPaths.length;

  return (
    <Drawer
      title="Finding Inspector"
      subtitle={item ? `ID: ${item.item_id}` : "No finding selected"}
      footer={
        <>
          <Button tone="secondary" className="w-full" onClick={onGeneratePlan} disabled={usingMock || busy || !item}>
            Generate Plan
          </Button>
          <Button
            tone="primary"
            className="w-full"
            onClick={onResolve}
            disabled={usingMock || busy || !item || item.review_status === "resolved"}
          >
            {item?.review_status === "resolved" ? "Resolved Until Refresh" : "Resolve Until Refresh"}
          </Button>
        </>
      }
    >
      {!item ? (
        <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/50 px-4 py-8 font-sans text-xs text-primary-subtle text-center">
          Select a review finding from the stream to inspect details and plan actions.
        </div>
      ) : (
        <>
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1.5">
                <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
              </div>
              <span className="font-mono text-[10px] text-primary-subtle truncate max-w-[140px]">{item.item_type}</span>
            </div>
            <h3 className="font-display text-base font-bold text-primary leading-snug">{item.summary}</h3>
            {item.reason_summary ? (
              <p className="font-editorial text-xs italic text-primary-muted leading-relaxed">{item.reason_summary}</p>
            ) : null}
          </div>

          <div className="space-y-1.5 rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
            <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">
              <span>Priority Score</span>
              <span className="font-bold text-primary">{item.priority_score} / 100</span>
            </div>
            <ProgressBar value={item.priority_score} />
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Reclaimable</p>
              <p className="mt-1 font-display text-lg font-bold tracking-tight text-emerald-400">{formatBytes(item.reclaimable_bytes ?? 0)}</p>
            </div>
            <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Confidence</p>
              <p className="mt-1 font-display text-lg font-bold tracking-tight text-accent">
                {item.confidence !== undefined && item.confidence !== null ? `${Math.round(item.confidence * 100)}%` : "n/a"}
              </p>
            </div>
          </div>

          <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/40 px-3.5 py-2.5 font-sans text-xs text-primary-muted leading-relaxed" role="note">
            <p className="font-medium text-accent/90 mb-0.5">Resolution Behavior</p>
            <p className="text-[11px] text-primary-subtle">
              Executing <span className="text-primary font-medium">Resolve Until Refresh</span> hides this finding without modifying files. If the next library refresh still detects it, NyxCore returns it as Seen.
            </p>
          </div>

          <div className="space-y-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-primary-subtle">Triage State</p>
            <div className="grid grid-cols-2 gap-2">
              <Button tone="secondary" disabled={usingMock || busy} onClick={onMarkSeen}>
                Mark Seen
              </Button>
              <Button tone="ghost" disabled={usingMock || busy} onClick={onIgnore}>
                Ignore
              </Button>
              <Button tone="ghost" disabled={usingMock || busy} onClick={onSnooze}>
                Snooze 7d
              </Button>
              <Button tone="secondary" disabled={usingMock || busy} onClick={onGeneratePlan}>
                Generate Plan
              </Button>
            </div>
          </div>

          {item.preferred_path ? (
            <LabeledValue
              label="Preferred Copy"
              value={<PathBlock value={item.preferred_path} tone="primary" />}
            />
          ) : null}

          <LabeledValue
            label={`Affected Files (${affectedPaths.length})`}
            value={
              affectedPaths.length > 0 ? (
                <div className="space-y-1.5">
                  {displayedAffectedPaths.map((path) => (
                    <PathBlock key={path} value={path} />
                  ))}
                  {remainingAffectedPathCount > 0 ? (
                    <p className="px-1 font-sans text-[11px] text-primary-subtle">
                      {remainingAffectedPathCount} more paths. Generate a plan to inspect full operations.
                    </p>
                  ) : null}
                </div>
              ) : (
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low px-3 py-2 font-sans text-xs text-primary-subtle">No affected paths attached.</div>
              )
            }
          />

          <p className="pt-2 text-center font-editorial text-xs italic text-primary-subtle/80">
            NyxCore records mutation intent in the local ledger before execution.
          </p>
        </>
      )}
    </Drawer>
  );
}
