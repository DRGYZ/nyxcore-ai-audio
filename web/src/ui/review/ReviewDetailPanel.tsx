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
      title="Review Details"
      subtitle={item ? `ITEM: ${item.item_id}` : "No item selected"}
      footer={
        <>
          <Button tone="ghost" className="w-full" onClick={onGeneratePlan} disabled={usingMock || busy || !item}>
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
        <div className="border border-border bg-surface-low px-4 py-8 font-mono text-xs text-primary-subtle">
          Select a review item to inspect its affected files, triage state, and plan options.
        </div>
      ) : (
        <>
          <Panel className="border border-border bg-surface-low p-4">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex size-8 items-center justify-center border border-border bg-surface text-accent">
                  <span className="material-symbols-outlined text-base">audio_file</span>
                </div>
                <div>
                  <h3 className="font-display text-sm font-bold uppercase tracking-wider text-primary">{item.summary}</h3>
                  <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-primary-subtle">{item.item_type}</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">
                <span>Priority Score</span>
                <span className="text-primary font-bold">{item.priority_score}</span>
              </div>
              <ProgressBar value={item.priority_score} />
            </div>
          </Panel>

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
              Plan
            </Button>
          </div>

          <div className="border border-border bg-surface-low px-4 py-3 font-mono text-xs text-primary-muted leading-relaxed" role="note">
            <span className="font-bold text-accent">Resolve Until Refresh</span> hides this finding without editing audio.
            If the next library refresh still detects it, NyxCore returns it as <span className="font-bold text-primary">Seen</span>.
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="border border-border bg-surface-low p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Reclaimable</p>
              <p className="mt-1 font-display text-xl font-bold tracking-tight text-emerald-400">{formatBytes(item.reclaimable_bytes ?? 0)}</p>
            </div>
            <div className="border border-border bg-surface-low p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-subtle">Confidence</p>
              <p className="mt-1 font-display text-xl font-bold tracking-tight text-accent">{item.confidence ? `${Math.round(item.confidence * 100)}%` : "n/a"}</p>
            </div>
          </div>

          <LabeledValue
            label="Reason Summary"
            value={<div className="break-words border border-border bg-surface-low px-4 py-3 font-mono text-xs text-primary-muted">{item.reason_summary}</div>}
          />

          {item.preferred_path ? (
            <LabeledValue
              label="Preferred Copy"
              value={<PathBlock value={item.preferred_path} tone="primary" />}
            />
          ) : null}

          <LabeledValue
            label="Affected Paths"
            value={
              affectedPaths.length > 0 ? (
                <div className="space-y-1.5">
                  {displayedAffectedPaths.map((path) => (
                    <PathBlock key={path} value={path} />
                  ))}
                  {remainingAffectedPathCount > 0 ? (
                    <p className="px-1 font-mono text-[10px] text-primary-subtle">
                      {remainingAffectedPathCount} more paths. Generate a plan to inspect the paginated operations.
                    </p>
                  ) : null}
                </div>
              ) : (
                <div className="border border-border bg-surface-low px-4 py-3 font-mono text-xs text-primary-subtle">No affected paths were attached to this finding.</div>
              )
            }
          />
        </>
      )}
    </Drawer>
  );
}
