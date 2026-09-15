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
        <div className="rounded-xl border border-dashed border-border-dark bg-background-dark/50 px-4 py-8 text-sm text-slate-500">
          Select a review item to inspect its affected files, triage state, and plan options.
        </div>
      ) : (
        <>
          <Panel className="border-primary/20 bg-background-dark/50 p-4">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="flex size-12 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
                  <span className="material-symbols-outlined">audio_file</span>
                </div>
                <div>
                  <h3 className="font-display text-base font-bold text-slate-200">{item.summary}</h3>
                  <p className="mt-1 text-[10px] font-mono uppercase tracking-[0.18em] text-slate-500">{item.item_type}</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Chip tone={reviewPriorityTone(item.priority_band)}>{item.priority_band}</Chip>
                <Chip tone={reviewStatusTone(item.review_status)}>{reviewStatusLabel(item.review_status)}</Chip>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">
                <span>Priority Score</span>
                <span className="font-mono text-primary">{item.priority_score}</span>
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

          <div className="rounded-xl border border-primary/15 bg-primary/5 px-4 py-3 text-xs leading-5 text-slate-400" role="note">
            <span className="font-bold text-primary">Resolve Until Refresh</span> hides this finding without editing audio.
            If the next library refresh still detects it, NyxCore returns it as <span className="font-bold text-slate-300">Seen</span>.
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Panel className="p-3">
              <p className="text-[10px] text-slate-500">Reclaimable</p>
              <p className="mt-1 text-lg font-bold text-amber-400">{formatBytes(item.reclaimable_bytes ?? 0)}</p>
            </Panel>
            <Panel className="p-3">
              <p className="text-[10px] text-slate-500">Confidence</p>
              <p className="mt-1 text-lg font-bold text-primary">{item.confidence ? `${Math.round(item.confidence * 100)}%` : "n/a"}</p>
            </Panel>
          </div>

          <LabeledValue
            label="Reason Summary"
            value={<div className="break-words rounded-xl bg-background-dark px-4 py-3 text-sm text-slate-300">{item.reason_summary}</div>}
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
                <div className="space-y-2">
                  {displayedAffectedPaths.map((path) => (
                    <PathBlock key={path} value={path} />
                  ))}
                  {remainingAffectedPathCount > 0 ? (
                    <p className="px-1 text-xs text-slate-500">
                      {remainingAffectedPathCount} more paths. Generate a plan to inspect the paginated operations.
                    </p>
                  ) : null}
                </div>
              ) : (
                <div className="rounded-xl bg-background-dark px-4 py-3 text-sm text-slate-500">No affected paths were attached to this finding.</div>
              )
            }
          />
        </>
      )}
    </Drawer>
  );
}
