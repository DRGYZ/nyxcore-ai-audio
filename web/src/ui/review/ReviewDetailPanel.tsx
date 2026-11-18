import { useState } from "react";
import { reviewPriorityTone, reviewStatusLabel, reviewStatusTone } from "../../lib/review-presenter";
import { duplicateKind, type DuplicateEvidence } from "../../lib/duplicate-evidence";
import type { ReviewItem } from "../../lib/types";
import { Button, Chip, Drawer, Icon, LabeledValue, PathBlock, ProgressBar, formatBytes } from "../components";
import { formatDuplicateScore } from "../duplicates/DuplicateComparison";
import { DuplicateDecisionEvidence } from "./DuplicateDecisionEvidence";
import { MetadataDraftLink, MetadataEditButton } from "../metadata/MetadataEditor";

const AFFECTED_PATH_PREVIEW_LIMIT = 20;

export function ReviewDetailPanel({
  item,
  duplicateEvidence,
  usingMock,
  busy,
  onGeneratePlan,
  onMarkSeen,
  onIgnore,
  onSnooze,
  onResolve,
  onClose,
}: {
  item?: ReviewItem;
  duplicateEvidence?: DuplicateEvidence;
  usingMock: boolean;
  busy: boolean;
  onGeneratePlan: () => void;
  onMarkSeen: () => void;
  onIgnore: () => void;
  onSnooze: () => void;
  onResolve: () => void;
  onClose?: () => void;
}) {
  const affectedPaths = item?.affected_paths ?? item?.sample_paths ?? [];
  const editable = !usingMock && (item?.item_type === "missing_metadata" || item?.item_type === "weak_or_placeholder_metadata");
  const [paging, setPaging] = useState({ item: "", page: 0 });
  const page = editable && paging.item === item?.item_id ? paging.page : 0;
  const start = page * AFFECTED_PATH_PREVIEW_LIMIT;
  const displayedAffectedPaths = affectedPaths.slice(start, start + AFFECTED_PATH_PREVIEW_LIMIT);
  const remainingAffectedPathCount = affectedPaths.length - displayedAffectedPaths.length;
  const kind = duplicateKind(item?.item_type);
  const detailedEvidence = duplicateEvidence?.status === "matched";

  return (
    <Drawer
      title="Finding Inspector"
      subtitle={item ? `ID: ${item.item_id}` : "No finding selected"}
      action={
        onClose ? (
          <button
            type="button"
            aria-label="Close inspector"
            onClick={onClose}
            className="rounded-lg p-1.5 text-primary-muted hover:bg-surface-mid hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <Icon name="close" />
          </button>
        ) : undefined
      }
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
        <div className="rounded-xl border border-white/[0.08] bg-surface-low/60 px-4 py-8 text-center font-sans text-xs text-primary-subtle">
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
              <span className="max-w-[160px] truncate font-mono text-[11px] text-primary-subtle">{item.item_type}</span>
            </div>
            <h3 className="font-display text-base font-semibold leading-snug text-primary">{item.summary}</h3>
            {!kind && item.reason_summary ? (
              <p className="text-xs leading-relaxed text-primary-muted">{item.reason_summary}</p>
            ) : null}
          </div>

          <div className="space-y-2 rounded-xl border border-white/[0.08] bg-surface-low/75 p-3.5">
            <div className="flex justify-between font-sans text-xs text-primary-subtle">
              <span>Priority Score</span>
              <span className="font-mono font-semibold text-primary">{item.priority_score} / 100</span>
            </div>
            <ProgressBar value={item.priority_score} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-white/[0.08] bg-surface-low/75 p-3.5">
              <p className="font-sans text-[11px] font-medium text-primary-subtle">{kind === "exact" ? "Size of other copies" : "Reclaimable"}</p>
              <p className="mt-1 font-display text-base font-semibold text-primary">
                {kind === "likely" || item.reclaimable_bytes == null ? "Unavailable" : formatBytes(item.reclaimable_bytes)}
              </p>
              {kind === "exact" && item.reclaimable_bytes != null ? (
                <p className="mt-1 text-xs text-primary-muted">Quarantine keeps files on disk.</p>
              ) : null}
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-surface-low/75 p-3.5">
              <p className="font-sans text-[11px] font-medium text-primary-subtle">
                {kind === "exact" ? "Match type" : kind === "likely" ? "Lowest reported pair score" : "Confidence"}
              </p>
              <p className="mt-1 font-display text-base font-semibold tracking-tight text-accent">
                {kind === "exact" ? "Byte identity" : item.confidence == null ? "Unavailable"
                  : kind === "likely" ? formatDuplicateScore(item.confidence) : Math.round(item.confidence * 100) + "%"}
              </p>
              {kind === "likely" ? <p className="mt-1 text-xs text-primary-muted">Heuristic similarity</p> : null}
            </div>
          </div>

          {duplicateEvidence ? <DuplicateDecisionEvidence key={item.item_id} evidence={duplicateEvidence} /> : null}

          <div className="rounded-xl border border-white/[0.08] bg-surface-low/60 px-3.5 py-3 font-sans text-xs leading-relaxed text-primary-muted" role="note">
            <p className="mb-0.5 font-medium text-accent">Resolution Behavior</p>
            <p className="text-[11px] text-primary-subtle">
              Executing <span className="font-medium text-primary">Resolve Until Refresh</span> hides this finding without modifying files. If the next library refresh still detects it, NyxCore returns it as Seen.
            </p>
          </div>

          <div className="space-y-2.5">
            <p className="font-sans text-xs font-medium text-primary-subtle">Triage State</p>
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

          {item.preferred_path && !detailedEvidence ? (
            <LabeledValue
              label="Preferred Copy"
              value={<PathBlock value={item.preferred_path} tone="primary" />}
            />
          ) : null}

          {!detailedEvidence ? (
            <LabeledValue
              label={`Affected Files (${affectedPaths.length})`}
              value={
                affectedPaths.length > 0 ? (
                  <div className="space-y-2">
                    {editable && <MetadataDraftLink />}
                    {displayedAffectedPaths.map((path) => (
                      <div key={path} className="space-y-1.5">
                        <PathBlock value={path} />
                        {editable && <MetadataEditButton path={path} />}
                      </div>
                    ))}
                    {editable && affectedPaths.length > AFFECTED_PATH_PREVIEW_LIMIT ? (
                      <nav aria-label="Metadata finding files" className="flex flex-wrap items-center gap-2 text-xs text-primary-muted">
                        <Button type="button" disabled={page === 0} onClick={() => setPaging({ item: item.item_id, page: page - 1 })}>Previous</Button>
                        <span>{start + 1}–{start + displayedAffectedPaths.length} of {affectedPaths.length}</span>
                        <Button type="button" disabled={start + displayedAffectedPaths.length >= affectedPaths.length}
                          onClick={() => setPaging({ item: item.item_id, page: page + 1 })}>Next</Button>
                      </nav>
                    ) : remainingAffectedPathCount > 0 ? (
                      <p className="px-1 font-sans text-[11px] text-primary-subtle">
                        {remainingAffectedPathCount} more paths. Generate a plan to inspect full operations.
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <div className="rounded-lg border border-white/[0.08] bg-surface-low px-3 py-2 font-sans text-xs text-primary-subtle">
                    No affected paths attached.
                  </div>
                )
              }
            />
          ) : null}

          <p className="pt-1 text-center font-sans text-[11px] text-primary-subtle">
            NyxCore records mutation intent in the local ledger before execution.
          </p>
        </>
      )}
    </Drawer>
  );
}
