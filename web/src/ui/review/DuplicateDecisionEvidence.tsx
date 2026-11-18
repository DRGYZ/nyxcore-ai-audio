import type { DuplicateEvidence } from "../../lib/duplicate-evidence";
import { Panel } from "../components";
import { DuplicateFiles, DuplicatePreference, PairEvidence, formatDuplicateScore } from "../duplicates/DuplicateComparison";

export function DuplicateDecisionEvidence({ evidence, preview = false }: { evidence: DuplicateEvidence; preview?: boolean }) {
  const exact = evidence.kind === "exact";
  return (
    <Panel className="space-y-3 bg-surface-low/65 p-3.5">
      <h4 className="font-display text-sm font-semibold text-primary">Duplicate evidence</h4>
      {evidence.status !== "matched" ? (
        <p className="text-xs leading-relaxed text-primary-muted" role="status">
          {evidence.status === "loading" ? "Checking duplicate evidence…"
            : "Detailed duplicate evidence is unavailable. Refresh Inbox to check this finding again."}
        </p>
      ) : (
        <>
          <p className="text-xs leading-relaxed text-primary-muted">
            {exact
              ? "SHA-256 verified identical file bytes. The copies carry the same audio and embedded tags."
              : "Manual review only. Reported pairs use heuristic similarity scores. Verify the recordings before deciding which copy to keep."}
          </p>
          {preview && !exact && evidence.group.relationships?.length ? (
            <p className="text-xs text-primary-muted">
              {evidence.group.relationships.length} reported {evidence.group.relationships.length === 1 ? "pair" : "pairs"}.
              {evidence.group.confidence == null ? null : " Lowest pair score " + formatDuplicateScore(evidence.group.confidence) + "."}
            </p>
          ) : null}
          <DuplicatePreference group={evidence.group} kind={evidence.kind} root={evidence.root} showPath />
          {exact ? (
            <p className="text-xs leading-relaxed text-primary-muted">
              The recommendation identifies the keeper. Select each eligible quarantine move explicitly. Quarantine preserves the moved files.
            </p>
          ) : null}
          {!preview ? (
            <details className="min-w-0">
              <summary className="cursor-pointer text-xs font-semibold text-accent focus-visible:outline-accent">
                {exact ? "Compare copies" : "Compare copies and reported pairs"}
              </summary>
              <div className="mt-2.5 space-y-3">
                {!exact ? <PairEvidence group={evidence.group} /> : null}
                <DuplicateFiles group={evidence.group} root={evidence.root} compact />
              </div>
            </details>
          ) : null}
        </>
      )}
    </Panel>
  );
}
