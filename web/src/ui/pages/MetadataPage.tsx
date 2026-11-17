import { useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { applyMetadataPlan, previewMetadataPlan } from "../../lib/api";
import { pathFilename } from "../../lib/library-path";
import { FIELD_LABELS, METADATA_FIELDS, tagValidation } from "../../lib/metadata";
import type { MetadataPlan, MetadataProposal, PlanInput } from "../../lib/metadata";
import type { ReviewPlanApplyResponse } from "../../lib/types";
import { ActionBanner, Button, Chip, PageHeader, Panel, PathBlock } from "../components";
import { MetadataEditButton, TAG_INPUT, useMetadataDrafts, useSaveMetadataDrafts } from "../metadata/MetadataEditor";

type Preview = { input: PlanInput; plan: MetadataPlan; revision: string };
type Outcome = { response: ReviewPlanApplyResponse; cleanupError?: string };

export function MetadataPage() {
  const cache = useQueryClient();
  const drafts = useMetadataDrafts();
  const save = useSaveMetadataDrafts();
  const generate = useMutation({ mutationFn: previewMetadataPlan });
  const apply = useMutation({ mutationFn: applyMetadataPlan });
  const [deselected, setDeselected] = useState<Set<string>>(new Set());
  const [batchField, setBatchField] = useState<"artist" | "album">("artist");
  const [batchValue, setBatchValue] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const data = drafts.data;
  const entries = data?.entries ?? [];
  const selected = entries.filter((entry) => !deselected.has(entry.path));
  const stale = selected.some((entry) => entry.generation !== data?.generation);
  const previewStale = !!preview && (preview.revision !== data?.revision ||
    preview.input.entries.some((entry) => entry.generation !== data?.generation));
  const batchError = batchValue ? tagValidation(batchValue) : null;
  const busy = save.isPending || generate.isPending || apply.isPending;

  function clearPreview() {
    setPreview(null);
    setConfirmed(false);
    generate.reset();
    apply.reset();
  }

  async function replaceDrafts(next: MetadataProposal[]) {
    if (!data) return;
    clearPreview();
    try {
      await save.mutateAsync({ revision: data.revision, entries: next });
    } catch {
      // The draft error is shown next to the controls.
    }
  }

  function updateSelected(event: FormEvent) {
    event.preventDefault();
    if (batchError || !batchValue.trim() || !selected.length || stale) return;
    return replaceDrafts(entries.map((entry) => deselected.has(entry.path) ? entry : {
      ...entry, values: { ...entry.values, [batchField]: batchValue.trim() },
    }));
  }

  async function generatePreview() {
    if (!data) return;
    clearPreview();
    setOutcome(null);
    const input = { entries: selected };
    try {
      setPreview({ input, plan: await generate.mutateAsync(input), revision: data.revision });
    } catch {
      void drafts.refetch();
    }
  }

  async function applyPreview() {
    if (!preview || !confirmed || preview.plan.review_only || previewStale) return;
    try {
      const response = await apply.mutateAsync({ ...preview.input, preview: preview.plan.preview, confirm: true });
      const successful = new Set(response.results.flatMap((result) => result.operation_results)
        .filter((result) => result.status === "ok").map((result) => result.path));
      setOutcome({ response });
      clearPreview();
      if (data && successful.size) {
        try {
          await save.mutateAsync({ revision: data.revision, entries: entries.filter((entry) => !successful.has(entry.path)) });
        } catch {
          setOutcome({ response, cleanupError: "Some draft entries could not be removed. Check the results before opening those drafts again." });
        }
      }
      void Promise.all(["health", "review", "duplicates", "history", "status", "search", "metadata"].map(
        (key) => cache.invalidateQueries({ queryKey: [key] }),
      ));
    } catch {
      void drafts.refetch();
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Library Repair" title="Staged metadata"
        description="Review a batch of replacements before writing tags. Each selected file counts as one operation." />
      {drafts.error && <ActionBanner tone="error" message={drafts.error.message} />}
      {drafts.isLoading && <p role="status" className="text-sm text-primary-muted">Loading workspace drafts…</p>}
      {data && entries.length === 0 && <Panel className="space-y-3 p-5">
        <p className="text-sm text-primary">No metadata edits staged.</p>
        <p className="text-xs text-primary-muted">Open a track from Library Health, Review Inbox, or Archive Search.</p>
        <Link className="text-xs text-accent underline" to="/health">Open Library Health</Link>
      </Panel>}
      {data && entries.length > 0 && (
        <Panel className="space-y-4 p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="font-display text-lg font-semibold text-primary">Files to review</h2>
            <p className="font-mono text-xs text-primary-muted">{selected.length} selected · {entries.length} of 50 staged</p>
          </div>
          <div className="divide-y divide-white/10 border-y border-white/10">
            {entries.map((entry) => (
              <article key={entry.path} className="grid min-w-0 gap-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto]">
                <div className="min-w-0 space-y-1.5">
                  <label className="flex items-start gap-2 text-sm text-primary">
                    <input className="mt-1 accent-accent" type="checkbox" checked={!deselected.has(entry.path)} disabled={busy}
                      onChange={(event) => {
                        const checked = event.target.checked;
                        clearPreview();
                        setDeselected((previous) => {
                          const next = new Set(previous);
                          if (checked) next.delete(entry.path);
                          else next.add(entry.path);
                          return next;
                        });
                      }} />
                    <span className="[overflow-wrap:anywhere]">{pathFilename(entry.path)}</span>
                  </label>
                  <PathBlock value={entry.path} />
                  <dl className="space-y-0.5 text-xs [overflow-wrap:anywhere]">
                    {METADATA_FIELDS.filter((field) => entry.values[field] !== undefined).map((field) => (
                      <div className="flex gap-2" key={field}>
                        <dt className="text-primary-subtle">{FIELD_LABELS[field]}</dt><dd className="text-primary">{entry.values[field]}</dd>
                      </div>
                    ))}
                  </dl>
                  {entry.generation !== data.generation && <Chip tone="warning">Stale · reopen and stage again</Chip>}
                </div>
                <div className="flex items-start gap-2">
                  <MetadataEditButton path={entry.path} />
                  <Button type="button" disabled={busy} onClick={() => void replaceDrafts(entries.filter((item) => item.path !== entry.path))}
                    aria-label={`Remove edit for ${pathFilename(entry.path)}`}>Remove</Button>
                </div>
              </article>
            ))}
          </div>
          <form className="flex flex-col gap-2 sm:flex-row sm:items-end" onSubmit={updateSelected}>
            <label className="space-y-1 text-xs text-primary-muted">
              <span>Shared field</span>
              <select value={batchField} onChange={(event) => setBatchField(event.target.value === "artist" ? "artist" : "album")} className={TAG_INPUT}>
                <option value="artist">Artist</option><option value="album">Album</option>
              </select>
            </label>
            <label className="min-w-0 flex-1 space-y-1 text-xs text-primary-muted">
              <span>Shared replacement value</span>
              <input className={TAG_INPUT} value={batchValue} maxLength={512} aria-invalid={!!batchError}
                aria-describedby={batchError ? "metadata-batch-error" : undefined}
                onChange={(event) => setBatchValue(event.target.value)} />
            </label>
            <Button type="submit" disabled={busy || stale || !!batchError || !batchValue.trim() || !selected.length}>
              Stage for {selected.length} selected
            </Button>
          </form>
          {batchError && <p id="metadata-batch-error" role="alert" className="text-xs text-rose-300">{batchError}</p>}
          {stale && <ActionBanner tone="warning" message="Selected drafts were staged before the current analysis. Reopen and stage them again, or deselect them." />}
          {save.error && <ActionBanner tone="error" message={save.error.message} />}
          {generate.error && <ActionBanner tone="error" message={generate.error.message} />}
          <Button type="button" tone="primary" disabled={busy || stale || !selected.length} onClick={() => void generatePreview()}>
            {generate.isPending ? "Preparing preview…" : "Generate metadata plan"}
          </Button>
        </Panel>
      )}
      {preview && (
        <Panel className="space-y-4 p-4 sm:p-5">
          <h2 className="font-display text-lg font-semibold text-primary">Metadata plan preview</h2>
          {preview.plan.review_only ? (
            <ActionBanner tone="info" message="Preview only. This launcher cannot apply edits or undo file changes." />
          ) : (
            <p className="text-sm text-primary-muted">Applying writes the tag replacements below to the original files and saves full backups to the workspace.</p>
          )}
          <ol className="divide-y divide-white/10 border-y border-white/10">
            {preview.plan.files.map((file) => (
              <li key={file.path} className="space-y-2 py-3">
                <PathBlock value={file.path} />
                <dl className="grid min-w-0 gap-2 text-xs [overflow-wrap:anywhere]">
                  {METADATA_FIELDS.filter((field) => file.changes[field] !== undefined).map((field) => (
                    <div key={field} className="grid gap-1 sm:grid-cols-[5rem_minmax(0,1fr)_minmax(0,1fr)]">
                      <dt className="font-semibold text-primary">{FIELD_LABELS[field]}</dt>
                      <dd className="text-primary-muted"><span className="text-primary-subtle">Before · </span>{file.before[field].join(" · ") || "Missing"}</dd>
                      <dd className="text-accent"><span className="text-primary-subtle">After · </span>{file.changes[field]}</dd>
                    </div>
                  ))}
                </dl>
              </li>
            ))}
          </ol>
          {previewStale && <ActionBanner tone="warning" message="The library or staged edits changed after this preview. Check the drafts and generate a fresh plan." />}
          {apply.error && <ActionBanner tone="error" message={apply.error.message} />}
          {!preview.plan.review_only && <div className="space-y-3">
            <label className="flex items-start gap-2 text-sm text-primary">
              <input type="checkbox" className="mt-1 accent-accent" checked={confirmed} disabled={busy || previewStale}
                onChange={(event) => setConfirmed(event.target.checked)} />
              <span>I approve the listed tag replacements on these {preview.plan.files.length} files.</span>
            </label>
            <Button type="button" tone="danger" disabled={!confirmed || busy || previewStale} onClick={() => void applyPreview()}>
              {apply.isPending ? "Applying metadata…" : "Apply approved metadata"}
            </Button>
          </div>}
        </Panel>
      )}
      {outcome && <Panel className="space-y-3 p-5">
        <h2 className="font-display text-lg font-semibold text-primary" aria-live="polite">Metadata apply results</h2>
        {outcome.cleanupError && <ActionBanner tone="warning" message={outcome.cleanupError} />}
        <ul className="space-y-3">
          {outcome.response.results.flatMap((result) => result.operation_results).map((operation) => (
            <li key={operation.operation_id} className="space-y-1">
              {operation.path && <PathBlock value={operation.path} />}
              <p className="text-sm text-primary">
                {operation.status === "ok"
                  ? "Applied and verified"
                  : `${operation.status === "skipped" ? "Skipped" : "Failed"} · ${operation.message}`}
              </p>
            </li>
          ))}
        </ul>
        {outcome.response.batch_id && <Link className="text-sm text-accent underline"
          to={`/history?batch=${encodeURIComponent(outcome.response.batch_id)}`}>Open history to inspect or undo</Link>}
      </Panel>}
    </div>
  );
}
