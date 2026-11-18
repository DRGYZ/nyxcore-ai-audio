import { useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { applyMetadataPlan, previewMetadataPlan } from "../../lib/api";
import { pathFilename } from "../../lib/library-path";
import { musicBrainzLink } from "../../lib/enrichment";
import { FIELD_LABELS, METADATA_FIELDS, tagValidation } from "../../lib/metadata";
import type { MetadataPlan, MetadataProposal, PlanInput } from "../../lib/metadata";
import type { ReviewPlanApplyResponse } from "../../lib/types";
import { ActionBanner, Button, Chip, EmptyState, Icon, PageHeader, Panel, PathBlock, SleeveArt } from "../components";
import { MetadataEditButton, TAG_INPUT, useMetadataDrafts, useSaveMetadataDrafts } from "../metadata/MetadataEditor";
import { MusicSourceLink } from "../metadata/MusicInfo";

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
      ...entry,
      values: { ...entry.values, [batchField]: batchValue.trim() },
      source_fields: entry.source_fields?.filter((field) => field !== batchField),
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
      <PageHeader
        eyebrow="Tag & Artwork Staging"
        title="Staged metadata"
        description="Review tag and front-cover replacements side by side before applying. Each selected file counts as one operation."
      />
      {drafts.error && <ActionBanner tone="error" message={drafts.error.message} />}
      {drafts.isLoading && <p role="status" className="text-sm text-primary-muted">Loading workspace drafts…</p>}
      {data && entries.length === 0 && (
        <EmptyState
          title="No metadata edits staged."
          description="Open a track from Library Health, Review Inbox, or Archive Search to inspect tags, look up MusicBrainz releases, or stage cover artwork."
          action={
            <div className="flex flex-wrap justify-center gap-2.5">
              <Link className="inline-flex items-center gap-1.5 rounded-lg border border-accent bg-accent px-4 py-2 text-xs font-semibold text-[#14110e] hover:bg-[#e8ae6d]" to="/health">
                Open Library Health
              </Link>
              <Link className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-surface-mid px-4 py-2 text-xs font-medium text-primary hover:border-accent/50" to="/search">
                Browse Archive Search
              </Link>
            </div>
          }
        />
      )}

      {data && entries.length > 0 && (
        <Panel className="space-y-5 p-5 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.08] pb-3.5">
            <div>
              <h2 className="font-display text-lg font-semibold text-primary">Files to review</h2>
              <p className="mt-0.5 text-xs text-primary-subtle">Select the staged track proposals you want to include in a preview plan.</p>
            </div>
            <Chip tone="accent">{selected.length} selected · {entries.length} of 50 staged</Chip>
          </div>

          <div className="space-y-3">
            {entries.map((entry) => {
              const filename = pathFilename(entry.path);
              const isSelected = !deselected.has(entry.path);
              return (
                <article
                  key={entry.path}
                  className={`grid min-w-0 gap-4 rounded-xl border p-4 transition-colors sm:grid-cols-[minmax(0,1fr)_auto] ${
                    isSelected ? "border-white/[0.12] bg-surface-low/80" : "border-white/[0.06] bg-surface-low/40 opacity-75"
                  }`}
                >
                  <div className="flex min-w-0 items-start gap-3.5">
                    <SleeveArt
                      title={entry.values.title ?? filename}
                      artist={entry.values.artist}
                      hasArtwork={Boolean(entry.artwork)}
                      size="md"
                      showVinyl
                    />
                    <div className="min-w-0 flex-1 space-y-2">
                      <label className="flex items-start gap-2.5 text-sm font-semibold text-primary">
                        <input
                          className="mt-1 size-4 accent-accent"
                          type="checkbox"
                          checked={isSelected}
                          disabled={busy}
                          onChange={(event) => {
                            const checked = event.target.checked;
                            clearPreview();
                            setDeselected((previous) => {
                              const next = new Set(previous);
                              if (checked) next.delete(entry.path);
                              else next.add(entry.path);
                              return next;
                            });
                          }}
                        />
                        <span className="font-display [overflow-wrap:anywhere]">{filename}</span>
                      </label>
                      <PathBlock value={entry.path} />
                      {entry.source && <MusicSourceLink source={entry.source} />}
                      <dl className="grid gap-1.5 rounded-lg border border-white/[0.06] bg-surface p-2.5 text-xs [overflow-wrap:anywhere] sm:grid-cols-3">
                        {METADATA_FIELDS.filter((field) => entry.values[field] !== undefined).map((field) => (
                          <div key={field}>
                            <dt className="text-[11px] text-primary-subtle">{FIELD_LABELS[field]}</dt>
                            <dd className="mt-0.5 font-medium text-accent">{entry.values[field]}</dd>
                          </div>
                        ))}
                      </dl>
                      <div className="flex flex-wrap gap-1.5">
                        {entry.artwork && <Chip tone="accent">{entry.artwork.replace_existing ? "Replace front cover" : "Add front cover"}</Chip>}
                        {entry.generation !== data.generation && <Chip tone="warning">Stale · reopen and stage again</Chip>}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <MetadataEditButton path={entry.path} />
                    <Button
                      type="button"
                      tone="ghost"
                      disabled={busy}
                      onClick={() => void replaceDrafts(entries.filter((item) => item.path !== entry.path))}
                      aria-label={`Remove edit for ${filename}`}
                    >
                      Remove
                    </Button>
                  </div>
                </article>
              );
            })}
          </div>

          <form className="flex flex-col gap-3 rounded-xl border border-white/[0.08] bg-surface-low/60 p-4 sm:flex-row sm:items-end" onSubmit={updateSelected}>
            <label className="space-y-1.5 text-xs font-medium text-primary-muted">
              <span>Shared field</span>
              <select
                value={batchField}
                onChange={(event) => setBatchField(event.target.value === "artist" ? "artist" : "album")}
                className={TAG_INPUT}
              >
                <option value="artist">Artist</option>
                <option value="album">Album</option>
              </select>
            </label>
            <label className="min-w-0 flex-1 space-y-1.5 text-xs font-medium text-primary-muted">
              <span>Shared replacement value</span>
              <input
                className={TAG_INPUT}
                value={batchValue}
                maxLength={512}
                aria-invalid={!!batchError}
                aria-describedby={batchError ? "metadata-batch-error" : undefined}
                onChange={(event) => setBatchValue(event.target.value)}
              />
            </label>
            <Button type="submit" tone="secondary" disabled={busy || stale || !!batchError || !batchValue.trim() || !selected.length}>
              Stage for {selected.length} selected
            </Button>
          </form>
          {batchError && <p id="metadata-batch-error" role="alert" className="text-xs text-[#f59f8a]">{batchError}</p>}
          {stale && <ActionBanner tone="warning" message="Selected drafts were staged before the current analysis. Reopen and stage them again, or deselect them." />}
          {save.error && <ActionBanner tone="error" message={save.error.message} />}
          {generate.error && <ActionBanner tone="error" message={generate.error.message} />}

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <Button type="button" tone="primary" disabled={busy || stale || !selected.length} onClick={() => void generatePreview()}>
              <Icon name="preview" className="text-base" />
              {generate.isPending ? "Preparing preview…" : "Generate metadata plan"}
            </Button>
          </div>
        </Panel>
      )}

      {preview && (
        <Panel className="space-y-5 p-5 sm:p-6">
          <div className="border-b border-white/[0.08] pb-3.5">
            <h2 className="font-display text-lg font-semibold text-primary">Metadata plan preview</h2>
            {preview.plan.review_only ? (
              <div className="mt-3">
                <ActionBanner tone="info" message="Preview only. This launcher cannot apply edits or undo file changes." />
              </div>
            ) : (
              <p className="mt-1 text-sm text-primary-muted">Applying writes the listed changes to the original files and saves full backups to the workspace.</p>
            )}
          </div>

          <ol className="space-y-4">
            {preview.plan.files.map((file) => (
              <li key={file.path} className="space-y-3 rounded-xl border border-white/[0.08] bg-surface-low/80 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-display text-sm font-semibold text-primary">{pathFilename(file.path)}</p>
                  {file.source && <MusicSourceLink source={file.source} />}
                </div>
                <PathBlock value={file.path} />

                <dl className="grid min-w-0 gap-2.5 text-xs [overflow-wrap:anywhere]">
                  {METADATA_FIELDS.filter((field) => file.changes[field] !== undefined).map((field) => (
                    <div key={field} className="grid gap-2 rounded-lg border border-white/[0.06] bg-surface p-3 sm:grid-cols-[5.5rem_minmax(0,1fr)_minmax(0,1fr)] sm:items-center">
                      <dt className="font-display font-semibold text-primary">{FIELD_LABELS[field]}</dt>
                      <dd className="text-primary-muted">
                        <span className="text-primary-subtle">Before · </span>
                        <span>{file.before[field].join(" · ") || "Missing"}</span>
                      </dd>
                      <dd className="font-medium text-accent">
                        <span className="text-primary-subtle font-normal">After · </span>
                        <span>{file.changes[field]}</span>
                        {file.source_fields?.includes(field) && <span className="ml-2 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] text-accent">MusicBrainz</span>}
                      </dd>
                    </div>
                  ))}
                </dl>

                {file.artwork && (
                  <div className="grid gap-4 border-t border-white/[0.08] pt-3 text-xs sm:grid-cols-2">
                    <div className="space-y-2 rounded-lg border border-white/[0.07] bg-surface p-3">
                      <p className="font-medium text-primary-muted">Before · {file.artwork.before_present ? "Existing front cover" : "No front cover"}</p>
                      {file.artwork.before ? (
                        <img className="size-24 rounded-lg border border-white/[0.12] object-cover shadow-sleeve" src={file.artwork.before} alt="Existing embedded front cover" />
                      ) : (
                        <SleeveArt title={pathFilename(file.path)} size="lg" />
                      )}
                    </div>
                    <div className="space-y-2 rounded-lg border border-accent/30 bg-accent/[0.05] p-3">
                      <p className="font-medium text-accent">After · {file.artwork.before_present ? "Replace front cover" : "Embed front cover"}</p>
                      {file.artwork.after && (
                        <img className="size-24 rounded-lg border border-accent/30 object-cover shadow-sleeve" src={file.artwork.after} alt="Approved new front cover" />
                      )}
                      <a
                        className="inline-block text-accent underline underline-offset-4"
                        target="_blank"
                        rel="noreferrer"
                        href={musicBrainzLink("release", file.artwork.release_id)}
                      >
                        Artwork source release
                      </a>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ol>

          {previewStale && <ActionBanner tone="warning" message="The library or staged edits changed after this preview. Check the drafts and generate a fresh plan." />}
          {apply.error && <ActionBanner tone="error" message={apply.error.message} />}
          {!preview.plan.review_only && (
            <div className="space-y-3 border-t border-white/[0.08] pt-4">
              <label className="flex items-start gap-2.5 text-sm text-primary">
                <input
                  type="checkbox"
                  className="mt-1 accent-accent"
                  checked={confirmed}
                  disabled={busy || previewStale}
                  onChange={(event) => setConfirmed(event.target.checked)}
                />
                <span>I approve the listed changes on these {preview.plan.files.length} files.</span>
              </label>
              <Button type="button" tone="danger" disabled={!confirmed || busy || previewStale} onClick={() => void applyPreview()}>
                {apply.isPending ? "Applying metadata…" : "Apply approved metadata"}
              </Button>
            </div>
          )}
        </Panel>
      )}

      {outcome && (
        <Panel className="space-y-3 p-5">
          <h2 className="font-display text-lg font-semibold text-primary" aria-live="polite">Metadata apply results</h2>
          {outcome.cleanupError && <ActionBanner tone="warning" message={outcome.cleanupError} />}
          <ul className="space-y-3">
            {outcome.response.results.flatMap((result) => result.operation_results).map((operation) => (
              <li key={operation.operation_id} className="space-y-1.5 rounded-lg border border-white/[0.07] bg-surface-low p-3">
                {operation.path && <PathBlock value={operation.path} />}
                <p className="text-sm text-primary">
                  {operation.status === "ok"
                    ? "Applied and verified"
                    : `${operation.status === "skipped" ? "Skipped" : "Failed"} · ${operation.message}`}
                </p>
              </li>
            ))}
          </ul>
          {outcome.response.batch_id && (
            <Link className="text-sm font-medium text-accent underline underline-offset-4" to={`/history?batch=${encodeURIComponent(outcome.response.batch_id)}`}>
              Open history to inspect or undo
            </Link>
          )}
        </Panel>
      )}
    </div>
  );
}
