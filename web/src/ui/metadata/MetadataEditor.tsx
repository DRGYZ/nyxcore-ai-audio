import { useId, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { fetchMetadataDrafts, fetchMetadataTrack, saveMetadataDrafts } from "../../lib/api";
import { pathFilename } from "../../lib/library-path";
import { FIELD_LABELS, ISSUE_LABELS, MAX_METADATA_FILES, METADATA_DRAFTS_KEY, METADATA_FIELDS, tagValidation } from "../../lib/metadata";
import type { DraftSet, MetadataField, MetadataProposal, MetadataTrack, TagValues } from "../../lib/metadata";
import { ActionBanner, Button, Chip, Modal, PathBlock } from "../components";

export const TAG_INPUT = "min-w-0 w-full rounded-[3px] border border-white/20 bg-surface-low px-3 py-2 text-sm text-primary focus-visible:outline focus-visible:outline-accent disabled:opacity-60";
const DRAFT_LINK = "text-xs font-medium text-accent underline-offset-4 hover:underline focus-visible:outline-accent";

export function MetadataDraftLink() {
  return <Link className={DRAFT_LINK} to="/metadata">Review staged metadata</Link>;
}

export function MetadataEditButton({ path }: { path: string }) {
  const [, setSearch] = useSearchParams();
  return (
    <Button type="button" aria-label={`Edit metadata for ${pathFilename(path)}`} tone="secondary"
      onClick={() => setSearch((params) => {
        const next = new URLSearchParams(params);
        next.set("editTrack", path);
        return next;
      }, { replace: true })}>
      Edit metadata
    </Button>
  );
}

export function useMetadataDrafts(enabled = true) {
  return useQuery({ queryKey: METADATA_DRAFTS_KEY, queryFn: fetchMetadataDrafts, enabled, retry: false });
}

export function useSaveMetadataDrafts() {
  const cache = useQueryClient();
  return useMutation({
    mutationFn: ({ revision, entries }: { revision: string; entries: MetadataProposal[] }) => saveMetadataDrafts(revision, entries),
    onSuccess: (updated) => cache.setQueryData(METADATA_DRAFTS_KEY, updated),
    onError: () => cache.invalidateQueries({ queryKey: METADATA_DRAFTS_KEY }),
  });
}

export function MetadataEditor() {
  const [params, setSearch] = useSearchParams();
  const path = params.get("editTrack");
  const track = useQuery({
    queryKey: ["metadata", "track", path], queryFn: () => fetchMetadataTrack(path!),
    enabled: Boolean(path), retry: false, refetchOnWindowFocus: false,
  });
  const drafts = useMetadataDrafts(Boolean(path));

  function close() {
    setSearch((previous) => {
      const next = new URLSearchParams(previous);
      next.delete("editTrack");
      return next;
    }, { replace: true });
  }

  if (!path) return null;
  const failure = track.error || drafts.error;
  return (
    <Modal open title="Edit track metadata" subtitle={pathFilename(path)} onClose={close}>
      {failure ? (
        <div className="space-y-3">
          <ActionBanner tone="error" message={failure.message} />
          <Button type="button" onClick={() => { void track.refetch(); void drafts.refetch(); }}>Retry inspection</Button>
        </div>
      ) : track.data && drafts.data && !track.isFetching ? (
        <MetadataForm key={track.data.inspection} track={track.data} drafts={drafts.data} />
      ) : <p className="text-sm text-primary-muted" role="status">Reading embedded metadata…</p>}
    </Modal>
  );
}

const SUGGESTION_LABELS = {
  swap: "Filename-matched swap", embedded: "Embedded value", formatting: "Formatting cleanup",
  filename: "Filename suggestion", unavailable: "No reliable suggestion",
};
const CONFIDENCE_LABELS = {
  matched: "Both tags match the swap", existing: "Already embedded", high: "High confidence",
  tentative: "Tentative", ambiguous: "Ambiguous",
};

function MetadataForm({ track, drafts }: { track: MetadataTrack; drafts: DraftSet }) {
  const id = useId();
  const existing = drafts.entries.find((entry) => entry.path === track.path);
  const [changes, setChanges] = useState<TagValues>(existing?.values ?? {});
  const [origins, setOrigins] = useState<Partial<Record<MetadataField, string>>>({});
  const [saved, setSaved] = useState(false);
  const save = useSaveMetadataDrafts();
  const fields = METADATA_FIELDS.filter((field) => changes[field] !== undefined);
  const invalid = fields.some((field) => tagValidation(changes[field]!));
  const full = drafts.entries.length >= MAX_METADATA_FILES && !existing;
  const generationChanged = drafts.generation !== track.generation;
  const staleDraft = existing && existing.inspection !== track.inspection;
  const changedSinceStaging = fields.some((field) => changes[field] !== existing?.values[field]) ||
    Object.keys(existing?.values ?? {}).length !== fields.length;
  const ready = fields.length > 0 && !invalid && !full && !generationChanged && (changedSinceStaging || staleDraft);

  function setField(field: MetadataField, value: string, source?: string) {
    setChanges((previous) => {
      const next = { ...previous };
      if (value === (track.tags[field][0] ?? "")) delete next[field];
      else next[field] = value;
      return next;
    });
    setOrigins((previous) => ({ ...previous, [field]: source }));
    setSaved(false);
  }

  async function stage(event: FormEvent) {
    event.preventDefault();
    if (!ready || !track.supported) return;
    const values = Object.fromEntries(fields.map((field) => [field, changes[field]!.trim()]));
    const entries = [
      ...drafts.entries.filter((entry) => entry.path !== track.path),
      { path: track.path, inspection: track.inspection, generation: track.generation, values },
    ];
    try {
      await save.mutateAsync({ revision: drafts.revision, entries });
      setChanges(values);
      setSaved(true);
    } catch {
      // The draft error is announced below.
    }
  }

  return (
    <form onSubmit={stage} className="space-y-4">
      <PathBlock value={track.path} />
      {track.issues.length > 0 && <div className="flex flex-wrap gap-1.5">
        {track.issues.map((issue) => <Chip tone="warning" key={issue}>{ISSUE_LABELS[issue]}</Chip>)}
      </div>}
      {!track.supported && <ActionBanner tone="info" message="This file can be inspected. Writing is available for readable MP3, FLAC, and M4A files with a single filesystem link." />}
      {staleDraft && <ActionBanner tone="warning" message="This staged edit is stale. Compare it with the current tags before staging it again." />}
      {generationChanged && <ActionBanner tone="warning" message="The library was refreshed. Close and reopen this track to inspect the latest values." />}
      {track.swap && track.supported && <Button type="button" tone="secondary" onClick={() => {
        for (const field of ["artist", "title"] as const) {
          const suggestion = track.suggestions[field];
          if (suggestion.value) setField(field, suggestion.value, SUGGESTION_LABELS.swap);
        }
      }}>Use filename-matched title and artist swap</Button>}
      <div className="divide-y divide-white/10 border-y border-white/10">
        {METADATA_FIELDS.map((field) => {
          const control = `${id}-${field}`;
          const suggestion = track.suggestions[field];
          const changed = changes[field] !== undefined;
          const error = changed ? tagValidation(changes[field]!) : null;
          const canSuggest = suggestion.value && suggestion.source !== "embedded";
          return (
            <div key={field} className="grid min-w-0 gap-3 py-3 sm:grid-cols-2">
              <div className="min-w-0 space-y-1 text-sm [overflow-wrap:anywhere]">
                <p className="font-display text-sm font-semibold text-primary">{FIELD_LABELS[field]}</p>
                <p className="text-primary-muted">Current · {track.tags[field].join(" · ") || "Missing"}</p>
                <p className="text-xs text-primary-muted">
                  {SUGGESTION_LABELS[suggestion.source]}{canSuggest && <> · <strong className="text-accent">{suggestion.value}</strong></>}
                </p>
                <p className="text-xs text-primary-muted">
                  {canSuggest ? `${CONFIDENCE_LABELS[suggestion.confidence]}. ` : ""}{suggestion.evidence}
                </p>
                {canSuggest && track.supported && <Button type="button" onClick={() => setField(field, suggestion.value!, SUGGESTION_LABELS[suggestion.source])}>
                  Use {field} suggestion
                </Button>}
              </div>
              <div className="min-w-0 space-y-1.5">
                <label className="block text-xs font-medium text-primary" htmlFor={control}>Proposed {field}</label>
                <input id={control} className={TAG_INPUT} maxLength={512} value={changes[field] ?? track.tags[field][0] ?? ""}
                  disabled={!track.supported || save.isPending}
                  aria-invalid={Boolean(error)} aria-describedby={`${control}-state`}
                  onChange={(event) => setField(field, event.target.value, "User-entered")} />
                <p id={`${control}-state`} className={`text-xs ${error ? "text-rose-300" : "text-primary-muted"}`} role={error ? "alert" : undefined}>
                  {error || (changed ? `Will change · ${origins[field] ?? "Staged value"}` : "Unchanged")}
                </p>
                {changed && <Button type="button" onClick={() => setField(field, track.tags[field][0] ?? "")}>Revert {field}</Button>}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-primary-muted">Clearing tags is unavailable. Staging saves a proposal to the workspace without touching audio.</p>
      {full && <ActionBanner tone="warning" message="The batch is full. Review or remove an edit before staging another file." />}
      {save.error && <ActionBanner tone="error" message={save.error.message} />}
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" tone="primary" disabled={!track.supported || !ready || save.isPending}>
          {save.isPending ? "Staging…" : "Stage metadata edit"}
        </Button>
        <Button type="button" disabled={!changedSinceStaging || save.isPending} onClick={() => {
          setChanges(existing?.values ?? {});
          setOrigins({});
          setSaved(false);
        }}>Discard unsaved edits</Button>
        <MetadataDraftLink />
      </div>
      <p role="status" className="text-xs text-primary-muted">{saved ? "Staged in the workspace." : ready ? "Ready to stage." : invalid ? "Correct the highlighted values to stage this edit." : existing ? "A draft is staged." : "Enter a replacement to stage this track."}</p>
    </form>
  );
}
