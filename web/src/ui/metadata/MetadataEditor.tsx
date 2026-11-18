import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { fetchMetadataDrafts, fetchMetadataTrack, saveMetadataDrafts } from "../../lib/api";
import { pathFilename } from "../../lib/library-path";
import { FIELD_LABELS, ISSUE_LABELS, MAX_METADATA_FILES, METADATA_DRAFTS_KEY, METADATA_FIELDS, tagValidation } from "../../lib/metadata";
import type { DraftSet, MetadataField, MetadataProposal, MetadataTrack, TagValues } from "../../lib/metadata";
import type { MusicSource, SelectedCover } from "../../lib/enrichment";
import { ActionBanner, Button, Chip, Icon, Modal, PathBlock, SleeveArt } from "../components";
import { MusicInfo, MusicSourceLink } from "./MusicInfo";

export const TAG_INPUT = "min-w-0 w-full rounded-lg border border-white/[0.15] bg-surface px-3 py-2 text-sm text-primary placeholder:text-primary-subtle focus-visible:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:opacity-60";
const DRAFT_LINK = "text-xs font-medium text-accent underline-offset-4 hover:underline focus-visible:outline-accent";

export function MetadataDraftLink() {
  return <Link className={DRAFT_LINK} to="/metadata">Review staged metadata</Link>;
}

export function MetadataEditButton({ path }: { path: string }) {
  const [, setSearch] = useSearchParams();
  return (
    <Button
      type="button"
      aria-label={`Edit metadata for ${pathFilename(path)}`}
      tone="secondary"
      data-metadata-editor-path={path}
      onClick={() => setSearch((params) => {
        const next = new URLSearchParams(params);
        next.set("editTrack", path);
        return next;
      }, { replace: true })}
    >
      <Icon name="edit" className="text-[15px] text-accent" />
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
    queryKey: ["metadata", "track", path],
    queryFn: () => fetchMetadataTrack(path!),
    enabled: Boolean(path),
    retry: false,
    refetchOnWindowFocus: false,
  });
  const drafts = useMetadataDrafts(Boolean(path));
  const [actionsTarget, setActionsTarget] = useState<HTMLDivElement | null>(null);
  const closingPath = useRef<string | null>(null);

  useEffect(() => {
    if (path || !closingPath.current) return;
    const targetPath = closingPath.current;
    const frame = requestAnimationFrame(() => {
      closingPath.current = null;
      const buttons = document.querySelectorAll<HTMLButtonElement>("button[data-metadata-editor-path]");
      const trigger = Array.from(buttons).find((button) =>
        button.dataset.metadataEditorPath === targetPath && button.getClientRects().length > 0,
      );
      trigger?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [path]);

  function close() {
    closingPath.current = path;
    setSearch((previous) => {
      const next = new URLSearchParams(previous);
      next.delete("editTrack");
      return next;
    }, { replace: true });
  }

  if (!path) return null;
  const failure = track.error || drafts.error;
  const ready = !failure && track.data && drafts.data && !track.isFetching;
  return (
    <Modal
      open
      title="Edit track metadata"
      subtitle={pathFilename(path)}
      onClose={close}
      footer={ready && track.data?.readable ? <div className="w-full space-y-2" ref={setActionsTarget} /> : undefined}
    >
      {failure ? (
        <div className="space-y-3">
          <ActionBanner tone="error" message={failure.message} />
          <Button type="button" onClick={() => { void track.refetch(); void drafts.refetch(); }}>Retry inspection</Button>
        </div>
      ) : track.data && drafts.data && !track.isFetching ? (
        track.data.readable ? (
          <MetadataForm key={track.data.inspection} track={track.data} drafts={drafts.data} actionsTarget={actionsTarget} />
        ) : (
          <div className="space-y-3">
            <PathBlock value={track.data.path} />
            <ActionBanner tone="error" message={track.data.write_blocker ?? "Embedded metadata could not be read."} />
          </div>
        )
      ) : <p className="text-sm text-primary-muted" role="status">Reading embedded metadata…</p>}
    </Modal>
  );
}

const SUGGESTION_LABELS = {
  swap: "Filename-matched swap",
  embedded: "Embedded value",
  formatting: "Formatting cleanup",
  filename: "Filename suggestion",
  unavailable: "No reliable suggestion",
};
const CONFIDENCE_LABELS = {
  matched: "Both tags match the swap",
  existing: "Already embedded",
  high: "High confidence",
  tentative: "Tentative",
  ambiguous: "Ambiguous",
};

function MetadataForm({ track, drafts, actionsTarget }: {
  track: MetadataTrack;
  drafts: DraftSet;
  actionsTarget: HTMLDivElement | null;
}) {
  const id = useId();
  const existing = drafts.entries.find((entry) => entry.path === track.path);
  const [changes, setChanges] = useState<TagValues>(existing?.values ?? {});
  const [origins, setOrigins] = useState<Partial<Record<MetadataField, string>>>(() =>
    Object.fromEntries(existing?.source_fields?.map((field) => [field, "MusicBrainz suggestion"]) ?? []),
  );
  const [source, setSource] = useState(existing?.source);
  const [cover, setCover] = useState<SelectedCover | undefined>(
    existing?.artwork ? { selection: existing.artwork, thumbnail: null } : undefined,
  );
  const [saved, setSaved] = useState(false);
  const save = useSaveMetadataDrafts();
  const fields = METADATA_FIELDS.filter((field) => changes[field] !== undefined);
  const suggestedValues = { title: source?.title, artist: source?.artist, album: source?.release?.title };
  const providerFields = fields.filter((field) =>
    origins[field] === "MusicBrainz suggestion" && changes[field]?.trim() === suggestedValues[field],
  );
  const invalid = fields.some((field) => tagValidation(changes[field]!));
  const full = drafts.entries.length >= MAX_METADATA_FILES && !existing;
  const generationChanged = drafts.generation !== track.generation;
  const staleDraft = existing && existing.inspection !== track.inspection;
  const changedSinceStaging = fields.some((field) => changes[field] !== existing?.values[field]) ||
    Object.keys(existing?.values ?? {}).length !== fields.length ||
    providerFields.join() !== (existing?.source_fields ?? []).join() ||
    source?.token !== existing?.source?.token ||
    cover?.selection.token !== existing?.artwork?.token ||
    cover?.selection.replace_existing !== existing?.artwork?.replace_existing;
  const artworkReady = !cover || (track.artwork_supported && (!track.cover_present || cover.selection.replace_existing));
  const ready = (fields.length > 0 || Boolean(cover)) && artworkReady && !invalid && !full && !generationChanged && (changedSinceStaging || staleDraft);

  const previewTitle = (changes.title ?? track.tags.title[0]) || pathFilename(track.path);
  const previewArtist = (changes.artist ?? track.tags.artist[0]) || "Unknown artist";
  const previewAlbum = (changes.album ?? track.tags.album[0]) || "Unknown album";

  function setField(field: MetadataField, value: string, origin?: string) {
    setChanges((previous) => {
      const next = { ...previous };
      if (value === (track.tags[field][0] ?? "")) delete next[field];
      else next[field] = value;
      return next;
    });
    setOrigins((previous) => ({ ...previous, [field]: origin }));
    setSaved(false);
  }

  function useSuggestedTags(selection: MusicSource) {
    if (selection.token !== source?.token) setCover(undefined);
    setSource(selection);
    const suggested = { title: selection.title, artist: selection.artist, album: selection.release?.title };
    for (const field of METADATA_FIELDS) {
      const value = suggested[field];
      if (value) setField(field, value, "MusicBrainz suggestion");
      else if (origins[field] === "MusicBrainz suggestion") setField(field, track.tags[field][0] ?? "");
    }
    setSaved(false);
  }

  async function stage(event: FormEvent) {
    event.preventDefault();
    if (!ready || !track.supported) return;
    const values = Object.fromEntries(fields.map((field) => [field, changes[field]!.trim()]));
    const entries = [
      ...drafts.entries.filter((entry) => entry.path !== track.path),
      {
        path: track.path,
        inspection: track.inspection,
        generation: track.generation,
        values,
        source,
        source_fields: providerFields.length ? providerFields : undefined,
        artwork: cover?.selection,
      },
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
    <form id={id} onSubmit={stage} className="space-y-5">
      {/* Track & Cover Sleeve Showcase */}
      <div className="flex flex-col gap-4 rounded-xl border border-white/[0.08] bg-surface-low/80 p-4 sm:flex-row sm:items-center">
        <SleeveArt
          title={previewTitle}
          artist={previewArtist}
          hasArtwork={Boolean(track.cover_present || cover)}
          src={cover?.thumbnail}
          alt={cover?.thumbnail ? "Selected front cover" : undefined}
          size="lg"
          showVinyl
        />
        <div className="min-w-0 flex-1 space-y-2">
          <div>
            <p className="font-display text-base font-semibold text-primary [overflow-wrap:anywhere]">{previewTitle}</p>
            <p className="text-xs text-primary-muted [overflow-wrap:anywhere]">
              {previewArtist} · {previewAlbum}
            </p>
          </div>
          <PathBlock value={track.path} />
          {track.issues.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {track.issues.map((issue) => <Chip tone="warning" key={issue}>{ISSUE_LABELS[issue]}</Chip>)}
            </div>
          )}
        </div>
      </div>

      {track.write_blocker && <ActionBanner tone="info" message={track.write_blocker} />}
      {staleDraft && <ActionBanner tone="warning" message="This staged edit is stale. Compare it with the current tags before staging it again." />}
      {generationChanged && <ActionBanner tone="warning" message="The library was refreshed. Close and reopen this track to inspect the latest values." />}

      <MusicInfo
        track={track}
        artwork={cover?.selection}
        artworkSourceToken={source?.token}
        onTags={useSuggestedTags}
        onCover={(selection, image) => {
          setSource(selection);
          setCover(image);
          setSaved(false);
        }}
      />

      {source && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-accent/30 bg-accent/[0.05] px-4 py-2.5 text-xs text-primary-muted">
          <MusicSourceLink source={source} />
          <Button
            type="button"
            onClick={() => {
              setSource(undefined);
              setCover(undefined);
              setOrigins(Object.fromEntries(fields.map((field) => [field, "User-entered"])));
              setSaved(false);
            }}
          >
            Clear online selection
          </Button>
        </div>
      )}

      {cover && (
        <div className="space-y-3 rounded-xl border border-accent/30 bg-surface-low/80 p-4">
          <div className="flex flex-wrap items-center gap-4">
            {cover.thumbnail ? (
              <img className="size-20 rounded-lg border border-white/15 object-cover shadow-sleeve" src={cover.thumbnail} alt="Selected front cover" />
            ) : null}
            <div className="min-w-0 flex-1 text-xs text-primary-muted">
              <p className="font-display text-sm font-semibold text-primary">Proposed front cover</p>
              <p className="mt-0.5 font-mono">{cover.selection.width} × {cover.selection.height}</p>
            </div>
            <Button type="button" onClick={() => { setCover(undefined); setSaved(false); }}>Remove cover from proposal</Button>
          </div>
          {track.cover_present && (
            <label className="flex items-start gap-2 text-xs text-primary">
              <input
                type="checkbox"
                className="mt-0.5 accent-accent"
                checked={cover.selection.replace_existing}
                onChange={(event) => {
                  setCover({ ...cover, selection: { ...cover.selection, replace_existing: event.target.checked } });
                  setSaved(false);
                }}
              />
              Replace this track’s existing front cover
            </label>
          )}
        </div>
      )}

      {track.swap && track.supported && (
        <Button
          type="button"
          tone="secondary"
          onClick={() => {
            for (const field of ["artist", "title"] as const) {
              const suggestion = track.suggestions[field];
              if (suggestion.value) setField(field, suggestion.value, SUGGESTION_LABELS.swap);
            }
          }}
        >
          <Icon name="swap_horiz" className="text-base text-accent" />
          Use filename-matched title and artist swap
        </Button>
      )}

      <div className="space-y-3">
        {METADATA_FIELDS.map((field) => {
          const control = `${id}-${field}`;
          const suggestion = track.suggestions[field];
          const changed = changes[field] !== undefined;
          const error = changed ? tagValidation(changes[field]!) : null;
          const canSuggest = suggestion.value && suggestion.source !== "embedded";
          return (
            <div
              key={field}
              className={`grid min-w-0 gap-4 rounded-xl border p-4 sm:grid-cols-2 ${
                changed ? "border-accent/40 bg-accent/[0.04]" : "border-white/[0.08] bg-surface-low/60"
              }`}
            >
              <div className="min-w-0 space-y-1.5 text-xs [overflow-wrap:anywhere]">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-display text-sm font-semibold text-primary">{FIELD_LABELS[field]}</p>
                  {changed ? <Chip tone="accent">Modified</Chip> : <Chip tone="neutral">Current</Chip>}
                </div>
                <p className="text-sm text-primary-muted">
                  Current · <span className="text-primary">{track.tags[field].join(" · ") || "Missing"}</span>
                </p>
                <p className="text-xs text-primary-muted">
                  {SUGGESTION_LABELS[suggestion.source]}{canSuggest && <> · <strong className="text-accent">{suggestion.value}</strong></>}
                </p>
                <p className="text-xs text-primary-subtle">
                  {canSuggest ? `${CONFIDENCE_LABELS[suggestion.confidence]}. ` : ""}{suggestion.evidence}
                </p>
                {canSuggest && track.supported && (
                  <div className="pt-1">
                    <Button type="button" tone="secondary" onClick={() => setField(field, suggestion.value!, SUGGESTION_LABELS[suggestion.source])}>
                      Use {field} suggestion
                    </Button>
                  </div>
                )}
              </div>

              <div className="min-w-0 space-y-2">
                <label className="block text-xs font-medium text-primary" htmlFor={control}>Proposed {field}</label>
                <input
                  id={control}
                  className={TAG_INPUT}
                  maxLength={512}
                  value={changes[field] ?? track.tags[field][0] ?? ""}
                  disabled={!track.supported || save.isPending}
                  aria-invalid={Boolean(error)}
                  aria-describedby={`${control}-state`}
                  onChange={(event) => setField(field, event.target.value, "User-entered")}
                />
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p
                    id={`${control}-state`}
                    className={`text-xs ${error ? "text-[#f59f8a]" : changed ? "text-accent" : "text-primary-subtle"}`}
                    role={error ? "alert" : undefined}
                  >
                    {error || (changed ? `Will change · ${providerFields.includes(field) ? "MusicBrainz suggestion"
                      : origins[field] === "MusicBrainz suggestion" ? "User-entered" : origins[field] ?? "Staged value"}` : "Unchanged")}
                  </p>
                  {changed && (
                    <Button type="button" tone="ghost" onClick={() => setField(field, track.tags[field][0] ?? "")}>
                      Revert {field}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-primary-subtle">Clearing tags is unavailable. Staging saves a proposal to the workspace without touching audio.</p>
      {full && <ActionBanner tone="warning" message="The batch is full. Review or remove an edit before staging another file." />}

      {actionsTarget && createPortal(
        <>
          {save.error && <ActionBanner tone="error" message={save.error.message} />}
          <div className="flex flex-wrap items-center gap-2.5">
            <Button type="submit" form={id} tone="primary" disabled={!track.supported || !ready || save.isPending}>
              {save.isPending ? "Staging…" : "Stage metadata edit"}
            </Button>
            <Button
              type="button"
              disabled={!changedSinceStaging || save.isPending}
              onClick={() => {
                setChanges(existing?.values ?? {});
                setSource(existing?.source);
                setCover(existing?.artwork ? { selection: existing.artwork, thumbnail: null } : undefined);
                setOrigins(Object.fromEntries(existing?.source_fields?.map((field) => [field, "MusicBrainz suggestion"]) ?? []));
                setSaved(false);
              }}
            >
              Discard unsaved edits
            </Button>
            <MetadataDraftLink />
          </div>
          <p role="status" className="text-xs text-primary-muted">
            {saved ? "Staged in the workspace." : ready ? "Ready to stage." : invalid ? "Correct the highlighted values to stage this edit." : cover && !artworkReady ? "Confirm replacement or remove the cover before staging." : existing ? "A draft is staged." : "Enter a replacement to stage this track."}
          </p>
        </>,
        actionsTarget,
      )}
    </form>
  );
}
