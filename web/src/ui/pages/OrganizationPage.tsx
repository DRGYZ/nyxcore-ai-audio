import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { applyOrganizationSelection, buildOrganizationPreview, fetchOrganizationPage, reviewOrganizationSelection } from "../../lib/api";
import type { OrganizationProposal, OrganizationStatus, OrganizationView } from "../../lib/organization";
import { ActionBanner, Button, Chip, EmptyState, PageHeader, Panel, PathBlock, formatNumber } from "../components";
import { MetadataEditButton } from "../metadata/MetadataEditor";

const LABELS = {
  ready: "Ready",
  needs_review: "Needs review",
  collision: "Collision",
  unchanged: "Unchanged",
} satisfies Record<OrganizationStatus, string>;
const STATUSES = Object.keys(LABELS) as OrganizationStatus[];
const MOVE_LIMIT = 50;

function Files({ rows, selection, busy, toggle }: {
  rows: OrganizationProposal[];
  selection: string[];
  busy: boolean;
  toggle?: (path: string) => void;
}) {
  return (
    <ul className="divide-y divide-white/[0.07]">
      {rows.map((row) => (
        <li key={row.path} className="space-y-2.5 p-4">
          <Chip tone={row.status === "ready" ? "accent" : row.status === "unchanged" ? "neutral" : "warning"}>
            {LABELS[row.status]}
          </Chip>
          <dl className="grid min-w-0 gap-2 sm:grid-cols-2">
            <div className="min-w-0">
              <dt className="mb-1 text-xs text-primary-muted">Current path</dt>
              <dd><PathBlock value={row.source} /></dd>
            </div>
            <div className="min-w-0">
              <dt className="mb-1 text-xs text-primary-muted">Proposed path</dt>
              <dd>{row.destination ? <PathBlock value={row.destination.join(" / ")} />
                : <span className="text-xs text-primary-muted">Awaiting review</span>}</dd>
            </div>
          </dl>
          <p className="text-xs leading-relaxed text-primary-muted">{row.reasons.join(" ")}</p>
          {toggle && row.status === "ready" &&
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" className="mt-1 accent-accent" checked={selection.includes(row.path)}
                disabled={busy || (selection.length >= MOVE_LIMIT && !selection.includes(row.path))}
                onChange={() => toggle(row.path)} />
              Select {row.source}
            </label>}
          {row.inspectable && (row.status === "needs_review" || row.status === "collision") &&
            <MetadataEditButton path={row.path} />}
        </li>
      ))}
    </ul>
  );
}

function Folders({ rows }: { rows: OrganizationProposal[] }) {
  const folders = new Map<string, Map<string, string[]>>();
  for (const row of rows) {
    if (!row.destination) continue;
    const [artist, album, file] = row.destination;
    let albums = folders.get(artist);
    if (!albums) {
      albums = new Map();
      folders.set(artist, albums);
    }
    let files = albums.get(album);
    if (!files) {
      files = [];
      albums.set(album, files);
    }
    files.push(file);
  }

  return (
    <ul aria-label="Proposed folders" className="space-y-3 p-4 text-sm [overflow-wrap:anywhere]">
      {[...folders].map(([artist, albums]) => (
        <li key={artist}>
          <details open>
            <summary className="cursor-pointer font-semibold text-primary">{artist}</summary>
            <ul className="ml-3 mt-1 space-y-2 border-l border-white/10 pl-3 sm:ml-4">
              {[...albums].map(([album, files]) => (
                <li key={album}>
                  <details open>
                    <summary className="cursor-pointer text-primary">{album}</summary>
                    <ul className="ml-3 mt-1 space-y-1 border-l border-white/10 pl-3 font-mono text-xs text-primary-muted">
                      {files.map((file) => <li key={file}>{file}</li>)}
                    </ul>
                  </details>
                </li>
              ))}
            </ul>
          </details>
        </li>
      ))}
    </ul>
  );
}

export function OrganizationPage() {
  const client = useQueryClient();
  const [view, setView] = useState<OrganizationView>("files");
  const [status, setStatus] = useState<OrganizationStatus | "all">("all");
  const [offset, setOffset] = useState(0);
  const [selection, setSelection] = useState<string[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const review = useMutation({ mutationFn: reviewOrganizationSelection });
  const apply = useMutation({
    mutationFn: applyOrganizationSelection,
    onSuccess: () => {
      build.reset();
      clearSelection();
      for (const key of ["history", "status", "search", "health", "duplicates", "review", "metadata"]) {
        void client.invalidateQueries({ queryKey: [key] });
      }
    },
  });
  const build = useMutation({
    mutationFn: buildOrganizationPreview,
    onSuccess: () => setOffset(0),
  });
  const preview = build.data;
  const page = useQuery({
    queryKey: ["organization", preview?.id, status, view, offset],
    queryFn: () => fetchOrganizationPage(preview!.id, status, view, offset),
    enabled: !!preview,
    retry: false,
  });

  function chooseView(next: OrganizationView) {
    setView(next);
    if (next === "folders" && (status === "needs_review" || status === "collision")) setStatus("all");
    setOffset(0);
  }

  const filters = view === "files" ? STATUSES : (["ready", "unchanged"] as const);
  const data = page.data;
  const busy = build.isPending || review.isPending || apply.isPending;

  function clearSelection() {
    setSelection([]);
    setConfirmed(false);
    review.reset();
  }

  function toggle(path: string) {
    setSelection((old) => old.includes(path) ? old.filter((selected) => selected !== path) : [...old, path]);
    setConfirmed(false);
    review.reset();
    apply.reset();
  }

  return (
    <div className="space-y-5">
      <PageHeader eyebrow={preview?.apply_enabled ? "Disposable development library" : "Preview only"} title="Library Organization"
        description="Preview artist and album folders from embedded tags. Review conflicts before organizing your files."
        actions={<Button tone="primary" disabled={busy} onClick={() => { clearSelection(); apply.reset(); build.mutate(); }}>
          {build.isPending ? "Building preview…" : preview ? "Rebuild preview" : "Build preview"}
        </Button>} />
      {build.error && <ActionBanner tone="error" message={build.error.message} />}
      {build.isPending && <p role="status" className="text-sm text-primary-muted">Checking tags and destinations…</p>}
      {apply.data && (
        <Panel className="space-y-3 p-4">
          <h2 className="font-semibold">Organization results</h2>
          <p role="status" className="text-sm">
            {apply.data.operations.filter((item) => item.status === "ok").length} of {apply.data.operations.length} files moved
          </p>
          <ul className="space-y-2">
            {apply.data.operations.map((item) => (
              <li key={item.operation_id} className="space-y-1">
                <PathBlock value={item.status === "ok" ? item.destination_path : item.path} />
                <p className="break-words text-sm">{item.status === "ok" ? "Moved" : `${item.status === "skipped" ? "Skipped" : "Failed"} · ${item.message}`}</p>
              </li>
            ))}
          </ul>
          <Link className="text-sm text-accent underline"
            to={`/history?batch=${encodeURIComponent(apply.data.batch_id)}`}>Open History to inspect or undo</Link>
        </Panel>
      )}
      {!preview && !build.isPending && !build.error && !apply.data && (
        <EmptyState title="Preview your folder layout"
          description="Build a preview to see proposed paths, collisions, and tracks that need metadata review." />
      )}
      {preview && (
        <>
          <Panel className="space-y-3 p-4">
            <details className="space-y-1 text-xs text-primary-muted">
              <summary className="cursor-pointer">Library root</summary>
              <PathBlock value={preview.root} />
            </details>
            <dl className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              {STATUSES.map((key) => (
                <div key={key} className="rounded border border-white/10 p-2">
                  <dt className="text-primary-muted">{LABELS[key]}</dt>
                  <dd className="mt-1 font-mono text-sm text-primary">{formatNumber(preview.counts[key])}</dd>
                </div>
              ))}
            </dl>
            <div className="flex flex-wrap items-end gap-3">
              <div role="group" aria-label="Preview view" className="flex flex-wrap gap-2">
                <Button aria-pressed={view === "files"} tone={view === "files" ? "primary" : "secondary"}
                  onClick={() => chooseView("files")}>Files</Button>
                <Button aria-pressed={view === "folders"} tone={view === "folders" ? "primary" : "secondary"}
                  onClick={() => chooseView("folders")}>Folder tree</Button>
              </div>
              <label className="grid gap-1 text-xs text-primary-muted">
                Show status
                <select value={status} onChange={(event) => { setStatus(event.target.value as typeof status); setOffset(0); }}
                  className="rounded-[3px] border border-white/20 bg-surface-low px-3 py-2 text-sm text-primary">
                  <option value="all">All statuses</option>
                  {filters.map((key) => <option value={key} key={key}>{LABELS[key]}</option>)}
                </select>
              </label>
            </div>
            {view === "folders" && <p className="text-xs text-primary-muted">Folder tree shows ready and unchanged tracks.</p>}
            {preview.apply_enabled && (
              <div className="space-y-2">
                <p className="text-xs text-primary-muted">Ready indicates that a proposed path passed the organization checks.</p>
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm" aria-live="polite">{selection.length} selected · {MOVE_LIMIT} maximum</p>
                  <Button disabled={busy || !selection.length} onClick={() => {
                    setConfirmed(false);
                    apply.reset();
                    review.mutate({ id: preview.id, paths: selection });
                  }}>{review.isPending ? "Checking selection…" : "Review selected moves"}</Button>
                  {selection.length > 0 && <Button disabled={busy} onClick={() => { clearSelection(); apply.reset(); }}>Clear selection</Button>}
                </div>
              </div>
            )}
          </Panel>
          {review.error && <ActionBanner tone="error" message={review.error.message} />}
          {apply.error && <div className="space-y-2">
            <ActionBanner tone="error" message={apply.error.message} />
            <Link className="text-sm text-accent underline" to="/history">Inspect History</Link>
          </div>}
          {review.data && <Panel className="space-y-4 p-4">
            <h2 className="font-semibold">
              Review {review.data.moves.length} selected {review.data.moves.length === 1 ? "move" : "moves"}
            </h2>
            <ul className="space-y-3">
              {review.data.moves.map((move) => (
                <li key={move.source} className="grid min-w-0 gap-2 text-xs sm:grid-cols-2">
                  <div className="min-w-0">Source<PathBlock value={move.source} /></div>
                  <div className="min-w-0">Destination<PathBlock value={move.destination} /></div>
                </li>
              ))}
            </ul>
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" className="mt-1 accent-accent" checked={confirmed} disabled={busy}
                onChange={(event) => setConfirmed(event.target.checked)} />
              I approve moving {review.data.moves.length === 1 ? "this file" : "these files"} to the listed destinations.
            </label>
            <Button tone="danger" disabled={!confirmed || busy || !!apply.error} onClick={() => {
              if (review.data && confirmed) apply.mutate({ id: preview.id, paths: selection, approval: review.data.approval });
            }}>{apply.isPending ? "Moving files…" : "Apply approved moves"}</Button>
          </Panel>}
          {page.isPending && <p role="status" className="text-sm text-primary-muted">Loading preview page…</p>}
          {page.error && <ActionBanner tone="error" message={page.error.message} />}
          {data && data.total === 0 && <EmptyState title="No entries in this view"
            description="Choose another status or check the file list for tracks that need review." />}
          {data && data.total > 0 && (
            <Panel className="overflow-hidden">
              {view === "files" ? <Files rows={data.rows} selection={selection} busy={busy}
                toggle={preview.apply_enabled ? toggle : undefined} /> : <Folders rows={data.rows} />}
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/10 p-4">
                <p role="status" className="text-xs text-primary-muted">
                  {formatNumber(offset + 1)}–{formatNumber(offset + data.rows.length)} of {formatNumber(data.total)}
                </p>
                {data.total > data.page_size && (
                  <div className="flex gap-2">
                    <Button disabled={offset === 0} onClick={() => setOffset(offset - data.page_size)}>Previous page</Button>
                    <Button disabled={offset + data.page_size >= data.total} onClick={() => setOffset(offset + data.page_size)}>Next page</Button>
                  </div>
                )}
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
