import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { applyOrganizationSelection, buildOrganizationPreview, fetchOrganizationPage, reviewOrganizationSelection } from "../../lib/api";
import type { OrganizationProposal, OrganizationStatus, OrganizationView } from "../../lib/organization";
import { ActionBanner, Button, Chip, EmptyState, Icon, PageHeader, Panel, PathBlock, formatNumber } from "../components";
import { MetadataEditButton } from "../metadata/MetadataEditor";

const LABELS = {
  ready: "Ready",
  needs_review: "Needs review",
  collision: "Collision",
  unchanged: "Unchanged",
} satisfies Record<OrganizationStatus, string>;
const STATUSES = Object.keys(LABELS) as OrganizationStatus[];
const MOVE_LIMIT = 50;

function statusChipTone(status: OrganizationStatus) {
  if (status === "ready") return "success" as const;
  if (status === "needs_review") return "warning" as const;
  if (status === "collision") return "danger" as const;
  return "neutral" as const;
}

function statusAccentBar(status: OrganizationStatus) {
  if (status === "ready") return "border-l-4 border-l-[#52b788]/70";
  if (status === "needs_review") return "border-l-4 border-l-[#e9b44c]/70";
  if (status === "collision") return "border-l-4 border-l-[#e06d53]/80";
  return "";
}

function Files({ rows, selection, busy, toggle }: {
  rows: OrganizationProposal[];
  selection: string[];
  busy: boolean;
  toggle?: (path: string) => void;
}) {
  return (
    <ul className="divide-y divide-white/[0.07]">
      {rows.map((row) => (
        <li key={row.path} className={`space-y-3 p-4 sm:p-5 ${statusAccentBar(row.status)}`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Chip tone={statusChipTone(row.status)}>
                {LABELS[row.status]}
              </Chip>
              <span className="text-xs leading-relaxed text-primary-muted">{row.reasons.join(" ")}</span>
            </div>
            {row.inspectable && (row.status === "needs_review" || row.status === "collision") && (
              <MetadataEditButton path={row.path} />
            )}
          </div>

          <dl className="grid min-w-0 gap-3 sm:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] sm:items-center">
            <div className="min-w-0">
              <dt className="mb-1 font-sans text-[11px] font-medium text-primary-subtle">Current path</dt>
              <dd><PathBlock value={row.source} /></dd>
            </div>
            <div aria-hidden="true" className="hidden pt-4 text-accent/70 sm:flex">
              <Icon name="arrow_forward" className="text-[18px]" />
            </div>
            <div className="min-w-0">
              <dt className="mb-1 font-sans text-[11px] font-medium text-primary-subtle">Proposed path</dt>
              <dd>
                {row.destination ? (
                  <PathBlock
                    value={row.destination.join(" / ")}
                    tone={row.status === "ready" ? "success" : row.status === "collision" ? "danger" : "default"}
                  />
                ) : (
                  <div className="rounded-lg border border-dashed border-white/[0.12] bg-surface-low/50 px-3 py-1.5 text-xs text-primary-subtle">
                    Awaiting review
                  </div>
                )}
              </dd>
            </div>
          </dl>

          {toggle && row.status === "ready" && (
            <label className="inline-flex items-center gap-2.5 rounded-lg border border-white/[0.08] bg-surface-low px-3 py-1.5 text-xs font-medium text-primary">
              <input
                type="checkbox"
                className="size-4 accent-accent"
                checked={selection.includes(row.path)}
                disabled={busy || (selection.length >= MOVE_LIMIT && !selection.includes(row.path))}
                onChange={() => toggle(row.path)}
              />
              <span>Select {row.source}</span>
            </label>
          )}
        </li>
      ))}
    </ul>
  );
}

function Folders({ rows }: { rows: OrganizationProposal[] }) {
  const folders = new Map<string, Map<string, OrganizationProposal[]>>();
  for (const row of rows) {
    if (!row.destination) continue;
    const [artist, album] = row.destination;
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
    files.push(row);
  }

  return (
    <ul aria-label="Proposed folders" className="space-y-3 p-4 text-sm [overflow-wrap:anywhere] sm:p-5">
      {[...folders].map(([artist, albums]) => (
        <li key={artist} className="rounded-xl border border-white/[0.08] bg-surface-low/65 p-3.5">
          <details open>
            <summary className="folder-toggle flex cursor-pointer items-center gap-2.5 font-display text-sm font-semibold text-primary">
              <Icon name="folder_shared" className="text-[18px] text-accent" />
              <span>{artist}</span>
              <span aria-hidden="true" className="ml-auto rounded-full bg-white/[0.05] px-2 py-0.5 font-sans text-[11px] font-normal text-primary-subtle">
                {albums.size} {albums.size === 1 ? "album" : "albums"}
              </span>
            </summary>
            <ul className="ml-3 mt-2.5 space-y-2.5 border-l border-white/[0.1] pl-3.5 sm:ml-4">
              {[...albums].map(([album, files]) => (
                <li key={album}>
                  <details open>
                    <summary className="folder-toggle flex cursor-pointer items-center gap-2 font-sans text-xs font-medium text-primary">
                      <Icon name="album" className="text-[16px] text-primary-muted" />
                      <span>{album}</span>
                      <span aria-hidden="true" className="ml-auto font-sans text-[11px] text-primary-subtle">
                        {files.length} {files.length === 1 ? "track" : "tracks"}
                      </span>
                    </summary>
                    <ul className="ml-3 mt-1.5 space-y-1 border-l border-white/[0.08] pl-3 font-mono text-xs text-primary-muted">
                      {files.map((row) => (
                        <li key={row.path} className="flex flex-wrap items-center gap-2 py-1">
                          <Icon name="music_note" className="shrink-0 text-[14px] text-accent/70" />
                          <span className="min-w-0 flex-1">{row.destination![2]}</span>
                          <Chip tone={statusChipTone(row.status)}>{LABELS[row.status]}</Chip>
                        </li>
                      ))}
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
    <div className="space-y-6">
      <PageHeader
        eyebrow={preview?.apply_enabled ? "Disposable development library" : "Preview only"}
        title="Library Organization"
        description="Preview clean artist and album folders derived from embedded tags, with safety and collision checks before anything moves."
        actions={
          <Button tone="primary" disabled={busy} onClick={() => { clearSelection(); apply.reset(); build.mutate(); }}>
            <Icon name="account_tree" className="text-base" />
            {build.isPending ? "Building preview…" : preview ? "Rebuild preview" : "Build preview"}
          </Button>
        }
      />
      {build.error && <ActionBanner tone="error" message={build.error.message} />}
      {build.isPending && <p role="status" className="text-sm text-primary-muted">Checking tags and destinations…</p>}

      {apply.data && (
        <Panel className="space-y-3 p-5">
          <h2 className="font-display text-lg font-semibold text-primary">Organization results</h2>
          <p role="status" className="text-sm text-primary-muted">
            {apply.data.operations.filter((item) => item.status === "ok").length} of {apply.data.operations.length} files moved
          </p>
          <ul className="space-y-2">
            {apply.data.operations.map((item) => (
              <li key={item.operation_id} className="space-y-1.5 rounded-lg border border-white/[0.07] bg-surface-low p-3">
                <PathBlock value={item.status === "ok" ? item.destination_path : item.path} />
                <p className="break-words text-sm text-primary">{item.status === "ok" ? "Moved" : `${item.status === "skipped" ? "Skipped" : "Failed"} · ${item.message}`}</p>
              </li>
            ))}
          </ul>
          <Link className="inline-block text-sm font-medium text-accent underline underline-offset-4" to={`/history?batch=${encodeURIComponent(apply.data.batch_id)}`}>
            Open History to inspect or undo
          </Link>
        </Panel>
      )}

      {!preview && !build.isPending && !build.error && !apply.data && (
        <EmptyState
          title="Preview your folder layout"
          description="Build a preview to see proposed paths, collisions, and tracks that need metadata review."
        />
      )}

      {preview && (
        <>
          <Panel className="overflow-hidden">
            <dl className="grid grid-cols-2 divide-x divide-y divide-white/[0.07] border-b border-white/[0.08] sm:grid-cols-4 sm:divide-y-0">
              {STATUSES.map((key) => {
                const badgeColor = key === "ready" ? "text-[#7cd4aa]" : key === "needs_review" ? "text-[#f2c86b]" : key === "collision" ? "text-[#f59f8a]" : "text-primary";
                return (
                  <div key={key} className="p-4">
                    <dt className="font-sans text-xs font-medium text-primary-muted">{LABELS[key]}</dt>
                    <dd className={`mt-1.5 font-display text-2xl font-semibold ${badgeColor}`}>{formatNumber(preview.counts[key])}</dd>
                  </div>
                );
              })}
            </dl>

            <div className="space-y-4 p-4 sm:p-5">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div className="inline-flex flex-wrap gap-1.5 rounded-xl border border-white/[0.08] bg-surface-low p-1" role="group" aria-label="Preview view">
                  <Button
                    aria-pressed={view === "files"}
                    tone={view === "files" ? "primary" : "ghost"}
                    className="border-0"
                    onClick={() => chooseView("files")}
                  >
                    <Icon name="view_list" className="text-[16px]" />
                    Files
                  </Button>
                  <Button
                    aria-pressed={view === "folders"}
                    tone={view === "folders" ? "primary" : "ghost"}
                    className="border-0"
                    onClick={() => chooseView("folders")}
                  >
                    <Icon name="folder_open" className="text-[16px]" />
                    Folder tree
                  </Button>
                </div>

                <label className="grid gap-1 text-xs font-medium text-primary-muted">
                  Show status
                  <select
                    value={status}
                    onChange={(event) => { setStatus(event.target.value as typeof status); setOffset(0); }}
                    className="rounded-lg border border-white/[0.14] bg-surface-low px-3 py-2 text-xs text-primary focus:border-accent focus:outline-none"
                  >
                    <option value="all">All statuses</option>
                    {filters.map((key) => <option value={key} key={key}>{LABELS[key]}</option>)}
                  </select>
                </label>
              </div>

              <details className="text-xs text-primary-muted">
                <summary className="cursor-pointer font-medium text-primary-muted hover:text-accent">Library root</summary>
                <div className="mt-2"><PathBlock value={preview.root} /></div>
              </details>

              {view === "folders" && <p className="text-xs text-primary-subtle">Folder tree shows ready and unchanged tracks.</p>}

              {preview.apply_enabled && (
                <div className="space-y-2 rounded-xl border border-white/[0.08] bg-surface-low/65 p-3.5">
                  <p className="text-xs text-primary-muted">Ready indicates that a proposed path passed the organization checks.</p>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <p className="text-sm font-medium text-primary" aria-live="polite">{selection.length} selected · {MOVE_LIMIT} maximum</p>
                    <Button
                      tone="secondary"
                      disabled={busy || !selection.length}
                      onClick={() => {
                        setConfirmed(false);
                        apply.reset();
                        review.mutate({ id: preview.id, paths: selection });
                      }}
                    >
                      {review.isPending ? "Checking selection…" : "Review selected moves"}
                    </Button>
                    {selection.length > 0 && (
                      <Button disabled={busy} onClick={() => { clearSelection(); apply.reset(); }}>Clear selection</Button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Panel>

          {review.error && <ActionBanner tone="error" message={review.error.message} />}
          {apply.error && (
            <div className="space-y-2">
              <ActionBanner tone="error" message={apply.error.message} />
              <Link className="text-sm font-medium text-accent underline underline-offset-4" to="/history">Inspect History</Link>
            </div>
          )}

          {review.data && (
            <Panel className="space-y-4 p-5">
              <h2 className="font-display text-base font-semibold text-primary">
                Review {review.data.moves.length} selected {review.data.moves.length === 1 ? "move" : "moves"}
              </h2>
              <ul className="space-y-3">
                {review.data.moves.map((move) => (
                  <li key={move.source} className="grid min-w-0 gap-3 rounded-xl border border-white/[0.07] bg-surface-low p-3 text-xs sm:grid-cols-2">
                    <div className="min-w-0 space-y-1"><span className="text-primary-subtle">Source</span><PathBlock value={move.source} /></div>
                    <div className="min-w-0 space-y-1"><span className="text-primary-subtle">Destination</span><PathBlock value={move.destination} tone="success" /></div>
                  </li>
                ))}
              </ul>
              <label className="flex items-start gap-2.5 text-sm text-primary">
                <input
                  type="checkbox"
                  className="mt-1 accent-accent"
                  checked={confirmed}
                  disabled={busy}
                  onChange={(event) => setConfirmed(event.target.checked)}
                />
                I approve moving {review.data.moves.length === 1 ? "this file" : "these files"} to the listed destinations.
              </label>
              <Button
                tone="danger"
                disabled={!confirmed || busy || !!apply.error}
                onClick={() => {
                  if (review.data && confirmed) apply.mutate({ id: preview.id, paths: selection, approval: review.data.approval });
                }}
              >
                {apply.isPending ? "Moving files…" : "Apply approved moves"}
              </Button>
            </Panel>
          )}

          {page.isPending && <p role="status" className="text-sm text-primary-muted">Loading preview page…</p>}
          {page.error && <ActionBanner tone="error" message={page.error.message} />}
          {data && data.total === 0 && (
            <EmptyState
              title="No entries in this view"
              description="Choose another status or check the file list for tracks that need review."
            />
          )}
          {data && data.total > 0 && (
            <Panel className="overflow-hidden">
              {view === "files" ? (
                <Files
                  rows={data.rows}
                  selection={selection}
                  busy={busy}
                  toggle={preview.apply_enabled ? toggle : undefined}
                />
              ) : (
                <Folders rows={data.rows} />
              )}
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.08] bg-surface-low/60 px-4 py-3">
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
