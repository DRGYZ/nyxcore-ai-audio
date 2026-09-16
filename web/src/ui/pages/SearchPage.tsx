import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useArchiveSearchQuery } from "../../lib/hooks";
import { ActionBanner, Button, Chip, EmptyState, Icon, PageHeader, Panel, PathBlock, formatBytes, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";

function formatDuration(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "Unknown duration";
  const totalSeconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function SearchPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const query = (searchParams.get("q") ?? "").trim();
  const [draft, setDraft] = useState(query);
  const searchQuery = useArchiveSearchQuery(query);

  useEffect(() => {
    setDraft(query);
  }, [query]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextQuery = draft.trim();
    if (nextQuery.length < 2) return;
    const params = new URLSearchParams({ q: nextQuery });
    navigate(`/search?${params.toString()}`);
  }

  const response = searchQuery.data;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Read-only Lookup"
        title="Archive Search"
        description="Find tracks by filename, title, artist, album, album artist, genre, or folder without modifying audio files."
      />

      <Panel className="p-4">
        <form className="flex flex-col gap-2.5 sm:flex-row" onSubmit={handleSubmit} role="search">
          <label className="relative min-w-0 flex-1">
            <span className="sr-only">Search the music archive</span>
            <Icon name="search" className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-sm text-primary-subtle" />
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              className="w-full rounded-[3px] border border-white/[0.08] bg-surface-low py-2 pl-9 pr-3.5 font-sans text-xs text-primary placeholder:text-primary-subtle focus:border-accent/60 focus:outline-none focus:ring-1 focus:ring-accent/40"
              placeholder="Search by title, artist, album, genre, or filename…"
              type="search"
              minLength={2}
              maxLength={200}
              required
            />
          </label>
          <Button type="submit" tone="primary" disabled={draft.trim().length < 2 || searchQuery.isFetching}>
            <Icon name="manage_search" className="text-base" />
            {searchQuery.isFetching ? "Scanning…" : "Search Library"}
          </Button>
        </form>
      </Panel>

      {!query ? (
        <EmptyState
          title="Search your local archive"
          description="Enter at least two characters. NyxCore reads the current library metadata directly from disk."
        />
      ) : searchQuery.isPending ? (
        <ActionBanner tone="info" message={`Scanning the local library for “${query}”…`} />
      ) : searchQuery.isError ? (
        <ApiUnavailableState contextLabel="Archive Search" />
      ) : response && response.items.length === 0 ? (
        <EmptyState
          title={`No matches for “${response.query}”`}
          description="Try fewer words, a partial filename, or another artist, album, or genre."
        />
      ) : response ? (
        <div className="space-y-3.5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="font-sans text-xs text-primary-subtle">
              Found <span className="font-mono font-semibold text-accent">{formatNumber(response.total_matches)}</span> matches for <span className="font-editorial italic text-primary">“{response.query}”</span>
            </p>
            <span className="font-mono text-[11px] text-primary-subtle">{response.returned_count} shown</span>
          </div>

          <div className="divide-y divide-white/[0.06] rounded-[3px] border border-white/[0.07] bg-surface">
            {response.items.map((track) => (
              <div key={track.path} className="p-4 transition-colors hover:bg-surface-raised/40">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start gap-3">
                      <div className="flex size-7 shrink-0 items-center justify-center rounded-[2px] border border-white/[0.08] bg-surface-low text-accent">
                        <Icon name="audio_file" className="text-sm" />
                      </div>
                      <div className="min-w-0">
                        <h2 className="break-words font-sans text-xs font-semibold text-primary">{track.title || track.filename}</h2>
                        <p className="mt-0.5 font-editorial text-xs italic text-primary-subtle">
                          {track.artist || "Unknown artist"}{track.album ? ` · ${track.album}` : ""}
                        </p>
                      </div>
                    </div>
                    <div className="mt-2.5">
                      <PathBlock value={track.path} />
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-wrap gap-1.5 lg:max-w-xs lg:justify-end">
                    {track.match_fields.map((field) => (
                      <Chip key={field} tone="default">
                        {field.replace("albumartist", "album artist")}
                      </Chip>
                    ))}
                    {track.warnings.length > 0 ? (
                      <Chip tone="warning">{track.warnings.length} warnings</Chip>
                    ) : null}
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 border-t border-white/[0.04] pt-2.5 font-mono text-[11px] text-primary-subtle">
                  <span>{formatDuration(track.duration_seconds)}</span>
                  <span>•</span>
                  <span>{formatBytes(track.file_size_bytes)}</span>
                  <span>•</span>
                  <span className={track.has_cover_art ? "text-accent" : "text-primary-subtle"}>
                    {track.has_cover_art ? "Artwork embedded" : "No artwork"}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {response.total_matches > response.returned_count ? (
            <p className="text-center font-mono text-xs text-primary-subtle">Showing the first {response.returned_count} of {response.total_matches} ranked matches.</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
