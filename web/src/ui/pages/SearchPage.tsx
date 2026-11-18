import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useArchiveSearchQuery, useStatusQuery } from "../../lib/hooks";
import { ActionBanner, Button, Chip, EmptyState, Icon, PageHeader, Panel, PathBlock, SleeveArt, formatBytes, formatNumber } from "../components";
import { ApiUnavailableState } from "../feedback";
import { MetadataDraftLink, MetadataEditButton } from "../metadata/MetadataEditor";

function formatDuration(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "Unknown duration";
  const totalSeconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function trackFormat(filename: string) {
  const dot = filename.lastIndexOf(".");
  if (dot <= 0 || dot === filename.length - 1) return null;
  return filename.slice(dot + 1).toUpperCase();
}

export function SearchPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const query = (searchParams.get("q") ?? "").trim();
  const [draft, setDraft] = useState(query);
  const searchQuery = useArchiveSearchQuery(query);
  const reviewOnly = useStatusQuery().data?.review_only === true;

  useEffect(() => {
    setDraft(query);
  }, [query]);

  function submitQuery(rawQuery: string) {
    const nextQuery = rawQuery.trim();
    if (nextQuery.length < 2) return;
    const params = new URLSearchParams({ q: nextQuery });
    navigate(`/search?${params.toString()}`);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submitQuery(draft);
  }

  const response = searchQuery.data;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Catalog Discovery"
        title="Archive Search"
        description="Find tracks by title, artist, album, album artist, genre, filename, or folder without modifying audio files."
        actions={<MetadataDraftLink />}
      />

      <Panel className="space-y-3.5 p-4 sm:p-5">
        <form className="flex flex-col gap-2.5 sm:flex-row" onSubmit={handleSubmit} role="search">
          <label className="relative min-w-0 flex-1">
            <span className="sr-only">Search the music archive</span>
            <Icon name="search" className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[17px] text-primary-subtle" />
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              className="w-full rounded-lg border border-white/[0.12] bg-surface-low py-2.5 pl-10 pr-4 font-sans text-sm text-primary placeholder:text-primary-subtle focus:border-accent/60 focus:outline-none focus:ring-2 focus:ring-accent/30"
              placeholder="Search by title, artist, album, genre, or filename…"
              type="search"
              minLength={2}
              maxLength={200}
              required
            />
          </label>
          <Button type="submit" tone="primary" className="sm:px-5" disabled={draft.trim().length < 2 || searchQuery.isFetching}>
            <Icon name="manage_search" className="text-base" />
            {searchQuery.isFetching ? "Searching…" : "Search Library"}
          </Button>
        </form>

      </Panel>

      {!query ? (
        <EmptyState
          title="Search your local archive"
          description="Enter at least two characters to search the library metadata."
        />
      ) : searchQuery.isLoading ? (
        <ActionBanner tone="info" message={`Searching for “${query}”…`} />
      ) : searchQuery.isError ? (
        reviewOnly
          ? <ActionBanner tone="error" message="Search could not load. Try Refresh Library again." />
          : <ApiUnavailableState contextLabel="Archive Search" />
      ) : response && response.items.length === 0 ? (
        <EmptyState
          title={`No matches for “${response.query}”`}
          description="Try fewer words, a partial filename, or another artist, album, or genre."
        />
      ) : response ? (
        <div className="space-y-4">
          <div className="flex flex-col gap-2 px-1 sm:flex-row sm:items-center sm:justify-between">
            <p className="font-sans text-sm text-primary-muted">
              Found <strong className="font-semibold text-accent">{formatNumber(response.total_matches)}</strong>{" "}
              {response.total_matches === 1 ? "track" : "tracks"} matching{" "}
              <span className="font-editorial text-base italic text-primary">“{response.query}”</span>
            </p>
            <span className="font-sans text-xs text-primary-subtle">{response.returned_count} shown</span>
          </div>

          <Panel className="overflow-hidden">
            <div className="divide-y divide-white/[0.07]">
              {response.items.map((track) => {
                const format = trackFormat(track.filename);
                return (
                  <article
                    key={track.path}
                    className="group p-4 transition-colors hover:bg-surface-mid/45 sm:p-5"
                  >
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                      <div className="flex min-w-0 flex-1 items-start gap-4">
                        <SleeveArt
                          title={track.title || track.filename}
                          artist={track.artist}
                          format={format}
                          hasArtwork={track.has_cover_art}
                          size="md"
                          showVinyl
                        />
                        <div className="min-w-0 flex-1 space-y-2">
                          <div>
                            <div className="flex flex-wrap items-center gap-2">
                              <h2 className="break-words font-display text-base font-semibold text-primary">
                                {track.title || track.filename}
                              </h2>
                              {format ? (
                                <span className="rounded-md border border-white/[0.1] bg-surface-low px-1.5 py-0.5 font-mono text-[10px] font-semibold text-accent">
                                  {format}
                                </span>
                              ) : null}
                            </div>
                            <p className="mt-0.5 font-sans text-xs text-primary-muted">
                              <span className="font-medium text-primary/90">{track.artist || "Unknown artist"}</span>
                              {track.album ? <span className="text-primary-subtle"> · {track.album}</span> : null}
                            </p>
                          </div>

                          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-primary-subtle">
                            <span className="inline-flex items-center gap-1 font-mono text-[11px] text-primary-muted">
                              <Icon name="schedule" className="text-[14px] text-primary-subtle" />
                              {formatDuration(track.duration_seconds)}
                            </span>
                            <span className="inline-flex items-center gap-1 font-mono text-[11px] text-primary-muted">
                              <Icon name="hard_drive" className="text-[14px] text-primary-subtle" />
                              {formatBytes(track.file_size_bytes)}
                            </span>
                            <Chip tone={track.has_cover_art ? "success" : "neutral"}>
                              {track.has_cover_art ? "Artwork embedded" : "No artwork"}
                            </Chip>
                          </div>

                          <PathBlock value={track.path} />
                        </div>
                      </div>

                      <div className="flex shrink-0 flex-col items-start gap-2.5 lg:items-end">
                        <div className="flex flex-wrap gap-1.5 lg:max-w-xs lg:justify-end">
                          {track.match_fields.map((field) => (
                            <Chip key={field} tone="accent">
                              {field.replace("albumartist", "album artist")}
                            </Chip>
                          ))}
                          {track.warnings.length > 0 ? (
                            <Chip tone="warning">{track.warnings.length} warnings</Chip>
                          ) : null}
                        </div>
                        <MetadataEditButton path={track.path} />
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </Panel>

          {response.total_matches > response.returned_count ? (
            <p className="text-center font-sans text-xs text-primary-subtle">
              Showing the first {response.returned_count} of {response.total_matches} ranked matches.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
