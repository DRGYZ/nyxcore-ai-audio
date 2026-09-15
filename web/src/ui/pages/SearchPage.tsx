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
        eyebrow="Read-only lookup"
        title="Archive Search"
        description="Find tracks by filename, title, artist, album, album artist, genre, or folder. A scan runs only when you submit a search."
      />

      <Panel className="p-5">
        <form className="flex flex-col gap-3 sm:flex-row" onSubmit={handleSubmit} role="search">
          <label className="relative min-w-0 flex-1">
            <span className="sr-only">Search the music archive</span>
            <Icon name="search" className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              className="w-full rounded-lg border border-border-dark bg-background-dark py-2.5 pl-10 pr-4 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-primary focus:ring-1 focus:ring-primary"
              placeholder="Try a title, artist, album, genre, or filename"
              type="search"
              minLength={2}
              maxLength={200}
              required
            />
          </label>
          <Button type="submit" tone="primary" disabled={draft.trim().length < 2 || searchQuery.isFetching}>
            <Icon name="manage_search" className="text-lg" />
            {searchQuery.isFetching ? "Scanning…" : "Search Library"}
          </Button>
        </form>
      </Panel>

      {!query ? (
        <EmptyState
          title="Search your local archive"
          description="Enter at least two characters. NyxCore reads the current library metadata from disk and does not modify audio files."
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
        <div className="space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-slate-400">
              Found <span className="font-bold text-primary">{formatNumber(response.total_matches)}</span> matches for “{response.query}”
            </p>
            <Chip tone="primary">{response.returned_count} shown</Chip>
          </div>
          {response.items.map((track) => (
            <Panel key={track.path} className="p-5">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex items-start gap-3">
                    <span className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                      <Icon name="audio_file" />
                    </span>
                    <div className="min-w-0">
                      <h2 className="break-words font-display text-lg font-bold text-slate-100">{track.title || track.filename}</h2>
                      <p className="mt-1 text-sm text-slate-400">
                        {track.artist || "Unknown artist"}{track.album ? ` · ${track.album}` : ""}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4"><PathBlock value={track.path} /></div>
                </div>
                <div className="flex shrink-0 flex-wrap gap-2 lg:max-w-xs lg:justify-end">
                  {track.match_fields.map((field) => <Chip key={field} tone="primary">{field.replace("albumartist", "album artist")}</Chip>)}
                  {track.warnings.length > 0 ? <Chip tone="warning">{track.warnings.length} warnings</Chip> : null}
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-primary/5 pt-4 text-xs text-slate-500">
                <span>{formatDuration(track.duration_seconds)}</span>
                <span>{formatBytes(track.file_size_bytes)}</span>
                <span>{track.has_cover_art ? "Artwork embedded" : "No embedded artwork"}</span>
              </div>
            </Panel>
          ))}
          {response.total_matches > response.returned_count ? (
            <p className="text-center text-xs text-slate-500">Showing the first {response.returned_count} of {response.total_matches} ranked matches.</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
