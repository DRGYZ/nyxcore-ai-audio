import { useId, useState } from "react";
import type { KeyboardEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { findFrontCovers, findLinkedReleases, findMusicInfo, selectFrontCover } from "../../lib/api";
import { musicBrainzLink, musicSearchDefaults } from "../../lib/enrichment";
import type { ArtworkReference, CoverChoice, MusicSource, Recording, ReleaseChoice, SelectedCover } from "../../lib/enrichment";
import type { MetadataTrack } from "../../lib/metadata";
import { ActionBanner, Button, Chip, Icon } from "../components";

const INPUT = "w-full min-w-0 rounded-lg border border-white/[0.14] bg-surface px-3 py-2 text-sm text-primary placeholder:text-primary-subtle focus-visible:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:opacity-60";
const REFERENCE = "text-xs font-medium text-accent underline decoration-accent/45 underline-offset-4 hover:text-[#ebb375] [overflow-wrap:anywhere]";
const INITIAL_CANDIDATES = 4;
const IDENTITY = {
  supported: "Recording clues agree",
  uncertain: "Recording version uncertain",
  conflicting: "Recording clues conflict",
};

function releaseKeyOf(choice: ReleaseChoice) {
  return choice.kind + ":" + choice.id;
}

function releaseLabel(choice: ReleaseChoice) {
  const details = choice.kind === "release-group" ? "Release group" : [choice.date, choice.country].filter(Boolean).join(" · ") || "Release";
  return `${choice.title} · ${details} · ${choice.id.slice(0, 8)}`;
}

function limitText(value: string, maxLength: number) {
  return Array.from(value).slice(0, maxLength).join("");
}

function duration(value: number | null) {
  if (value === null) return "Duration unavailable";
  const seconds = Math.round(value / 1000);
  return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
}

function recordingDuration(item: Recording) {
  if (item.duration_delta_ms === null) return duration(item.duration_ms);
  const seconds = item.duration_delta_ms / 1000;
  const difference = seconds < 60 ? `${Number(seconds.toFixed(1))}s` : duration(item.duration_delta_ms);
  return `${duration(item.duration_ms)} · ${difference} difference`;
}

export function MusicSourceLink({ source }: { source: Omit<MusicSource, "token"> }) {
  const release = source.release;
  return (
    <a
      className={REFERENCE}
      target="_blank"
      rel="noreferrer"
      href={release ? musicBrainzLink(release.kind, release.id) : musicBrainzLink("recording", source.recording_id)}
    >
      {release ? release.title : source.title} on MusicBrainz
    </a>
  );
}

export function MusicInfo({
  track, artwork, artworkSourceToken, onTags, onCover,
}: {
  track: MetadataTrack;
  artwork?: ArtworkReference;
  artworkSourceToken?: string;
  onTags: (source: MusicSource) => void;
  onCover: (source: MusicSource, cover: SelectedCover) => void;
}) {
  const defaults = musicSearchDefaults(track);
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState(defaults.title);
  const [artist, setArtist] = useState(defaults.artist);
  const [recordingId, setRecordingId] = useState("");
  const [releaseKey, setReleaseKey] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [releaseFilter, setReleaseFilter] = useState("");
  const [linked, setLinked] = useState<{ choices: ReleaseChoice[]; nextPage: number | null }>({ choices: [], nextPage: 1 });
  const panel = useId();
  const candidateList = useId();
  const search = useMutation({ mutationFn: findMusicInfo });
  const browse = useMutation({
    mutationFn: findLinkedReleases,
    onSuccess: (data) => {
      setLinked((previous) => ({ choices: [...previous.choices, ...data.releases], nextPage: data.next_page }));
    },
  });
  const covers = useMutation({ mutationFn: findFrontCovers });
  const fetchCover = useMutation({
    mutationFn: ({ source, choice }: { source: MusicSource; choice: CoverChoice }) => selectFrontCover(source, choice),
  });
  const candidates = search.data?.candidates ?? [];
  const recording = candidates.find((item) => item.id === recordingId);
  const available = new Map((recording?.releases ?? []).map((choice) => [releaseKeyOf(choice), choice]));
  for (const choice of linked.choices) {
    const key = releaseKeyOf(choice);
    if (!available.has(key)) available.set(key, choice);
  }
  const allReleases = [...available.values()];
  const release = allReleases.find((item) => releaseKeyOf(item) === releaseKey);
  const selected = release?.source ?? recording?.source;
  const busy = search.isPending || browse.isPending || covers.isPending || fetchCover.isPending;

  function forgetCoverResults() {
    covers.reset();
    fetchCover.reset();
  }

  function selectRecording(id: string) {
    setRecordingId(id);
    setReleaseKey("");
    setReleaseFilter("");
    setLinked({ choices: [], nextPage: 1 });
    browse.reset();
    forgetCoverResults();
  }

  function runSearch() {
    if (busy || !(title.trim() || artist.trim())) return;
    selectRecording("");
    setShowAll(false);
    search.mutate({
      title: limitText(title.trim(), 160),
      artist: limitText(artist.trim(), 160),
      album: limitText(defaults.album, 160),
      duration_ms: track.duration_ms ?? null,
      filename_title: track.suggestions.title.source === "filename" ? limitText(track.suggestions.title.value ?? "", 160) : "",
      filename_artist: track.suggestions.artist.source === "filename" ? limitText(track.suggestions.artist.value ?? "", 160) : "",
      version_hint: defaults.version,
      version_detail: limitText(defaults.versionDetail, 100),
    });
  }

  function searchOnEnter(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" && !event.nativeEvent.isComposing) {
      event.preventDefault();
      runSearch();
    }
  }

  return (
    <section className="space-y-3.5 rounded-xl border border-white/[0.1] bg-surface-low/80 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Button
          type="button"
          tone="secondary"
          aria-expanded={open}
          aria-controls={panel}
          disabled={busy}
          onClick={() => setOpen(!open)}
        >
          <Icon name="album" className="text-base text-accent" /> Find Music Info
        </Button>
        <span className="font-sans text-xs text-primary-subtle">MusicBrainz &amp; Cover Art Archive lookup</span>
      </div>

      <div id={panel} hidden={!open} className="space-y-4 border-t border-white/[0.07] pt-3.5">
        <p className="text-xs leading-relaxed text-primary-muted">
          Search sends the title and artist you enter to MusicBrainz. Album, duration, and filename clues are compared locally.
          Cover lookup sends the selected release identifier to Cover Art Archive. Manual editing works offline.
        </p>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1.5 text-xs font-medium text-primary">
            <span>Search title</span>
            <input className={INPUT} value={title} maxLength={160} disabled={busy} onKeyDown={searchOnEnter} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="space-y-1.5 text-xs font-medium text-primary">
            <span>Search artist</span>
            <input className={INPUT} value={artist} maxLength={160} disabled={busy} onKeyDown={searchOnEnter} onChange={(event) => setArtist(event.target.value)} />
          </label>
        </div>

        {(defaults.title !== defaults.originalTitle || defaults.artist !== defaults.originalArtist) && (
          <details className="rounded-lg border border-white/[0.07] bg-surface p-2.5 text-xs text-primary-muted">
            <summary className="cursor-pointer font-medium text-primary">Original search text</summary>
            <p className="mt-1.5 [overflow-wrap:anywhere]">{defaults.originalTitle}{defaults.originalArtist && ` · ${defaults.originalArtist}`}</p>
            <button
              className={`${REFERENCE} mt-1 inline-block`}
              type="button"
              disabled={busy}
              onClick={() => { setTitle(defaults.originalTitle); setArtist(defaults.originalArtist); }}
            >
              Use original search text
            </button>
          </details>
        )}

        <Button type="button" tone="secondary" disabled={busy || !(title.trim() || artist.trim())} onClick={runSearch}>
          <Icon name="search" className="text-[16px] text-accent" />
          {search.isPending ? "Searching…" : "Search MusicBrainz"}
        </Button>

        {search.error && <ActionBanner tone="error" message={search.error.message} />}
        {search.data?.status === "no_match" && <ActionBanner tone="info" message="No recordings matched. Try another title or artist." />}

        {candidates.length ? (
          <div className="space-y-3">
            <div>
              <h4 className="font-display text-sm font-semibold text-primary">Recording candidates</h4>
              <p className="mt-1 text-xs leading-relaxed text-primary-muted">
                {search.data?.status === "ambiguous" ? "Several matches may be plausible. " : ""}
                Text match compares title and artist. Check the version, duration, and release before choosing.
                {track.duration_ms && <> This file has a duration of {duration(track.duration_ms)}.</>}
                {defaults.versionDetail && <> File version clue · {defaults.versionDetail}.</>}
              </p>
            </div>
            <ul id={candidateList} className="space-y-2.5">
              {(showAll ? candidates : candidates.slice(0, INITIAL_CANDIDATES)).map((item) => {
                const active = item.id === recordingId;
                const albums = item.releases.filter((choice) => choice.kind === "release");
                return (
                  <li
                    key={item.id}
                    className={`space-y-2.5 rounded-xl border p-3.5 transition-colors ${
                      active ? "border-accent/45 bg-accent/[0.06]" : "border-white/[0.08] bg-surface"
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0 space-y-0.5 [overflow-wrap:anywhere]">
                        <p className="font-display text-sm font-semibold text-primary">{item.title}</p>
                        <p className="text-xs text-primary-muted">{item.artist || "Artist unavailable"} · {recordingDuration(item)}</p>
                      </div>
                      <Chip tone={active ? "accent" : "neutral"}>Text match {item.text_score}%</Chip>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {item.version && <Chip tone="neutral"><span className="capitalize">{item.version}</span></Chip>}
                      <Chip tone={item.identity === "conflicting" ? "warning" : "neutral"}>{IDENTITY[item.identity]}</Chip>
                      {item.evidence.album === "agrees" && <Chip tone="success">Album clue agrees</Chip>}
                      {item.version_context_agrees && <Chip tone="success">Version clue agrees</Chip>}
                    </div>
                    {item.disambiguation && <p className="text-xs text-primary-muted [overflow-wrap:anywhere]">{item.disambiguation}</p>}
                    <p className="text-xs text-primary-subtle [overflow-wrap:anywhere]">
                      {albums.length ? <>
                        Releases · {[...new Set(albums.map((album) => album.title))].slice(0, 2).join(" · ")}
                      </> : "Release information unavailable"}
                    </p>
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                      <Button type="button" tone={active ? "primary" : "secondary"} aria-pressed={active} disabled={busy} onClick={() => selectRecording(item.id)}>
                        {active ? "Recording selected" : "Select recording"}
                      </Button>
                      <a className={REFERENCE} target="_blank" rel="noreferrer" href={musicBrainzLink("recording", item.id)}>Recording details</a>
                      {active && item.artists.map((match) => (
                        <a
                          className={REFERENCE}
                          key={match.id}
                          target="_blank"
                          rel="noreferrer"
                          href={musicBrainzLink("artist", match.id)}
                        >
                          {match.name} artist details
                        </a>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
            {candidates.length > INITIAL_CANDIDATES && (
              <Button
                type="button"
                tone="secondary"
                aria-expanded={showAll}
                aria-controls={candidateList}
                onClick={() => setShowAll(!showAll)}
              >
                {showAll ? "Show first four candidates" : `Show all ${candidates.length} candidates`}
              </Button>
            )}
          </div>
        ) : null}

        {recording && selected && (
          <div className="space-y-3.5 rounded-xl border border-accent/35 bg-surface p-4">
            {allReleases.length > 6 && (
              <label className="block space-y-1 text-xs font-medium text-primary">
                <span>Filter available releases</span>
                <input className={INPUT} value={releaseFilter} onChange={(event) => setReleaseFilter(event.target.value)} />
              </label>
            )}
            <label className="block space-y-1.5 text-xs font-medium text-primary">
              <span>Release or release group</span>
              <select
                className={INPUT}
                value={releaseKey}
                disabled={busy}
                onChange={(event) => {
                  setReleaseKey(event.target.value);
                  forgetCoverResults();
                }}
              >
                <option value="">Leave the album unchanged</option>
                {allReleases
                  .filter((item) => releaseKeyOf(item) === releaseKey ||
                    releaseLabel(item).toLowerCase().includes(releaseFilter.trim().toLowerCase()))
                  .map((item) => (
                    <option key={releaseKeyOf(item)} value={releaseKeyOf(item)}>{releaseLabel(item)}</option>
                  ))}
              </select>
            </label>

            {linked.nextPage !== null && (
              <div className="space-y-1">
                <Button
                  type="button"
                  tone="secondary"
                  disabled={busy}
                  onClick={() => linked.nextPage && browse.mutate({ source: recording.source, page: linked.nextPage })}
                >
                  {browse.isPending ? "Loading releases…" : "Load more linked releases"}
                </Button>
                <p className="text-xs text-primary-subtle">Loads a page from MusicBrainz using the selected recording ID.</p>
              </div>
            )}

            {browse.isSuccess && <p role="status" className="text-xs text-primary-muted">{allReleases.length} release choices loaded.</p>}
            {browse.error && <ActionBanner tone="error" message={browse.error.message} />}
            <MusicSourceLink source={selected} />

            <div className="flex flex-wrap gap-2">
              <Button type="button" tone="primary" disabled={busy || !track.supported} onClick={() => onTags(selected)}>
                Use suggested tags
              </Button>
              {release && (
                <Button type="button" tone="secondary" disabled={busy} onClick={() => covers.mutate(selected)}>
                  {covers.isPending ? "Looking for covers…" : "Find front covers"}
                </Button>
              )}
            </div>

            {covers.error && <ActionBanner tone="error" message={covers.error.message} />}
            {covers.data?.covers.length === 0 && <ActionBanner tone="info" message="No approved front cover was found for this selection." />}

            {covers.data?.covers.length ? (
              <div className="space-y-2.5 border-t border-white/[0.08] pt-3">
                <div>
                  <h4 className="font-display text-sm font-semibold text-primary">Front covers</h4>
                  <p className="text-xs text-primary-muted">Artwork retains its owners’ copyright. Check the source release before using a cover.</p>
                  {!track.artwork_supported && (
                    <p className="mt-1 text-xs text-primary-muted">This track’s tag layout is available for artwork inspection only.</p>
                  )}
                </div>
                <ul className="grid gap-3 sm:grid-cols-2">
                  {covers.data.covers.map((choice, index) => {
                    const isChosen = artworkSourceToken === selected.token && artwork?.image_id === choice.id && artwork.release_id === choice.release_id;
                    return (
                      <li
                        key={choice.id}
                        className={`flex min-w-0 gap-3 rounded-xl border p-3 ${
                          isChosen ? "border-accent/50 bg-accent/[0.07]" : "border-white/[0.08] bg-surface-low"
                        }`}
                      >
                        {choice.thumbnail ? (
                          <img className="size-16 shrink-0 rounded-lg border border-white/[0.12] object-cover shadow-sleeve" src={choice.thumbnail} alt={`Front cover ${index + 1}`} />
                        ) : (
                          <span className="flex size-16 shrink-0 items-center justify-center rounded-lg border border-white/[0.08] bg-surface text-center text-[10px] text-primary-muted">
                            Preview unavailable
                          </span>
                        )}
                        <div className="min-w-0 flex-1 space-y-1.5">
                          <p className="text-xs font-medium text-primary [overflow-wrap:anywhere]">{choice.label || `Front cover ${index + 1}`}</p>
                          <a className={REFERENCE} target="_blank" rel="noreferrer" href={musicBrainzLink("release", choice.release_id)}>Source release</a>
                          <Button
                            type="button"
                            tone={isChosen ? "primary" : "secondary"}
                            className="w-full px-2"
                            disabled={busy || !track.artwork_supported}
                            onClick={() => fetchCover.mutate({ source: selected, choice }, {
                              onSuccess: (cover) => {
                                onCover(selected, cover);
                              },
                            })}
                          >
                            {isChosen ? "Cover selected" : `Choose cover ${index + 1}`}
                          </Button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : null}
            {fetchCover.error && <ActionBanner tone="error" message={fetchCover.error.message} />}
          </div>
        )}
      </div>
    </section>
  );
}
