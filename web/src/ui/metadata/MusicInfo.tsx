import { useId, useState } from "react";
import type { KeyboardEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { findFrontCovers, findMusicInfo, selectFrontCover } from "../../lib/api";
import { musicBrainzLink } from "../../lib/enrichment";
import type { ArtworkReference, CoverChoice, MusicSource, SelectedCover } from "../../lib/enrichment";
import { pathFilename } from "../../lib/library-path";
import type { MetadataTrack } from "../../lib/metadata";
import { ActionBanner, Button, Chip, Icon } from "../components";

const INPUT = "w-full min-w-0 rounded-[3px] border border-white/20 bg-surface-low px-3 py-2 text-sm text-primary focus-visible:outline focus-visible:outline-accent";
const REFERENCE = "text-xs text-accent underline decoration-accent/50 underline-offset-4 [overflow-wrap:anywhere]";

function duration(value: number | null) {
  if (value === null) return "Duration unavailable";
  const seconds = Math.round(value / 1000);
  return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
}

export function MusicSourceLink({ source }: { source: Omit<MusicSource, "token"> }) {
  const release = source.release;
  return <a className={REFERENCE} target="_blank" rel="noreferrer"
    href={release ? musicBrainzLink(release.kind, release.id) : musicBrainzLink("recording", source.recording_id)}>
    {release ? release.title : source.title} on MusicBrainz
  </a>;
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
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState(track.suggestions.title.value ?? (track.tags.title[0] || pathFilename(track.path).replace(/\.[^.]+$/, "")));
  const [artist, setArtist] = useState(track.suggestions.artist.value ?? track.tags.artist[0] ?? "");
  const [recordingId, setRecordingId] = useState("");
  const [releaseKey, setReleaseKey] = useState("");
  const panel = useId();
  const search = useMutation({ mutationFn: findMusicInfo });
  const covers = useMutation({ mutationFn: findFrontCovers });
  const fetchCover = useMutation({ mutationFn: ({ source, choice }: { source: MusicSource; choice: CoverChoice }) =>
    selectFrontCover(source, choice) });
  const recording = search.data?.candidates.find((item) => item.id === recordingId);
  const release = recording?.releases.find((item) => item.kind + ":" + item.id === releaseKey);
  const selected = release?.source ?? recording?.source;
  const busy = search.isPending || covers.isPending || fetchCover.isPending;

  function forgetCoverResults() {
    covers.reset();
    fetchCover.reset();
  }

  function runSearch() {
    if (busy || !(title.trim() || artist.trim())) return;
    forgetCoverResults();
    setRecordingId("");
    setReleaseKey("");
    search.mutate({
      title: title.trim().slice(0, 160), artist: artist.trim().slice(0, 160),
      album: track.tags.album[0]?.slice(0, 160) ?? "", duration_ms: track.duration_ms ?? null,
      filename_title: track.suggestions.title.source === "filename" ? track.suggestions.title.value?.slice(0, 160) ?? "" : "",
      filename_artist: track.suggestions.artist.source === "filename" ? track.suggestions.artist.value?.slice(0, 160) ?? "" : "",
    });
  }

  function searchOnEnter(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" && !event.nativeEvent.isComposing) {
      event.preventDefault();
      runSearch();
    }
  }

  return (
    <section className="space-y-3 rounded-[3px] border border-white/10 bg-surface-low p-3 sm:p-4">
      <Button type="button" tone="secondary" aria-expanded={open} aria-controls={panel}
        disabled={busy} onClick={() => setOpen(!open)}>
        <Icon name="manage_search" className="text-base text-accent" /> Find Music Info
      </Button>
      <div id={panel} hidden={!open} className="space-y-4">
        <p className="text-xs leading-relaxed text-primary-muted">
          Search sends the title and artist you enter to MusicBrainz. Album, duration, and filename clues are compared locally.
          Cover lookup sends the selected release identifier to Cover Art Archive. Manual editing works offline.
        </p>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="space-y-1 text-xs text-primary"><span>Search title</span>
            <input className={INPUT} value={title} maxLength={160} disabled={busy} onKeyDown={searchOnEnter} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="space-y-1 text-xs text-primary"><span>Search artist</span>
            <input className={INPUT} value={artist} maxLength={160} disabled={busy} onKeyDown={searchOnEnter} onChange={(event) => setArtist(event.target.value)} />
          </label>
        </div>
        <Button type="button" tone="secondary" disabled={busy || !(title.trim() || artist.trim())} onClick={runSearch}>
          {search.isPending ? "Searching…" : "Search MusicBrainz"}
        </Button>
        {search.error && <ActionBanner tone="error" message={search.error.message} />}
        {search.data?.status === "no_match" && <ActionBanner tone="info" message="No recordings matched. Try another title or artist." />}
        {search.data?.candidates.length ? (
          <div className="space-y-2">
            <h4 className="font-display text-sm font-semibold text-primary">Recording candidates</h4>
            <p className="text-xs text-primary-muted">
              {search.data.status === "ambiguous" ? "Several matches may be plausible. " : ""}
              The match score compares your search clues. Check the recording and then choose its release.
            </p>
            <ul className="divide-y divide-white/10 border-y border-white/10">
              {search.data.candidates.map((item) => {
                const active = item.id === recordingId;
                const albums = item.releases.filter((choice) => choice.kind === "release");
                return (
                  <li key={item.id} className="space-y-2 py-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0 space-y-1 [overflow-wrap:anywhere]">
                        <p className="font-display text-sm font-semibold text-primary">{item.title}</p>
                        <p className="text-xs text-primary-muted">{item.artist || "Artist unavailable"} · {duration(item.duration_ms)}</p>
                      </div>
                      <Chip tone={active ? "accent" : "neutral"}>Match clues {item.score}%</Chip>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(item.evidence).map(([field, evidence]) => (
                        <Chip key={field} tone={evidence === "differs" ? "warning" : "neutral"}>{field} · {evidence}</Chip>
                      ))}
                    </div>
                    <p className="text-xs text-primary-muted [overflow-wrap:anywhere]">
                      {albums.length ? <>
                        {albums.length} {albums.length === 1 ? "release" : "releases"} · {albums.slice(0, 2).map((album) => album.title).join(" · ")}
                      </> : "Release information unavailable"}
                    </p>
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                      <Button type="button" aria-pressed={active} disabled={busy} onClick={() => {
                        setRecordingId(item.id);
                        setReleaseKey("");
                        forgetCoverResults();
                      }}>{active ? "Recording selected" : "Select recording"}</Button>
                      <a className={REFERENCE} target="_blank" rel="noreferrer" href={musicBrainzLink("recording", item.id)}>Recording details</a>
                      {active && item.artists.map((match) => (
                        <a className={REFERENCE} key={match.id} target="_blank" rel="noreferrer"
                          href={musicBrainzLink("artist", match.id)}>{match.name} artist details</a>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}
        {recording && selected && (
          <div className="space-y-3 border-l border-accent/40 pl-3">
            <label className="block space-y-1 text-xs text-primary">
              <span>Release or release group</span>
              <select className={INPUT} value={releaseKey} disabled={busy} onChange={(event) => {
                setReleaseKey(event.target.value);
                forgetCoverResults();
              }}>
                <option value="">Leave the album unchanged</option>
                {recording.releases.map((item) => <option key={item.kind + ":" + item.id} value={item.kind + ":" + item.id}>
                  {item.title} · {item.kind === "release-group" ? "Release group" : [item.date, item.country].filter(Boolean).join(" · ") || "Release"} · {item.id.slice(0, 8)}
                </option>)}
              </select>
            </label>
            <MusicSourceLink source={selected} />
            <div className="flex flex-wrap gap-2">
              <Button type="button" tone="primary" disabled={busy || !track.supported} onClick={() => onTags(selected)}>Use suggested tags</Button>
              {release && <Button type="button" tone="secondary" disabled={busy} onClick={() => covers.mutate(selected)}>
                {covers.isPending ? "Looking for covers…" : "Find front covers"}
              </Button>}
            </div>
            {covers.error && <ActionBanner tone="error" message={covers.error.message} />}
            {covers.data?.covers.length === 0 && <ActionBanner tone="info" message="No approved front cover was found for this selection." />}
            {covers.data?.covers.length ? (
              <div className="space-y-2">
                <h4 className="font-display text-sm font-semibold text-primary">Front covers</h4>
                <p className="text-xs text-primary-muted">Artwork retains its owners’ copyright. Check the source release before using a cover.</p>
                {!track.artwork_supported && <p className="text-xs text-primary-muted">This track’s tag layout is available for artwork inspection only.</p>}
                <ul className="grid gap-2 sm:grid-cols-2">
                  {covers.data.covers.map((choice, index) => (
                    <li key={choice.id} className="flex min-w-0 gap-2 rounded-[3px] border border-white/10 p-2">
                      {choice.thumbnail ? <img className="size-14 shrink-0 rounded-[2px] object-contain" src={choice.thumbnail} alt={`Front cover ${index + 1}`} />
                        : <span className="w-14 shrink-0 text-[10px] text-primary-muted">Preview unavailable</span>}
                      <div className="min-w-0 space-y-1.5">
                        <p className="text-xs text-primary [overflow-wrap:anywhere]">{choice.label || `Front cover ${index + 1}`}</p>
                        <a className={REFERENCE} target="_blank" rel="noreferrer" href={musicBrainzLink("release", choice.release_id)}>Source release</a>
                        <Button type="button" className="w-full px-1.5" disabled={busy || !track.artwork_supported}
                          onClick={() => fetchCover.mutate({ source: selected, choice }, { onSuccess: (cover) => {
                            onCover(selected, cover);
                          } })}>{artworkSourceToken === selected.token && artwork?.image_id === choice.id && artwork.release_id === choice.release_id
                            ? "Cover selected" : `Choose cover ${index + 1}`}</Button>
                      </div>
                    </li>
                  ))}
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
