import { pathFilename } from "./library-path";
import type { MetadataTrack } from "./metadata";

export type ReleaseKind = "release" | "release-group";
type Version = "live" | "demo" | "instrumental" | "remix" | "edit" | "acoustic" | "karaoke" | "studio" | "video" |
  "orchestral" | "single version";

export type MusicSource = {
  recording_id: string;
  title: string;
  artist: string;
  release: { kind: ReleaseKind; id: string; title: string } | null;
  token: string;
};

export type ReleaseChoice = NonNullable<MusicSource["release"]> & {
  country: string;
  date: string;
  source: MusicSource;
  group_id: string | null;
};

export type Recording = {
  id: string;
  title: string;
  artist: string;
  artists: { id: string; name: string }[];
  duration_ms: number | null;
  duration_delta_ms: number | null;
  version: Version | null;
  disambiguation: string;
  identity: "supported" | "uncertain" | "conflicting";
  version_context_agrees: boolean;
  evidence: Record<"title" | "artist" | "album" | "duration" | "filename", "agrees" | "similar" | "differs" | "unavailable">;
  text_score: number;
  releases: ReleaseChoice[];
  source: MusicSource;
};

export type LookupQuery = {
  title: string; artist: string; album: string; duration_ms: number | null;
  filename_title: string; filename_artist: string;
  version_hint: Version | "";
  version_detail: string;
};
export type LookupResult = { candidates: Recording[]; status: "matches" | "no_match" | "ambiguous" };
export type LinkedReleases = { releases: ReleaseChoice[]; next_page: number | null };

const SEPARATOR = /\s+[-–—]\s+/u;
const UPLOAD_LABEL = /(?:\s*[-–—]\s*Topic|\s+VEVO|\s*[-–—]\s*Official (?:Music )?Channel)$/iu;
const VIDEO_LABEL = /\s*[\[(]\s*(?:official\s+)?(?:music video|lyric(?:s| video)?(?:[ _/]letra)?|lyrics video|audio|visuali[sz]er)\s*[)\]]\s*$/iu;
const VERSION_WORDS = /\b(live|demo|instrumental|remix|edit|acoustic|karaoke|studio|orchestral|single version)\b/iu;

function words(value: string) {
  return value.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{M}\p{N}]+/gu, " ").trim();
}

function channelCredit(value: string) {
  const cleaned = value.replace(UPLOAD_LABEL, "").replace(/(?<=[\p{Ll}])VEVO$/u, "").trim();
  const parts = cleaned.split(/\s+[-–—]\s+|\s*[,;]\s*/u);
  return parts.length === 2 && words(parts[0]) === words(parts[1]) ? parts[0] : cleaned;
}

function searchVersion(value: string): { type: Version; detail: string } | undefined {
  const bracketed = [...value.matchAll(/[\[(]([^()\[\]]+)[)\]]/gu)].map((match) => match[1]);
  const suffix = value.match(/(?:\s+[-–—]\s+|\s+\|\s*)([^()\[\]|]+)$/u)?.[1];
  for (const qualifier of [suffix ?? "", ...bracketed]) {
    const version = qualifier.match(VERSION_WORDS)?.[1].toLowerCase();
    if (version) return { type: version as Version, detail: qualifier };
  }
}

export function musicSearchDefaults(track: MetadataTrack) {
  const filename = pathFilename(track.path).replace(/\.[^.]+$/, "");
  const originalTitle = track.suggestions.title.value ?? (track.tags.title[0] || filename);
  const originalArtist = track.suggestions.artist.value ?? track.tags.artist[0] ?? "";
  let title = originalTitle.replace(VIDEO_LABEL, "").trim();
  if (!track.tags.title.length && track.suggestions.title.source === "unavailable") {
    title = title.replace(/^\d{1,3}[.)_-]\s+/u, "");
  }
  let artist = channelCredit(originalArtist);
  let album = track.tags.album[0] ?? "";
  const parts = title.split(SEPARATOR);
  if (parts.length === 3 && words(parts[0]) === words(parts[1])) parts.splice(1, 1);
  if (parts.length === 2) {
    const [left, right] = parts;
    const rightCredit = right.match(/^(.+?)\s+\(([^()]+)\)$/u);
    const rightArtist = channelCredit(rightCredit?.[1] ?? right);
    const leftArtist = channelCredit(left);
    if (artist && words(rightArtist) === words(artist)) {
      title = left;
      const fileCredit = filename.match(/\s+\(([^()]+)\)$/u);
      if (!album && fileCredit && !searchVersion(filename) &&
        words(filename.slice(0, fileCredit.index)) === words(left + " " + rightArtist)) {
        album = fileCredit[1];
      }
    } else {
      const firstName = words(leftArtist);
      const song = words(right.replace(/\s+\/[^/]+\/\s*$/u, "").replace(/\s*\|\s*edit$/iu, ""));
      const prefix = words(filename);
      const expectedStart = words(left) + " " + song;
      const corroborates = song && (prefix === expectedStart || prefix.startsWith(expectedStart + " "));
      const uploadStyle = VIDEO_LABEL.test(filename) || Boolean(searchVersion(filename)) || artist !== originalArtist;
      // A conflicting credit needs a matching filename plus a clue that it came from a video or channel.
      if (firstName && (firstName === words(artist) || corroborates && (!artist || uploadStyle))) {
        title = right;
        artist = leftArtist;
      }
    }
  }
  const filenameParts = filename.split(SEPARATOR);
  const clue = searchVersion(title) ?? searchVersion(
    filenameParts.length === 2 && artist && words(filenameParts[1]) === words(artist) ? filenameParts[0] : filename,
  );
  const version: Version | "" = clue?.type ?? "";
  if (version === "edit") title = title.replace(/\s*\|\s*edit$/iu, "");
  return { title, artist, album, version, versionDetail: clue?.detail ?? "", originalTitle, originalArtist };
}

export type CoverChoice = { id: string; release_id: string; extension: string; token: string; thumbnail: string | null; label: string };
export type ArtworkReference = {
  digest: string;
  image_id: string;
  release_id: string;
  width: number;
  height: number;
  token: string;
  replace_existing: boolean;
};
export type SelectedCover = { selection: ArtworkReference; thumbnail: string | null };
export type ArtworkPreview = {
  before_present: boolean;
  before: string | null;
  after: string | null;
  release_id: string;
  digest: string;
};

export function musicBrainzLink(kind: "artist" | "recording" | ReleaseKind, id: string) {
  return `https://musicbrainz.org/${kind}/${encodeURIComponent(id)}`;
}
