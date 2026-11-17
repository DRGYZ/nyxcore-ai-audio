export type ReleaseKind = "release" | "release-group";

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
  evidence: Record<"title" | "artist" | "album" | "duration" | "filename", "agrees" | "similar" | "differs" | "unavailable">;
  score: number;
  releases: ReleaseChoice[];
  source: MusicSource;
};

export type LookupQuery = {
  title: string; artist: string; album: string; duration_ms: number | null;
  filename_title: string; filename_artist: string;
};
export type LookupResult = { candidates: Recording[]; status: "matches" | "no_match" | "ambiguous" };

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
