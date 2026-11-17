export const METADATA_FIELDS = ["title", "artist", "album"] as const;
export type MetadataField = typeof METADATA_FIELDS[number];

export const FIELD_LABELS: Record<MetadataField, string> = { title: "Title", artist: "Artist", album: "Album" };
export const ISSUE_LABELS = {
  missing_title: "Missing Title", missing_artist: "Missing Artist", missing_album: "Missing Album",
  placeholder_metadata: "Placeholder Metadata", suspicious_swaps: "Suspicious Swaps",
};
export type MetadataIssue = keyof typeof ISSUE_LABELS;
export type TagValues = Partial<Record<MetadataField, string>>;
export type EmbeddedTags = Record<MetadataField, string[]>;

export type MetadataTrack = {
  path: string;
  tags: EmbeddedTags;
  supported: boolean;
  readable: boolean;
  write_blocker: string | null;
  issues: MetadataIssue[];
  suggestions: Record<MetadataField, {
    value: string | null;
    source: "swap" | "embedded" | "formatting" | "filename" | "unavailable";
    confidence: "matched" | "existing" | "high" | "tentative" | "ambiguous";
    evidence: string;
  }>;
  swap: boolean;
  generation: string;
  inspection: string;
  duration_ms?: number | null;
  cover_present?: boolean;
  artwork_supported?: boolean;
};

export type MetadataProposal = Pick<MetadataTrack, "path" | "generation" | "inspection"> & {
  values: TagValues;
  source?: MusicSource;
  source_fields?: MetadataField[];
  artwork?: ArtworkReference;
};
export type DraftSet = { entries: MetadataProposal[]; generation: string; revision: string };
export type PlanInput = { entries: MetadataProposal[] };
export type MetadataPlan = {
  preview: string;
  files: Array<{
    path: string; before: EmbeddedTags; changes: TagValues;
    source?: Omit<MusicSource, "token">; source_fields?: MetadataField[]; artwork?: ArtworkPreview;
  }>;
  review_only: boolean;
};
export const METADATA_DRAFTS_KEY = ["metadata", "drafts"] as const;
export const MAX_METADATA_FILES = 50;

export function tagValidation(value: string) {
  if (!value.trim()) return "Enter a value or revert this field. Clearing a tag is unavailable.";
  if (value.trim().toUpperCase() === "UNKNOWN") return "Enter a specific value in place of UNKNOWN.";
  if (value.trim().length > 512 || /[\p{Cc}\p{Cs}]/u.test(value)) return "Use up to 512 characters without control characters.";
  return null;
}
import type { ArtworkPreview, ArtworkReference, MusicSource } from "./enrichment";
