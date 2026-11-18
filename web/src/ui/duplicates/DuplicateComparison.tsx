import { useId } from "react";
import type { DuplicateKind } from "../../lib/duplicate-evidence";
import { libraryRelativePath, pathFilename } from "../../lib/library-path";
import type { DuplicateFileInfo, DuplicateGroup } from "../../lib/types";
import { Button, Chip, Icon, Panel, PathBlock, SleeveArt, formatBytes } from "../components";

const UNKNOWN = "Unavailable";
const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const score = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

export function formatDuplicateScore(value: number) {
  return score.format(value) + " / 1";
}

const matchReasons: Record<string, string> = {
  title_match_strong: "Very similar titles",
  title_match_good: "Similar titles",
  artist_match_strong: "Very similar artist names",
  artist_match_good: "Similar artist names",
  album_match: "Similar album tags",
  filename_match: "Similar filenames",
  bitrate_close: "Similar reported bitrates",
  file_size_close: "Similar file sizes",
};

function matchReason(reason: string) {
  const delta = /^duration_delta=(\d+(?:\.\d+)?)s$/.exec(reason);
  if (delta) {
    const seconds = Number(delta[1]);
    return seconds === 0 ? "Reported durations match at this precision" : "Duration differs by " + number.format(seconds) + (seconds === 1 ? " second" : " seconds");
  }
  return matchReasons[reason] ?? "Other reported evidence (" + reason + ")";
}

function format(file: DuplicateFileInfo) {
  return file.extension?.replace(/^\./, "").toUpperCase() || null;
}

function duration(file: DuplicateFileInfo) {
  return file.duration_seconds == null ? null : number.format(file.duration_seconds) + " s";
}

function bitrate(file: DuplicateFileInfo) {
  return file.bitrate_bps != null && file.bitrate_bps > 0 ? number.format(file.bitrate_bps / 1000) + " kbps" : null;
}

function metadataCount(file: DuplicateFileInfo) {
  return file.metadata_fields_present == null ? null : file.metadata_fields_present + " of 7 fields";
}

function preferenceReason(reason: string, file: DuplicateFileInfo) {
  switch (reason) {
    case "preferred_lossless_format": return "Preferred lossless extension" + (format(file) ? " (" + format(file) + ")" : "");
    case "highest_bitrate": return "Highest reported bitrate in the group" + (bitrate(file) ? " (" + bitrate(file) + ")" : "");
    case "richest_metadata": return "Most tracked metadata fields present" + (metadataCount(file) ? " (" + metadataCount(file) + ")" : "");
    case "has_cover_art": return "Embedded artwork is available in this copy";
    case "largest_file_size": return "Largest file in the group (" + formatBytes(file.file_size_bytes) + ")";
    case "deterministic_tie_break": return "Deterministic tie-break. NyxCore records a consistent choice without identifying a quality advantage.";
    default: return "Other reported preference (" + reason + ")";
  }
}

function FileCard({ file, index, preferred, root }: {
  file: DuplicateFileInfo;
  index: number;
  preferred: boolean;
  root?: string;
}) {
  const filename = pathFilename(file.path);
  const relative = libraryRelativePath(file.path, root);
  const folderLabel = relative !== null ? (relative.slice(0, -(filename.length + 1)) || "Library root") : null;
  const ext = format(file);

  return (
    <li className={`min-w-0 rounded-xl border p-4 transition-colors ${
      preferred
        ? "border-[#52b788]/40 bg-[#52b788]/[0.05] shadow-[inset_0_1px_0_rgba(82,183,136,0.12)]"
        : "border-white/[0.08] bg-surface-low/75"
    }`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] text-primary-subtle">File {index}</span>
          {preferred ? <Chip tone="success">Preferred copy</Chip> : <Chip>Other copy</Chip>}
        </div>
      </div>

      <h4 className="mt-2.5 font-display text-sm font-semibold text-primary [overflow-wrap:anywhere]">{filename}</h4>
      {folderLabel !== null ? (
        <p className="mt-1 font-mono text-[11px] leading-relaxed text-primary-muted [overflow-wrap:anywhere]">
          {folderLabel}
        </p>
      ) : null}

      <details className="mt-2">
        <summary className="cursor-pointer font-sans text-xs font-medium text-accent focus-visible:outline-accent" aria-label={"Full path for file " + index}>
          Full path
        </summary>
        <PathBlock className="mt-1.5 select-all" value={file.path} />
      </details>

      <dl className="mt-3.5 grid grid-cols-2 gap-x-3 gap-y-2.5 border-t border-white/[0.07] pt-3.5 text-xs">
        {[
          ["Format (extension)", ext],
          ["Size", formatBytes(file.file_size_bytes)],
          ["Duration", duration(file)],
          ["Reported bitrate", bitrate(file)],
          ["Artwork", file.has_cover_art === true ? "Embedded" : file.has_cover_art === false ? "None found" : null],
          ["Metadata", metadataCount(file)],
        ].map(([label, value]) => (
          <div key={label}>
            <dt className="text-[11px] text-primary-subtle">{label}</dt>
            <dd className="mt-0.5 font-medium text-primary [overflow-wrap:anywhere]">{value ?? UNKNOWN}</dd>
          </div>
        ))}
      </dl>
    </li>
  );
}

function pairRows(left: DuplicateFileInfo, right: DuplicateFileInfo) {
  function row(label: string, first: string | null | undefined, second: string | null | undefined, verdict?: string) {
    const difference = first && second ? verdict ?? (first === second ? "Same" : "Different") : "Incomplete data";
    return { label, first: first || UNKNOWN, second: second || UNKNOWN, difference };
  }
  const delta = left.duration_seconds != null && right.duration_seconds != null
    ? Math.abs(left.duration_seconds - right.duration_seconds)
    : null;
  return [
    row("Title", left.title, right.title),
    row("Artist", left.artist, right.artist),
    row("Album", left.album, right.album),
    row("Duration", duration(left), duration(right), delta === null ? undefined : number.format(delta) + " s apart"),
    row("Format", format(left), format(right)),
    row("Reported bitrate", bitrate(left), bitrate(right)),
  ];
}

function orderedFiles(group: DuplicateGroup) {
  const preferred = group.files.find((file) => file.path === group.preferred.path);
  return preferred ? [preferred, ...group.files.filter((file) => file !== preferred)] : group.files;
}

export function DuplicateFiles({ group, root, compact = false }: { group: DuplicateGroup; root?: string; compact?: boolean }) {
  return (
    <ul aria-label="Files in this group" className={"grid min-w-0 gap-3.5 " + (compact ? "" : group.files.length > 2 ? "sm:grid-cols-2 xl:grid-cols-3" : "sm:grid-cols-2")}>
      {orderedFiles(group).map((file, index) => (
        <FileCard key={file.path} file={file} index={index + 1} preferred={file.path === group.preferred.path} root={root} />
      ))}
    </ul>
  );
}

export function DuplicatePreference({ group, kind, root, showPath = false }: {
  group: DuplicateGroup;
  kind: DuplicateKind;
  root?: string;
  showPath?: boolean;
}) {
  const preferred = group.files.find((file) => file.path === group.preferred.path);
  const relative = preferred && libraryRelativePath(preferred.path, root);
  const Heading = showPath ? "h4" : "h3";
  return (
    <div className={showPath ? "border-l-2 border-[#52b788]/50 pl-3" : "rounded-xl border border-[#52b788]/25 bg-[#52b788]/[0.04] p-4"}>
      <Heading className="flex items-center gap-2 font-display text-sm font-semibold text-primary">
        <Icon name="recommend" className="text-[17px] text-[#7cd4aa]" />
        <span>Why NyxCore prefers this copy</span>
      </Heading>
      {preferred && showPath ? (
        <p className="mt-1.5 select-all font-mono text-xs text-primary-muted [overflow-wrap:anywhere]">{relative ?? preferred.path}</p>
      ) : null}
      {preferred && group.preferred.reasons.length ? (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-relaxed text-primary-muted">
          {group.preferred.reasons.map((reason) => <li key={reason}>{preferenceReason(reason, preferred)}</li>)}
        </ul>
      ) : <p className="mt-1.5 text-xs text-primary-muted">Recommendation details are unavailable.</p>}
      {!showPath || kind === "likely" ? (
        <p className="mt-2.5 text-xs leading-relaxed text-primary-subtle">
          {kind === "exact"
            ? "The preference identifies the copy to keep in place if you choose to move the others."
            : "Format, bitrate and file size alone cannot establish recording fidelity."}
        </p>
      ) : null}
    </div>
  );
}

export function PairEvidence({ group, id }: { group: DuplicateGroup; id?: string }) {
  const files = orderedFiles(group);
  const byPath = new Map(files.map((file, index) => [file.path, { file, number: index + 1 }]));
  const pairs = (group.relationships ?? []).flatMap((relationship) => {
    if (relationship.paths.length !== 2) return [];
    const [leftPath, rightPath] = relationship.paths;
    const left = byPath.get(leftPath);
    const right = byPath.get(rightPath);
    return left && right && leftPath !== rightPath ? [{ relationship, left, right }] : [];
  });

  return (
    <div id={id} className="scroll-mt-36 space-y-3">
      <div>
        <h3 className="font-display text-sm font-semibold text-primary">Reported pair evidence</h3>
        <p className="mt-1 text-xs leading-relaxed text-primary-muted">
          Similarity scores are heuristic values on a 0–1 scale and have no probability interpretation. A group can include files linked through another copy. Each comparison below is a directly reported pair.
        </p>
      </div>
      {pairs.length ? pairs.map(({ relationship, left, right }, index) => {
        const title = "File " + left.number + " ↔ File " + right.number;
        return (
          <details key={title} open={index === 0} className="min-w-0 overflow-hidden rounded-xl border border-white/[0.08] bg-surface-low/70">
            <summary className="flex cursor-pointer flex-wrap items-center justify-between gap-2 px-4 py-3 text-sm text-primary hover:bg-white/[0.02] focus-visible:outline-accent">
              <span className="font-display font-semibold">{title}</span>
              <span className="rounded-full border border-accent/30 bg-accent/12 px-2.5 py-0.5 font-mono text-xs text-accent">
                Similarity score {formatDuplicateScore(relationship.confidence)}
              </span>
            </summary>
            <div className="space-y-4 border-t border-white/[0.08] p-4">
              {relationship.reasons.length ? (
                <ul aria-label={"Match evidence for " + title} className="flex flex-wrap gap-2">
                  {relationship.reasons.map((reason) => (
                    <li key={reason} className="max-w-full [overflow-wrap:anywhere]">
                      <Chip>{matchReason(reason)}</Chip>
                    </li>
                  ))}
                </ul>
              ) : <p className="text-xs text-primary-muted">The detector recorded no reasons for this pair.</p>}

              <div className="overflow-x-auto rounded-lg border border-white/[0.07] bg-surface">
                <table aria-label={"Stored properties for " + title} className="w-full table-fixed text-left text-xs leading-relaxed">
                  <thead className="border-b border-white/[0.08] bg-surface-low/90 text-primary-muted">
                    <tr>
                      <th scope="col" className="w-[30%] px-3 py-2.5">Property</th>
                      {[left, right].map(({ file, number: fileNumber }) => (
                        <th key={file.path} scope="col" className="px-3 py-2.5 align-top">
                          <span className="text-primary">File {fileNumber}</span>
                          <span className="mt-0.5 block font-normal text-primary-subtle [overflow-wrap:anywhere]">{pathFilename(file.path)}</span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.06]">
                    {pairRows(left.file, right.file).map(({ label, first, second, difference }) => {
                      const differs = difference !== "Same";
                      return (
                        <tr key={label} className="align-top [overflow-wrap:anywhere]">
                          <th scope="row" className="px-3 py-2.5 font-medium text-primary-muted">
                            {label}
                            <span className={`mt-0.5 block text-[11px] font-normal ${differs ? "text-[#f2c86b]" : "text-primary-subtle"}`}>
                              {difference}
                            </span>
                          </th>
                          <td className="px-3 py-2.5 text-primary">{first}</td>
                          <td className="px-3 py-2.5 text-primary">{second}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        );
      }) : <p className="text-xs text-primary-muted">No file-to-file evidence is available for this group.</p>}
    </div>
  );
}

export function DuplicateComparison({ group, kind, root, onReview, matchedFinding }: {
  group: DuplicateGroup;
  kind: DuplicateKind;
  root?: string;
  onReview: () => void;
  matchedFinding: boolean;
}) {
  const id = useId();
  const pairEvidenceId = id + "-pair-evidence";
  const exact = kind === "exact";
  const files = orderedFiles(group);
  const lead = files[0];

  return (
    <Panel className={"overflow-hidden " + (exact ? "" : "border-l-4 border-l-[#e9b44c]/70")}>
      <article aria-labelledby={id}>
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08] bg-surface-low/85 p-4 sm:p-5">
          <div className="flex min-w-0 items-center gap-3.5">
            <SleeveArt
              title={lead?.title || (lead ? pathFilename(lead.path) : "Duplicate")}
              artist={lead?.artist}
              format={lead ? format(lead) : null}
              hasArtwork={lead?.has_cover_art}
              size="md"
              showVinyl
            />
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Chip tone={exact ? "success" : "warning"}>{exact ? "Exact duplicate" : "Likely duplicate"}</Chip>
                {!exact ? <Chip>Manual review only</Chip> : null}
                <span className="text-xs text-primary-subtle">{files.length} files</span>
              </div>
              <h2 id={id} className="mt-1.5 font-display text-base font-semibold text-primary [overflow-wrap:anywhere]">
                {lead ? pathFilename(lead.path) : "Duplicate group"}
              </h2>
            </div>
          </div>
          <Button tone="secondary" onClick={onReview}>
            {matchedFinding ? (exact ? "Review quarantine options" : "Open manual review") : "Find in Review Inbox"}
          </Button>
        </div>

        <div className="space-y-5 p-4 sm:p-5">
          <p className="text-sm leading-relaxed text-primary-muted">
            {exact
              ? "SHA-256 verified identical file bytes. These copies carry the same audio and embedded tags, even if their filenames differ."
              : "NyxCore found similarities in the reported file properties and tags. Verify the recordings before deciding whether to keep both copies."}
          </p>

          {exact && group.content_hash ? (
            <details className="rounded-lg border border-white/[0.07] bg-surface-low/50 px-3.5 py-2">
              <summary className="cursor-pointer font-sans text-xs font-medium text-accent focus-visible:outline-accent">SHA-256 digest</summary>
              <PathBlock className="mt-2 select-all" value={group.content_hash} />
            </details>
          ) : null}

          {!exact && group.relationships?.length ? (
            <a href={"#" + pairEvidenceId} className="inline-flex items-center gap-1.5 text-xs font-medium text-accent underline-offset-4 hover:underline focus-visible:outline-accent">
              <Icon name="compare_arrows" className="text-[15px]" />
              Jump to reported pair evidence
            </a>
          ) : null}

          <DuplicateFiles group={group} root={root} />
          <DuplicatePreference group={group} kind={kind} />
          {exact ? (
            <p className="text-xs leading-relaxed text-primary-subtle">
              Select each move in Review Inbox. Quarantine preserves the selected files and they still occupy disk space.
            </p>
          ) : <PairEvidence group={group} id={pairEvidenceId} />}
        </div>
      </article>
    </Panel>
  );
}
