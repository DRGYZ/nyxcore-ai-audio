import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import type { DuplicateReport } from "../../lib/types";
import { renderWithProviders } from "../../test/test-utils";
import { DuplicatesPage } from "../pages/DuplicatesPage";

const root = "C:\\Music Library";
const meta = { music_path: root, out_path: "C:\\Reports", active_profile: "default", mode: "live" };
const directory = "録音 Archives " + "Long folder ".repeat(6).trim();
const exact = {
  path: root + "\\Artists\\Blue Hour.flac", extension: ".flac", file_size_bytes: 1024,
  bitrate_bps: 320000, duration_seconds: 6, has_cover_art: true, metadata_fields_present: 3,
};
const copy = { ...exact, path: root + "\\" + directory + "\\Blue Hour Édition.flac" };
const other = { ...exact, path: root + "\\Imports\\Blue Hour copy.flac" };

const preferred = {
  path: root + "\\Studio\\Night Drift.flac", extension: ".flac", file_size_bytes: 4096000, bitrate_bps: 480000,
  duration_seconds: 8, has_cover_art: true, metadata_fields_present: 5,
  title: "Night Drift", artist: "Signal Path", album: "Focus Studies",
};
const bridge = {
  path: root + "\\Imports\\Night Drift.mp3", extension: ".mp3", file_size_bytes: 3072000, bitrate_bps: 456000,
  duration_seconds: 8.4, has_cover_art: false, metadata_fields_present: 3,
  title: "Night Drift", artist: "Signal Paths", album: "Focus Studies",
};
const third = {
  path: "D:\\Outside\\夜の曲.mp3", extension: ".mp3", file_size_bytes: 2048000, title: "Night Drift", album: null,
};

const duplicateReport: DuplicateReport = {
  summary: { total_tracks: 6, exact_group_count: 1, likely_group_count: 1, exact_duplicate_file_count: 3, likely_duplicate_file_count: 3 },
  exact_duplicates: [{
    group_id: "exact-001", preferred: { path: exact.path, reasons: ["deterministic_tie_break"] },
    files: [copy, exact, other], content_hash: "a".repeat(64),
  }],
  likely_duplicates: [{
    group_id: "likely-001", files: [preferred, bridge, third],
    preferred: { path: preferred.path, reasons: [
      "preferred_lossless_format", "highest_bitrate", "richest_metadata", "has_cover_art", "largest_file_size",
    ] },
    relationships: [
      { paths: [preferred.path, bridge.path], confidence: 0.913, reasons: ["title_match_strong", "artist_match_strong", "duration_delta=0.40s", "bitrate_close"] },
      { paths: [bridge.path, third.path], confidence: 0.83, reasons: ["title_match_strong"] },
    ],
  }],
};
const wrongFinding = {
  item_id: "unrelated-exact", item_type: "exact_duplicate_group", details: { source_group_id: "exact-001" },
  affected_paths: [exact.path, "/somewhere/else.flac"],
};
const matchingFindings = [
  { item_id: "correct-exact", item_type: "exact_duplicate_group", affected_paths: [other.path, copy.path, exact.path] },
  { item_id: "correct-likely", item_type: "likely_duplicate_group", affected_paths: [third.path, bridge.path, preferred.path] },
];

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Current route">{location.pathname + location.search}</output>;
}

describe("Duplicate comparison", () => {
  let report: DuplicateReport;
  let reviewItems: Array<typeof wrongFinding | typeof matchingFindings[number]>;

  beforeEach(() => {
    report = duplicateReport;
    reviewItems = [wrongFinding, ...matchingFindings];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const path = new URL(url, "http://localhost").pathname;
      if (path === "/api/status") return Response.json({ status: "ok", review_only: false });
      if (path === "/api/duplicates") return Response.json({ meta, data: report });
      if (path === "/api/review") return Response.json({ meta, data: { items: reviewItems } });
      throw new Error("Unexpected request to " + path);
    }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("pages large duplicate sets and starts each tab on its first page", async () => {
    report = {
      ...duplicateReport,
      exact_duplicates: Array.from({ length: 30 }, (_, index) => ({
        ...duplicateReport.exact_duplicates[0],
        group_id: "exact-" + index,
      })),
    };
    const user = userEvent.setup();
    renderWithProviders(<DuplicatesPage />);
    await screen.findByRole("navigation", { name: "duplicate groups pages" });
    expect(screen.getAllByRole("article")).toHaveLength(24);
    expect(screen.getByText("1–24 of 30 duplicate groups")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getAllByRole("article")).toHaveLength(6);
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /Likely Duplicates/ }));
    expect(screen.getAllByRole("article")).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: /Exact Duplicates/ }));
    expect(screen.getByText("1–24 of 30 duplicate groups")).toBeVisible();
  });

  it("explains verified byte identity, deterministic preference and quarantine for three exact copies", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DuplicatesPage />);
    const group = within(await screen.findByRole("article", { name: "Blue Hour.flac" }));
    expect(group.getByText(/SHA-256 verified identical file bytes/)).toBeVisible();
    expect(group.getByText(/same audio and embedded tags/)).toBeVisible();
    expect(group.getByText(/Deterministic tie-break/)).toBeVisible();
    expect(group.getByText(/Select each move in Review Inbox.*still occupy disk space/)).toBeVisible();
    expect(screen.getByText("Size of additional exact copies")).toBeVisible();
    expect(screen.getByText("2.0 KB")).toBeVisible();
    const files = within(group.getByRole("list", { name: "Files in this group" })).getAllByRole("listitem");
    expect(files).toHaveLength(3);
    expect(within(files[0]).getByText("Preferred copy")).toBeVisible();
    expect(within(files[0]).getByText("FLAC")).toBeVisible();
    expect(within(files[0]).getByText("Embedded")).toBeVisible();
    expect(within(files[0]).getByText("3 of 7 fields")).toBeVisible();
    expect(within(files[1]).getByRole("heading", { name: "Blue Hour Édition.flac" })).toBeVisible();
    expect(within(files[1]).getByText(directory)).toBeVisible();
    expect(within(files[1]).getByText(copy.path)).not.toBeVisible();
    await user.click(within(files[1]).getByLabelText("Full path for file 2"));
    expect(within(files[1]).getByText(copy.path)).toBeVisible();
  });

  it("uses only reported pair scores and presents stored differences and preference evidence", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DuplicatesPage />);
    const exactTab = await screen.findByRole("button", { name: /Exact Duplicates/ });
    const likelyTab = screen.getByRole("button", { name: /Likely Duplicates/ });
    expect(exactTab).toHaveAttribute("aria-pressed", "true");
    await user.click(likelyTab);
    expect(likelyTab).toHaveAttribute("aria-pressed", "true");
    expect(exactTab).toHaveAttribute("aria-pressed", "false");

    const group = within(screen.getByRole("article", { name: "Night Drift.flac" }));
    expect(group.getByText("Manual review only")).toBeVisible();
    expect(group.getByText(/heuristic values on a 0–1 scale and have no probability interpretation/)).toBeVisible();
    expect(group.getByText("Similarity score 0.913 / 1")).toBeVisible();
    expect(group.getByText("Similarity score 0.83 / 1")).toBeVisible();
    expect(group.queryByText("File 1 ↔ File 3")).not.toBeInTheDocument();
    const evidence = within(group.getByRole("list", { name: "Match evidence for File 1 ↔ File 2" }));
    expect(evidence.getByText("Very similar titles")).toBeVisible();
    expect(evidence.getByText("Very similar artist names")).toBeVisible();
    expect(evidence.getByText("Duration differs by 0.4 seconds")).toBeVisible();
    expect(evidence.getByText("Similar reported bitrates")).toBeVisible();
    const table = within(group.getByRole("table", { name: "Stored properties for File 1 ↔ File 2" }));
    for (const value of ["Signal Path", "Signal Paths", "0.4 s apart", "FLAC", "MP3", "480 kbps", "456 kbps"]) {
      expect(table.getByText(value)).toBeVisible();
    }
    for (const reason of [
      /Preferred lossless extension.*FLAC/, /Highest reported bitrate.*480 kbps/,
      /Most tracked metadata fields.*5 of 7/, /Embedded artwork is available in this copy/, /Largest file in the group/,
    ]) {
      expect(group.getByText(reason)).toBeVisible();
    }

    const unknownFile = within(within(group.getByRole("list", { name: "Files in this group" })).getAllByRole("listitem")[2]);
    expect(unknownFile.getAllByText("Unavailable")).toHaveLength(4);
    expect(unknownFile.getByText(third.path)).not.toBeVisible();
    await user.click(unknownFile.getByLabelText("Full path for file 3"));
    expect(unknownFile.getByText(third.path)).toBeVisible();
    await user.click(group.getByText("File 2 ↔ File 3"));
    const second = within(group.getByRole("table", { name: "Stored properties for File 2 ↔ File 3" }));
    expect(second.getAllByText("Incomplete data").length).toBeGreaterThan(0);
    expect(group.queryByRole("button", { name: /quarantine/i })).not.toBeInTheDocument();
  });

  it("does not substitute a group score when pair evidence is missing", async () => {
    report = {
      ...report,
      likely_duplicates: [{ ...report.likely_duplicates[0], confidence: 0.91, relationships: undefined }],
    };
    const user = userEvent.setup();
    renderWithProviders(<DuplicatesPage />);
    await user.click(await screen.findByRole("button", { name: /Likely Duplicates/ }));
    expect(screen.getByText("No file-to-file evidence is available for this group.")).toBeVisible();
    expect(screen.queryByText(/Similarity score 0.91/)).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it.each([
    ["exact", "Review quarantine options", "correct-exact"],
    ["likely", "Open manual review", "correct-likely"],
  ])("opens the %s finding whose files match the comparison", async (kind, label, item) => {
    const user = userEvent.setup();
    renderWithProviders(<><DuplicatesPage /><CurrentLocation /></>);
    if (kind === "likely") await user.click(await screen.findByRole("button", { name: /Likely Duplicates/ }));
    await user.click(await screen.findByRole("button", { name: label }));
    expect(screen.getByLabelText("Current route")).toHaveTextContent("/review?type=" + kind + "_duplicate_group&item=" + item);
  });

  it("uses the Review Inbox filter when the current finding has a different file set", async () => {
    reviewItems = [wrongFinding];
    const user = userEvent.setup();
    renderWithProviders(<><DuplicatesPage /><CurrentLocation /></>);
    await user.click(await screen.findByRole("button", { name: "Find in Review Inbox" }));
    expect(screen.getByLabelText("Current route")).toHaveTextContent("/review?type=exact_duplicate_group");
    expect(screen.getByLabelText("Current route")).not.toHaveTextContent("item=");
  });
});
