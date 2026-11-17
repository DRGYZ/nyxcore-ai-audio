import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockDuplicateReport, mockHealthReport, mockHistoryResponse, mockReviewReport, mockStatus } from "../../lib/mock-data";
import type { MetadataProposal, MetadataTrack, PlanInput } from "../../lib/metadata";
import { renderWithProviders } from "../../test/test-utils";
import { App } from "../App";

const first = "/library/jazz/A Love Supreme.flac";
const second = "/library/ambient/Starlight.m4a";
const finding = mockReviewReport.items.find((item) => item.item_type === "missing_metadata")!;

describe("metadata repair", () => {
  let entries: MetadataProposal[];
  let generation: string;
  let previewOnly: boolean;
  let simulateFailure: boolean;
  let applyRequests: unknown[];
  let revision: string;
  let planned: PlanInput | undefined;

  const track = (path: string): MetadataTrack => ({
    path, generation, supported: true, inspection: `inspection-${path}-${generation}`,
    tags: { title: [], artist: ["Coltrane"], album: [] },
    issues: ["missing_title", "missing_album"], swap: false,
    suggestions: {
      title: { value: "Filename title", source: "filename", confidence: "tentative", evidence: "Parsed from the filename." },
      artist: { value: "Coltrane", source: "embedded", confidence: "existing", evidence: "Embedded value." },
      album: { value: null, source: "unavailable", confidence: "ambiguous", evidence: "Enter the album if you know it." },
    },
  });

  beforeEach(() => {
    entries = [];
    generation = "1";
    previewOnly = false;
    simulateFailure = false;
    revision = "1";
    applyRequests = [];
    planned = undefined;
    vi.stubGlobal("fetch", vi.fn(async (raw: string, options?: RequestInit) => {
      const url = new URL(raw, "http://127.0.0.1");
      const body = options?.body ? JSON.parse(String(options.body)) : undefined;
      switch (url.pathname) {
        case "/api/status": return Response.json({ ...mockStatus, review_only: previewOnly });
        case "/api/history": return Response.json(mockHistoryResponse);
        case "/api/library": return Response.json({
          meta: { ...mockHistoryResponse.meta, mode: "snapshot", analysis_generation: generation },
          analyzed_at: "2026-11-17T00:00:00Z", health: mockHealthReport, review: mockReviewReport, duplicates: mockDuplicateReport,
        });
        case "/api/library/refresh":
          generation = "2";
          return Response.json({});
        case "/api/health": return Response.json({ data: mockHealthReport });
        case "/api/review": return Response.json({ data: mockReviewReport });
        case "/api/duplicates": return Response.json({ data: mockDuplicateReport });
        case "/api/search": return Response.json({
          query: "Starlight", total_matches: 1, returned_count: 1, items: [{
            path: second, filename: "Starlight.m4a", title: null, album: null, artist: null,
            match_fields: ["filename"], warnings: [], duration_seconds: null, file_size_bytes: 100, has_cover_art: false,
          }],
        });
        case "/api/metadata/issues": return Response.json({ paths: [first], total: 78 });
        case "/api/metadata/track": return Response.json(track(url.searchParams.get("path")!));
        case "/api/metadata/drafts":
          if (body) {
            entries = body.entries;
            revision = String(Number(revision) + 1);
          }
          return Response.json({ entries, revision, generation });
        case "/api/metadata/plan": {
          planned = body;
          return Response.json({ preview: "approved-preview", review_only: previewOnly,
            files: (body as PlanInput).entries.map((entry) => ({ path: entry.path, before: track(entry.path).tags, changes: entry.values })),
          });
        }
        case "/api/metadata/plan/apply":
          applyRequests.push(body);
          return Response.json({ batch_id: "batch-metadata", results: [{
            operation_results: body.entries.map((entry: MetadataProposal, index: number) => ({
              operation_id: String(index), path: entry.path, status: simulateFailure ? "error" : "ok",
              message: simulateFailure ? "Source changed after approval" : "applied",
            })),
          }] });
        default: throw new Error(`Unexpected request: ${url.pathname}`);
      }
    }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("opens from Health, validates and reverts edits, then stages another correction from Search", async () => {
    const user = userEvent.setup();
    renderWithProviders(<App />, { initialEntries: ["/health"] });
    await user.click(await screen.findByRole("button", { name: /Missing Title/ }));
    expect(await screen.findByRole("navigation", { name: "Affected tracks pages" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Edit metadata for A Love Supreme.flac" }));
    let dialog = await screen.findByRole("dialog");
    await user.click(await within(dialog).findByRole("button", { name: "Use title suggestion" }));
    const title = within(dialog).getByRole("textbox", { name: "Proposed title" });
    expect(title).toHaveValue("Filename title");
    await user.click(within(dialog).getByRole("button", { name: "Revert title" }));
    expect(title).toHaveValue("");
    await user.type(title, "UNKNOWN");
    expect(within(dialog).getByRole("alert")).toHaveTextContent("specific value");
    expect(within(dialog).getByRole("button", { name: "Stage metadata edit" })).toBeDisabled();
    await user.clear(title);
    await user.type(title, "Manual title");
    await user.click(within(dialog).getByRole("button", { name: "Stage metadata edit" }));
    expect(await within(dialog).findByText("Staged in the workspace.")).toBeVisible();
    await user.click(within(dialog).getByRole("button", { name: "Close dialog" }));
    await user.click(screen.getByRole("link", { name: "Archive Search" }));
    await user.type(screen.getByRole("searchbox", { name: "Search the music archive" }), "Starlight");
    await user.click(screen.getByRole("button", { name: "Search Library" }));
    await user.click(await screen.findByRole("button", { name: "Edit metadata for Starlight.m4a" }));
    dialog = await screen.findByRole("dialog");
    await user.type(await within(dialog).findByRole("textbox", { name: "Proposed album" }), "Manual album");
    await user.click(within(dialog).getByRole("button", { name: "Stage metadata edit" }));
    expect(await within(dialog).findByText("Staged in the workspace.")).toBeVisible();
    expect(entries.map((entry) => entry.values)).toEqual([{ title: "Manual title" }, { album: "Manual album" }]);
  });

  it("opens the same editor from a selected Review Inbox finding and retains its context", async () => {
    const user = userEvent.setup();
    renderWithProviders(<App />, { initialEntries: [`/review?item=${finding.item_id}`] });
    const edit = (await screen.findAllByRole("button", { name: "Edit metadata for A Love Supreme.flac" }))[0];
    await user.click(edit);
    expect(await within(screen.getByRole("dialog", { name: "Edit track metadata" })).findByRole("textbox", { name: "Proposed title" })).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Edit track metadata" })).toBeNull();
    expect(screen.getAllByRole("button", { name: "Edit metadata for A Love Supreme.flac" }).length).toBeGreaterThan(0);
  });

  it("applies a selected shared value, previews both files and requires confirmation", async () => {
    entries = [
      { ...track(first), values: { title: "Title One" } }, { ...track(second), values: { title: "Title Two" } },
    ].map(({ path, generation, inspection, values }) => ({ path, generation, inspection, values }));
    const user = userEvent.setup();
    renderWithProviders(<App />, { initialEntries: ["/metadata"] });
    await user.click(await screen.findByRole("checkbox", { name: "Starlight.m4a" }));
    await user.type(screen.getByRole("textbox", { name: "Shared replacement value" }), "Shared artist");
    await user.click(screen.getByRole("button", { name: "Stage for 1 selected" }));
    expect(await screen.findByText("Shared artist")).toBeVisible();
    await user.click(screen.getByRole("checkbox", { name: "Starlight.m4a" }));
    await user.click(screen.getByRole("button", { name: "Generate metadata plan" }));
    expect(await screen.findByRole("heading", { name: "Metadata plan preview" })).toBeVisible();
    expect(planned?.entries.map((entry) => entry.values)).toEqual([
      { title: "Title One", artist: "Shared artist" }, { title: "Title Two" },
    ]);
    const apply = screen.getByRole("button", { name: "Apply approved metadata" });
    expect(apply).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: /I approve/ }));
    await user.click(apply);
    expect(await screen.findAllByText(/Applied and verified/)).toHaveLength(2);
    expect(screen.getByRole("link", { name: "Open history to inspect or undo" })).toHaveAttribute("href", "/history?batch=batch-metadata");
    expect(applyRequests).toHaveLength(1);
  });

  it("shows a preview-only boundary and marks its persisted drafts stale after library refresh", async () => {
    previewOnly = true;
    const { path, inspection } = track(first);
    entries = [{ path, generation, inspection, values: { title: "Title One" } }];
    const user = userEvent.setup();
    renderWithProviders(<App />, { initialEntries: ["/metadata"] });
    await user.click(await screen.findByRole("button", { name: "Generate metadata plan" }));
    expect(await screen.findByText(/Preview only. This launcher cannot apply/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Apply approved metadata" })).toBeNull();
    await user.click(await screen.findByRole("button", { name: "Refresh Library" }));
    expect(await screen.findByText(/Selected drafts were staged before the current analysis/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Generate metadata plan" })).toBeDisabled();
    expect(applyRequests).toHaveLength(0);
  });

  it("reports a file failure and keeps its draft", async () => {
    simulateFailure = true;
    const { path, inspection } = track(first);
    entries = [{ path, generation, inspection, values: { artist: "Replacement" } }];
    const user = userEvent.setup();
    renderWithProviders(<App />, { initialEntries: ["/metadata"] });
    await user.click(await screen.findByRole("button", { name: "Generate metadata plan" }));
    await user.click(await screen.findByRole("checkbox", { name: /I approve/ }));
    await user.click(screen.getByRole("button", { name: "Apply approved metadata" }));
    expect(await screen.findByText(/Failed · Source changed after approval/)).toBeVisible();
    expect(entries).toHaveLength(1);
  });
});
