import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ActionPlanOperation, ActionPlanReport, ApiMeta, DuplicateFileInfo, DuplicateGroup, DuplicateReport, ReviewItem } from "../../lib/types";
import { renderWithProviders } from "../../test/test-utils";
import { ReviewPage } from "../pages/ReviewPage";

const meta: ApiMeta = { music_path: "/fixture", out_path: "/reports", active_profile: "default", mode: "live" };
const hash = "b".repeat(64);
const paths = ["/fixture/Archive/Blue.flac", "/fixture/Imports/Blue copy.flac", "/fixture/Imports/Blue extra.flac"];
const likelyPaths = ["/fixture/Archive/Signal.flac", "/fixture/Imports/Signal.mp3", "/fixture/Tests/Signal.mp3"];

function track(path: string, fields: Partial<DuplicateFileInfo> = {}): DuplicateFileInfo {
  return { path, file_size_bytes: 1024, extension: ".flac", ...fields };
}

function finding(id: string, type: string, affected_paths: string[], fields: Partial<ReviewItem> = {}): ReviewItem {
  return { item_id: id, item_type: type, affected_paths, summary: id, priority_score: 80,
    priority_band: "high", reason_summary: "", review_status: "new", ...fields };
}

const exact = finding("Exact finding", "exact_duplicate_group", paths, {
  preferred_path: paths[0], details: { source_group_id: "exact-decoy", content_hash: hash, preferred_reasons: ["deterministic_tie_break"] },
  reclaimable_bytes: 2048, reason_summary: "2048 reclaimable bytes",
});
const likely = finding("Likely finding", "likely_duplicate_group", likelyPaths, {
  preferred_path: likelyPaths[0], confidence: 0.83, details: { relationship_count: 2 },
});
const unrelated = finding("Missing artist", "missing_metadata", ["/fixture/Unrelated.mp3"]);
const exactGroup: DuplicateGroup = {
  group_id: "exact-current", content_hash: hash, preferred: { path: paths[0], reasons: ["deterministic_tie_break"] },
  files: [track(paths[2]), track(paths[0]), track(paths[1])],
};
const reports: DuplicateReport = {
  summary: { total_tracks: 7, exact_group_count: 2, likely_group_count: 1, exact_duplicate_file_count: 5, likely_duplicate_file_count: 3 },
  exact_duplicates: [{
    ...exactGroup, group_id: "exact-decoy", files: [track(paths[0]), track("/fixture/Other.flac")],
  }, exactGroup],
  likely_duplicates: [{
    group_id: "likely-current", confidence: 0.83,
    preferred: { path: likelyPaths[0], reasons: ["preferred_lossless_format"] },
    files: [
      track(likelyPaths[0], { title: "Signal", album: "First release", duration_seconds: 8, bitrate_bps: 96000 }),
      track(likelyPaths[1], { title: "Signal", album: "First release", duration_seconds: 8.2, bitrate_bps: 96000, extension: ".mp3" }),
      track(likelyPaths[2], { title: "Signal", album: null, duration_seconds: 11, bitrate_bps: 192000, extension: ".mp3" }),
    ],
    relationships: [
      { paths: likelyPaths.slice(0, 2), confidence: 0.99, reasons: ["title_match_strong", "duration_delta=0.20s", "bitrate_close"] },
      { paths: likelyPaths.slice(1), confidence: 0.83, reasons: ["title_match_strong", "duration_delta=2.80s"] },
    ],
  }],
};

const ops: ActionPlanOperation[] = paths.map((path, index) => ({
  operation_id: "op-" + index, operation_type: index === 0 ? "keep_preferred" : "quarantine_move",
  path, destination_path: index === 0 ? null : path.replace("/fixture/", "/fixture/.nyxcore_quarantine/"),
  fields: [], values: {}, notes: [], apply_supported: index > 0, expected_hash: hash,
}));
const exactPlan: ActionPlanReport = {
  created_at: "", source_review_item_ids: [exact.item_id],
  plans: [{
    plan_id: "exact-plan", source_review_item_ids: [exact.item_id], action_type: "exact_duplicate_keep_plan",
    affected_files: paths, proposed_operations: ops, confidence: 1, safety_level: "low-risk",
    reasons: [], notes: [], apply_supported: true, resolves_review_items: true,
  }],
  unsupported_items: [], summary: {
    requested_item_count: 1, generated_plan_count: 1, unsupported_item_count: 0, apply_supported_plan_count: 1, max_automatic_operations: 2,
  },
};

function openFinding(item: ReviewItem) {
  return renderWithProviders(<ReviewPage />, { initialEntries: ["/review?item=" + encodeURIComponent(item.item_id)] });
}

describe("Duplicate evidence in Review Inbox", () => {
  let duplicateReport: DuplicateReport;
  let duplicateMeta: ApiMeta;
  let duplicateStatus: number;
  let generatedPlan: ActionPlanReport;
  const capturedApply = vi.fn();
  const duplicateRequests = vi.fn();

  beforeEach(() => {
    duplicateReport = reports;
    duplicateMeta = meta;
    duplicateStatus = 200;
    generatedPlan = exactPlan;
    capturedApply.mockClear();
    duplicateRequests.mockClear();
    vi.stubGlobal("innerWidth", 1440);
    vi.stubGlobal("fetch", vi.fn(async (url: string, options?: RequestInit) => {
      const path = new URL(url, "http://localhost").pathname;
      if (path === "/api/status") return Response.json({ status: "ok", review_only: false });
      if (path === "/api/review") return Response.json({ meta, data: { items: [exact, likely, unrelated] } });
      if (path === "/api/duplicates") {
        duplicateRequests();
        return Response.json({ meta: duplicateMeta, data: duplicateReport }, { status: duplicateStatus });
      }
      if (path === "/api/review/plan") {
        const itemId = JSON.parse(String(options?.body)).item_ids[0];
        const plan = itemId === likely.item_id ? {
          ...exactPlan, source_review_item_ids: [itemId], plans: [],
          unsupported_items: [{ source_review_item_id: itemId, reason: "Likely matches require manual review." }],
        } : generatedPlan;
        return Response.json({ meta, data: plan });
      }
      if (path === "/api/review/plan/apply") {
        capturedApply(JSON.parse(String(options?.body)));
        return Response.json({ result_count: 0, resolved_review_item_ids: [], results: [] });
      }
      throw new Error("Unexpected API request " + path);
    }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("uses full file sets and keeps evidence out of the selected operation payload and confirmation", async () => {
    const user = userEvent.setup();
    openFinding(exact);
    expect(await screen.findByText(/SHA-256 verified identical file bytes/)).toBeVisible();
    expect(screen.getByText(/Deterministic tie-break/)).toBeVisible();
    expect(screen.getByText("Archive/Blue.flac")).toBeVisible();
    expect(screen.queryByText(/reclaimable bytes|100%/)).not.toBeInTheDocument();
    const compare = screen.getByText("Compare copies");
    expect(compare.closest("details")).not.toHaveAttribute("open");
    await user.click(compare);
    expect(within(screen.getByRole("list", { name: "Files in this group" })).getAllByRole("listitem")).toHaveLength(3);
    expect(screen.queryByText("Other.flac")).not.toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    expect(preview.getByText(/SHA-256 verified identical file bytes/)).toBeVisible();
    expect(preview.getByText(/Deterministic tie-break/)).toBeVisible();
    const [selected, excluded] = preview.getAllByRole("checkbox");
    await user.click(selected);
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    const confirmation = within(screen.getByRole("dialog", { name: "Confirm Plan Apply" }));
    for (const label of ["Keep", "Move from", "Move to"]) expect(confirmation.getByText(label)).toBeVisible();
    expect(confirmation.queryByText("Duplicate evidence")).not.toBeInTheDocument();
    expect(confirmation.queryByText(/Blue extra/)).not.toBeInTheDocument();
    await user.click(confirmation.getByRole("button", { name: "Cancel" }));
    expect(await preview.findByRole("checkbox", { name: selected.getAttribute("aria-label")! })).toBeChecked();
    expect(preview.getByRole("checkbox", { name: excluded.getAttribute("aria-label")! })).not.toBeChecked();
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    await user.click(confirmation.getByRole("button", { name: "Apply Plan" }));
    expect(capturedApply).toHaveBeenCalledWith({ plan_report: {
      ...exactPlan, plans: [{ ...exactPlan.plans[0], affected_files: paths.slice(0, 2),
        proposed_operations: ops.slice(0, 2), resolves_review_items: false }],
    } });
  });

  it("shows two reported pairs in a three-file mobile finding and preserves manual review", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("innerWidth", 390);
    openFinding(likely);
    const inspector = within(await screen.findByRole("dialog", { name: /Finding Inspector: Likely finding/ }));
    expect(await inspector.findByText(/Manual review only.*heuristic similarity/)).toBeVisible();
    expect(inspector.getByText(/Preferred lossless extension.*FLAC/)).toBeVisible();
    expect(inspector.getByText("0.83 / 1")).toBeVisible();
    expect(within(inspector.getByText("Reclaimable").parentElement!).getByText("Unavailable")).toBeVisible();
    await user.click(inspector.getByText("Compare copies and reported pairs"));
    expect(inspector.getByText("Similarity score 0.99 / 1")).toBeVisible();
    expect(inspector.getByText("Similarity score 0.83 / 1")).toBeVisible();
    expect(inspector.queryByText("File 1 ↔ File 3")).not.toBeInTheDocument();
    expect(inspector.getByText("Duration differs by 0.2 seconds")).toBeVisible();
    await user.click(inspector.getByText("File 2 ↔ File 3"));
    expect(within(inspector.getByRole("table", { name: "Stored properties for File 2 ↔ File 3" })).getByText("192 kbps")).toBeVisible();
    await user.click(inspector.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    expect(preview.getByText(/Manual review only.*heuristic similarity/)).toBeVisible();
    expect(preview.getByText(/2 reported pairs\. Lowest pair score 0.83 \/ 1/)).toBeVisible();
    expect(preview.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(preview.getByRole("button", { name: "Review Selected Changes" })).toBeDisabled();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog", { name: /Finding Inspector: Likely finding/ })).toBeVisible();
  });

  it.each(["missing group", "changed hash", "different root", "different profile", "wrong type", "ambiguous match"])(
    "withholds detailed evidence for %s", async (mismatch) => {
      if (mismatch === "different root") duplicateMeta = { ...meta, music_path: "/other" };
      if (mismatch === "different profile") duplicateMeta = { ...meta, active_profile: "strict" };
      if (mismatch === "missing group") duplicateReport = { ...reports, exact_duplicates: [reports.exact_duplicates[0]] };
      if (mismatch === "changed hash") duplicateReport = { ...reports, exact_duplicates: [{ ...exactGroup, content_hash: "c".repeat(64) }] };
      if (mismatch === "wrong type") duplicateReport = { ...reports, exact_duplicates: [], likely_duplicates: [exactGroup] };
      if (mismatch === "ambiguous match") duplicateReport = { ...reports, exact_duplicates: [exactGroup, { ...exactGroup, group_id: "second" }] };
      openFinding(exact);
      expect(await screen.findByText(/Detailed duplicate evidence is unavailable/)).toBeVisible();
      expect(screen.queryByText(/Deterministic tie-break/)).not.toBeInTheDocument();
    },
  );

  it("withholds stale plan evidence while leaving the generated operation choices intact", async () => {
    const user = userEvent.setup();
    generatedPlan = { ...exactPlan, plans: [{ ...exactPlan.plans[0],
      proposed_operations: ops.map((operation) => ({ ...operation, expected_hash: "c".repeat(64) })),
    }] };
    openFinding(exact);
    expect(await screen.findByText(/Deterministic tie-break/)).toBeVisible();
    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    expect(preview.getByText(/Detailed duplicate evidence is unavailable/)).toBeVisible();
    expect(preview.getAllByRole("checkbox")).toHaveLength(2);
  });

  it("removes previously matched evidence when an inbox refresh reports changed content", async () => {
    const user = userEvent.setup();
    openFinding(exact);
    expect(await screen.findByText(/Deterministic tie-break/)).toBeVisible();
    duplicateReport = { ...reports, exact_duplicates: [{ ...exactGroup, content_hash: "c".repeat(64) }] };
    await user.click(screen.getByRole("button", { name: "Refresh Inbox" }));
    expect(await screen.findByText(/Detailed duplicate evidence is unavailable/)).toBeVisible();
    expect(screen.queryByText(/Deterministic tie-break/)).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Mark Seen" })).toBeEnabled());
  });

  it("keeps the inbox usable after a duplicate endpoint failure", async () => {
    const user = userEvent.setup();
    duplicateStatus = 500;
    openFinding(unrelated);
    expect(await screen.findByRole("heading", { name: "Missing artist" })).toBeVisible();
    expect(duplicateRequests).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /Inspect finding: Exact finding/ }));
    expect(await screen.findByText(/Detailed duplicate evidence is unavailable/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Mark Seen" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: /Inspect finding: Missing artist/ }));
    expect(screen.getByRole("heading", { name: "Missing artist" })).toBeVisible();
    expect(screen.queryByText("Duplicate evidence")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mark Seen" })).toBeEnabled();
  });
});
