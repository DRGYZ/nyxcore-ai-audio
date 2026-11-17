import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReviewPage } from "../pages/ReviewPage";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";
import { mockReviewReport, mockStatus } from "../../lib/mock-data";
import type { ActionPlan, ActionPlanOperation, ActionPlanReport, ReviewPlanApplyResponse } from "../../lib/types";

const musicRoot = "C:\\Synthetic Library";
const keeper = musicRoot + "\\Archive\\Original Café.flac";
const duplicate = musicRoot + "\\Imports\\重複 copy.flac";
const secondDuplicate = musicRoot + "\\Imports\\Other copy.flac";
const destination = musicRoot + "\\.nyxcore_quarantine\\Imports\\重複 copy.flac";
const outside = "D:\\Other Library\\持出.flac";

function operation(id: string, type: string, fields: Partial<ActionPlanOperation> = {}): ActionPlanOperation {
  return { operation_id: id, operation_type: type, apply_supported: true, notes: [], fields: [], values: {}, ...fields };
}

const duplicateOperations = [
  operation("keep", "keep_preferred", { path: keeper, apply_supported: false }),
  operation("move-1", "quarantine_move", { path: duplicate, destination_path: destination, expected_hash: "approved-fingerprint" }),
  operation("move-2", "quarantine_move", { path: secondDuplicate, destination_path: musicRoot + "\\.nyxcore_quarantine\\Imports\\Other copy.flac" }),
];

const makePlan = (id: string, action: string, operations: ActionPlanOperation[]): ActionPlan => ({
  plan_id: id, action_type: action, proposed_operations: operations, source_review_item_ids: ["item-1"],
  confidence: 1, safety_level: "low-risk", affected_files: [], reasons: [], notes: [],
  apply_supported: true, resolves_review_items: true,
});

const report: ActionPlanReport = {
  created_at: "", source_review_item_ids: ["item-1"],
  plans: [
    makePlan("duplicates", "exact_duplicate_keep_plan", duplicateOperations),
    makePlan("tags", "metadata_fix_plan", [
      operation("tag-1", "write_metadata", { path: musicRoot + "\\Tracks\\Café.mp3", fields: ["title", "artist"], values: { title: "Été", artist: "Mira Vale", album: "Unselected value" } }),
    ]),
    makePlan("rename", "rename_plan", [
      operation("rename-1", "rename_file", { path: musicRoot + "\\Tracks\\Old name.mp3", destination_path: musicRoot + "\\Tracks\\New name.mp3" }),
      operation("review-1", "inspect_file", { path: outside, apply_supported: false }),
    ]),
  ],
  unsupported_items: [{ source_review_item_id: "item-1", item_type: "manual", reason: "Identify this file manually." }],
  summary: { requested_item_count: 1, generated_plan_count: 3, unsupported_item_count: 1, apply_supported_plan_count: 3, max_automatic_operations: 2 },
};

function result(statuses: Array<"ok" | "error">): ReviewPlanApplyResponse {
  return {
    result_count: 1, batch_id: "batch-fixture", resolved_review_item_ids: [],
    results: [{
      plan_id: "duplicates", source_review_item_ids: ["item-1"], resolved_review_item_ids: [],
      action_type: "exact_duplicate_keep_plan", status: statuses.includes("error") ? "partial_failure" : "ok",
      operation_results: [
        { ...duplicateOperations[0], status: "skipped", message: "operation is review-only" },
        ...statuses.map((status, index) => ({
          ...duplicateOperations[index + 1], status, message: status === "error" ? "destination already exists" : "applied",
        })),
      ],
    }],
  };
}

describe("Review Inbox", () => {
  const mockReviewMutation = { mutateAsync: vi.fn(), isPending: false };
  const mockPlanMutation = {
    mutateAsync: vi.fn().mockResolvedValue({ data: report, meta: { music_path: musicRoot } }),
    isPending: false,
  };
  const mockApplyMutation = { mutateAsync: vi.fn(), isPending: false };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(hooks, "useStatusQuery").mockReturnValue({ data: mockStatus } as ReturnType<typeof hooks.useStatusQuery>);

    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: mockReviewReport } as any,
      isLoading: false,
      isError: false,
      isFetching: false,
    } as any);

    vi.spyOn(hooks, "useReviewStateMutation").mockReturnValue(mockReviewMutation as any);
    vi.spyOn(hooks, "useDuplicatesQuery").mockReturnValue({ isError: true } as ReturnType<typeof hooks.useDuplicatesQuery>);
    vi.spyOn(hooks, "useGenerateReviewPlanMutation").mockReturnValue(mockPlanMutation as any);
    vi.spyOn(hooks, "useApplyReviewPlanMutation").mockReturnValue(mockApplyMutation as any);
    vi.spyOn(hooks, "useCheckConnection").mockReturnValue({
      checkConnection: vi.fn(),
      checking: false,
    });
  });

  it("opens the page containing a deep-linked finding and keeps navigation bounded", async () => {
    const item = mockReviewReport.items[0];
    const items = Array.from({ length: 110 }, (_, index) => ({
      ...item, item_id: "finding-" + index, summary: "Library finding " + index,
    }));
    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: { ...mockReviewReport, items } },
      isLoading: false,
      isError: false,
      isFetching: false,
      isSuccess: true,
    } as ReturnType<typeof hooks.useReviewQuery>);
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />, { initialEntries: ["/review?item=finding-103"] });
    const table = within(screen.getByRole("table"));
    expect(table.getAllByRole("row")).toHaveLength(11);
    expect(table.getByRole("button", { name: /^Inspect finding: Library finding 103,/ })).toBeVisible();
    expect(screen.getByText("101–110 of 110 findings")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Previous page" }));
    expect(table.getAllByRole("row")).toHaveLength(51);
    expect(screen.getByText("51–100 of 110 findings")).toBeVisible();
    expect(table.queryByRole("button", { name: /^Inspect finding: Library finding 103,/ })).toBeNull();
  });

  it("filters findings by priority band using aria-pressed buttons", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);

    // Initially all 5 findings are shown
    expect(screen.getByText(/5 \/ 5 findings/i)).toBeInTheDocument();

    // Click 'High' priority filter button
    const highFilter = screen.getByRole("button", { name: /high \(2\)/i });
    expect(highFilter).toHaveAttribute("aria-pressed", "false");

    await user.click(highFilter);
    expect(highFilter).toHaveAttribute("aria-pressed", "true");

    // High count is 2 findings
    expect(screen.getByText(/2 \/ 5 findings/i)).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getByText(/Blue Train\.flac/i)).toBeInTheDocument();
    expect(within(table).getByText(/Upscaled lossy source suspected/i)).toBeInTheDocument();
    expect(within(table).queryByText(/Night Drive/i)).not.toBeInTheDocument();
  });

  it("keeps priority separate from exact identity and heuristic pair scores", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);

    // Finding 1 (Blue Train) has priority score 98 and confidence: undefined
    const firstFindingTrigger = screen.getByRole("button", {
      name: /inspect finding: exact duplicate found: 'blue train\.flac'/i,
    });
    await user.click(firstFindingTrigger);

    // Inspector shows details
    expect(screen.getByText(/ID: exact_duplicate_group-7abb604ec008/i)).toBeInTheDocument();
    expect(screen.getByText(/98 \/ 100/i)).toBeInTheDocument();

    expect(screen.getByText("Byte identity")).toBeInTheDocument();
    expect(screen.queryByText("100%")).not.toBeInTheDocument();

    // Finding 2 (Night Drive) has confidence 0.82 (82%)
    const secondFindingTrigger = screen.getByRole("button", {
      name: /inspect finding: review likely duplicate set: 'night drive'/i,
    });
    await user.click(secondFindingTrigger);

    expect(screen.getByText(/ID: likely_duplicate_group-21f984264b20/i)).toBeInTheDocument();
    const pairScore = screen.getByText("Lowest reported pair score").closest("div")!;
    expect(within(pairScore).getByText("0.82 / 1")).toBeInTheDocument();
    expect(within(pairScore).getByText("Heuristic similarity")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
  });

  it("exposes all triage actions in the inspector panel", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);

    const findingTrigger = screen.getByRole("button", {
      name: /inspect finding: exact duplicate found: 'blue train\.flac'/i,
    });
    await user.click(findingTrigger);

    // Verify all 5 triage actions are present and accessible
    expect(screen.getByRole("button", { name: "Mark Seen" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ignore" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Snooze 7d" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Generate Plan" })[0]).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resolve Until Refresh" })).toBeInTheDocument();
  });

  it("opens PlanReportModal, traps focus inside, and restores focus upon closing", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);

    const generatePlanButtons = screen.getAllByRole("button", { name: "Generate Plan" });
    const generateBtn = generatePlanButtons[0];
    generateBtn.focus();
    expect(document.activeElement).toBe(generateBtn);

    // Click to generate plan
    await user.click(generateBtn);

    // Modal opens
    const modal = await screen.findByRole("dialog", { name: /generated action plan/i });
    expect(modal).toBeInTheDocument();

    // Close modal via Escape
    await user.keyboard("{Escape}");

    // Modal closes
    expect(screen.queryByRole("dialog", { name: /generated action plan/i })).not.toBeInTheDocument();
    expect(generateBtn).toHaveFocus();
  });

  it("explains operation-specific consequences and offers the original full paths", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);
    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const modal = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    expect(modal.getByText("Imports\\重複 copy.flac")).toBeVisible();
    expect(modal.getByText(".nyxcore_quarantine\\Imports\\重複 copy.flac")).toBeVisible();
    expect(modal.getAllByText("Archive\\Original Café.flac")[0]).toBeVisible();
    expect(modal.getByText(duplicate)).not.toBeVisible();
    await user.click(modal.getByText(duplicate).closest("details")!.querySelector("summary")!);
    expect(modal.getByText(duplicate)).toBeVisible();
    expect(modal.getByText(outside)).toBeVisible();
    expect(modal.getByText("Été")).toBeVisible();
    expect(modal.getByText("Mira Vale")).toBeVisible();
    expect(modal.queryByText("Unselected value")).not.toBeInTheDocument();
    expect(modal.getByText("Old name.mp3")).toBeVisible();
    expect(modal.getByText("New name.mp3")).toBeVisible();
    expect(modal.getByText("This operation cannot be applied.")).toBeVisible();
    expect(modal.getAllByRole("checkbox")).toHaveLength(4);
  });

  it("previews selected changes without an apply action in review-only mode", async () => {
    const user = userEvent.setup();
    vi.spyOn(hooks, "useStatusQuery").mockReturnValue({
      data: { ...mockStatus, review_only: true },
    } as ReturnType<typeof hooks.useStatusQuery>);
    renderWithProviders(<ReviewPage />);

    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    await user.click(preview.getByRole("checkbox", { name: /quarantine duplicate.*重複/ }));
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));

    const selection = within(screen.getByRole("dialog", { name: "Selected Changes" }));
    expect(selection.getByText(/file changes are disabled/i)).toBeVisible();
    expect(selection.getByText("Imports\\重複 copy.flac")).toBeVisible();
    expect(selection.queryByRole("button", { name: "Apply Plan" })).toBeNull();
    expect(mockApplyMutation.mutateAsync).not.toHaveBeenCalled();
  });

  it("keeps only selected changes in confirmation and sends the same approved paths and fingerprint", async () => {
    const user = userEvent.setup();
    mockApplyMutation.mutateAsync.mockResolvedValue(result(["ok"]));
    renderWithProviders(<ReviewPage />);
    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    const first = preview.getByRole("checkbox", { name: /quarantine duplicate.*重複/ });
    const second = preview.getByRole("checkbox", { name: /quarantine duplicate.*Other copy/ });
    await user.click(first);
    await user.click(second);
    expect(preview.getByRole("checkbox", { name: /write metadata/ })).toBeDisabled();
    await user.click(second);
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    const confirm = within(screen.getByRole("dialog", { name: "Confirm Plan Apply" }));
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(confirm.getByText("1 selected change")).toBeInTheDocument();
    expect(confirm.getByText("Duplicate moves · 1")).toBeInTheDocument();
    expect(confirm.getByText("Keep")).toBeInTheDocument();
    expect(confirm.getByText("Move from")).toBeInTheDocument();
    expect(confirm.getByText("Move to")).toBeInTheDocument();
    expect(confirm.queryByText(/Other copy|Metadata edits|Renames/)).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(mockApplyMutation.mutateAsync).not.toHaveBeenCalled();
    expect(preview.getByRole("checkbox", { name: /quarantine duplicate.*重複/ })).toBeChecked();
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    await user.click(confirm.getByRole("button", { name: "Apply Plan" }));

    expect(mockApplyMutation.mutateAsync).toHaveBeenCalledTimes(1);
    const sent = mockApplyMutation.mutateAsync.mock.calls[0][0].plan_report;
    expect(sent.plans).toHaveLength(1);
    expect(sent.plans[0].proposed_operations).toEqual(duplicateOperations.slice(0, 2));
    expect(await screen.findByText("All selected changes applied. 1 applied · 0 failed · 0 not applied.")).toBeInTheDocument();
  });

  it.each([
    { statuses: ["ok", "error"] as const, summary: "Partially applied. 1 applied · 1 failed · 0 not applied." },
    { statuses: ["error", "error"] as const, summary: "No changes applied. 0 applied · 2 failed · 0 not applied." },
  ])("reports $summary and keeps the details and History link", async ({ statuses, summary }) => {
    const user = userEvent.setup();
    mockApplyMutation.mutateAsync.mockResolvedValue(result([...statuses]));
    renderWithProviders(<ReviewPage />);
    await user.click(screen.getAllByRole("button", { name: "Generate Plan" })[0]);
    const preview = within(await screen.findByRole("dialog", { name: "Generated Action Plan" }));
    for (const move of preview.getAllByRole("checkbox", { name: /quarantine duplicate/ })) await user.click(move);
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    const confirm = within(screen.getByRole("dialog", { name: "Confirm Plan Apply" }));
    await user.click(confirm.getByRole("button", { name: "Cancel" }));
    expect(mockApplyMutation.mutateAsync).not.toHaveBeenCalled();
    await user.click(preview.getByRole("button", { name: "Review Selected Changes" }));
    await user.click(confirm.getByRole("button", { name: "Apply Plan" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(summary);
    expect(screen.getAllByText(/destination already exists/)[0]).toBeVisible();
    expect(screen.getByRole("link", { name: "Inspect History Batch" })).toHaveAttribute("href", "/history?batch=batch-fixture");
  });

  it("preserves selection and restores focus when closing responsive inspector", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewPage />);

    const trigger = screen.getByRole("button", {
      name: /inspect finding: exact duplicate found: 'blue train\.flac'/i,
    });
    await user.click(trigger);

    // Responsive drawer has Back to Findings button
    const backButton = screen.queryByRole("button", { name: /back to findings/i });
    if (backButton) {
      await user.click(backButton);
      expect(document.activeElement).toBe(trigger);
    }
  });
});
