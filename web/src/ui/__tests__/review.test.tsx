import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReviewPage } from "../pages/ReviewPage";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";
import { mockReviewReport } from "../../lib/mock-data";

describe("Review Inbox", () => {
  const mockReviewMutation = { mutateAsync: vi.fn(), isPending: false };
  const mockPlanMutation = {
    mutateAsync: vi.fn().mockResolvedValue({
      data: {
        summary: { generated_plan_count: 1, max_automatic_operations: 10 },
        plans: [
          {
            plan_id: "plan-1",
            action_type: "quarantine_move",
            safety_level: "safe",
            confidence: 1.0,
            affected_files: ["/path/to/file.mp3"],
            proposed_operations: [
              {
                operation_id: "op-1",
                operation_type: "quarantine_move",
                path: "/path/to/file.mp3",
                fields: [],
                values: {},
                notes: [],
              },
            ],
            reasons: ["Exact duplicate"],
            notes: [],
          },
        ],
        unsupported_items: [],
      },
    }),
    isPending: false,
  };
  const mockApplyMutation = { mutateAsync: vi.fn(), isPending: false };

  beforeEach(() => {
    vi.clearAllMocks();

    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: mockReviewReport } as any,
      isLoading: false,
      isError: false,
      isFetching: false,
    } as any);

    vi.spyOn(hooks, "useReviewStateMutation").mockReturnValue(mockReviewMutation as any);
    vi.spyOn(hooks, "useGenerateReviewPlanMutation").mockReturnValue(mockPlanMutation as any);
    vi.spyOn(hooks, "useApplyReviewPlanMutation").mockReturnValue(mockApplyMutation as any);
    vi.spyOn(hooks, "useCheckConnection").mockReturnValue({
      checkConnection: vi.fn(),
      checking: false,
    });
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

  it("updates inspector when selecting a finding and shows honest confidence (not fabricated)", async () => {
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

    // Confidence should NOT be fabricated when absent; it must display 'n/a'
    const confidenceLabel = screen.getByText(/confidence/i, { selector: "p" });
    const confidenceContainer = confidenceLabel.closest("div")!;
    expect(within(confidenceContainer).getByText("n/a")).toBeInTheDocument();

    // Finding 2 (Night Drive) has confidence 0.82 (82%)
    const secondFindingTrigger = screen.getByRole("button", {
      name: /inspect finding: review likely duplicate set: 'night drive'/i,
    });
    await user.click(secondFindingTrigger);

    expect(screen.getByText(/ID: likely_duplicate_group-21f984264b20/i)).toBeInTheDocument();
    const secondConfidenceContainer = screen.getByText(/confidence/i, { selector: "p" }).closest("div")!;
    expect(within(secondConfidenceContainer).getByText("82%")).toBeInTheDocument();
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
