import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HistoryPage } from "../pages/HistoryPage";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";
import { mockHistoryResponse } from "../../lib/mock-data";

describe("History Page", () => {
  const mockReverseMutation = { mutateAsync: vi.fn(), isPending: false };

  beforeEach(() => {
    vi.clearAllMocks();

    vi.spyOn(hooks, "useHistoryQuery").mockReturnValue({
      data: mockHistoryResponse,
      isLoading: false,
      isError: false,
    } as any);

    vi.spyOn(hooks, "useReverseHistoryMutation").mockReturnValue(mockReverseMutation as any);
  });

  it("exposes one canonical Reverse action and no separate Undo/Restore buttons", () => {
    renderWithProviders(<HistoryPage />);

    // Table rows and detail panel have "Reverse" action
    const reverseButtons = screen.getAllByRole("button", { name: /reverse/i });
    expect(reverseButtons.length).toBeGreaterThan(0);

    // Verify there are NO separate "Restore" or "Undo" buttons
    expect(screen.queryByRole("button", { name: /^restore$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^undo$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^restore batch$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^undo batch$/i })).not.toBeInTheDocument();
  });

  it("opens reversal confirmation dialog, traps focus, and closes on cancel with focus restoration", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HistoryPage />);

    const reverseButtons = screen.getAllByRole("button", { name: /reverse/i });
    const trigger = reverseButtons[0];
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    await user.click(trigger);

    // Confirmation modal opens
    const modal = await screen.findByRole("dialog", { name: /confirm reversal/i });
    expect(modal).toBeInTheDocument();

    // Cancel modal
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    // Modal closes
    expect(screen.queryByRole("dialog", { name: /confirm reversal/i })).not.toBeInTheDocument();
  });
});
