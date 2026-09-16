import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DuplicatesPage } from "../pages/DuplicatesPage";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";
import { mockDuplicateReport, mockReviewReport } from "../../lib/mock-data";

describe("Duplicates Page", () => {
  it("keeps exact and likely duplicate groups semantically distinct via aria-pressed buttons", async () => {
    const user = userEvent.setup();

    vi.spyOn(hooks, "useDuplicatesQuery").mockReturnValue({
      data: { data: mockDuplicateReport } as any,
      isLoading: false,
      isError: false,
    } as any);

    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: mockReviewReport } as any,
      isLoading: false,
      isError: false,
    } as any);

    renderWithProviders(<DuplicatesPage />);

    // Check filter buttons
    const exactButton = screen.getByRole("button", { name: /exact duplicates/i });
    const likelyButton = screen.getByRole("button", { name: /likely duplicates/i });

    expect(exactButton).toHaveAttribute("aria-pressed", "true");
    expect(likelyButton).toHaveAttribute("aria-pressed", "false");

    // Initially exact duplicates list is shown
    expect(screen.getByText(/100% \(Bit-Exact\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Preferred Copy/i)).toBeInTheDocument();

    // Click likely duplicates button
    await user.click(likelyButton);

    expect(exactButton).toHaveAttribute("aria-pressed", "false");
    expect(likelyButton).toHaveAttribute("aria-pressed", "true");

    // Likely duplicates content is shown
    expect(screen.getByText(/Match Score/i)).toBeInTheDocument();
    expect(screen.getByText(/Similar content detected with metadata or tag variance/i)).toBeInTheDocument();
  });
});
