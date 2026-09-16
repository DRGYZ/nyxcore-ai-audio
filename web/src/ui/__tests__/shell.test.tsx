import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AppShell } from "../shell";
import { renderWithProviders } from "../../test/test-utils";
import * as hooks from "../../lib/hooks";
import { mockStatus, mockReviewReport } from "../../lib/mock-data";

describe("AppShell", () => {
  it("renders all six primary navigation links and omits Saved Playlists", () => {
    vi.spyOn(hooks, "useStatusQuery").mockReturnValue({
      data: mockStatus,
      isError: false,
      isLoading: false,
    } as any);

    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: mockReviewReport } as any,
      isError: false,
      isLoading: false,
    } as any);

    renderWithProviders(
      <AppShell>
        <div>Content</div>
      </AppShell>,
      { initialEntries: ["/"] },
    );

    // Verify all six approved routes are in the navigation
    expect(screen.getByRole("link", { name: /overview/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /archive search/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /review inbox/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /duplicates/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /library health/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /history/i })).toBeInTheDocument();

    // Verify Saved Playlists is NOT in primary navigation
    expect(screen.queryByRole("link", { name: /saved playlists/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /smart playlists/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /playlists/i })).not.toBeInTheDocument();
  });

  it("handles mobile navigation toggle expansion, collapse, and Escape key", async () => {
    const user = userEvent.setup();

    vi.spyOn(hooks, "useStatusQuery").mockReturnValue({
      data: mockStatus,
      isError: false,
      isLoading: false,
    } as any);

    vi.spyOn(hooks, "useReviewQuery").mockReturnValue({
      data: { data: mockReviewReport } as any,
      isError: false,
      isLoading: false,
    } as any);

    renderWithProviders(
      <AppShell>
        <div>Content</div>
      </AppShell>,
      { initialEntries: ["/"] },
    );

    const toggleButton = screen.getByRole("button", { name: /toggle navigation menu/i });
    expect(toggleButton).toHaveAttribute("aria-expanded", "false");

    // Click to expand
    await user.click(toggleButton);
    expect(toggleButton).toHaveAttribute("aria-expanded", "true");

    // Press Escape to collapse
    await user.keyboard("{Escape}");
    expect(toggleButton).toHaveAttribute("aria-expanded", "false");
  });
});
