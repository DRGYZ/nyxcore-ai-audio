import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockDuplicateReport, mockHealthReport, mockHistoryResponse, mockReviewReport, mockStatus } from "../../lib/mock-data";
import { renderWithProviders } from "../../test/test-utils";
import { DashboardPage } from "../pages/DashboardPage";

describe("Overview first run", () => {
  const status = { ...mockStatus, music_path: "C:\\Synthetic Library\\音楽" };
  let historyStatus: number;
  let reviewOnly: boolean;

  beforeEach(() => {
    historyStatus = 200;
    reviewOnly = false;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const path = new URL(url, "http://localhost").pathname;
      if (path === "/api/status") return Response.json({ ...status, review_only: reviewOnly });
      if (path === "/api/health") return Response.json({ data: mockHealthReport });
      if (path === "/api/review") return Response.json({ data: mockReviewReport });
      if (path === "/api/duplicates") return Response.json({ data: mockDuplicateReport });
      if (path === "/api/history") return Response.json(mockHistoryResponse, { status: historyStatus });
      throw new Error("Unexpected request to " + path);
    }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("shows the configured library and links to a read-only finding", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DashboardPage />);
    expect(await screen.findByRole("heading", { name: "Active library" })).toBeInTheDocument();
    expect(screen.getByText(status.music_path)).toBeInTheDocument();
    expect(screen.getByText(/generating plan previews leave audio files unchanged/i)).toBeInTheDocument();
    expect(await screen.findByRole("link", { name: "Inspect the first finding" }))
      .toHaveAttribute("href", "/review?item=" + mockReviewReport.items[0].item_id);
    expect(screen.getByRole("link", { name: "Open Review Inbox" })).toHaveAttribute("href", "/review");
    await user.click(screen.getByText("Inspect a different library"));
    expect(screen.getByText("NYXCORE_WEB_MUSIC_DIR")).toBeVisible();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("shows launcher instructions for switching libraries in review-only mode", async () => {
    reviewOnly = true;
    const user = userEvent.setup();
    renderWithProviders(<DashboardPage />);

    await user.click(await screen.findByText("Inspect a different library"));
    expect(screen.getByText("--music")).toBeVisible();
    expect(screen.getByText("--out")).toBeVisible();
    expect(screen.queryByText(/Use a disposable library to test Apply/)).toBeNull();
    expect(screen.queryByText("NYXCORE_WEB_MUSIC_DIR")).toBeNull();
  });

  it("keeps the configured library visible when only History fails and refreshes on retry", async () => {
    historyStatus = 500;
    const user = userEvent.setup();
    renderWithProviders(<DashboardPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The local API is connected. Overview could not load recent operations.");
    expect(screen.getByText(status.music_path)).toBeInTheDocument();
    expect(screen.queryByText("Setup Instructions")).not.toBeInTheDocument();
    historyStatus = 200;
    await user.click(screen.getByRole("button", { name: "Refresh Overview" }));
    expect(await screen.findByRole("heading", { name: "Recent Operations" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
