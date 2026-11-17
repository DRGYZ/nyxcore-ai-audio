import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockDuplicateReport, mockHealthReport, mockHistoryResponse, mockReviewReport, mockStatus } from "../../lib/mock-data";
import { renderWithProviders } from "../../test/test-utils";
import { DashboardPage } from "../pages/DashboardPage";
import { AppShell } from "../shell";

describe("Overview first run", () => {
  const status = { ...mockStatus, music_path: "C:\\Synthetic Library\\音楽" };
  let historyStatus: number;
  let reviewOnly: boolean;
  let refreshStatus: number;
  let libraryStatus: number;
  let tracks: number;
  let libraryReads: number;

  beforeEach(() => {
    historyStatus = 200;
    reviewOnly = false;
    refreshStatus = 200;
    libraryStatus = 200;
    tracks = 1007;
    libraryReads = 0;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const path = new URL(url, "http://localhost").pathname;
      if (path === "/api/status") return Response.json({ ...status, review_only: reviewOnly });
      if (path === "/api/library") {
        libraryReads += 1;
        if (libraryStatus !== 200) return Response.json({ detail: "Reports unavailable." }, { status: libraryStatus });
        return Response.json({
          meta: { ...mockHistoryResponse.meta, mode: "snapshot", analysis_generation: String(tracks) },
          analyzed_at: "2026-11-17T00:00:00Z",
          health: { ...mockHealthReport, overview: { ...mockHealthReport.overview, total_audio_files: tracks } },
          review: mockReviewReport,
          duplicates: mockDuplicateReport,
        });
      }
      if (path === "/api/library/refresh") {
        return Response.json(
          refreshStatus === 200 ? {} : { detail: "Library analysis failed. Check the library folder." },
          { status: refreshStatus },
        );
      }
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
    await user.click(await screen.findByRole("button", { name: "Refresh Overview" }));
    expect(await screen.findByRole("heading", { name: "Recent Operations" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shares reports and preserves them when a library refresh or its follow-up read fails", async () => {
    reviewOnly = true;
    const user = userEvent.setup();
    renderWithProviders(<AppShell><DashboardPage /></AppShell>);
    expect(await screen.findByText("1,007")).toBeVisible();
    expect(libraryReads).toBe(1);

    const refresh = () => user.click(screen.getByRole("button", { name: "Refresh Library" }));
    refreshStatus = 503;
    await refresh();
    expect(await screen.findByRole("alert")).toHaveTextContent("Library analysis failed");
    expect(screen.getByText("1,007")).toBeVisible();
    expect(screen.queryByText("Setup Instructions")).toBeNull();
    expect(libraryReads).toBe(1);

    refreshStatus = 200;
    libraryStatus = 503;
    await refresh();
    expect(await screen.findByText(/scanned, but the updated reports could not be loaded/)).toBeVisible();
    expect(screen.getByText("1,007")).toBeVisible();

    libraryStatus = 200;
    tracks = 1008;
    await refresh();
    expect(await screen.findByText("1,008")).toBeVisible();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(libraryReads).toBe(3);
  });
});
