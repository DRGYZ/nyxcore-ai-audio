import { afterEach, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Outlet, Route, Routes, useSearchParams } from "react-router-dom";
import type { OrganizationProposal } from "../../lib/organization";
import { renderWithProviders } from "../../test/test-utils";
import { OrganizationPage } from "../pages/OrganizationPage";

const rows: OrganizationProposal[] = [
  { path: "/music/inbox/live.flac", source: "inbox/live.flac", status: "ready", inspectable: true,
    reasons: ["Grouped by embedded album artist and album."], destination: ["Artist", "Album", "Glass (Live).flac"] },
  { path: "/music/review.flac", source: "review.flac", status: "needs_review", inspectable: true,
    reasons: ["Add a reliable album tag."], destination: null },
  { path: "/music/conflict.flac", source: "conflict.flac", status: "collision", inspectable: true,
    reasons: ["Multiple tracks propose the same destination."], destination: ["Artist", "Album", "Occupied.flac"] },
];

afterEach(() => vi.unstubAllGlobals());

function LocationProbe() {
  const [params] = useSearchParams();
  return <><p>{params.get("editTrack")}</p><Outlet /></>;
}

it("builds on request, pages and filters files, and points to the existing editor", async () => {
  const user = userEvent.setup();
  const request = vi.fn(async (input: string, settings?: RequestInit) => {
    const url = new URL(input);
    if (settings?.method === "POST") return Response.json({
      id: "one", root: "/music", counts: { ready: 1, needs_review: 1, collision: 1, unchanged: 0 },
    });
    const offset = Number(url.searchParams.get("offset"));
    const status = url.searchParams.get("status");
    const found = status === "all" ? rows : rows.filter((row) => row.status === status);
    return Response.json({ rows: found.slice(offset, offset + 2), total: found.length, page_size: 2 });
  });
  vi.stubGlobal("fetch", request);
  renderWithProviders(<Routes><Route element={<LocationProbe />}><Route path="/" element={<OrganizationPage />} /></Route></Routes>);
  expect(request).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Build preview" }));
  expect(await screen.findByText("Artist / Album / Glass (Live).flac")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Next page" }));
  expect(await screen.findByText("Multiple tracks propose the same destination.")).toBeVisible();
  await user.selectOptions(screen.getByRole("combobox", { name: "Show status" }), "needs_review");
  expect(await screen.findByText("Add a reliable album tag.")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Edit metadata for review.flac" }));
  expect(screen.getByText("/music/review.flac")).toBeVisible();
  expect(screen.queryByRole("button", { name: /apply|move/i })).not.toBeInTheDocument();
  expect(request.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
});

it("shows only safe destinations in the expandable tree and handles expiration", async () => {
  const user = userEvent.setup();
  let expired = false;
  vi.stubGlobal("fetch", vi.fn(async (input: string, settings?: RequestInit) => {
    if (settings?.method === "POST") return Response.json({
      id: "two", root: "/music", counts: { ready: 1, needs_review: 1, collision: 1, unchanged: 0 },
    });
    if (expired) return Response.json({ detail: "This preview is no longer current. Build a new preview." }, { status: 409 });
    const folderView = new URL(input).searchParams.get("view") === "folders";
    return Response.json({ rows: folderView ? rows.slice(0, 1) : rows, total: folderView ? 1 : 3, page_size: 50 });
  }));
  renderWithProviders(<OrganizationPage />);
  await user.click(screen.getByRole("button", { name: "Build preview" }));
  await screen.findByText("inbox/live.flac");
  await user.click(screen.getByRole("button", { name: "Folder tree" }));
  const tree = await screen.findByRole("list", { name: "Proposed folders" });
  expect(within(tree).getByText("Artist")).toBeVisible();
  expect(within(tree).getByText("Album")).toBeVisible();
  expect(within(tree).getByText("Glass (Live).flac")).toBeVisible();
  expect(within(tree).queryByText("Occupied.flac")).not.toBeInTheDocument();
  expired = true;
  await user.selectOptions(screen.getByRole("combobox", { name: "Show status" }), "ready");
  expect(await screen.findByText(/This preview is no longer current/)).toBeVisible();
});
