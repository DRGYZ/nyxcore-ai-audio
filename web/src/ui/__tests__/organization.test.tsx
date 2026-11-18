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
  const unchanged: OrganizationProposal = {
    path: "/music/Artist/Album/Cataloged.flac", source: "Artist/Album/Cataloged.flac",
    status: "unchanged", inspectable: true,
    reasons: ["The track is already at its proposed destination."],
    destination: ["Artist", "Album", "Cataloged.flac"],
  };
  let expired = false;
  vi.stubGlobal("fetch", vi.fn(async (input: string, settings?: RequestInit) => {
    if (settings?.method === "POST") return Response.json({
      id: "two", root: "/music", counts: { ready: 1, needs_review: 1, collision: 1, unchanged: 0 },
    });
    if (expired) return Response.json({ detail: "This preview is no longer current. Build a new preview." }, { status: 409 });
    const folderView = new URL(input).searchParams.get("view") === "folders";
    return Response.json({ rows: folderView ? [rows[0], unchanged] : rows, total: folderView ? 2 : 3, page_size: 50 });
  }));
  renderWithProviders(<OrganizationPage />);
  await user.click(screen.getByRole("button", { name: "Build preview" }));
  await screen.findByText("inbox/live.flac");
  await user.click(screen.getByRole("button", { name: "Folder tree" }));
  const tree = await screen.findByRole("list", { name: "Proposed folders" });
  expect(within(tree).getByText("Artist")).toBeVisible();
  expect(within(tree).getByText("Album")).toBeVisible();
  expect(within(tree).getByText("Glass (Live).flac")).toBeVisible();
  expect(within(tree).getByText("Cataloged.flac")).toBeVisible();
  expect(within(tree).getByText("Ready")).toBeVisible();
  expect(within(tree).getByText("Unchanged")).toBeVisible();
  expect(within(tree).queryByText("Occupied.flac")).not.toBeInTheDocument();
  expired = true;
  await user.selectOptions(screen.getByRole("combobox", { name: "Show status" }), "ready");
  expect(await screen.findByText(/This preview is no longer current/)).toBeVisible();
});

it("applies only the exact reviewed selection after a separate confirmation", async () => {
  const user = userEvent.setup();
  const request = vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(input).pathname;
    if (path.endsWith("/review")) {
      expect(JSON.parse(String(init?.body))).toEqual({ paths: [rows[0].path] });
      return Response.json({ approval: "one-time", moves: [
        { source: rows[0].path, destination: "/music/Artist/Album/Glass (Live).flac" },
      ] });
    }
    if (path.endsWith("/apply")) {
      expect(JSON.parse(String(init?.body))).toEqual({ paths: [rows[0].path], confirm: true, approval: "one-time" });
      return Response.json({ batch_id: "organization-batch", operations: [
        { operation_id: "1", path: rows[0].path, destination_path: "/music/Artist/Album/Glass (Live).flac",
          status: "ok", message: "applied" },
      ] });
    }
    if (init?.method === "POST") return Response.json({
      id: "enabled", root: "/music", apply_enabled: true, counts: { ready: 1, needs_review: 1, collision: 1, unchanged: 0 },
    });
    return Response.json({ rows, total: rows.length, page_size: 50 });
  });
  vi.stubGlobal("fetch", request);
  renderWithProviders(<OrganizationPage />);
  await user.click(screen.getByRole("button", { name: "Build preview" }));
  const track = await screen.findByRole("checkbox", { name: "Select inbox/live.flac" });
  expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  expect(track).not.toBeChecked();
  await user.click(track);
  await user.click(screen.getByRole("button", { name: "Review selected moves" }));
  expect(await screen.findByText("/music/Artist/Album/Glass (Live).flac")).toBeVisible();
  expect(screen.getByRole("button", { name: "Apply approved moves" })).toBeDisabled();
  await user.click(screen.getByRole("checkbox", { name: /I approve moving/ }));
  await user.click(track);
  expect(screen.queryByRole("button", { name: "Apply approved moves" })).not.toBeInTheDocument();
  await user.click(track);
  await user.click(screen.getByRole("button", { name: "Review selected moves" }));
  expect(await screen.findByRole("button", { name: "Apply approved moves" })).toBeDisabled();
  await user.click(screen.getByRole("checkbox", { name: /I approve moving/ }));
  await user.click(screen.getByRole("button", { name: "Apply approved moves" }));
  expect(await screen.findByText("1 of 1 files moved")).toBeVisible();
  expect(screen.getByRole("link", { name: "Open History to inspect or undo" })).toHaveAttribute(
    "href", "/history?batch=organization-batch",
  );
});
