import { afterEach, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { MusicSource } from "../../lib/enrichment";
import type { MetadataTrack } from "../../lib/metadata";
import { renderWithProviders } from "../../test/test-utils";
import { MusicInfo } from "../metadata/MusicInfo";

const unavailable = { value: null, source: "unavailable", confidence: "ambiguous", evidence: "No clue" } as const;
const track: MetadataTrack = {
  path: "/synthetic/Artist - Meridian.flac", tags: { title: [], artist: [], album: [] },
  supported: true, readable: true, write_blocker: null, issues: ["missing_title"], swap: false,
  suggestions: {
    title: { value: "Meridian", source: "filename", confidence: "tentative", evidence: "Filename" },
    artist: unavailable, album: unavailable,
  },
  artwork_supported: true, cover_present: false, duration_ms: 192000, generation: "generation", inspection: "inspection",
};
const recording: MusicSource = { recording_id: "recording-one", title: "Méridian", artist: "Écho", release: null, token: "r" };
const release: MusicSource = { ...recording, release: { kind: "release", id: "jp-one", title: "Lumière 夜" }, token: "a" };

afterEach(() => vi.unstubAllGlobals());

it("looks up on explicit submit, distinguishes releases, and forwards only a selected cover", async () => {
  const user = userEvent.setup();
  const useTags = vi.fn();
  const useCover = vi.fn();
  const accidentalSubmit = vi.fn((event) => event.preventDefault());
  const requests = vi.fn(async (value: string) => {
    const path = new URL(value).pathname;
    if (path.endsWith("/lookup")) return Response.json({
      status: "ambiguous",
      candidates: [{
        id: "recording-one", title: "Méridian", artist: "Écho", artists: [], duration_ms: 192000,
        evidence: { title: "agrees", artist: "agrees", duration: "agrees", album: "unavailable", filename: "similar" },
        source: recording, score: 98,
        releases: [
          { ...release.release, date: "2024", country: "JP", source: release },
          { kind: "release", id: "fr-two", title: "Reissued", date: "2025", country: "FR",
            source: { ...recording, release: { kind: "release", id: "fr-two", title: "Reissued" }, token: "b" } },
        ],
      }],
    });
    if (path.endsWith("/covers")) return Response.json({ covers: [{
      id: "1", release_id: "jp-one", extension: ".jpg", token: "cover", label: "Original front", thumbnail: null,
    }] });
    if (path.endsWith("/cover-select")) return Response.json({
      selection: { digest: "image-sha", release_id: "jp-one", image_id: "1", width: 300, height: 200,
        token: "approved-cover", replace_existing: false },
      thumbnail: "data:image/jpeg;base64,AA==",
    });
    throw new Error(path);
  });
  vi.stubGlobal("fetch", requests);
  renderWithProviders(<form onSubmit={accidentalSubmit}><MusicInfo track={track} onTags={useTags} onCover={useCover} /></form>);
  await user.click(screen.getByRole("button", { name: "Find Music Info" }));
  expect(requests).not.toHaveBeenCalled();
  const query = screen.getByRole("textbox", { name: "Search title" });
  expect(query).toHaveValue("Meridian");
  await user.type(query, "{Enter}");
  expect(accidentalSubmit).not.toHaveBeenCalled();
  expect(await screen.findByText(/Several matches may be plausible/)).toBeVisible();
  expect(useTags).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Select recording" }));
  expect(screen.getByRole("combobox", { name: "Release or release group" })).toHaveValue("");
  await user.selectOptions(screen.getByRole("combobox", { name: "Release or release group" }), "release:jp-one");
  await user.click(screen.getByRole("button", { name: "Use suggested tags" }));
  expect(useTags).toHaveBeenCalledWith(release);
  await user.click(screen.getByRole("button", { name: "Find front covers" }));
  expect(await screen.findByText("Preview unavailable")).toBeVisible();
  expect(useCover).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Choose cover 1" }));
  await waitFor(() => expect(useCover).toHaveBeenCalledWith(release, expect.objectContaining({
    selection: expect.objectContaining({ release_id: "jp-one", replace_existing: false }),
  })));
});

it("shows provider failure with no automatic metadata selection", async () => {
  const request = vi.fn(async () => Response.json({ detail: "The provider could not be reached." }, { status: 503 }));
  vi.stubGlobal("fetch", request);
  const user = userEvent.setup();
  const useTags = vi.fn();
  renderWithProviders(<MusicInfo track={track} onTags={useTags} onCover={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Find Music Info" }));
  await user.click(screen.getByRole("button", { name: "Search MusicBrainz" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The provider could not be reached.");
  expect(useTags).not.toHaveBeenCalled();
});
