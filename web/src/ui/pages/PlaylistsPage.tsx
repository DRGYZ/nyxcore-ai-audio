import { useMemo, useState } from "react";
import { useCreatePlaylistMutation, usePlaylistsQuery, useRefreshPlaylistMutation } from "../../lib/hooks";
import { mockPlaylistsResponse } from "../../lib/mock-data";
import { resolveQueryData, toQueryNoticeState } from "../../lib/query-state";
import { useUrlBackedSelection } from "../../lib/url-selection";
import { ActionBanner, Button, Chip, Drawer, EmptyState, Modal, PageHeader, PageQueryStateNotice, Panel, PathBlock, formatDate } from "../components";
import { SplitScreen } from "../shell";

type PlaylistFilter = "all" | "recent" | "never";

export function PlaylistsPage() {
  const playlistsQuery = usePlaylistsQuery();
  const createMutation = useCreatePlaylistMutation();
  const refreshMutation = useRefreshPlaylistMutation();
  const playlistsState = resolveQueryData(playlistsQuery, mockPlaylistsResponse);
  const response = playlistsState.data;
  const usingMock = playlistsState.usingMock;
  const [playlistFilter, setPlaylistFilter] = useState<PlaylistFilter>("all");
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [query, setQuery] = useState("");
  const [maxTracks, setMaxTracks] = useState("25");
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);
  const filtered = useMemo(() => response.items.filter((item) => {
    if (playlistFilter === "all") return true;
    if (playlistFilter === "never") return !item.last_refreshed_at;
    if (!item.last_refreshed_at) return false;
    return Date.now() - Date.parse(item.last_refreshed_at) <= 7 * 24 * 60 * 60 * 1000;
  }), [playlistFilter, response.items]);
  const { selected, selectById } = useUrlBackedSelection({
    items: filtered,
    param: "playlist",
    idKey: "playlist_id",
  });
  const busy = createMutation.isPending || refreshMutation.isPending;

  async function handleCreate() {
    const parsedMaxTracks = Number.parseInt(maxTracks, 10);
    try {
      const created = await createMutation.mutateAsync({
        name: name.trim(),
        query: query.trim(),
        max_tracks: Number.isFinite(parsedMaxTracks) ? parsedMaxTracks : undefined,
        export_m3u: true,
      });
      setBanner({
        tone: "success",
        message: `Created ${created.item.name} with ${created.item.track_count} tracks${created.m3u_path ? ` and exported ${created.m3u_path}` : ""}.`,
      });
      setCreateOpen(false);
      setName("");
      setQuery("");
      setMaxTracks("25");
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to create playlist." });
    }
  }

  async function handleRefresh() {
    if (!selected) return;
    try {
      const refreshed = await refreshMutation.mutateAsync({ playlistId: selected.playlist_id, export_m3u: true });
      setBanner({
        tone: "success",
        message: `Refreshed ${refreshed.item.name}: ${refreshed.item.track_count} tracks${refreshed.m3u_path ? `; M3U: ${refreshed.m3u_path}` : ""}.`,
      });
    } catch (error) {
      setBanner({ tone: "error", message: error instanceof Error ? error.message : "Unable to refresh playlist." });
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Saved Playlists"
        description="Dynamic collections synthesized from natural-language queries and refreshed against the current local library."
        actions={
          <>
            <Button tone="ghost" disabled={usingMock || busy || !selected} onClick={() => void handleRefresh()}>
              {refreshMutation.isPending ? "Refreshing..." : "Refresh Selected"}
            </Button>
            <Button tone="primary" disabled={usingMock || busy} onClick={() => setCreateOpen(true)}>
              Create Playlist
            </Button>
          </>
        }
      />
      <PageQueryStateNotice {...toQueryNoticeState(playlistsState)} />
      {banner ? <ActionBanner tone={banner.tone} message={banner.message} /> : null}
      <div className="flex flex-wrap gap-3">
        {([
          ["all", "All Queries"],
          ["recent", "Refreshed This Week"],
          ["never", "Never Refreshed"],
        ] as Array<[PlaylistFilter, string]>).map(([value, label]) => (
          <button key={value} type="button" onClick={() => setPlaylistFilter(value)}>
            <Chip tone={playlistFilter === value ? "primary" : "neutral"} active={playlistFilter === value}>
              {label}
            </Chip>
          </button>
        ))}
      </div>
      <SplitScreen
        main={
          <div className="space-y-4">
            {filtered.length === 0 ? (
              <EmptyState
                title={response.items.length === 0 ? "No saved playlists" : "No playlists match this filter"}
                description={response.items.length === 0
                  ? "Create a playlist from a natural-language query, then refresh it whenever the local library changes."
                  : "Choose another refresh filter to bring playlists back into view."}
                action={response.items.length === 0 && !usingMock
                  ? <Button tone="primary" onClick={() => setCreateOpen(true)}>Create Playlist</Button>
                  : undefined}
              />
            ) : (
              filtered.map((item) => {
                const active = item.playlist_id === selected?.playlist_id;
                return (
                  <button
                    key={item.playlist_id}
                    type="button"
                    className="block w-full text-left"
                    onClick={() => selectById(item.playlist_id)}
                  >
                    <Panel className={`p-5 transition-all ${active ? "border-primary/40 shadow-[0_0_20px_rgba(37,226,244,0.08)]" : "hover:border-primary/30"}`}>
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <p className="mb-1 truncate font-mono text-[10px] text-primary/60">QUERY: {item.query}</p>
                          <h3 className="truncate font-display text-xl font-bold text-slate-100">{item.name}</h3>
                        </div>
                        <div className="flex shrink-0 gap-2">
                          <span className="rounded-lg bg-primary/10 p-2 text-primary">
                            <span className="material-symbols-outlined text-xl">play_arrow</span>
                          </span>
                          <span className="p-2 text-slate-500">
                            <span className="material-symbols-outlined">more_vert</span>
                          </span>
                        </div>
                      </div>
                      <div className="mt-5 flex flex-wrap items-center gap-6 border-t border-primary/5 pt-4">
                        <div className="flex flex-col">
                          <span className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Track Count</span>
                          <span className="font-mono text-sm text-slate-200">{item.track_count} items</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Last Refreshed</span>
                          <span className="font-mono text-sm text-slate-200">{formatDate(item.last_refreshed_at)}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Profile</span>
                          <Chip tone="primary">{item.profile}</Chip>
                        </div>
                      </div>
                    </Panel>
                  </button>
                );
              })
            )}
          </div>
        }
        side={
          <Drawer
            title="Playlist Details"
            subtitle={selected ? `ID: ${selected.playlist_id}` : undefined}
            footer={selected ? (
              <Button tone="primary" className="w-full" disabled={usingMock || busy} onClick={() => void handleRefresh()}>
                {refreshMutation.isPending ? "Refreshing..." : "Refresh and Export M3U"}
              </Button>
            ) : undefined}
          >
            {selected ? (
              <>
                <div className="rounded-xl border border-primary/10 bg-primary/5 p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <Chip tone="primary">Engine Analysis</Chip>
                    <span className="font-mono text-[10px] text-slate-400">{selected.playlist_id}</span>
                  </div>
                  <h3 className="font-display text-2xl font-bold text-slate-100">{selected.name}</h3>
                  <p className="mt-2 text-sm text-slate-400">
                    Comparing saved manifest against the latest local-library refresh.
                  </p>
                </div>
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-primary">queue_music</span>
                    <h4 className="text-xs font-bold uppercase tracking-[0.24em] text-primary">
                      Current Tracks ({selected.latest_tracks.length})
                    </h4>
                  </div>
                  {selected.latest_tracks.length > 0 ? (
                    <div className="max-h-96 space-y-2 overflow-y-auto pr-1">
                      {selected.latest_tracks.slice(0, 50).map((track, index) => (
                        <div key={`${track.path}-${index}`} className="rounded-lg border border-primary/10 bg-background-dark/40 p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-slate-200">{track.title || track.path.split(/[\\/]/).pop()}</p>
                              <p className="mt-1 truncate text-xs text-slate-500">{track.artist || "Unknown artist"}</p>
                            </div>
                            <Chip tone="primary">{track.score.toFixed(1)}</Chip>
                          </div>
                          <div className="mt-2"><PathBlock value={track.path} /></div>
                        </div>
                      ))}
                      {selected.latest_tracks.length > 50 ? (
                        <p className="text-center text-xs text-slate-500">Showing the first 50 tracks.</p>
                      ) : null}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border-dark bg-background-dark/40 px-4 py-3 text-sm text-slate-500">
                      Refresh this playlist to generate its current track list.
                    </div>
                  )}
                </div>
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-emerald-400">add_circle</span>
                    <h4 className="text-xs font-bold uppercase tracking-[0.24em] text-emerald-400">
                      Added to Collection ({((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).length})
                    </h4>
                  </div>
                  {(((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).length > 0) ? (
                    <div className="space-y-2">
                      {(((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).slice(0, 3)).map((path) => (
                        <div key={path} className="rounded-lg border border-emerald-500/10 bg-emerald-500/5 p-3">
                          <p className="text-sm font-medium text-slate-200">{path.split(/[\\/]/).pop()}</p>
                          <div className="mt-2">
                            <PathBlock value={path} tone="success" />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border-dark bg-background-dark/40 px-4 py-3 text-sm text-slate-500">No tracks were added on the latest refresh.</div>
                  )}
                </div>
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-rose-400">do_not_disturb_on</span>
                    <h4 className="text-xs font-bold uppercase tracking-[0.24em] text-rose-400">
                      Removed from Collection ({((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).length})
                    </h4>
                  </div>
                  {(((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).length > 0) ? (
                    <div className="space-y-2">
                      {(((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).slice(0, 3)).map((path) => (
                        <div key={path} className="rounded-lg border border-rose-500/10 bg-rose-500/5 p-3 opacity-70">
                          <p className="text-sm font-medium text-slate-400 line-through">{path.split(/[\\/]/).pop()}</p>
                          <div className="mt-2">
                            <PathBlock value={path} tone="danger" strike />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border-dark bg-background-dark/40 px-4 py-3 text-sm text-slate-500">No tracks dropped out on the latest refresh.</div>
                  )}
                </div>
                <div className="rounded-xl border border-primary/10 bg-background-dark/40 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.24em] text-slate-500">Refresh Summary</p>
                  <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-slate-500">Track Count Delta</p>
                      <p className="font-bold text-primary">{String(selected.latest_refresh_diff.track_count_delta ?? 0)}</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Duration Delta</p>
                      <p className="font-bold text-slate-200">{String(selected.latest_refresh_diff.estimated_duration_delta_seconds ?? 0)} sec</p>
                    </div>
                  </div>
                </div>
              </>
            ) : null}
          </Drawer>
        }
      />

      <Modal
        open={createOpen}
        title="Create Saved Playlist"
        subtitle="The playlist is generated immediately from the current local library."
        onClose={() => {
          if (!createMutation.isPending) setCreateOpen(false);
        }}
        footer={
          <>
            <Button tone="ghost" disabled={createMutation.isPending} onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button
              tone="primary"
              disabled={
                createMutation.isPending
                || !name.trim()
                || !query.trim()
                || Number(maxTracks) < 1
                || Number(maxTracks) > 500
              }
              onClick={() => void handleCreate()}
            >
              {createMutation.isPending ? "Creating..." : "Create and Refresh"}
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <label className="block">
            <span className="mb-2 block text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={120}
              placeholder="Night Drive"
              className="w-full rounded-lg border border-border-dark bg-background-dark px-3 py-2 text-sm text-slate-200 outline-none focus:border-primary"
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Query</span>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              maxLength={500}
              rows={4}
              placeholder="dark electronic under 5 minutes no vocals"
              className="w-full resize-none rounded-lg border border-border-dark bg-background-dark px-3 py-2 text-sm text-slate-200 outline-none focus:border-primary"
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Maximum Tracks</span>
            <input
              type="number"
              min={1}
              max={500}
              value={maxTracks}
              onChange={(event) => setMaxTracks(event.target.value)}
              className="w-full rounded-lg border border-border-dark bg-background-dark px-3 py-2 text-sm text-slate-200 outline-none focus:border-primary"
            />
          </label>
          <p className="text-xs text-slate-500">A refreshed M3U file is exported beside the saved playlist data.</p>
        </div>
      </Modal>
    </div>
  );
}
