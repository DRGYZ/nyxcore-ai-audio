import { useMemo, useState } from "react";
import { useCreatePlaylistMutation, usePlaylistsQuery, useRefreshPlaylistMutation } from "../../lib/hooks";
import { useUrlBackedSelection } from "../../lib/url-selection";
import { ActionBanner, Button, Chip, Drawer, EmptyState, Modal, PageHeader, Panel, PathBlock, formatDate } from "../components";
import { ApiUnavailableState } from "../feedback";
import { SplitScreen } from "../shell";

type PlaylistFilter = "all" | "recent" | "never";

export function PlaylistsPage() {
  const playlistsQuery = usePlaylistsQuery();
  const createMutation = useCreatePlaylistMutation();
  const refreshMutation = useRefreshPlaylistMutation();
  const [playlistFilter, setPlaylistFilter] = useState<PlaylistFilter>("all");
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [query, setQuery] = useState("");
  const [maxTracks, setMaxTracks] = useState("25");
  const [banner, setBanner] = useState<{ tone: "info" | "success" | "error"; message: string } | null>(null);

  const response = playlistsQuery.data;
  const usingMock = false;
  const items = useMemo(() => response?.items ?? [], [response?.items]);
  const filtered = useMemo(() => items.filter((item) => {
    if (playlistFilter === "all") return true;
    if (playlistFilter === "never") return !item.last_refreshed_at;
    if (!item.last_refreshed_at) return false;
    return Date.now() - Date.parse(item.last_refreshed_at) <= 7 * 24 * 60 * 60 * 1000;
  }), [playlistFilter, items]);
  const { selected, selectById } = useUrlBackedSelection({
    items: filtered,
    param: "playlist",
    idKey: "playlist_id",
  });
  const busy = createMutation.isPending || refreshMutation.isPending;

  if (playlistsQuery.isError) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Playlists"
          title="Saved Playlists"
          description="Dynamic collections synthesized from natural-language queries and refreshed against the current local library."
        />
        <ApiUnavailableState contextLabel="Saved Playlists" />
      </div>
    );
  }

  if (playlistsQuery.isLoading || !response) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Playlists"
          title="Saved Playlists"
          description="Dynamic collections synthesized from natural-language queries and refreshed against the current local library."
        />
        <Panel className="p-8 text-center text-sm text-slate-400">
          Loading saved playlists from local API…
        </Panel>
      </div>
    );
  }

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
                    <Panel className={`p-5 transition-all ${active ? "border-accent/40 bg-surface-raised" : "hover:border-white/[0.12]"}`}>
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <p className="mb-1 truncate font-mono text-[10px] text-accent/80">QUERY: {item.query}</p>
                          <h3 className="truncate font-sans text-base font-semibold text-primary">{item.name}</h3>
                        </div>
                        <div className="flex shrink-0 gap-2">
                          <span className="rounded-[2px] bg-accent/10 p-1.5 text-accent">
                            <span className="material-symbols-outlined text-lg">play_arrow</span>
                          </span>
                        </div>
                      </div>
                      <div className="mt-4 flex flex-wrap items-center gap-5 border-t border-white/[0.05] pt-3">
                        <div className="flex flex-col">
                          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Track Count</span>
                          <span className="font-mono text-xs text-primary">{item.track_count} items</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Last Refreshed</span>
                          <span className="font-mono text-xs text-primary">{formatDate(item.last_refreshed_at)}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Profile</span>
                          <Chip tone="default">{item.profile}</Chip>
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
                <div className="rounded-[3px] border border-white/[0.07] bg-surface-low/80 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <Chip tone="primary">Dynamic Playlist</Chip>
                    <span className="font-mono text-[10px] text-primary-subtle">{selected.playlist_id}</span>
                  </div>
                  <h3 className="font-sans text-base font-semibold text-primary">{selected.name}</h3>
                  <p className="mt-1 font-editorial text-xs italic text-primary-subtle">
                    Comparing saved manifest against the latest local-library refresh.
                  </p>
                </div>
                <div>
                  <div className="mb-2.5 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-accent">queue_music</span>
                    <h4 className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-accent">
                      Current Tracks ({selected.latest_tracks.length})
                    </h4>
                  </div>
                  {selected.latest_tracks.length > 0 ? (
                    <div className="max-h-96 space-y-2 overflow-y-auto pr-1">
                      {selected.latest_tracks.slice(0, 50).map((track, index) => (
                        <div key={`${track.path}-${index}`} className="rounded-[3px] border border-white/[0.06] bg-surface-low/50 p-2.5">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <p className="truncate font-sans text-xs font-medium text-primary">{track.title || track.path.split(/[\\/]/).pop()}</p>
                              <p className="mt-0.5 truncate font-editorial text-xs italic text-primary-subtle">{track.artist || "Unknown artist"}</p>
                            </div>
                            <Chip tone="default">{track.score.toFixed(1)}</Chip>
                          </div>
                          <div className="mt-2"><PathBlock value={track.path} /></div>
                        </div>
                      ))}
                      {selected.latest_tracks.length > 50 ? (
                        <p className="text-center font-mono text-xs text-primary-subtle">Showing the first 50 tracks.</p>
                      ) : null}
                    </div>
                  ) : (
                    <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/40 px-4 py-3 text-xs text-primary-subtle">
                      Refresh this playlist to generate its current track list.
                    </div>
                  )}
                </div>
                <div>
                  <div className="mb-2.5 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-emerald-400">add_circle</span>
                    <h4 className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-400">
                      Added to Collection ({((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).length})
                    </h4>
                  </div>
                  {(((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).length > 0) ? (
                    <div className="space-y-2">
                      {(((selected.latest_refresh_diff.tracks_added as string[] | undefined) ?? []).slice(0, 3)).map((path) => (
                        <div key={path} className="rounded-[3px] border border-emerald-500/20 bg-emerald-500/[0.03] p-2.5">
                          <p className="font-sans text-xs font-medium text-primary">{path.split(/[\\/]/).pop()}</p>
                          <div className="mt-2">
                            <PathBlock value={path} tone="success" />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/40 px-4 py-3 text-xs text-primary-subtle">No tracks were added on the latest refresh.</div>
                  )}
                </div>
                <div>
                  <div className="mb-2.5 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm text-rose-400">do_not_disturb_on</span>
                    <h4 className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-rose-400">
                      Removed from Collection ({((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).length})
                    </h4>
                  </div>
                  {(((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).length > 0) ? (
                    <div className="space-y-2">
                      {(((selected.latest_refresh_diff.tracks_removed as string[] | undefined) ?? []).slice(0, 3)).map((path) => (
                        <div key={path} className="rounded-[3px] border border-rose-500/20 bg-rose-500/[0.03] p-2.5 opacity-70">
                          <p className="font-sans text-xs font-medium text-primary-muted line-through">{path.split(/[\\/]/).pop()}</p>
                          <div className="mt-2">
                            <PathBlock value={path} tone="danger" strike />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/40 px-4 py-3 text-xs text-primary-subtle">No tracks dropped out on the latest refresh.</div>
                  )}
                </div>
                <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3.5">
                  <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Refresh Summary</p>
                  <div className="mt-2.5 grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-primary-subtle">Track Count Delta</p>
                      <p className="font-mono font-semibold text-accent">{String(selected.latest_refresh_diff.track_count_delta ?? 0)}</p>
                    </div>
                    <div>
                      <p className="text-primary-subtle">Duration Delta</p>
                      <p className="font-mono font-semibold text-primary">{String(selected.latest_refresh_diff.estimated_duration_delta_seconds ?? 0)} sec</p>
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
            <span className="mb-1.5 block font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={120}
              placeholder="Night Drive"
              className="w-full rounded-[3px] border border-white/[0.08] bg-surface-low px-3 py-2 font-sans text-xs text-primary outline-none focus:border-accent"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Query</span>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              maxLength={500}
              rows={4}
              placeholder="dark electronic under 5 minutes no vocals"
              className="w-full resize-none rounded-[3px] border border-white/[0.08] bg-surface-low px-3 py-2 font-sans text-xs text-primary outline-none focus:border-accent"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block font-mono text-[10px] uppercase tracking-[0.16em] text-primary-subtle">Maximum Tracks</span>
            <input
              type="number"
              min={1}
              max={500}
              value={maxTracks}
              onChange={(event) => setMaxTracks(event.target.value)}
              className="w-full rounded-[3px] border border-white/[0.08] bg-surface-low px-3 py-2 font-sans text-xs text-primary outline-none focus:border-accent"
            />
          </label>
          <p className="font-editorial text-xs italic text-primary-subtle">A refreshed M3U file is exported beside the saved playlist data.</p>
        </div>
      </Modal>
    </div>
  );
}
