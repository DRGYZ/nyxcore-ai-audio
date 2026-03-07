import { usePlaylistsQuery } from "../../lib/hooks";
import { mockPlaylistsResponse } from "../../lib/mock-data";
import { resolveQueryData, toQueryNoticeState } from "../../lib/query-state";
import { useUrlBackedSelection } from "../../lib/url-selection";
import { Button, Chip, Drawer, EmptyState, PageHeader, PageQueryStateNotice, Panel, PathBlock, formatDate } from "../components";
import { SplitScreen } from "../shell";

export function PlaylistsPage() {
  const playlistsQuery = usePlaylistsQuery();
  const playlistsState = resolveQueryData(playlistsQuery, mockPlaylistsResponse);
  const response = playlistsState.data;
  const { selected, selectById } = useUrlBackedSelection({
    items: response.items,
    param: "playlist",
    idKey: "playlist_id",
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Saved Playlists"
        description="Dynamic collections synthesized from natural-language queries and refreshed against the current local library."
        actions={
          <>
            <Button tone="ghost" disabled>Result History</Button>
            <Button tone="primary" disabled>Create via CLI</Button>
          </>
        }
      />
      <PageQueryStateNotice {...toQueryNoticeState(playlistsState)} />
      <div className="flex flex-wrap gap-3">
        <Chip tone="primary">All Queries</Chip>
        <Chip tone="neutral">High-Res Only</Chip>
        <Chip tone="neutral">Recent Syncs</Chip>
      </div>
      <SplitScreen
        main={
          <div className="space-y-4">
            {response.items.length === 0 ? (
              <EmptyState
                title="No saved playlists"
                description={'Create one with `python -m nyxcore.cli save-playlist <music-dir> --out data/reports --name "Ambient Focus" --query "ambient focus instrumental"`, then inspect its latest refresh diff here.'}
              />
            ) : (
              response.items.map((item) => {
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
          <Drawer title="Diff Analysis" subtitle={selected ? `ID: ${selected.playlist_id}` : undefined}>
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
    </div>
  );
}
