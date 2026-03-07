import type { PropsWithChildren, ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useReviewQuery, useStatusQuery } from "../lib/hooks";
import { mockReviewReport, mockStatus } from "../lib/mock-data";
import { Chip, Icon } from "./components";

const navItems = [
  { to: "/", label: "Mission Control", icon: "dashboard" },
  { to: "/review", label: "Review Inbox", icon: "inbox" },
  { to: "/playlists", label: "Saved Playlists", icon: "auto_awesome" },
  { to: "/history", label: "Operation History", icon: "history_edu" },
  { to: "/duplicates", label: "Duplicates", icon: "copy_all" },
  { to: "/health", label: "Health", icon: "analytics" },
];

function SidebarLink({ to, label, icon, badge }: { to: string; label: string; icon: string; badge?: number }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
          isActive
            ? "border border-primary/20 bg-primary/10 text-primary"
            : "text-slate-400 hover:bg-primary/5 hover:text-primary"
        }`
      }
    >
      <Icon name={icon} className="text-xl" />
      <span>{label}</span>
      {badge ? <span className="ml-auto rounded bg-primary px-1.5 py-0.5 text-[10px] font-bold text-background-dark">{badge}</span> : null}
    </NavLink>
  );
}

export function AppShell({ children }: PropsWithChildren) {
  const statusQuery = useStatusQuery();
  const reviewQuery = useReviewQuery();
  const status = statusQuery.data ?? mockStatus;
  const review = reviewQuery.data?.data ?? mockReviewReport;
  const reviewCount = review.items.filter((item) => item.review_status === "new").length;
  const unresolvedReviewCount = review.items.filter((item) => item.review_status === "new" || item.review_status === "seen").length;
  const usingMock = !statusQuery.data || !reviewQuery.data;

  return (
    <div className="min-h-screen bg-background-dark text-slate-100">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="border-r border-primary/10 bg-background-dark lg:w-72 lg:shrink-0">
          <div className="flex items-center gap-3 px-6 py-6">
            <div className="flex size-10 items-center justify-center rounded-lg border border-primary/30 bg-primary/20 text-primary">
              <Icon name="waves" className="text-xl" />
            </div>
            <div>
              <h1 className="font-display text-lg font-bold">NyxCore</h1>
              <p className="text-xs font-medium text-primary/60">Local library control surface</p>
            </div>
          </div>
          <nav className="space-y-6 px-4 pb-6">
            <div>
              <p className="px-3 text-[10px] font-bold uppercase tracking-[0.28em] text-slate-500">Navigation</p>
              <div className="mt-3 space-y-1">
                {navItems.map((item) => (
                  <SidebarLink key={item.to} {...item} badge={item.to === "/review" ? reviewCount : undefined} />
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-primary/10 bg-gradient-to-br from-primary/10 to-transparent p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-bold uppercase tracking-[0.24em] text-primary">Workspace Snapshot</p>
                <Chip tone={usingMock ? "warning" : "primary"}>{usingMock ? "mock" : "live"}</Chip>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Open Review</p>
                  <p className="mt-1 text-lg font-bold text-slate-100">{unresolvedReviewCount}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Saved Playlists</p>
                  <p className="mt-1 text-lg font-bold text-primary">{status.saved_playlist_count}</p>
                </div>
              </div>
              <p className="mt-3 text-[10px] text-slate-400">
                {usingMock ? "Showing fallback examples because live API data is unavailable." : "Live status is sourced from the current API session."}
              </p>
            </div>
          </nav>
        </aside>
        <div className="flex min-h-screen min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex items-center justify-between border-b border-primary/10 bg-background-dark/70 px-6 py-4 backdrop-blur-md">
            <div className="flex min-w-0 items-center gap-6">
              <div className="hidden items-center gap-2 text-primary sm:flex">
                <Icon name="folder_open" className="text-lg" />
                <span className="truncate font-mono text-xs text-slate-300">{status.music_path}</span>
              </div>
              <div className="hidden h-4 w-px bg-primary/20 md:block" />
              <div className="hidden items-center gap-2 md:flex">
                <span className={`size-2 rounded-full ${usingMock ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.55)]" : "bg-primary shadow-[0_0_8px_#25e2f4]"}`} />
                <span className="text-[10px] font-bold uppercase tracking-[0.28em] text-slate-500">
                  {usingMock ? "Mock Fallback" : "Live API"}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="relative hidden md:block">
                <Icon name="search" className="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-slate-500" />
                <input
                  className="w-64 rounded-full border border-primary/10 bg-primary/5 py-1.5 pl-10 pr-4 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-primary focus:ring-1 focus:ring-primary"
                  placeholder="Search archive..."
                  type="text"
                />
              </div>
              <button className="relative rounded-lg bg-primary/10 p-2 text-primary transition-colors hover:bg-primary/20">
                <Icon name="notifications" className="text-xl" />
                <span className="absolute right-2 top-2 size-2 rounded-full border-2 border-background-dark bg-rose-500" />
              </button>
              <Chip tone="primary">{status.active_profile}</Chip>
              <div className="size-10 rounded-full border border-primary/30 bg-gradient-to-br from-primary/40 to-secondary/40" />
            </div>
          </header>
          <main className="flex-1 overflow-y-auto bg-[radial-gradient(circle_at_top_right,_rgba(37,226,244,0.08),transparent_28%),linear-gradient(180deg,#102122_0%,#0a0f0f_100%)] px-6 py-6 lg:px-8">
            <div className="mx-auto max-w-[1400px]">{children}</div>
          </main>
          <footer className="border-t border-primary/5 bg-background-dark/80 px-6 py-4 text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex flex-wrap items-center gap-6">
                <span className="flex items-center gap-2">
                  <span className={`size-1.5 rounded-full ${usingMock ? "bg-amber-400" : "animate-pulse bg-primary"}`} />
                  {usingMock ? "Mock Fallback Active" : "Live API Connected"}
                </span>
                <span>Profile: {status.active_profile}</span>
                <span>Saved Playlists: {status.saved_playlist_count}</span>
              </div>
              <div className="flex flex-wrap gap-6">
                <span>Review State: {status.review_state_exists ? "Loaded" : "Missing"}</span>
                <span>History: {status.history_exists ? "Available" : "Empty"}</span>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}

export function SplitScreen({
  main,
  side,
}: {
  main: ReactNode;
  side: ReactNode;
}) {
  return (
    <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="min-w-0">{main}</div>
      <div className="min-w-0 xl:sticky xl:top-24">{side}</div>
    </div>
  );
}
