import { useEffect, useRef, useState } from "react";
import type { FormEvent, PropsWithChildren, ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useReviewQuery, useStatusQuery } from "../lib/hooks";
import { Chip, Icon } from "./components";

const navItems = [
  { to: "/", label: "Overview", icon: "dashboard" },
  { to: "/search", label: "Archive Search", icon: "manage_search" },
  { to: "/review", label: "Review Inbox", icon: "inbox" },
  { to: "/duplicates", label: "Duplicates", icon: "copy_all" },
  { to: "/health", label: "Library Health", icon: "health_and_safety" },
  { to: "/history", label: "History", icon: "history" },
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
  const location = useLocation();
  const navigate = useNavigate();
  const statusQuery = useStatusQuery();
  const reviewQuery = useReviewQuery();
  const notificationRef = useRef<HTMLDivElement | null>(null);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const isConnected = !!statusQuery.data && !statusQuery.isError;
  const status = statusQuery.data;
  const review = isConnected ? reviewQuery.data?.data : undefined;
  const reviewCount = review ? review.items.filter((item) => item.review_status === "new").length : 0;
  const unresolvedReviewCount = review ? review.items.filter((item) => item.review_status === "new" || item.review_status === "seen").length : 0;
  const notificationItems = review
    ? [...review.items]
        .filter((item) => item.priority_band === "high" && (item.review_status === "new" || item.review_status === "seen"))
        .sort((left, right) => right.priority_score - left.priority_score || left.item_id.localeCompare(right.item_id))
        .slice(0, 5)
    : [];

  useEffect(() => {
    if (location.pathname === "/search") {
      setSearchValue(new URLSearchParams(location.search).get("q") ?? "");
    }
    setNotificationsOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (!notificationsOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
        setNotificationsOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNotificationsOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [notificationsOpen]);

  function handleSearchSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = searchValue.trim();
    if (query.length < 2) return;
    const params = new URLSearchParams({ q: query });
    navigate(`/search?${params.toString()}`);
  }

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
              <p className="text-xs font-medium text-primary/60">Local music-library review toolkit</p>
            </div>
          </div>
          <nav className="space-y-6 px-4 pb-6">
            <div>
              <p className="px-3 text-[10px] font-bold uppercase tracking-[0.28em] text-slate-500">Navigation</p>
              <div className="mt-3 space-y-1">
                {navItems.map((item) => (
                  <SidebarLink key={item.to} {...item} badge={item.to === "/review" ? (reviewCount || undefined) : undefined} />
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-primary/10 bg-gradient-to-br from-primary/10 to-transparent p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-bold uppercase tracking-[0.24em] text-primary">Workspace Status</p>
                <Chip tone={isConnected ? "primary" : "warning"}>{isConnected ? "connected" : "offline"}</Chip>
              </div>
              {isConnected ? (
                <>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Open Review</p>
                      <p className="mt-1 text-lg font-bold text-slate-100">{unresolvedReviewCount}</p>
                    </div>
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Ledger</p>
                      <p className="mt-1 text-lg font-bold text-primary">{status?.history_exists ? "Active" : "Empty"}</p>
                    </div>
                  </div>
                  <p className="mt-3 text-[10px] text-slate-400">
                    Live status is sourced from the local API session.
                  </p>
                </>
              ) : (
                <p className="mt-3 text-[11px] leading-relaxed text-slate-400">
                  Local API is not connected. Start the local backend to inspect and review your library.
                </p>
              )}
            </div>
          </nav>
        </aside>
        <div className="flex min-h-screen min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex items-center justify-between border-b border-primary/10 bg-background-dark/70 px-6 py-4 backdrop-blur-md">
            <div className="flex min-w-0 items-center gap-6">
              <div className="hidden items-center gap-2 text-primary sm:flex">
                <Icon name={isConnected ? "folder_open" : "cloud_off"} className="text-lg" />
                <span className="truncate font-mono text-xs text-slate-300">
                  {isConnected ? status?.music_path : "Local API Disconnected"}
                </span>
              </div>
              <div className="hidden h-4 w-px bg-primary/20 md:block" />
              <div className="hidden items-center gap-2 md:flex">
                <span className={`size-2 rounded-full ${isConnected ? "bg-primary shadow-[0_0_8px_#25e2f4]" : "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.55)]"}`} />
                <span className="text-[10px] font-bold uppercase tracking-[0.28em] text-slate-500">
                  {isConnected ? "Local API" : "Disconnected"}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <form className="relative hidden md:block" role="search" onSubmit={handleSearchSubmit}>
                <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-slate-500" />
                <input
                  aria-label="Search archive"
                  className="w-64 rounded-full border border-primary/10 bg-primary/5 py-1.5 pl-10 pr-10 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-primary focus:ring-1 focus:ring-primary"
                  placeholder="Search archive..."
                  type="search"
                  value={searchValue}
                  onChange={(event) => setSearchValue(event.target.value)}
                  minLength={2}
                  maxLength={200}
                  required
                />
                <button
                  type="submit"
                  aria-label="Submit archive search"
                  className="absolute right-1 top-1/2 flex size-7 -translate-y-1/2 items-center justify-center rounded-full text-slate-500 transition-colors hover:bg-primary/10 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                >
                  <Icon name="arrow_forward" className="text-base" />
                </button>
              </form>
              <div ref={notificationRef} className="relative">
                <button
                  type="button"
                  aria-label={`Review notifications (${notificationItems.length})`}
                  aria-expanded={notificationsOpen}
                  aria-haspopup="menu"
                  onClick={() => setNotificationsOpen((open) => !open)}
                  className="relative rounded-lg bg-primary/10 p-2 text-primary transition-colors hover:bg-primary/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                >
                  <Icon name="notifications" className="text-xl" />
                  {notificationItems.length > 0 ? (
                    <span className="absolute -right-1 -top-1 flex min-w-5 items-center justify-center rounded-full border-2 border-background-dark bg-rose-500 px-1 text-[9px] font-bold leading-4 text-white">
                      {notificationItems.length}
                    </span>
                  ) : null}
                </button>
                {notificationsOpen ? (
                  <div
                    role="menu"
                    aria-label="High-priority review notifications"
                    className="absolute right-0 top-full z-50 mt-3 w-[min(24rem,calc(100vw-2rem))] overflow-hidden rounded-xl border border-primary/20 bg-surface-dark shadow-2xl"
                  >
                    <div className="border-b border-border-dark px-4 py-3">
                      <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">Priority Review</p>
                      <p className="mt-1 text-xs text-slate-500">High-priority items that still need attention.</p>
                    </div>
                    {notificationItems.length > 0 ? (
                      <div className="max-h-80 overflow-y-auto p-2">
                        {notificationItems.map((item) => (
                          <Link
                            key={item.item_id}
                            role="menuitem"
                            to={`/review?item=${encodeURIComponent(item.item_id)}`}
                            className="block rounded-lg px-3 py-3 transition-colors hover:bg-primary/10 focus-visible:bg-primary/10 focus-visible:outline-none"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-rose-400">High priority</span>
                              <span className="font-mono text-[10px] text-slate-500">{item.priority_score.toFixed(1)}</span>
                            </div>
                            <p className="mt-1 line-clamp-2 text-sm font-medium text-slate-200">{item.summary}</p>
                          </Link>
                        ))}
                      </div>
                    ) : (
                      <p className="px-4 py-6 text-center text-sm text-slate-500">No high-priority review items right now.</p>
                    )}
                    <Link
                      role="menuitem"
                      to="/review"
                      className="flex items-center justify-between border-t border-border-dark px-4 py-3 text-xs font-bold text-primary hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
                    >
                      Open Review Inbox
                      <Icon name="arrow_forward" className="text-base" />
                    </Link>
                  </div>
                ) : null}
              </div>
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
                  <span className={`size-1.5 rounded-full ${isConnected ? "animate-pulse bg-primary" : "bg-amber-400"}`} />
                  {isConnected ? "Local API Connected (127.0.0.1:8000)" : "Local API Unavailable"}
                </span>
                {isConnected && status?.music_path ? <span>Library: {status.music_path}</span> : null}
              </div>
              {isConnected ? (
                <div className="flex flex-wrap gap-6">
                  <span>Review State: {status?.review_state_exists ? "Loaded" : "Empty"}</span>
                  <span>History Ledger: {status?.history_exists ? "Available" : "Empty"}</span>
                </div>
              ) : (
                <span>NyxCore runs locally</span>
              )}
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
