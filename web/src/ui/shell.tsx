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
        `group relative flex items-center gap-3 px-3 py-2 text-xs font-mono tracking-wider uppercase transition-colors ${
          isActive
            ? "border-l-2 border-accent bg-accent/5 text-accent font-medium pl-[10px]"
            : "border-l-2 border-transparent text-primary-muted hover:bg-surface-low hover:text-primary pl-[10px]"
        }`
      }
    >
      <Icon name={icon} className="text-base" />
      <span>{label}</span>
      {badge ? (
        <span className="ml-auto bg-accent text-background px-1.5 py-0.5 font-mono text-[10px] font-bold">
          {badge}
        </span>
      ) : null}
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
    <div className="min-h-screen bg-background text-primary">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="border-r border-border bg-surface lg:w-72 lg:shrink-0">
          <div className="flex items-center gap-3 border-b border-border px-6 py-5">
            <div className="flex size-8 items-center justify-center border border-border-bright bg-surface-low text-accent">
              <span className="size-2 bg-accent" />
            </div>
            <div>
              <h1 className="font-display text-sm font-bold uppercase tracking-wider text-primary">NyxCore</h1>
              <p className="font-mono text-[10px] tracking-wider text-primary-muted uppercase">Audio Library Toolkit</p>
            </div>
          </div>
          <nav className="space-y-6 p-4">
            <div>
              <p className="px-3 font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">Navigation</p>
              <div className="mt-2 space-y-0.5">
                {navItems.map((item) => (
                  <SidebarLink key={item.to} {...item} badge={item.to === "/review" ? (reviewCount || undefined) : undefined} />
                ))}
              </div>
            </div>
            <div className="border border-border bg-surface-low p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Workspace Status</p>
                <Chip tone={isConnected ? "accent" : "neutral"}>{isConnected ? "connected" : "offline"}</Chip>
              </div>
              {isConnected ? (
                <>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-wider text-primary-subtle">Open Review</p>
                      <p className="mt-1 font-mono text-lg font-bold text-primary">{unresolvedReviewCount}</p>
                    </div>
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-wider text-primary-subtle">Ledger</p>
                      <p className="mt-1 font-mono text-lg font-bold text-accent">{status?.history_exists ? "Active" : "Empty"}</p>
                    </div>
                  </div>
                  <p className="mt-3 font-mono text-[10px] text-primary-subtle">
                    Live status is sourced from the local API session.
                  </p>
                </>
              ) : (
                <p className="mt-3 text-xs leading-relaxed text-primary-muted">
                  Local API is not connected. Start the local backend to inspect and review your library.
                </p>
              )}
            </div>
          </nav>
        </aside>
        <div className="flex min-h-screen min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-background/95 px-6 py-3.5 backdrop-blur-md">
            <div className="flex min-w-0 items-center gap-6">
              <div className="hidden items-center gap-2 text-primary-muted sm:flex">
                <Icon name={isConnected ? "folder_open" : "cloud_off"} className="text-base text-accent" />
                <span className="truncate font-mono text-xs text-primary-muted">
                  {isConnected ? status?.music_path : "Local API Disconnected"}
                </span>
              </div>
              <div className="hidden h-4 w-px bg-border md:block" />
              <div className="hidden items-center gap-2 md:flex">
                <span className={`size-1.5 ${isConnected ? "bg-accent" : "bg-amber-400"}`} />
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-subtle">
                  {isConnected ? "Local API Active" : "Disconnected"}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <form className="relative hidden md:block" role="search" onSubmit={handleSearchSubmit}>
                <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-primary-subtle" />
                <input
                  aria-label="Search archive"
                  className="w-64 border border-border bg-surface-low py-1.5 pl-9 pr-9 font-mono text-xs text-primary placeholder:text-primary-subtle focus:border-accent focus:outline-none"
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
                  className="absolute right-1 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center text-primary-subtle transition-colors hover:bg-surface-mid hover:text-accent focus-visible:outline-none"
                >
                  <Icon name="arrow_forward" className="text-sm" />
                </button>
              </form>
              <div ref={notificationRef} className="relative">
                <button
                  type="button"
                  aria-label={`Review notifications (${notificationItems.length})`}
                  aria-expanded={notificationsOpen}
                  aria-haspopup="menu"
                  onClick={() => setNotificationsOpen((open) => !open)}
                  className="relative border border-border bg-surface-low p-2 text-primary-muted transition-colors hover:border-border-bright hover:text-primary focus-visible:outline-none"
                >
                  <Icon name="notifications" className="text-lg" />
                  {notificationItems.length > 0 ? (
                    <span className="absolute -right-1 -top-1 flex min-w-4 items-center justify-center bg-rose-500 px-1 font-mono text-[9px] font-bold leading-4 text-white">
                      {notificationItems.length}
                    </span>
                  ) : null}
                </button>
                {notificationsOpen ? (
                  <div
                    role="menu"
                    aria-label="High-priority review notifications"
                    className="absolute right-0 top-full z-50 mt-2 w-[min(24rem,calc(100vw-2rem))] border border-border bg-surface shadow-2xl"
                  >
                    <div className="border-b border-border bg-surface-low px-4 py-3">
                      <p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Priority Review</p>
                      <p className="mt-1 text-xs text-primary-subtle">High-priority items that still need attention.</p>
                    </div>
                    {notificationItems.length > 0 ? (
                      <div className="max-h-80 overflow-y-auto divide-y divide-border">
                        {notificationItems.map((item) => (
                          <Link
                            key={item.item_id}
                            role="menuitem"
                            to={`/review?item=${encodeURIComponent(item.item_id)}`}
                            className="block px-4 py-3 transition-colors hover:bg-surface-low focus-visible:bg-surface-low focus-visible:outline-none"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-rose-400">High priority</span>
                              <span className="font-mono text-[10px] text-primary-subtle">{item.priority_score.toFixed(1)}</span>
                            </div>
                            <p className="mt-1 line-clamp-2 text-xs font-medium text-primary">{item.summary}</p>
                          </Link>
                        ))}
                      </div>
                    ) : (
                      <p className="px-4 py-6 text-center text-xs font-mono text-primary-subtle">No high-priority review items right now.</p>
                    )}
                    <Link
                      role="menuitem"
                      to="/review"
                      className="flex items-center justify-between border-t border-border bg-surface-low px-4 py-2.5 font-mono text-xs font-bold uppercase tracking-wider text-accent hover:bg-surface-mid focus-visible:outline-none"
                    >
                      <span>Open Review Inbox</span>
                      <Icon name="arrow_forward" className="text-sm" />
                    </Link>
                  </div>
                ) : null}
              </div>
              <div className="flex h-8 items-center border border-border bg-surface-low px-2.5 font-mono text-[10px] font-bold uppercase tracking-wider text-primary-subtle">
                LOCAL
              </div>
            </div>
          </header>
          <main className="flex-1 overflow-y-auto bg-background px-6 py-6 lg:px-8">
            <div className="mx-auto max-w-[1400px]">{children}</div>
          </main>
          <footer className="border-t border-border bg-surface-low px-6 py-3 font-mono text-[10px] uppercase tracking-[0.18em] text-primary-subtle">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex flex-wrap items-center gap-6">
                <span className="flex items-center gap-2">
                  <span className={`size-1.5 ${isConnected ? "bg-accent" : "bg-amber-400"}`} />
                  {isConnected ? "Local API Connected (127.0.0.1:8000)" : "Local API Unavailable"}
                </span>
                {isConnected && status?.music_path ? <span className="text-primary-muted">Library: {status.music_path}</span> : null}
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
