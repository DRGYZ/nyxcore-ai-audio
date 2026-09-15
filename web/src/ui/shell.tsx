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
        `group relative flex items-center gap-2.5 rounded-[3px] px-3 py-2 font-sans text-xs font-medium transition-colors ${
          isActive
            ? "border-l-2 border-accent bg-white/[0.06] text-primary font-medium pl-2.5"
            : "border-l-2 border-transparent text-primary-muted hover:bg-white/[0.03] hover:text-primary pl-2.5"
        }`
      }
    >
      <Icon name={icon} className="text-[18px] opacity-75 group-hover:opacity-100" />
      <span>{label}</span>
      {badge ? (
        <span className="ml-auto rounded-[3px] border border-accent/30 bg-accent/15 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-accent">
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
        <aside className="border-r border-white/[0.07] bg-surface lg:w-64 lg:shrink-0">
          <div className="flex items-center gap-3 border-b border-white/[0.07] px-5 py-4">
            <div className="flex size-7 items-center justify-center rounded-[3px] border border-white/[0.08] bg-surface-low text-accent">
              <span className="size-2 rounded-[1px] bg-accent" />
            </div>
            <div>
              <h1 className="font-display text-sm font-bold tracking-tight text-primary">NyxCore</h1>
              <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-primary-subtle">Audio Library Toolkit</p>
            </div>
          </div>
          <nav className="space-y-5 p-3.5">
            <div>
              <p className="px-2.5 font-mono text-[9px] font-semibold uppercase tracking-[0.2em] text-primary-subtle">Navigation</p>
              <div className="mt-1.5 space-y-0.5">
                {navItems.map((item) => (
                  <SidebarLink key={item.to} {...item} badge={item.to === "/review" ? (reviewCount || undefined) : undefined} />
                ))}
              </div>
            </div>
            <div className="rounded-[3px] border border-white/[0.06] bg-surface-low/60 p-3.5">
              <div className="flex items-center justify-between gap-2">
                <p className="font-sans text-[11px] font-medium text-primary-subtle">Engine Status</p>
                <Chip tone={isConnected ? "accent" : "neutral"}>{isConnected ? "Connected" : "Offline"}</Chip>
              </div>
              {isConnected ? (
                <>
                  <div className="mt-3 grid grid-cols-2 gap-2 border-t border-white/[0.05] pt-2.5">
                    <div>
                      <p className="font-mono text-[9px] uppercase tracking-wider text-primary-subtle">Open Review</p>
                      <p className="mt-0.5 font-display text-base font-bold text-primary">{unresolvedReviewCount}</p>
                    </div>
                    <div>
                      <p className="font-mono text-[9px] uppercase tracking-wider text-primary-subtle">Ledger</p>
                      <p className="mt-0.5 font-display text-base font-bold text-accent">{status?.history_exists ? "Active" : "Empty"}</p>
                    </div>
                  </div>
                  <p className="mt-2.5 truncate font-mono text-[10px] text-primary-subtle">
                    {status?.music_path?.split(/[\\/]/).pop() ?? "Local session"}
                  </p>
                </>
              ) : (
                <p className="mt-2 font-sans text-xs leading-relaxed text-primary-subtle">
                  Local backend unreachable. Start uvicorn on port 8000.
                </p>
              )}
            </div>
          </nav>
        </aside>
        <div className="flex min-h-screen min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b border-white/[0.07] bg-background/90 px-6 py-2.5 backdrop-blur-md">
            <div className="flex min-w-0 flex-1 items-center gap-3">
              <div className="hidden min-w-0 max-w-sm items-center gap-2 text-primary-muted sm:flex">
                <Icon name={isConnected ? "folder_open" : "cloud_off"} className="shrink-0 text-sm text-accent" />
                <span className="truncate font-mono text-xs text-primary-muted">
                  {isConnected ? status?.music_path : "Local API Disconnected"}
                </span>
              </div>
              <div className="hidden h-3.5 w-px shrink-0 bg-white/[0.08] lg:block" />
              <div className="hidden shrink-0 items-center gap-1.5 lg:flex">
                <span className={`size-1.5 rounded-full ${isConnected ? "bg-accent" : "bg-amber-400"}`} />
                <span className="font-sans text-[11px] text-primary-subtle">
                  {isConnected ? "Local Engine Active" : "Disconnected"}
                </span>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-3">
              <form className="relative hidden md:block" role="search" onSubmit={handleSearchSubmit}>
                <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-xs text-primary-subtle" />
                <input
                  aria-label="Search archive"
                  className="w-48 lg:w-60 rounded-[3px] border border-white/[0.08] bg-surface-low py-1.5 pl-8 pr-8 font-sans text-xs text-primary placeholder:text-primary-subtle focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/50"
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
                  className="absolute right-1 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center text-primary-subtle transition-colors hover:text-accent focus-visible:outline-none"
                >
                  <Icon name="arrow_forward" className="text-xs" />
                </button>
              </form>
              <div ref={notificationRef} className="relative">
                <button
                  type="button"
                  aria-label={`Review notifications (${notificationItems.length})`}
                  aria-expanded={notificationsOpen}
                  aria-haspopup="menu"
                  onClick={() => setNotificationsOpen((open) => !open)}
                  className="relative rounded-[3px] border border-white/[0.08] bg-surface-low p-1.5 text-primary-muted transition-colors hover:border-white/[0.14] hover:text-primary focus-visible:outline-none"
                >
                  <Icon name="notifications" className="text-base" />
                  {notificationItems.length > 0 ? (
                    <span className="absolute -right-1 -top-1 flex min-w-3.5 items-center justify-center rounded-full bg-rose-500 px-1 font-sans text-[9px] font-bold leading-3.5 text-white">
                      {notificationItems.length}
                    </span>
                  ) : null}
                </button>
                {notificationsOpen ? (
                  <div
                    role="menu"
                    aria-label="High-priority review notifications"
                    className="absolute right-0 top-full z-50 mt-2 w-[min(24rem,calc(100vw-2rem))] rounded-[4px] border border-white/[0.1] bg-surface shadow-2xl"
                  >
                    <div className="border-b border-white/[0.07] bg-surface-low/80 px-4 py-3">
                      <p className="font-sans text-xs font-semibold text-primary">Priority Review</p>
                      <p className="mt-0.5 font-sans text-[11px] text-primary-subtle">High-priority items that still need attention.</p>
                    </div>
                    {notificationItems.length > 0 ? (
                      <div className="max-h-80 overflow-y-auto divide-y divide-white/[0.04]">
                        {notificationItems.map((item) => (
                          <Link
                            key={item.item_id}
                            role="menuitem"
                            to={`/review?item=${encodeURIComponent(item.item_id)}`}
                            className="block px-4 py-2.5 transition-colors hover:bg-white/[0.02] focus-visible:bg-white/[0.02] focus-visible:outline-none"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-sans text-[10px] font-semibold text-rose-400">High priority</span>
                              <span className="font-mono text-[10px] text-primary-subtle">{item.priority_score.toFixed(1)}</span>
                            </div>
                            <p className="mt-1 line-clamp-2 font-sans text-xs text-primary">{item.summary}</p>
                          </Link>
                        ))}
                      </div>
                    ) : (
                      <p className="px-4 py-5 text-center font-sans text-xs text-primary-subtle">No high-priority review items right now.</p>
                    )}
                    <Link
                      role="menuitem"
                      to="/review"
                      className="flex items-center justify-between border-t border-white/[0.07] bg-surface-low/80 px-4 py-2 font-sans text-xs font-medium text-accent hover:bg-surface-mid focus-visible:outline-none"
                    >
                      <span>Open Review Inbox</span>
                      <Icon name="arrow_forward" className="text-xs" />
                    </Link>
                  </div>
                ) : null}
              </div>
              <div className="flex h-7 items-center rounded-[3px] border border-white/[0.08] bg-surface-low px-2 font-mono text-[10px] text-primary-subtle">
                LOCAL
              </div>
            </div>
          </header>
          <main className="flex-1 overflow-y-auto bg-background px-6 py-6 lg:px-8">
            <div className="mx-auto max-w-[1400px]">{children}</div>
          </main>
          <footer className="border-t border-white/[0.07] bg-surface-low/40 px-6 py-2.5 font-sans text-[11px] text-primary-subtle">
            <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className={`size-1.5 rounded-full ${isConnected ? "bg-accent" : "bg-amber-400"}`} />
                  {isConnected ? "Local API Connected (127.0.0.1:8000)" : "Local API Unavailable"}
                </span>
                {isConnected && status?.music_path ? <span className="font-mono text-primary-muted">Library: {status.music_path}</span> : null}
              </div>
              {isConnected ? (
                <div className="flex flex-wrap gap-4 font-mono text-[10px]">
                  <span>Review State: {status?.review_state_exists ? "Loaded" : "Empty"}</span>
                  <span>Ledger: {status?.history_exists ? "Available" : "Empty"}</span>
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
