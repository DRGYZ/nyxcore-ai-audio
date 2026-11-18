import { useEffect, useRef, useState } from "react";
import type { FormEvent, PropsWithChildren, ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useLibraryAnalysis, useRefreshLibrary, useReviewQuery, useStatusQuery } from "../lib/hooks";
import { Button, Chip, Icon } from "./components";

type NavItem = {
  to: string;
  label: string;
  icon: string;
  group: "collection" | "curation";
  desktopOnly?: boolean;
};

const navItems: NavItem[] = [
  { to: "/", label: "Overview", icon: "album", group: "collection" },
  { to: "/search", label: "Archive Search", icon: "search", group: "collection" },
  { to: "/health", label: "Library Health", icon: "graphic_eq", group: "collection" },
  { to: "/review", label: "Review Inbox", icon: "inbox", group: "curation" },
  { to: "/duplicates", label: "Duplicates", icon: "library_music", group: "curation" },
  { to: "/metadata", label: "Metadata", icon: "sell", group: "curation", desktopOnly: true },
  { to: "/organization", label: "Library Organization", icon: "folder_open", group: "curation" },
  { to: "/history", label: "History", icon: "history", group: "curation" },
];

function BrandMark() {
  return (
    <div
      aria-hidden="true"
      className="relative flex size-8 shrink-0 items-center justify-center rounded-full border border-accent/40 bg-gradient-to-br from-[#2a231b] to-[#161310] shadow-sm"
    >
      <span className="size-4 rounded-full border border-accent/40" />
      <span className="absolute size-1.5 rounded-full bg-accent" />
    </div>
  );
}

function SidebarLink({
  to,
  label,
  icon,
  badge,
  onClick,
}: {
  to: string;
  label: string;
  icon: string;
  badge?: number;
  onClick?: () => void;
}) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      onClick={onClick}
      className={({ isActive }) =>
        `group relative flex items-center gap-2.5 rounded-lg px-3 py-2 font-sans text-[13px] transition-colors ${
          isActive
            ? "bg-accent/[0.14] font-semibold text-primary shadow-[inset_0_1px_0_rgba(255,248,235,0.06)]"
            : "font-medium text-primary-muted hover:bg-white/[0.04] hover:text-primary"
        }`
      }
    >
      {({ isActive }) => (
        <>
          <Icon
            name={icon}
            className={`text-[18px] transition-colors ${
              isActive ? "text-accent" : "text-primary-subtle group-hover:text-primary-muted"
            }`}
          />
          <span className="truncate">{label}</span>
          {badge ? (
            <span className="ml-auto rounded-full bg-accent/20 px-2 py-0.5 font-sans text-[11px] font-semibold text-accent">
              {badge}
            </span>
          ) : null}
        </>
      )}
    </NavLink>
  );
}

function LibraryAnalysisStatus() {
  const analysis = useLibraryAnalysis();
  const refresh = useRefreshLibrary();
  const error = refresh.error ?? analysis.error;
  const date = analysis.data && new Date(analysis.data.analyzed_at).toLocaleString();

  return (
    <section
      aria-label="Library analysis"
      className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-accent/25 bg-surface px-4 py-3 text-xs text-primary shadow-[inset_0_1px_0_rgba(224,159,90,0.08)]"
    >
      <div className="flex min-w-0 items-start gap-3">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-accent">
          <Icon name="verified_user" className="text-[16px]" />
        </span>
        <div className="min-w-0">
          <p className="text-[13px] text-primary">
            <strong className="font-semibold">Review-only mode.</strong>{" "}
            <span className="text-primary-muted">Applying plans and reversing file changes are disabled.</span>
          </p>
          <p role="status" className="mt-0.5 text-xs text-primary-subtle">
            {refresh.isPending
              ? (date ? "Refreshing library. Previous results remain available." : "Refreshing library…")
              : date
                ? `Analyzed ${date}. Use Refresh Library after changing files.`
                : (error ? "Library analysis is unavailable." : "Analyzing library…")}
          </p>
          {error && !refresh.isPending ? (
            <p role="alert" className="mt-1 break-words text-xs text-[#f59f8a]">
              {error.message} {date ? "Showing the last successful analysis." : null}
            </p>
          ) : null}
        </div>
      </div>
      <Button tone="secondary" disabled={refresh.isPending || analysis.isFetching} onClick={() => refresh.mutate()}>
        <Icon name="refresh" className={`text-[16px] ${refresh.isPending ? "animate-spin" : ""}`} />
        {refresh.isPending ? "Refreshing Library…" : "Refresh Library"}
      </Button>
    </section>
  );
}

export function AppShell({ children, desktop = false }: PropsWithChildren<{ desktop?: boolean }>) {
  const location = useLocation();
  const navigate = useNavigate();
  const statusQuery = useStatusQuery();
  const reviewQuery = useReviewQuery();
  const notificationRef = useRef<HTMLDivElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const isConnected = !!statusQuery.data && !statusQuery.isError;
  const isConnecting = !isConnected && statusQuery.isPending;
  const engineState = isConnected ? "Connected" : isConnecting ? "Connecting" : "Offline";
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
    setMobileNavOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (desktop && mainRef.current) mainRef.current.scrollTop = 0;
  }, [desktop, location.pathname]);

  useEffect(() => {
    if (!mobileNavOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNavOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mobileNavOpen]);

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

  const visibleNavItems = navItems.filter((item) => desktop || !item.desktopOnly);
  const collectionItems = visibleNavItems.filter((item) => item.group === "collection");
  const curationItems = visibleNavItems.filter((item) => item.group === "curation");
  const libraryFolderName = status?.music_path?.split(/[\\/]/).filter(Boolean).pop() ?? "Local session";

  return (
    <div className={`bg-background text-primary ${desktop ? "flex h-screen flex-col overflow-hidden" : "min-h-screen"}`}>
      <div className={`flex shrink-0 items-center justify-between border-b border-white/[0.08] bg-surface-low px-4 py-3 ${desktop ? "md:hidden" : "lg:hidden"}`}>
        <div className="flex items-center gap-2.5">
          <BrandMark />
          <div>
            <span className="font-display text-sm font-semibold tracking-tight text-primary">NyxCore</span>
            <span className="block font-sans text-[11px] text-primary-subtle">Music Collection</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Chip tone={isConnected ? "accent" : "neutral"}>{engineState}</Chip>
          <button
            type="button"
            aria-label="Toggle navigation menu"
            aria-expanded={mobileNavOpen}
            aria-controls="main-sidebar-nav"
            onClick={() => setMobileNavOpen((open) => !open)}
            className="flex size-8 items-center justify-center rounded-lg border border-white/[0.1] bg-surface text-primary-muted hover:border-white/[0.18] hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <Icon name={mobileNavOpen ? "close" : "menu"} className="text-lg" />
          </button>
        </div>
      </div>

      <div className={`flex flex-col ${desktop ? "min-h-0 flex-1 md:flex-row" : "min-h-screen lg:flex-row"}`}>
        <aside
          id="main-sidebar-nav"
          className={`flex flex-col border-r border-white/[0.08] bg-surface-low ${desktop ? "overflow-y-auto md:w-52 md:shrink-0 lg:w-60" : "lg:w-64 lg:shrink-0"} ${
            mobileNavOpen ? `block border-b border-white/[0.08] ${desktop ? "md:border-b-0" : "lg:border-b-0"}`
              : desktop ? "hidden md:flex" : "hidden lg:flex"
          }`}
        >
          <div className={`hidden items-center gap-3 border-b border-white/[0.07] px-4 py-3.5 ${desktop ? "md:flex" : "lg:flex"}`}>
            <BrandMark />
            <div className="min-w-0">
              <h1 className="font-display text-base font-semibold tracking-tight text-primary">NyxCore</h1>
              <p className="font-sans text-[11px] text-primary-subtle">Music Collection</p>
            </div>
          </div>

          <nav className="flex flex-1 flex-col justify-between gap-3.5 p-3" aria-label="Main navigation">
            <div className="space-y-4">
              <div>
                <p className="px-3 pb-1 font-sans text-[11px] font-medium text-primary-subtle">Collection</p>
                <div className="space-y-0.5">
                  {collectionItems.map((item) => (
                    <SidebarLink
                      key={item.to}
                      {...item}
                      onClick={() => setMobileNavOpen(false)}
                    />
                  ))}
                </div>
              </div>
              <div>
                <p className="px-3 pb-1 font-sans text-[11px] font-medium text-primary-subtle">Curation &amp; Cleanup</p>
                <div className="space-y-0.5">
                  {curationItems.map((item) => (
                    <SidebarLink
                      key={item.to}
                      {...item}
                      badge={item.to === "/review" ? (reviewCount || undefined) : undefined}
                      onClick={() => setMobileNavOpen(false)}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-white/[0.08] bg-surface p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="font-sans text-xs font-medium text-primary-muted">Collection Deck</p>
                <Chip tone={isConnected ? "success" : "neutral"}>{engineState}</Chip>
              </div>
              {isConnected ? (
                <>
                  <div className="mt-2.5 grid grid-cols-2 gap-2 border-t border-white/[0.06] pt-2">
                    <div>
                      <p className="font-sans text-[11px] text-primary-subtle">Open Review</p>
                      <p className="mt-0.5 font-display text-sm font-semibold text-primary">{unresolvedReviewCount}</p>
                    </div>
                    <div>
                      <p className="font-sans text-[11px] text-primary-subtle">Ledger</p>
                      <p className="mt-0.5 font-display text-sm font-semibold text-accent">{status?.history_exists ? "Active" : "Empty"}</p>
                    </div>
                  </div>
                  <p className="mt-2 truncate rounded-md bg-surface-low px-2.5 py-1 font-mono text-[11px] text-primary-subtle" title={status?.music_path}>
                    {libraryFolderName}
                  </p>
                </>
              ) : (
                <p className="mt-2 font-sans text-xs leading-relaxed text-primary-subtle">
                  {isConnecting ? "Connecting to the local engine…" : import.meta.env.PROD
                    ? "Local backend unreachable. Reopen NyxCore."
                    : "Local backend unreachable. Start uvicorn on port 8000."}
                </p>
              )}
            </div>
          </nav>
        </aside>

        <div className={`flex min-w-0 flex-1 flex-col ${desktop ? "min-h-0" : "min-h-screen"}`}>
          <header className="sticky top-0 z-30 flex shrink-0 items-center justify-between gap-3 border-b border-white/[0.08] bg-background/95 px-4 py-2.5 sm:px-6">
            <div className="flex min-w-0 flex-1 items-center gap-3">
              <div
                title={status?.music_path}
                className="hidden min-w-0 max-w-xs items-center gap-2 rounded-lg border border-white/[0.08] bg-surface px-3 py-1.5 text-primary-muted sm:flex"
              >
                <Icon name={isConnected ? "library_music" : "cloud_off"} className="shrink-0 text-[15px] text-accent" />
                <span className="truncate font-sans text-xs font-medium text-primary">
                  {isConnected ? libraryFolderName : isConnecting ? "Connecting to library" : "Local API Disconnected"}
                </span>
              </div>
              <div className="hidden shrink-0 items-center gap-1.5 lg:flex">
                <span className={`size-1.5 rounded-full ${isConnected ? "bg-[#52b788]" : "bg-[#e9b44c]"}`} />
                <span className="font-sans text-xs text-primary-subtle">
                  {isConnected ? "Local library ready" : isConnecting ? "Connecting" : "Disconnected"}
                </span>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2.5">
              <form className="relative hidden md:block" role="search" onSubmit={handleSearchSubmit}>
                <Icon name="search" className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[15px] text-primary-subtle" />
                <input
                  aria-label="Search archive"
                  className="w-44 rounded-lg border border-white/[0.1] bg-surface py-1.5 pl-8 pr-8 font-sans text-xs text-primary placeholder:text-primary-subtle focus:border-accent/60 focus:outline-none focus:ring-1 focus:ring-accent/40 lg:w-64"
                  placeholder="Jump to artist, album, track..."
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
                  className="absolute right-1.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded text-primary-subtle transition-colors hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <Icon name="arrow_forward" className="text-[14px]" />
                </button>
              </form>

              <div ref={notificationRef} className="relative">
                <button
                  type="button"
                  aria-label={`Review notifications (${notificationItems.length})`}
                  aria-expanded={notificationsOpen}
                  aria-haspopup="menu"
                  onClick={() => setNotificationsOpen((open) => !open)}
                  className="relative flex size-8 items-center justify-center rounded-lg border border-white/[0.1] bg-surface text-primary-muted transition-colors hover:border-white/[0.18] hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <Icon name="notifications" className="text-[18px]" />
                  {notificationItems.length > 0 ? (
                    <span className="absolute -right-1 -top-1 flex min-w-4 items-center justify-center rounded-full bg-accent px-1 font-sans text-[10px] font-bold leading-4 text-[#14110e]">
                      {notificationItems.length}
                    </span>
                  ) : null}
                </button>
                {notificationsOpen ? (
                  <div
                    role="menu"
                    aria-label="High-priority review notifications"
                    className="absolute right-0 top-full z-50 mt-2 w-[min(24rem,calc(100vw-2rem))] overflow-hidden rounded-xl border border-white/[0.12] bg-surface shadow-soft"
                  >
                    <div className="border-b border-white/[0.08] bg-surface-low/90 px-4 py-3">
                      <p className="font-display text-xs font-semibold text-primary">Priority Review</p>
                      <p className="mt-0.5 font-sans text-[11px] text-primary-subtle">High-priority items that still need attention.</p>
                    </div>
                    {notificationItems.length > 0 ? (
                      <div className="max-h-80 divide-y divide-white/[0.05] overflow-y-auto">
                        {notificationItems.map((item) => (
                          <Link
                            key={item.item_id}
                            role="menuitem"
                            to={`/review?item=${encodeURIComponent(item.item_id)}`}
                            className="block px-4 py-3 transition-colors hover:bg-white/[0.03] focus-visible:bg-white/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-sans text-[11px] font-semibold text-[#f59f8a]">High priority</span>
                              <span className="font-mono text-[11px] text-primary-subtle">{item.priority_score.toFixed(1)}</span>
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
                      className="flex items-center justify-between border-t border-white/[0.08] bg-surface-low/90 px-4 py-2.5 font-sans text-xs font-medium text-accent hover:bg-surface-mid focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent"
                    >
                      <span>Open Review Inbox</span>
                      <Icon name="arrow_forward" className="text-xs" />
                    </Link>
                  </div>
                ) : null}
              </div>
            </div>
          </header>

          <main ref={mainRef} className={`flex-1 overflow-y-auto bg-background px-4 py-5 sm:px-6 ${desktop ? "lg:px-8 lg:py-6" : "lg:px-8 lg:py-7"}`}>
            <div className="mx-auto max-w-[1380px]">
              {status?.review_only ? <LibraryAnalysisStatus /> : null}
              {children}
            </div>
          </main>

          <footer className="shrink-0 border-t border-white/[0.08] bg-surface-low/75 px-4 py-2 font-sans text-[11px] text-primary-subtle sm:px-6">
            <div className={`flex gap-2 ${desktop ? "items-center justify-between" : "flex-col lg:flex-row lg:items-center lg:justify-between"}`}>
              <div className={`flex min-w-0 items-center gap-4 ${desktop ? "" : "flex-wrap"}`}>
                <span className="flex shrink-0 items-center gap-1.5">
                  <span className={`size-1.5 rounded-full ${isConnected ? "bg-[#52b788]" : "bg-[#e9b44c]"}`} />
                  {isConnected ? "Local API Connected" : isConnecting ? "Connecting to local engine" : "Local API Unavailable"}
                </span>
                {isConnected && status?.music_path ? (
                  <span
                    title={desktop ? status.music_path : undefined}
                    className={`min-w-0 font-mono text-primary-muted ${desktop ? "truncate" : "[overflow-wrap:anywhere]"}`}
                  >
                    Library: {status.music_path}
                  </span>
                ) : null}
              </div>
              {isConnected ? (
                <div className={`gap-4 font-sans text-[11px] ${desktop ? "hidden shrink-0 lg:flex" : "flex flex-wrap"}`}>
                  <span>Review State: <strong className="font-medium text-primary-muted">{status?.review_state_exists ? "Loaded" : "Empty"}</strong></span>
                  <span>Ledger: <strong className="font-medium text-primary-muted">{status?.history_exists ? "Available" : "Empty"}</strong></span>
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
      <div className="min-w-0 xl:sticky xl:top-20">{side}</div>
    </div>
  );
}
