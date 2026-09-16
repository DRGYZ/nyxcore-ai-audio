import { useEffect, useId, useRef } from "react";
import type { PropsWithChildren, ReactNode } from "react";
import { useCheckConnection } from "../lib/hooks";
import { Button, Icon, Panel } from "./primitives";

export function ApiUnavailableState({
  contextLabel,
  onRetry,
  checking,
}: {
  contextLabel?: string;
  onRetry?: () => void;
  checking?: boolean;
}) {
  const { checkConnection: defaultCheck, checking: defaultChecking } = useCheckConnection();
  const handleCheck = onRetry ?? defaultCheck;
  const isChecking = checking !== undefined ? checking : defaultChecking;

  return (
    <Panel className="p-8">
      <div className="flex max-w-3xl flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-[3px] border border-amber-500/30 bg-amber-500/10 text-amber-400">
            <Icon name="cloud_off" className="text-xl" />
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-amber-400">Local Connection</p>
            <h2 className="font-display text-xl font-bold text-primary">
              Local API is not connected {contextLabel ? `• ${contextLabel}` : ""}
            </h2>
          </div>
        </div>

        <p className="font-sans text-sm leading-relaxed text-primary-muted">
          NyxCore is an experimental local-first toolkit. It connects to a local FastAPI backend running on your machine
          at <code className="rounded-[2px] border border-white/[0.08] bg-surface-low px-1.5 py-0.5 font-mono text-xs text-accent">http://127.0.0.1:8000</code>.
          The local server is not currently reachable.
        </p>

        <div className="min-w-0 max-w-full space-y-4 rounded-[3px] border border-white/[0.07] bg-surface-low/60 p-5">
          <p className="font-sans text-xs font-semibold text-primary">Setup Instructions</p>

          <div className="min-w-0 max-w-full space-y-2">
            <p className="font-sans text-xs text-primary-muted">
              1. Start the local API pointed at your music library & reports directory:
            </p>
            <pre className="min-w-0 max-w-full overflow-x-auto rounded-[3px] border border-white/[0.06] bg-surface p-3 font-mono text-xs text-primary/90 select-all">
{`export NYXCORE_WEB_MUSIC_DIR="/path/to/your/music"
export NYXCORE_WEB_OUT_DIR="data/reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000`}
            </pre>
          </div>

          <div className="min-w-0 max-w-full space-y-2">
            <p className="font-sans text-xs text-primary-muted">
              2. Or generate and inspect the built-in demo library:
            </p>
            <pre className="min-w-0 max-w-full overflow-x-auto rounded-[3px] border border-white/[0.06] bg-surface p-3 font-mono text-xs text-primary/90 select-all">
{`python demo/create_demo_library.py demo/demo-library
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/demo-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000`}
            </pre>
          </div>

          <p className="font-sans text-xs text-primary-subtle">
            For step-by-step setup guides (including Windows WSL2 setup and dependencies), refer to <span className="text-accent">INSTALL_WSL.md</span> in the project root.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <Button tone="primary" onClick={handleCheck} disabled={isChecking}>
            <Icon name="refresh" className={`text-base ${isChecking ? "animate-spin" : ""}`} />
            {isChecking ? "Checking Connection…" : "Check Connection"}
          </Button>
          <span className="font-sans text-xs text-primary-subtle">
            NyxCore operates locally. No cloud services or external network requests are made.
          </span>
        </div>
      </div>
    </Panel>
  );
}

export function QueryNotice({
  loading,
  error,
  usingMock,
}: {
  loading: boolean;
  error: unknown;
  usingMock?: boolean;
}) {
  if (loading) {
    return (
      <div className="rounded-[3px] border border-white/[0.07] bg-surface-low px-4 py-2.5 font-sans text-xs text-primary-muted">
        Loading library data from local API…
      </div>
    );
  }
  if (error || usingMock) {
    return (
      <div className="rounded-[3px] border border-amber-500/20 bg-amber-500/10 px-4 py-2.5 font-sans text-xs text-amber-300">
        Local API unavailable. Connect the backend to inspect live library data.
      </div>
    );
  }
  return (
    <div className="rounded-[3px] border border-emerald-500/20 bg-emerald-500/10 px-4 py-2.5 font-sans text-xs text-emerald-300">
      Live API connected. Data and review state are current.
    </div>
  );
}

export function PageQueryStateNotice({
  loading,
  error,
  usingMock,
  fallbackMessage,
}: {
  loading: boolean;
  error: unknown;
  usingMock: boolean;
  fallbackMessage?: string;
}) {
  return (
    <>
      <QueryNotice loading={loading} error={error} usingMock={usingMock} />
      {usingMock && fallbackMessage ? <ActionBanner tone="info" message={fallbackMessage} /> : null}
    </>
  );
}

export function ActionBanner({
  tone = "info",
  message,
  action,
}: {
  tone?: "info" | "success" | "error";
  message: string;
  action?: ReactNode;
}) {
  const styles = {
    info: "border-l-2 border-l-accent border-white/[0.07] bg-surface-low text-primary",
    success: "border-l-2 border-l-emerald-400 border-white/[0.07] bg-surface-low text-primary",
    error: "border-l-2 border-l-rose-400 border-white/[0.07] bg-surface-low text-primary",
  };
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
      aria-atomic="true"
      className={`flex flex-col gap-3 rounded-[3px] border px-4 py-3 text-xs md:text-sm md:flex-row md:items-center md:justify-between ${styles[tone]}`}
    >
      <span>{message}</span>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <Panel className="rounded-[3px] border border-white/[0.06] bg-surface-low/40 px-6 py-12 text-center">
      <p className="font-display text-base font-bold text-primary">{title}</p>
      <p className="mx-auto mt-1.5 max-w-xl font-sans text-xs leading-relaxed text-primary-muted">{description}</p>
      {action ? <div className="mt-4 flex justify-center">{action}</div> : null}
    </Panel>
  );
}

export function Modal({
  open,
  title,
  subtitle,
  children,
  footer,
  onClose,
}: PropsWithChildren<{ open: boolean; title: string; subtitle?: string; footer?: ReactNode; onClose: () => void }>) {
  const titleId = useId();
  const subtitleId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;

    // Record currently active element for defensive focus restoration
    previousFocusRef.current = (document.activeElement as HTMLElement | null) ?? null;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // Focus close button or first interactive element inside dialog
    const timer = requestAnimationFrame(() => {
      if (closeRef.current) {
        closeRef.current.focus();
      }
    });

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key === "Tab") {
        const dialog = dialogRef.current;
        if (!dialog) return;

        const focusableSelectors = [
          'a[href]',
          'button:not([disabled])',
          'input:not([disabled])',
          'select:not([disabled])',
          'textarea:not([disabled])',
          '[tabindex]:not([tabindex="-1"])',
        ].join(', ');

        const focusables = Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelectors)).filter(
          (el) => el.offsetParent !== null || el === closeRef.current
        );

        if (focusables.length === 0) {
          event.preventDefault();
          return;
        }

        const first = focusables[0];
        const last = focusables[focusables.length - 1];

        if (event.shiftKey) {
          if (document.activeElement === first || !dialog.contains(document.activeElement)) {
            event.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last || !dialog.contains(document.activeElement)) {
            event.preventDefault();
            first.focus();
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      cancelAnimationFrame(timer);
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      // Defensive focus restoration to trigger element if it still exists
      if (previousFocusRef.current && document.contains(previousFocusRef.current)) {
        try {
          previousFocusRef.current.focus();
        } catch {
          // ignore focus failure
        }
      }
    };
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 px-4 py-8 backdrop-blur-sm"
      role="presentation"
      onMouseDown={onClose}
    >
      <div
        ref={dialogRef}
        className="w-full max-w-4xl rounded-[4px] border border-white/[0.1] bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={subtitle ? subtitleId : undefined}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between border-b border-white/[0.07] bg-surface-low/80 px-6 py-4">
          <div className="min-w-0">
            <h3 id={titleId} className="font-display text-lg font-bold text-primary">
              {title}
            </h3>
            {subtitle ? (
              <p id={subtitleId} className="mt-1 min-w-0 break-words [overflow-wrap:anywhere] font-mono text-xs text-primary-subtle">
                {subtitle}
              </p>
            ) : null}
          </div>
          <button
            ref={closeRef}
            type="button"
            aria-label="Close dialog"
            onClick={onClose}
            className="rounded-[2px] p-1.5 text-primary-muted hover:bg-surface-mid hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
          >
            <Icon name="close" />
          </button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-6 py-6">{children}</div>
        {footer ? (
          <div className="flex flex-wrap justify-end gap-2.5 border-t border-white/[0.07] bg-surface-low/80 px-6 py-3.5">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  );
}
