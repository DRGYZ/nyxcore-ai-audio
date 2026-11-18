import { useEffect, useId, useRef } from "react";
import type { PropsWithChildren, ReactNode } from "react";
import { useCheckConnection } from "../lib/hooks";
import { Button, Icon, Panel } from "./primitives";

const DEMO_COMMANDS = [
  {
    shell: "WSL / Bash",
    command: [
      "python demo/create_demo_library.py",
      'export NYXCORE_WEB_MUSIC_DIR="$PWD/demo/generated/sample-library"',
      'export NYXCORE_WEB_OUT_DIR="$PWD/data/demo-reports"',
    ],
  },
  {
    shell: "Native PowerShell",
    command: [
      "python demo/create_demo_library.py",
      '$env:NYXCORE_WEB_MUSIC_DIR = Join-Path $PWD "demo/generated/sample-library"',
      '$env:NYXCORE_WEB_OUT_DIR = Join-Path $PWD "data/demo-reports"',
    ],
  },
];

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
    <Panel className="min-w-0 p-5 sm:p-8">
      <div className="flex max-w-3xl flex-col gap-5">
        <div className="flex items-center gap-3.5">
          <div className="flex size-10 items-center justify-center rounded-xl border border-[#e9b44c]/30 bg-[#e9b44c]/12 text-[#f2c86b]">
            <Icon name="cloud_off" className="text-xl" />
          </div>
          <div>
            <p className="font-sans text-xs font-medium text-[#f2c86b]">Local Connection</p>
            <h2 className="font-display text-xl font-semibold text-primary">
              Local API is not connected {contextLabel ? `• ${contextLabel}` : ""}
            </h2>
          </div>
        </div>

        <p className="font-sans text-sm leading-relaxed text-primary-muted">
          NyxCore needs its local FastAPI backend running with a library folder and a reports folder.
        </p>

        <div className="space-y-4 text-sm text-primary-muted">
          <h3 className="font-sans font-semibold text-primary">Setup Instructions</h3>
          <ol className="list-decimal space-y-4 pl-5">
            {import.meta.env.PROD ? (
              <li>
                <p><strong className="text-primary">Restart the NyxCore launcher.</strong> Use the same library, workspace, and port, then reopen the printed URL.</p>
              </li>
            ) : <li>
              <p><strong className="text-primary">Start the backend with the synthetic library.</strong> Run the commands from the repository root with your NyxCore Python environment active. The generator needs FFmpeg and can be skipped if the demo already exists.</p>
              {DEMO_COMMANDS.map(({ shell, command }) => (
                <details key={shell} className="mt-2 min-w-0 rounded-lg border border-white/[0.08] bg-surface-low/80 p-3">
                  <summary className="cursor-pointer font-medium text-accent focus-visible:outline-accent">{shell} commands</summary>
                  <pre className="mt-2 select-all overflow-x-auto whitespace-pre-wrap rounded-lg bg-surface p-3 font-mono text-xs text-primary [overflow-wrap:anywhere]"><code>{[
                    ...command, "python -m uvicorn nyxcore.webapi.app:app --host 127.0.0.1 --port 8000",
                  ].join("\n")}</code></pre>
                </details>
              ))}
              <p className="mt-2 text-xs text-primary-subtle">For installation on Windows via WSL2, see <code>INSTALL_WSL.md</code> in the project root. PowerShell commands assume the dependencies are already installed.</p>
            </li>}
            <li>
              <p><strong className="text-primary">Confirm the connection.</strong> Leave the backend running and check the connection here.</p>
              <Button className="mt-2" tone="primary" onClick={handleCheck} disabled={isChecking}>
                <Icon name="refresh" className={`text-base ${isChecking ? "animate-spin" : ""}`} />
                {isChecking ? "Checking Connection…" : "Check Connection"}
              </Button>
            </li>
            <li><strong className="text-primary">Open Overview and inspect a finding.</strong> Browsing and plan previews leave audio files unchanged.</li>
          </ol>
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
      <div className="rounded-lg border border-white/[0.08] bg-surface-low px-4 py-2.5 font-sans text-xs text-primary-muted">
        Loading library data from local API…
      </div>
    );
  }
  if (error || usingMock) {
    return (
      <div className="rounded-lg border border-[#e9b44c]/25 bg-[#e9b44c]/12 px-4 py-2.5 font-sans text-xs text-[#f2c86b]">
        Local API unavailable. Connect the backend to inspect live library data.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-[#52b788]/25 bg-[#52b788]/12 px-4 py-2.5 font-sans text-xs text-[#88dcb3]">
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
  tone?: "info" | "success" | "warning" | "error";
  message: string;
  action?: ReactNode;
}) {
  const styles = {
    info: "border-l-2 border-l-accent border-white/[0.08] bg-surface-mid/90 text-primary",
    success: "border-l-2 border-l-[#52b788] border-white/[0.08] bg-surface-mid/90 text-primary",
    warning: "border-l-2 border-l-[#e9b44c] border-white/[0.08] bg-surface-mid/90 text-primary",
    error: "border-l-2 border-l-[#e06d53] border-white/[0.08] bg-surface-mid/90 text-primary",
  };
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
      aria-atomic="true"
      className={`flex flex-col gap-3 rounded-lg border px-4 py-3 text-xs md:flex-row md:items-center md:justify-between md:text-sm ${styles[tone]}`}
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
    <Panel className="border border-white/[0.07] bg-surface-low/60 px-6 py-12 text-center">
      <div className="mx-auto mb-3 flex size-11 items-center justify-center rounded-full border border-white/[0.08] bg-surface-mid text-accent">
        <Icon name="album" className="text-xl" />
      </div>
      <p className="font-display text-base font-semibold text-primary">{title}</p>
      <p className="mx-auto mt-1.5 max-w-xl font-sans text-sm leading-relaxed text-primary-muted">{description}</p>
      {action ? <div className="mt-5 flex justify-center">{action}</div> : null}
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
  dismissible = true,
}: PropsWithChildren<{ open: boolean; title: string; subtitle?: string; footer?: ReactNode; onClose: () => void; dismissible?: boolean }>) {
  const titleId = useId();
  const subtitleId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = dismissible ? onClose : () => {};
  }, [dismissible, onClose]);

  useEffect(() => {
    if (!open) return;
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    const timer = requestAnimationFrame(() => closeRef.current?.focus());
    return () => cancelAnimationFrame(timer);
  }, [open, title]);

  useEffect(() => {
    if (!open) return;

    previousFocusRef.current = (document.activeElement as HTMLElement | null) ?? null;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
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
          'summary',
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
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      if (previousFocusRef.current && document.contains(previousFocusRef.current)) {
        try {
          previousFocusRef.current.focus();
        } catch {
          // ignore focus failure
        }
      }
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 px-4 py-4 sm:py-8"
      role="presentation"
      onMouseDown={dismissible ? onClose : undefined}
    >
      <div
        ref={dialogRef}
        className="flex max-h-full w-full min-w-0 max-w-4xl flex-col overflow-hidden rounded-xl border border-white/[0.12] bg-surface shadow-soft"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={subtitle ? subtitleId : undefined}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex shrink-0 items-start justify-between border-b border-white/[0.08] bg-surface-low/90 px-5 py-4 sm:px-6">
          <div className="min-w-0">
            <h3 id={titleId} className="font-display text-lg font-semibold text-primary">
              {title}
            </h3>
            {subtitle ? (
              <p id={subtitleId} className="mt-1 min-w-0 break-words [overflow-wrap:anywhere] font-sans text-xs text-primary-muted">
                {subtitle}
              </p>
            ) : null}
          </div>
          <button
            ref={closeRef}
            type="button"
            aria-label="Close dialog"
            onClick={dismissible ? onClose : undefined}
            aria-disabled={!dismissible}
            className="rounded-lg p-1.5 text-primary-muted hover:bg-surface-mid hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <Icon name="close" />
          </button>
        </div>
        <div ref={bodyRef} className="min-h-0 overflow-y-auto px-5 py-5 sm:px-6">{children}</div>
        {footer ? (
          <div className="flex shrink-0 flex-wrap justify-end gap-2.5 border-t border-white/[0.08] bg-surface-low/90 px-5 py-3.5 sm:px-6">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  );
}
