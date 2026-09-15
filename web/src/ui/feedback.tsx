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
          <div className="flex size-10 items-center justify-center rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-400">
            <Icon name="cloud_off" className="text-xl" />
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.24em] text-amber-400">Local Connection</p>
            <h2 className="font-display text-xl font-bold text-slate-100">
              Local API is not connected {contextLabel ? `• ${contextLabel}` : ""}
            </h2>
          </div>
        </div>

        <p className="text-sm leading-relaxed text-slate-300">
          NyxCore is an experimental local-first toolkit. It connects to a local FastAPI backend running on your machine
          at <code className="rounded bg-background-dark px-1.5 py-0.5 font-mono text-xs text-primary">http://127.0.0.1:8000</code>.
          The local server is not currently reachable.
        </p>

        <div className="space-y-4 rounded-xl border border-border-dark bg-background-dark/70 p-5">
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400">Setup Instructions</p>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-200">
              1. Start the local API pointed at your music library & reports directory:
            </p>
            <pre className="overflow-x-auto rounded-lg bg-background-dark p-3 font-mono text-xs text-slate-300">
{`export NYXCORE_WEB_MUSIC_DIR="/path/to/your/music"
export NYXCORE_WEB_OUT_DIR="data/reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000`}
            </pre>
          </div>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-200">
              2. Or generate and inspect the built-in demo library:
            </p>
            <pre className="overflow-x-auto rounded-lg bg-background-dark p-3 font-mono text-xs text-slate-300">
{`python demo/create_demo_library.py --force
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000`}
            </pre>
          </div>

          <p className="text-xs text-slate-400">
            For step-by-step setup guides (including Windows WSL2 setup and dependencies), refer to <span className="font-mono text-primary">INSTALL_WSL.md</span> in the project root.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <Button tone="primary" onClick={handleCheck} disabled={isChecking}>
            <Icon name="refresh" className={`text-base ${isChecking ? "animate-spin" : ""}`} />
            {isChecking ? "Checking Connection…" : "Check Connection"}
          </Button>
          <span className="text-xs text-slate-500">
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
      <div className="rounded-lg border border-border-dark bg-background-dark/70 px-4 py-3 text-sm text-slate-400">
        Loading library data from local API…
      </div>
    );
  }
  if (error || usingMock) {
    return (
      <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
        Local API unavailable. Connect the backend to inspect live library data.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-300">
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
    info: "border-primary/20 bg-primary/5 text-primary",
    success: "border-emerald-500/20 bg-emerald-500/10 text-emerald-400",
    error: "border-rose-500/20 bg-rose-500/10 text-rose-400",
  };
  return (
    <div className={`flex flex-col gap-3 rounded-lg border px-4 py-3 text-sm md:flex-row md:items-center md:justify-between ${styles[tone]}`}>
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
    <Panel className="border-dashed px-6 py-10 text-center">
      <p className="font-display text-lg font-bold text-slate-100">{title}</p>
      <p className="mx-auto mt-2 max-w-xl text-sm text-slate-400">{description}</p>
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
}: PropsWithChildren<{ open: boolean; title: string; subtitle?: string; footer?: ReactNode; onClose: () => void }>) {
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 px-4 py-8 backdrop-blur-sm" role="presentation" onMouseDown={onClose}>
      <div
        className="w-full max-w-4xl rounded-2xl border border-primary/20 bg-surface-dark shadow-[0_0_40px_rgba(0,0,0,0.45)]"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between border-b border-border-dark px-6 py-5">
          <div className="min-w-0">
            <h3 id={titleId} className="font-display text-xl font-bold text-slate-100">{title}</h3>
            {subtitle ? <p className="mt-1 break-all text-xs text-slate-500">{subtitle}</p> : null}
          </div>
          <button ref={closeRef} onClick={onClose} className="rounded-full p-2 text-slate-400 hover:bg-white/5 hover:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            <Icon name="close" />
          </button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-6 py-6">{children}</div>
        {footer ? <div className="flex flex-wrap justify-end gap-3 border-t border-border-dark px-6 py-4">{footer}</div> : null}
      </div>
    </div>
  );
}
