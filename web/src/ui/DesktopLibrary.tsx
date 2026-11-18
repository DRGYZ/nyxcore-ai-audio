import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { desktopAction, fetchDesktopSession } from "../lib/api";
import type { DesktopSession } from "../lib/api";
import { ActionBanner, Button, Chip, Icon, Modal, PathBlock, SleeveArt } from "./components";

type ChoiceAction = "choose-folder" | "open-folder" | "open-demo" | "retry-library";

function useLibraryChoice(onKeepCurrent?: () => void) {
  const [path, setPath] = useState<string | null>(null);
  const [busy, setBusy] = useState<ChoiceAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef(false);

  async function run(action: ChoiceAction) {
    if (pending.current) return;
    pending.current = true;
    setBusy(action);
    setError(null);
    let navigating = false;
    try {
      if (action === "choose-folder") {
        const selected = await desktopAction<{ path: string | null }>(action);
        if (selected.path) setPath(selected.path);
      } else {
        const result = await desktopAction<{ origin: string | null }>(action);
        if (result.origin) {
          // A different loopback origin gives the app new query and document state.
          window.location.replace(result.origin + "/");
          navigating = true;
        } else {
          onKeepCurrent?.();
        }
      }
    } catch (failure) {
      if (action === "choose-folder") setPath(null);
      setError(failure instanceof Error ? failure.message : "NyxCore could not open the library.");
    } finally {
      if (!navigating) {
        pending.current = false;
        setBusy(null);
      }
    }
  }

  return { path, busy, error, run };
}

function LibraryChoices({ choice, canRetry }: { choice: ReturnType<typeof useLibraryChoice>; canRetry?: boolean }) {
  const { path, busy, error, run } = choice;
  const wasPicking = useRef(false);
  const choicesRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (busy === "choose-folder") wasPicking.current = true;
    if (!busy && wasPicking.current) {
      wasPicking.current = false;
      choicesRef.current?.querySelector("button")?.focus();
    }
  }, [busy]);

  return (
    <div className="space-y-4">
      {path && (
        <div className="space-y-1.5" role="status">
          <p className="text-xs font-medium text-primary-muted">Selected music folder</p>
          <PathBlock value={path} />
        </div>
      )}
      <div ref={choicesRef} className="flex flex-wrap items-center gap-2.5">
        {path && (
          <Button tone="primary" disabled={!!busy} onClick={() => void run("open-folder")}>
            <Icon name="arrow_forward" className="text-base" />
            {busy === "open-folder" ? "Opening Library…" : "Open Library"}
          </Button>
        )}
        <Button tone={path ? "secondary" : "primary"} disabled={!!busy} onClick={() => void run("choose-folder")}>
          <Icon name="folder_open" className="text-base" />
          {busy === "choose-folder" ? "Waiting for folder picker…" : "Choose Music Folder"}
        </Button>
        {canRetry && (
          <Button tone="secondary" disabled={!!busy} onClick={() => void run("retry-library")}>
            {busy === "retry-library" ? "Retrying…" : "Retry Library"}
          </Button>
        )}
      </div>
      {error && <ActionBanner tone="error" message={error} />}
      {busy && <p role="status" className="text-xs text-primary-muted">
        {busy === "choose-folder" ? "Select a folder in the Windows dialog." : "Opening the library and preparing its workspace…"}
      </p>}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border pt-4">
        <Button disabled={!!busy} onClick={() => void run("open-demo")}>Explore Demo Library</Button>
        <span className="text-xs text-primary-muted">Synthetic sample content</span>
      </div>
    </div>
  );
}

function Welcome({ session }: { session: DesktopSession }) {
  const choice = useLibraryChoice();
  return (
    <main className="grid min-h-screen place-items-center bg-background px-5 py-7 sm:px-8">
      <section className="w-full max-w-4xl overflow-hidden rounded-xl border border-border-bright bg-surface">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-6 py-4 sm:px-8">
          <div className="flex items-center gap-2.5 text-primary">
            <Icon name="album" className="text-2xl text-accent" />
            <span className="font-display text-base font-semibold">NyxCore</span>
          </div>
          <Chip tone="success">Local · Review only</Chip>
        </header>
        <div className="grid gap-6 p-6 sm:p-8 md:grid-cols-[minmax(0,1fr)_13rem] md:gap-10">
          <div className="min-w-0">
            <p className="text-xs font-medium text-accent">{session.issue ? "Library setup" : "Your music collection"}</p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight text-primary sm:text-4xl">
              {session.issue ? "Open a music library" : "Get to know your collection"}
            </h1>
            <p className="mt-3 max-w-[52ch] text-sm leading-relaxed text-primary-muted">
              Choose a folder to find duplicates, check tags, and plan how to organize your music. NyxCore analyzes it on this PC and leaves your audio files unchanged.
            </p>
            {session.issue && (
              <div className="mt-4 space-y-2">
                <ActionBanner tone="error" message={session.issue} />
                {session.retry_path && !session.issue.includes(session.retry_path) && <PathBlock value={session.retry_path} />}
              </div>
            )}
            <div className="mt-6">
              <LibraryChoices choice={choice} canRetry={session.can_retry} />
            </div>
          </div>
          <div aria-hidden="true" className="hidden flex-col items-center justify-center gap-4 border-l border-border pl-5 md:flex">
            <SleeveArt size="xl" title="Music collection" artist="NyxCore" showVinyl showMonogram={false} />
            <span className="font-editorial text-lg italic text-primary-muted">Pressed &amp; Cataloged</span>
          </div>
        </div>
      </section>
    </main>
  );
}

function ChangeLibraryDialog({ onClose }: { onClose: () => void }) {
  const choice = useLibraryChoice(onClose);
  return (
    <Modal open title="Change Library" onClose={onClose} dismissible={!choice.busy}>
      <p className="mb-5 text-sm text-primary-muted">
        Choose another folder to review. Each library keeps its own findings and staged changes.
      </p>
      <LibraryChoices choice={choice} />
    </Modal>
  );
}

export function ChangeLibraryButton() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button tone="secondary" onClick={() => setOpen(true)}>
        <Icon name="folder_open" className="text-base" />
        Change Library
      </Button>
      {open && <ChangeLibraryDialog onClose={() => setOpen(false)} />}
    </>
  );
}

export function DesktopEntry({ renderApp }: { renderApp: (session: DesktopSession) => ReactNode }) {
  const session = useQuery({ queryKey: ["desktop-session"], queryFn: fetchDesktopSession, retry: false, staleTime: Infinity });
  if (session.data?.configured) return renderApp(session.data);
  if (session.data) return <Welcome session={session.data} />;
  return (
    <main className="grid min-h-screen place-items-center bg-background p-5">
      <div className="w-full max-w-md space-y-4 rounded-xl border border-border bg-surface p-6">
        <h1 className="font-display text-2xl font-semibold text-primary">NyxCore</h1>
        {session.isError ? (
          <ActionBanner tone="error" message="Could not connect to the desktop session." action={
            <Button onClick={() => void session.refetch()}>Retry connection</Button>
          } />
        ) : <p role="status" className="text-sm text-primary-muted">Connecting to the local engine…</p>}
      </div>
    </main>
  );
}
