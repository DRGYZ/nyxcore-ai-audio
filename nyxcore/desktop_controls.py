"""Only the owning WebView can request these native desktop actions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from secrets import compare_digest
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from nyxcore.core.atomic import atomic_write_text
from nyxcore.desktop_library import DesktopLibraries, LibraryChoice, LibrarySession, overlaps

CONTROL_PATH = "/api/desktop/"
WINDOW_HEADER = "X-NyxCore-Window"


class LibraryRequests:
    """Pause admission only when old workspace requests have finished."""

    def __init__(self):
        self.lock = Lock()
        self.active = 0
        self.switching = False

    def freeze(self):
        with self.lock:
            if self.active or self.switching:
                raise HTTPException(409, "The library is busy. Wait for the current work to finish, then try again.")
            self.switching = True

    def resume(self):
        with self.lock:
            self.switching = False

    def install(self, app: FastAPI):
        @app.middleware("http")
        async def count_requests(request: Request, call_next):
            path = request.url.path
            if not path.startswith("/api/") or path.startswith(CONTROL_PATH):
                return await call_next(request)
            with self.lock:
                if self.switching:
                    return JSONResponse({"detail": "NyxCore is switching libraries."}, status_code=409)
                self.active += 1
            try:
                return await call_next(request)
            finally:
                with self.lock:
                    self.active -= 1


def install_desktop_controls(
    app: FastAPI,
    *,
    libraries: DesktopLibraries,
    session: LibrarySession | None,
    issue: str | None,
    retry_choice: LibraryChoice | None,
    window_key: str,
    origin,
    choose_folder,
    choose_export,
    activate,
):
    gate = LibraryRequests()
    gate.install(app)
    control_lock = Lock()
    selected: LibraryChoice | None = None

    def authorize(request: Request):
        supplied = request.headers.get(WINDOW_HEADER, "").encode("latin-1")
        if not compare_digest(supplied, window_key.encode("ascii")):
            raise HTTPException(403, "This action requires the NyxCore desktop window.")
        if request.method == "POST" and request.headers.get("origin") != origin():
            raise HTTPException(403, "This action requires the current desktop origin.")

    def exclusive():
        if not control_lock.acquire(blocking=False):
            raise HTTPException(409, "Finish the open NyxCore action and try again.")
        try:
            if gate.switching:
                raise HTTPException(409, "NyxCore is switching libraries.")
            yield
        finally:
            control_lock.release()

    access = [Depends(authorize)]
    action = [Depends(authorize), Depends(exclusive)]

    @app.get(CONTROL_PATH + "session", dependencies=access, include_in_schema=False)
    def desktop_session():
        return {
            "configured": session is not None,
            "demo": bool(session and session.demo),
            "issue": issue,
            "retry_path": str(retry_choice.music) if retry_choice and retry_choice.music else None,
            "can_retry": retry_choice is not None,
        }

    @app.post(CONTROL_PATH + "choose-folder", dependencies=action, include_in_schema=False)
    def select_folder():
        nonlocal selected
        path = choose_folder(session.music if session and not session.demo else None)
        if path is None:
            return {"path": None}
        selected = None
        try:
            prepared = libraries.resolve(LibraryChoice(path))
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        selected = LibraryChoice(prepared.music)
        return {"path": str(prepared.music)}

    def open_library(choice: LibraryChoice, *, remember: bool):
        gate.freeze()
        try:
            target = libraries.resolve(choice)
            if target == session:
                if remember:
                    libraries.remember(target)
                gate.resume()
                return {"origin": None}
            if session and (
                target.workspace == session.workspace
                or overlaps(target.workspace, session.music)
                or overlaps(target.music, session.workspace)
            ):
                raise ValueError("Choose a music folder and workspace that are separate from the current library’s workspace and music folder.")
            return {"origin": activate(target, remember)}
        except (ValueError, OSError, RuntimeError) as exc:
            gate.resume()
            raise HTTPException(422, str(exc)) from exc
        except BaseException:
            gate.resume()
            raise

    @app.post(CONTROL_PATH + "open-folder", dependencies=action, include_in_schema=False)
    def open_folder():
        if selected is None:
            raise HTTPException(409, "Choose a music folder first.")
        return open_library(selected, remember=True)

    @app.post(CONTROL_PATH + "open-demo", dependencies=action, include_in_schema=False)
    def open_demo():
        return open_library(LibraryChoice(None), remember=True)

    @app.post(CONTROL_PATH + "retry-library", dependencies=action, include_in_schema=False)
    def retry_library():
        if retry_choice is None:
            raise HTTPException(409, "Choose a music folder first.")
        return open_library(retry_choice, remember=False)

    @app.post(CONTROL_PATH + "export-health", dependencies=action, include_in_schema=False)
    def export_health():
        report = app.state.desktop_health_report() if session else None
        if report is None:
            raise HTTPException(409, "Wait for Library Health to finish loading before exporting.")
        path = choose_export(f"nyxcore-health-{date.today().isoformat()}.json")
        if path is None:
            return {"path": None}
        try:
            destination = path.resolve()
            if destination.suffix.lower() != ".json":
                raise ValueError("Choose a file with the .json extension.")
            protected = (session.music, session.workspace, libraries.state.resolve(), libraries.application)
            if any(root is not None and destination.is_relative_to(root) for root in protected):
                raise ValueError("Save the report outside the music library and NyxCore folders.")
            if not destination.parent.is_dir():
                raise ValueError("Choose an existing destination folder.")
            atomic_write_text(destination, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f"Could not save the Health report. {exc}") from exc
        return {"path": str(destination)}
