"""Run the source checkout's web application in review-only mode."""

from __future__ import annotations

import argparse
import re
import socket
import sys
import webbrowser
from pathlib import Path
from tempfile import TemporaryFile
from threading import Thread

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from yaml import YAMLError

from nyxcore.config import load_config
from nyxcore.webapi.app import configured_config_path, create_app

SOURCE_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
SPA_ROUTES = ("/", "/search", "/review", "/duplicates", "/health", "/history", "/playlists")


def create_local_app(*, frontend_dir: Path, music_root: Path, workspace_root: Path) -> FastAPI:
    index = frontend_dir / "index.html"
    build_error = f"Frontend build is missing or incomplete at {frontend_dir}. Run npm ci and npm run build in web/."
    if not index.is_file():
        raise ValueError(build_error)
    assets = re.findall(r'(?:href|src)="(/assets/[^"]+)"', index.read_text(encoding="utf-8"))
    if not any(asset.endswith(".js") for asset in assets) or any(
        not (frontend_dir / asset.lstrip("/")).is_file() for asset in assets
    ):
        raise ValueError(build_error)

    app = create_app(music_root=music_root, workspace_root=workspace_root, review_only=True)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[HOST])
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="frontend-assets")

    def frontend() -> FileResponse:
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    for route in SPA_ROUTES:
        app.add_api_route(route, frontend, methods=["GET", "HEAD"], include_in_schema=False)
    return app


def _open_browser(url: str) -> None:
    try:
        if webbrowser.open(url):
            return
    except (OSError, webbrowser.Error):
        pass
    print(f"Could not open a browser automatically. Open {url} in your browser.", file=sys.stderr, flush=True)


class LocalServer(uvicorn.Server):
    def __init__(self, app: FastAPI, *, port: int, music: Path, out: Path, open_browser: bool) -> None:
        super().__init__(uvicorn.Config(app, host=HOST, port=port, workers=1, proxy_headers=False))
        self.url = f"http://{HOST}:{port}/"
        self.music = music
        self.out = out
        self.open_browser = open_browser

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if not self.started:
            return
        print(f"NyxCore is ready at {self.url} (review-only)\nLibrary: {self.music}\nWorkspace: {self.out}", flush=True)
        if self.open_browser:
            Thread(target=_open_browser, args=(self.url,), daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch NyxCore locally in review-only mode.")
    library = parser.add_mutually_exclusive_group(required=True)
    library.add_argument("--demo", action="store_true", help="Use the generated synthetic library in this checkout")
    library.add_argument("--music", type=Path, help="Folder to inspect")
    parser.add_argument("--out", type=Path, required=True, help="Separate workspace for reports and review state")
    parser.add_argument("--port", type=int, default=8000, help="Loopback port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        if args.demo:
            music = (SOURCE_ROOT / "demo/generated/sample-library").resolve()
            if not (music / "nyxcore_demo_manifest.json").is_file():
                parser.error("Synthetic demo is missing. Run python demo/create_demo_library.py first.")
        else:
            music = args.music.expanduser().resolve()
        out = args.out.expanduser().resolve()
        if not music.is_dir():
            parser.error(f"Music directory does not exist or is inaccessible: {music}")
        next(music.iterdir(), None)
        if out.is_relative_to(music) or music.is_relative_to(out):
            parser.error("--out and the music directory must be separate, non-overlapping folders")
        try:
            load_config(configured_config_path())
        except (OSError, ValueError, TypeError, AttributeError, YAMLError) as exc:
            parser.error(f"Invalid NyxCore configuration: {exc}")
        app = create_local_app(frontend_dir=SOURCE_ROOT / "web/dist", music_root=music, workspace_root=out)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            if sys.platform == "win32":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((HOST, args.port))
        except OSError as exc:
            parser.error(f"Cannot bind {HOST}:{args.port}. Try another --port. ({exc})")
        try:
            out.mkdir(parents=True, exist_ok=True)
            with TemporaryFile(dir=out):
                pass
        except OSError as exc:
            parser.error(f"Cannot write to workspace {out}. Choose a writable --out directory. ({exc})")

        server = LocalServer(app, port=args.port, music=music, out=out, open_browser=not args.no_browser)
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            return 0
        return 0 if server.started else 1


if __name__ == "__main__":
    raise SystemExit(main())
