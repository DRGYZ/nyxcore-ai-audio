"""Run the review-only local app in one Windows WebView2 window."""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import socket
import sys
import webbrowser
from contextlib import contextmanager
from ctypes import wintypes
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from urllib.parse import urlsplit


HOST = "127.0.0.1"
RESOURCES = Path(__file__).with_name("resources")
BACKGROUND = "#11100e"
CAPTION_BACKGROUND = "#141311"
FOREGROUND = "#f5f0e6"
WINDOW_SIZE = (1320, 860)
MIN_SIZE = (740, 560)
STARTUP_TIMEOUT = 15
LOG = logging.getLogger(__name__)


def local_navigation(url: str, origin: str | None) -> bool:
    if url == "about:blank":
        return True
    try:
        target = urlsplit(url)
        return origin is not None and f"{target.scheme}://{target.netloc}" == origin
    except ValueError:
        return False


def external_reference(url: str) -> bool:
    """Allow the app's explicit MusicBrainz reference links in the system browser."""
    try:
        target = urlsplit(url)
        return target.scheme == "https" and target.netloc.lower() == "musicbrainz.org"
    except ValueError:
        return False


class DesktopBackend:
    """Own exactly one loopback socket and one in-process Uvicorn thread."""

    def __init__(self, create_app):
        self.create_app = create_app
        self.origin: str | None = None
        self.error: Exception | None = None
        self.stopping = Event()
        self.finished = Event()
        self.server = None
        self.thread = Thread(target=self._run, name="nyxcore-desktop-api", daemon=True)

    def _run(self) -> None:
        try:
            from uvicorn import Config, Server

            app = self.create_app()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                if sys.platform == "win32":
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind((HOST, 0))
                port = listener.getsockname()[1]
                self.origin = f"http://{HOST}:{port}"
                self.server = Server(Config(
                    app, host=HOST, port=port, workers=1, proxy_headers=False,
                    access_log=False, log_config=None, timeout_graceful_shutdown=3,
                ))
                if not self.stopping.is_set():
                    self.server.run(sockets=[listener])
        except (Exception, SystemExit) as exc:
            LOG.exception("Local engine failed")
            self.error = exc if isinstance(exc, Exception) else RuntimeError("The local engine stopped during startup.")
        finally:
            self.finished.set()

    def start(self) -> None:
        self.thread.start()

    def ready(self) -> bool:
        return self.server is not None and self.server.started and not self.finished.is_set()

    def stop(self) -> None:
        self.stopping.set()
        if self.server is not None:
            self.server.should_exit = True

    def close(self) -> None:
        self.stop()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            LOG.warning("The local engine is still completing an active request.")


def create_desktop_app(music_path: Path | None, workspace: Path):
    from nyxcore.launch import SOURCE_ROOT, check_workspace, create_local_app, resolve_library_workspace

    music, out = resolve_library_workspace(music_path, workspace)
    app = create_local_app(frontend_dir=SOURCE_ROOT / "web/dist", music_root=music, workspace_root=out)
    check_workspace(out)
    return app


@contextmanager
def desktop_instance():
    """Keep double-launches from starting a second desktop engine in this session."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateMutexW
    create.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    handle = create(None, False, r"Local\NyxCoreDesktop01")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
    finally:
        close(handle)


def place_window(form, screen) -> None:
    from System.Drawing import ColorTranslator, Rectangle, Size
    from System.Windows.Forms import FormStartPosition, SystemInformation

    area = screen.WorkingArea
    form.StartPosition = FormStartPosition.Manual
    form.Bounds = area
    dpi = ctypes.windll.user32.GetDpiForWindow(wintypes.HWND(form.Handle.ToInt64()))
    scale = (dpi or 96) / 96
    width = min(round(WINDOW_SIZE[0] * scale), area.Width)
    height = min(round(WINDOW_SIZE[1] * scale), area.Height)
    form.BackColor = ColorTranslator.FromHtml(BACKGROUND)
    form.MinimumSize = Size(min(round(MIN_SIZE[0] * scale), width), min(round(MIN_SIZE[1] * scale), height))
    form.Bounds = Rectangle(area.X + (area.Width - width) // 2, area.Y + (area.Height - height) // 2, width, height)
    if not SystemInformation.HighContrast:
        hwnd = wintypes.HWND(form.Handle.ToInt64())
        for attribute, color in ((35, CAPTION_BACKGROUND), (36, FOREGROUND)):  # DWM caption and text colors
            value = wintypes.DWORD(ColorTranslator.ToWin32(ColorTranslator.FromHtml(color)))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    LOG.info("Display %s, %d DPI, window %dx%d", screen.DeviceName, round(96 * scale), width, height)


def native_status(form, message: str):
    from System.Drawing import ColorTranslator, ContentAlignment, Font
    from System.Windows.Forms import DockStyle, Label

    label = Label()
    label.Dock = DockStyle.Fill
    label.TextAlign = ContentAlignment.MiddleCenter
    label.Font = Font("Segoe UI", 13)
    label.BackColor = ColorTranslator.FromHtml(BACKGROUND)
    label.ForeColor = ColorTranslator.FromHtml(FOREGROUND)
    label.Text = message
    form.Controls.Add(label)
    label.BringToFront()
    return label


def require_webview2() -> None:
    import winreg

    key = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for hive, view in ((winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY), (winreg.HKEY_CURRENT_USER, 0)):
        try:
            with winreg.OpenKey(hive, key, access=winreg.KEY_READ | view) as handle:
                version, _ = winreg.QueryValueEx(handle, "pv")
        except OSError:
            continue
        if isinstance(version, str) and version.strip() not in {"", "0.0.0.0"}:
            return
    raise RuntimeError("Microsoft Edge WebView2 Runtime is required. Install or repair it, then reopen NyxCore.")


def run_window(music: Path | None, workspace: Path, display: int, started_at: float) -> int:
    require_webview2()
    # Set per-monitor DPI awareness before pywebview/WinForms create window handles.
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NyxCore.Desktop")
    try:
        import webview
        from webview.guilib import initialize
    except ImportError as exc:
        raise RuntimeError('Install the desktop extra with python -m pip install -e ".[desktop]".') from exc

    gui = initialize("edgechromium")
    if gui.renderer != "edgechromium":
        raise RuntimeError("Microsoft Edge WebView2 Runtime is required. Install or repair it, then reopen NyxCore.")
    screens = sorted(gui.WinForms.Screen.AllScreens, key=lambda screen: (not screen.Primary, screen.DeviceName))
    if not 1 <= display <= len(screens):
        raise ValueError(f"Display {display} is unavailable. Choose a display from 1 to {len(screens)}.")

    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
    webview.settings["ALLOW_FILE_URLS"] = False
    webview.settings["ALLOW_DOWNLOADS"] = False
    backend = DesktopBackend(lambda: create_desktop_app(music, workspace))
    renderer_ready = Event()
    renderer_failed = Event()
    launch_failed = Event()
    status_surface = None
    window = webview.create_window(
        "NyxCore", width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
        min_size=MIN_SIZE, background_color=BACKGROUND, text_select=True,
    )

    def guard_navigation(_, args):
        args.Cancel = not local_navigation(str(args.Uri), backend.origin)

    def open_reference(_, args):
        args.Handled = True
        url = str(args.Uri)
        if args.IsUserInitiated and external_reference(url):
            Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    def navigation_completed(sender, args):
        url = str(sender.Source)
        if url != "about:blank" and local_navigation(url, backend.origin):
            if args.IsSuccess:
                status_surface.Visible = False
                LOG.info("Desktop document loaded after %.3fs", monotonic() - started_at)
            else:
                status_surface.Text = "Could not load NyxCore\n\nClose this window and reopen the app."
                status_surface.Visible = True
                LOG.warning("Desktop navigation failed: %s", args.WebErrorStatus)

    def renderer_ready_on_ui(sender, args):
        try:
            if not args.IsSuccess:
                raise RuntimeError(str(args.InitializationException))
            core = sender.CoreWebView2
            # pywebview 6.2.1 otherwise opens or navigates every popup itself.
            core.NewWindowRequested -= window.native.browser.on_new_window_request
            core.NewWindowRequested += open_reference
            core.NavigationStarting += guard_navigation
            core.FrameNavigationStarting += guard_navigation
            core.NavigationCompleted += navigation_completed
            core.Settings.AreHostObjectsAllowed = False
            core.Settings.IsWebMessageEnabled = False
            renderer_ready.set()
        except Exception:
            LOG.exception("Could not initialize the guarded WebView2 renderer")
            renderer_failed.set()
            backend.stop()
            status_surface.Text = "Could not start NyxCore\n\nRepair the WebView2 runtime, then reopen NyxCore."

    def configure_native():
        nonlocal status_surface
        try:
            place_window(window.native, screens[display - 1])
            status_surface = native_status(window.native, "Starting NyxCore…")
            window.native.webview.CoreWebView2InitializationCompleted += renderer_ready_on_ui
        except Exception:
            LOG.exception("Could not configure the native window")
            renderer_failed.set()
            backend.stop()
            # Prevent a bad monitor placement from showing on the wrong display.
            from System import Action

            window.native.Opacity = 0
            window.native.BeginInvoke(Action(window.native.Close))

    def coordinate_startup():
        deadline = monotonic() + STARTUP_TIMEOUT
        while not backend.stopping.wait(0.025) and not renderer_failed.is_set():
            if renderer_ready.is_set() and backend.ready():
                LOG.info("Local engine ready at %s after %.3fs", backend.origin, monotonic() - started_at)
                window.load_url(f"{backend.origin}/?desktop=1")
                return
            if (backend.finished.is_set() and status_surface is not None) or monotonic() >= deadline:
                launch_failed.set()
                problem = (
                    str(backend.error) if backend.error else
                    "The local engine did not become ready. Close this window and reopen NyxCore."
                )
                if status_surface is not None:
                    from System import Action

                    window.native.BeginInvoke(Action(lambda: setattr(
                        status_surface, "Text", f"Could not start NyxCore\n\n{problem}",
                    )))
                backend.stop()
                return

    window.events.before_show += configure_native
    window.events.closed += backend.stop
    window.events.shown += lambda: LOG.info("Window shown after %.3fs", monotonic() - started_at)
    backend.start()
    try:
        with TemporaryDirectory(prefix="nyxcore-webview2-") as profile:
            webview.start(
                coordinate_startup, gui="edgechromium", debug=False, http_server=False,
                private_mode=True, storage_path=profile, icon=str(RESOURCES / "desktop.ico"),
            )
    finally:
        backend.close()
    LOG.info("Desktop closed")
    return int(renderer_failed.is_set() or launch_failed.is_set())


def main(argv: list[str] | None = None) -> int:
    started_at = monotonic()
    parser = argparse.ArgumentParser(description="Open the NyxCore review-only desktop prototype.")
    library = parser.add_mutually_exclusive_group()
    library.add_argument("--demo", action="store_true", help="Use the synthetic demo library (default)")
    library.add_argument("--music", type=Path, help="Existing library to inspect")
    parser.add_argument("--out", type=Path, help="Separate workspace, required with --music")
    parser.add_argument("--display", type=int, default=1, help="Display number, with the primary display first")
    args = parser.parse_args(argv)
    if args.music is not None and args.out is None:
        parser.error("--music requires a separate --out workspace")
    if sys.platform != "win32":
        parser.error("The desktop prototype requires Windows and Microsoft Edge WebView2.")

    state = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NyxCore" / "Desktop"
    try:
        state.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(state / "desktop.log", maxBytes=256_000, backupCount=2, encoding="utf-8")
        logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        with desktop_instance() as unique:
            if not unique:
                LOG.info("NyxCore Desktop is already running.")
                return 0
            LOG.info("Desktop starting [pid=%d]", os.getpid())
            return run_window(args.music, args.out or state / "demo-workspace", args.display, started_at)
    except Exception as exc:
        LOG.exception("NyxCore Desktop could not start")
        if args.display == 1:
            ctypes.windll.user32.MessageBoxW(None, f"{exc}\n\nSee {state / 'desktop.log'}.", "NyxCore", 0x10)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
