"""Verify both Windows ZIPs and smoke-test the extracted GUI executable."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from ctypes import wintypes
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from zipfile import ZipFile

import pefile

from windows.notices import digest

MARKER = b'<meta name="nyxcore-desktop" content="1">'
HTTP = build_opener(ProxyHandler({}))
READY = re.compile(r"Local engine ready at (http://127\.0\.0\.1:\d+)")


def require(condition: bool, problem: str) -> None:
    if not condition:
        raise ValueError(problem)


def verify_zip(path: Path) -> None:
    expected, name = path.with_name(path.name + ".sha256").read_text(encoding="ascii").split()
    require(name == path.name and digest(path) == expected, f"Checksum mismatch for {path.name}.")
    with ZipFile(path) as archive:
        require(archive.testzip() is None, f"Damaged ZIP {path.name}.")
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            require(bool(parts) and parts[0] == path.stem and ".." not in parts and "\\" not in name,
                    f"Invalid archive member {name}.")


def verify_source(package: Path, provenance: dict) -> None:
    companion = provenance["source_companion"]
    source = package.with_name(companion["file"])
    verify_zip(source)
    require(digest(source) == companion["sha256"], "The source companion belongs to another binary build.")
    with ZipFile(source) as archive:
        root = source.stem + "/"
        manifest = json.loads(archive.read(root + "PREVIEW_BUILD.json"))
        require(manifest["source_commit"] == provenance["source_commit"], "Source commit mismatch.")
        recorded = json.loads(archive.read(root + "SOURCE_FILES.json"))
        for name, expected in recorded.items():
            member = archive.read(root + "project/" + name)
            require(hashlib.sha256(member).hexdigest() == expected, f"Source snapshot differs at {name}.")
        require(any(name.startswith(root + "upstream/python/mutagen-") for name in archive.namelist()),
                "The exact source distribution for Mutagen is missing.")


def read_http(origin: str, path: str, data=None, headers=None) -> tuple[int, bytes]:
    body = None if data is None else json.dumps(data).encode()
    request = Request(origin + path, body, {"Content-Type": "application/json", **(headers or {})})
    try:
        with HTTP.open(request, timeout=20) as reply:
            return reply.status, reply.read()
    except HTTPError as error:
        return error.code, error.read()


def engine_address(process: subprocess.Popen, log: Path) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process.poll() is None:
        content = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if match := READY.search(content):
            return match[1]
        time.sleep(0.05)
    raise RuntimeError(f"The extracted executable did not start.\n{log.read_text(encoding='utf-8') if log.exists() else ''}")


def close_window(pid: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    window = []

    @callback_type
    def find(handle, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(handle):
            window.append(handle)
            return False
        return True

    user32.EnumWindows(find, 0)
    return bool(window and user32.PostMessageW(window[0], 0x10, 0, 0))


def hash_files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}


def smoke_executable(app: Path, sandbox: Path, display: int) -> dict:
    exe = app / "NyxCore.exe"
    with pefile.PE(str(exe), fast_load=True) as pe:
        require(pe.FILE_HEADER.Machine == 0x8664 and pe.OPTIONAL_HEADER.Subsystem == 2, "Expected a windowed x64 executable.")

    system = Path(os.environ["SystemRoot"])
    isolated = sandbox / "Isolated App Data"
    state = isolated / "NyxCore/Desktop"
    log = state / "desktop.log"
    env = {
        name: value for name, value in os.environ.items()
        if name.upper() not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"}
    }
    env.update(LOCALAPPDATA=str(isolated), PATH=f"{system / 'System32'};{system}")
    before = hash_files(app)
    args = [str(exe), "--demo", "--display", str(display)]
    started = time.monotonic()
    process = subprocess.Popen(args, cwd=sandbox, env=env)
    try:
        origin = engine_address(process, log)
        status, raw = read_http(origin, "/api/status")
        settings = json.loads(raw)
        require(status == 200 and settings["review_only"], "Preview did not report review-only mode.")
        require(Path(settings["music_path"]).is_relative_to(app.resolve()), "Demo loaded outside the extracted ZIP.")
        require(Path(settings["out_path"]).is_relative_to(state.resolve()), "Workspace is outside disposable user state.")
        status, page = read_http(origin, "/health")
        require(status == 200 and MARKER in page, "Desktop HTML or assets could not be loaded from the package.")
        progress, raw = read_http(origin, "/api/library")
        overview = json.loads(raw)["health"]["overview"]
        require(progress == 200 and overview["total_audio_files"] == 7, "The synthetic demo analysis is incomplete.")
        usable = round(time.monotonic() - started, 3)
        preview_code, raw = read_http(origin, "/api/organization/preview", {})
        preview = json.loads(raw)
        require(preview_code == 200 and not preview["apply_enabled"], "Organization did not stay review-only.")
        targets = [
            ("/api/review/plan/apply", {"plan_report": {}}),
            ("/api/history/fixture/restore", {}),
            ("/api/history/fixture/undo", {}),
            ("/api/metadata/plan/apply", {}),
            (f"/api/organization/preview/{preview['id']}/review", {"paths": ["fixture.flac"]}),
            (f"/api/organization/preview/{preview['id']}/apply", {"paths": ["fixture.flac"]}),
        ]
        muted = {path: read_http(origin, path, body)[0] for path, body in targets}
        require(all(code == 403 for code in muted.values()), f"A review-only route responded unexpectedly: {muted}")
        status, _ = read_http(origin, "/api/desktop/session")
        denied, _ = read_http(origin, "/api/desktop/choose-folder", {}, headers={"Origin": origin})
        require(status == denied == 403, "A desktop action accepted an uncredentialed local caller.")
        duplicate = subprocess.run(args, cwd=sandbox, env=env, timeout=15, check=True)
        require(duplicate.returncode == 0 and "already running" in log.read_text(encoding="utf-8"),
                "Duplicate launch did not take the singleton exit path.")
        require(close_window(process.pid), "Could not close the packaged NyxCore window.")
        shutdown_start = time.monotonic()
        require(process.wait(timeout=15) == 0, "The packaged NyxCore window did not close cleanly.")
        shutdown = round(time.monotonic() - shutdown_start, 3)
        with socket.socket() as probe:
            probe.settimeout(2)
            require(probe.connect_ex(("127.0.0.1", int(origin.rsplit(":", 1)[1]))) != 0,
                    "The desktop engine listener remained open.")
        require(before == hash_files(app), "The executable changed its extracted files during acceptance.")
        return {"demo_tracks": 7, "review_only_statuses": muted, "desktop_controls_without_key": [status, denied],
                "duplicate_exit": duplicate.returncode, "bundle_unchanged": True,
                "http_overview_ready_seconds": usable, "shutdown_seconds": shutdown,
                "extracted_bytes": sum(path.stat().st_size for path in app.rglob("*") if path.is_file())}
    finally:
        if process.poll() is None:
            if close_window(process.pid):
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--display", type=int, default=1)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    verify_zip(package)
    with TemporaryDirectory(prefix="NyxCore release 音楽 ", ignore_cleanup_errors=True) as directory, ZipFile(package) as archive:
        sandbox = Path(directory)
        root = package.stem + "/"
        provenance = json.loads(archive.read(root + "PREVIEW_BUILD.json"))
        verify_source(package, provenance)
        names = archive.namelist()
        for name in names:
            require(not any(part.lower() in {"node_modules", ".venv", "tests"} for part in PurePosixPath(name).parts)
                    and not re.search(r"/(?:pythonw?|node|npm|ffmpeg)\.exe$", name, re.I), f"Development payload in {name}.")
        archive.extractall(sandbox)
        result = smoke_executable(sandbox / package.stem, sandbox, args.display)
    receipt = {"source_commit": provenance["source_commit"], "zip_sha256": digest(package), **result}
    report = json.dumps(receipt, indent=2)
    if args.receipt:
        args.receipt.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
