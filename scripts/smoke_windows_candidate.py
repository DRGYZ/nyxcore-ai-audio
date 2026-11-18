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

from windows.notices import GPL_NOTICES, REQUIRED_SOURCE_DISTRIBUTIONS, digest

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


def verify_source_materials(
    binary: ZipFile, source: ZipFile, binary_root: str, source_root: str, recorded: dict[str, str],
) -> None:
    for notice, original in GPL_NOTICES.items():
        for archive, root in ((binary, binary_root), (source, source_root)):
            member = root + notice
            require(member in archive.namelist(), f"Missing GPL notice: {member}")
            require(hashlib.sha256(archive.read(member)).hexdigest() == recorded.get(original),
                    f"GPL notice differs from the source snapshot: {member}")

    inventory = source.read(source_root + "WINDOWS_DEPENDENCIES.json")
    require(binary.read(binary_root + "WINDOWS_DEPENDENCIES.json") == inventory,
            "Binary and source dependency inventories differ.")
    dependencies = json.loads(inventory)
    for name in REQUIRED_SOURCE_DISTRIBUTIONS:
        package = dependencies["python"].get(name, {})
        require(bool(package.get("file") and package.get("notices")), f"Missing source or copying terms for {name}.")
        for notice in package["notices"]:
            member = f"licenses/python/{name}/{notice}"
            require(binary_root + member in binary.namelist() and source_root + member in source.namelist(),
                    f"Missing upstream license notice: {member}")
            require(binary.read(binary_root + member) == source.read(source_root + member),
                    f"Upstream license notice differs across the archives: {member}")
    inputs = {
        **{f"upstream/python/{item['file']}": item["sha256"]
           for item in dependencies["python"].values() if "file" in item},
        **{f"upstream/rust/{item['name']}-{item['version']}.crate": item["sha256"]
           for item in dependencies["pydantic_core_locked_rust_sources"]},
        **{f"upstream/native/{name}": item["sha256"]
           for name, item in dependencies["runtime"]["source_archives"].items()},
    }
    source_members = set(source.namelist())
    for name, expected in inputs.items():
        member = source_root + name
        require(member in source_members, f"Missing upstream source: {name}")
        with source.open(member) as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                    f"Upstream source checksum mismatch: {name}")


def verify_pair(
    package: Path, binary: ZipFile, provenance: dict, *,
    expected_commit: str | None = None, approved_license_sha256: str | None = None,
) -> None:
    if expected_commit:
        require(provenance["source_state"] == "commit" and provenance["source_commit"] == expected_commit,
                "The binary was not built from the expected commit.")

    companion = provenance["source_companion"]
    require(companion["file"] == f"{package.stem}-source.zip", "Unexpected source companion filename.")
    source = package.with_name(companion["file"])
    verify_zip(source)
    require(digest(source) == companion["sha256"], "The source companion belongs to another binary build.")

    license_sha256 = provenance.get("first_party_license_sha256")
    if approved_license_sha256 is not None:
        require(expected_commit is not None and license_sha256 == approved_license_sha256,
                "The archive pair must include the approved first-party LICENSE from the expected commit.")

    with ZipFile(source) as archive:
        root = source.stem + "/"
        manifest = json.loads(archive.read(root + "PREVIEW_BUILD.json"))
        require(manifest["binary_archive"] == package.name, "The source companion names a different binary ZIP.")
        for field in ("source_commit", "source_state", "base_commit", "source_files_sha256", "first_party_license_sha256"):
            require(manifest.get(field) == provenance.get(field), f"Binary and source provenance differ at {field}.")
        recorded = json.loads(archive.read(root + "SOURCE_FILES.json"))
        inventory_sha256 = hashlib.sha256(json.dumps(recorded, sort_keys=True).encode("utf-8")).hexdigest()
        require(inventory_sha256 == provenance["source_files_sha256"], "Source file inventory digest mismatch.")
        require(recorded.get("LICENSE") == license_sha256, "First-party license is inconsistent with the source inventory.")
        for zip_file, name in (
            (binary, package.stem + "/LICENSE"),
            (archive, root + "LICENSE"),
            (archive, root + "project/LICENSE"),
        ):
            entries = zip_file.namelist().count(name)
            if license_sha256 is None:
                require(entries == 0, f"License is present without provenance: {name}.")
            else:
                require(entries == 1, f"Expected exactly one first-party license at {name}.")
                contents = zip_file.read(name)
                require(bool(contents.strip()) and hashlib.sha256(contents).hexdigest() == license_sha256,
                        f"First-party license checksum mismatch at {name}.")
        for name, expected in recorded.items():
            member = archive.read(root + "project/" + name)
            require(hashlib.sha256(member).hexdigest() == expected, f"Source snapshot differs at {name}.")
        verify_source_materials(binary, archive, package.stem + "/", root, recorded)


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
    import pefile

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
    parser.add_argument("--verify-only", action="store_true", help="Verify the archive pair without launching the app.")
    parser.add_argument("--expected-commit")
    parser.add_argument("--approved-license-sha256")
    args = parser.parse_args()
    approved_license = args.approved_license_sha256
    if approved_license is not None and (not args.expected_commit or not re.fullmatch(r"[0-9a-fA-F]{64}", approved_license)):
        parser.error("--approved-license-sha256 requires a SHA-256 digest and --expected-commit.")
    package = args.package.resolve()
    verify_zip(package)
    with TemporaryDirectory(prefix="NyxCore release 音楽 ", ignore_cleanup_errors=True) as directory, ZipFile(package) as archive:
        sandbox = Path(directory)
        root = package.stem + "/"
        provenance = json.loads(archive.read(root + "PREVIEW_BUILD.json"))
        verify_pair(package, archive, provenance, expected_commit=args.expected_commit,
                    approved_license_sha256=approved_license.lower() if approved_license is not None else None)
        if args.verify_only:
            print("Verified the binary and source archive pair.")
            return 0
        names = archive.namelist()
        for name in names:
            require(not any(part.lower() in {"node_modules", ".venv", "tests"} for part in PurePosixPath(name).parts)
                    and not re.search(r"/(?:pythonw?|node|npm|ffmpeg)\.exe$", name, re.I), f"Development payload in {name}.")
        archive.extractall(sandbox)
        result = smoke_executable(sandbox / package.stem, sandbox, args.display)
    receipt = {"source_commit": provenance["source_commit"], "zip_sha256": digest(package),
               "first_party_license_sha256": provenance.get("first_party_license_sha256"), **result}
    report = json.dumps(receipt, indent=2)
    if args.receipt:
        args.receipt.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
