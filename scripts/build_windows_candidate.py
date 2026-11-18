"""Build the windowed NyxCore x64 ZIP and its matching source companion."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from build_preview import ROOT, demo_files, frontend_files, frontend_licenses, output, tool, write_archive
from windows.notices import collect_notices, digest, save_json

ZIP_NAME = "NyxCore-Windows-x64.zip"
SOURCE_NAME = "NyxCore-Windows-x64-source.zip"
SOURCE_PATHS = (
    "nyxcore", "web", "demo/create_demo_library.py", "scripts", "tests", ".github/workflows",
    ".gitignore", "pyproject.toml", "README.md", "INSTALL_DESKTOP.md", "INSTALL_PREVIEW.md",
    "INSTALL_WINDOWS.md", "docs/WINDOWS_DISTRIBUTION.md", "docs/WINDOWS_ACCEPTANCE.md",
)


def stage_project(git: str, stage: Path, commit: str, working_tree: bool) -> dict[str, Path]:
    if working_tree:
        paths = output(
            git, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SOURCE_PATHS, "LICENSE", cwd=ROOT,
        )
        for name in sorted(set(paths.decode("utf-8").split("\0")) - {""}):
            source = ROOT / name
            if source.is_file():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    else:
        license_entry = output(git, "ls-tree", "--name-only", commit, "--", "LICENSE", cwd=ROOT)
        paths = (*SOURCE_PATHS, "LICENSE") if license_entry.strip() == b"LICENSE" else SOURCE_PATHS
        archive = output(git, "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=zip", commit, *paths, cwd=ROOT)
        with ZipFile(io.BytesIO(archive)) as snapshot:
            snapshot.extractall(stage)
    return {path.relative_to(stage).as_posix(): path for path in stage.rglob("*") if path.is_file()}


def prepare_assets(stage: Path, npm: str) -> tuple[list[str], dict[str, Path], str]:
    import imageio_ffmpeg

    assets_environment = os.environ.copy()
    assets_environment.pop("NODE_ENV", None)
    tools = stage / "_tools"
    tools.mkdir()
    ffmpeg = tools / "ffmpeg.exe"
    shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    assets_environment["PATH"] = str(tools) + os.pathsep + assets_environment["PATH"]
    web = stage / "web"
    subprocess.run((npm, "ci", "--engine-strict", "--include=dev"), cwd=web, env=assets_environment, check=True)
    subprocess.run((npm, "run", "build", "--", "--manifest"), cwd=web, env=assets_environment, check=True)
    demo = stage / "demo/generated/sample-library"
    subprocess.run((sys.executable, "-B", "demo/create_demo_library.py", str(demo)),
                   cwd=stage, env=assets_environment, check=True)
    return frontend_files(web / "dist"), demo_files(demo), output(str(ffmpeg), "-version").decode("utf-8").splitlines()[0]


def build(destination: Path, working_tree: bool) -> tuple[Path, Path]:
    if sys.platform != "win32" or platform.machine().upper() != "AMD64":
        raise ValueError("Build on native Windows x64.")
    git, node, npm = (tool(name) for name in ("git", "node", "npm.cmd"))
    commit = output(git, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if not working_tree and output(git, "status", "--porcelain"):
        raise ValueError("Commit the candidate before building a commit-addressed ZIP. Use --working-tree for local experiments.")
    with TemporaryDirectory(prefix="nyxcore-rc-") as directory:
        stage = Path(directory).resolve()
        sources = stage_project(git, stage, commit, working_tree)
        source_hashes = {name: digest(path) for name, path in sources.items()}
        runtime = json.loads(sources["scripts/windows/runtime.json"].read_bytes())
        if (platform.python_version() != runtime["python_version"]
                or digest(Path(sys.base_prefix) / "python312.dll") != runtime["python_dll_sha256"]):
            raise ValueError("Use the pinned x64 Windows Python build in scripts/windows/runtime.json.")
        frontend, demo, ffmpeg = prepare_assets(stage, npm)
        web = stage / "web"
        provenance = {
            "kind": "windows-portable-candidate",
            "source_state": "working-tree" if working_tree else "commit",
            "source_commit": None if working_tree else commit,
            "base_commit": commit if working_tree else None,
            "source_files_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
            "platform": "Windows x64",
            "python": platform.python_version(),
            "pyinstaller": metadata.version("pyinstaller"),
            "pyinstaller_hooks": metadata.version("pyinstaller-hooks-contrib"),
            "node": output(node, "--version").decode().strip(),
            "npm": output(npm, "--version").decode().strip(),
            "ffmpeg": ffmpeg,
            "npm_lock_sha256": digest(web / "package-lock.json"),
            "requirements_sha256": digest(sources["scripts/windows/requirements.txt"]),
            "frontend_files": frontend,
            "distribution_status": "internal release candidate",
            "first_party_license_sha256": source_hashes.get("LICENSE"),
        }
        resources = stage / "_desktop_payload"
        files = {
            **demo,
            **{f"web/dist/{name}": web / "dist" / name for name in frontend},
            **{f"nyxcore/resources/{name}": stage / "nyxcore/resources" / name for name in ("default.yaml", "desktop.ico")},
        }
        for name, source in files.items():
            target = resources / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        save_json(resources / "PREVIEW_BUILD.json", provenance)
        subprocess.run(
            (sys.executable, "-m", "PyInstaller", "--noconfirm", "scripts/windows/NyxCore.spec"),
            cwd=stage, check=True,
        )
        bundle = stage / "dist/NyxCore"
        licenses, dependency_sources, inventory = collect_notices(stage, bundle, runtime, web / "package-lock.json")
        licenses.update(frontend_licenses(web, npm))
        provenance["python_dependencies"] = {name: data["version"] for name, data in inventory["python"].items()}
        inventory_path = save_json(stage / "WINDOWS_DEPENDENCIES.json", inventory)
        source_zip = stage / SOURCE_NAME
        common = {
            **licenses,
            "WINDOWS_DEPENDENCIES.json": inventory_path,
            "WINDOWS_DISTRIBUTION.md": sources["docs/WINDOWS_DISTRIBUTION.md"],
            "WINDOWS_ACCEPTANCE.md": sources["docs/WINDOWS_ACCEPTANCE.md"],
        }
        if "LICENSE" in sources:
            common["LICENSE"] = sources["LICENSE"]
        write_archive(source_zip, {
            **{f"project/{name}": path for name, path in sources.items()},
            **dependency_sources,
            **common,
            "SOURCE_FILES.json": save_json(stage / "SOURCE_FILES.json", source_hashes),
        }, {**provenance, "kind": "windows-source-companion", "binary_archive": ZIP_NAME})

        provenance["source_companion"] = {"file": SOURCE_NAME, "sha256": digest(source_zip)}
        save_json(bundle / "_internal/PREVIEW_BUILD.json", provenance)
        payload = {path.relative_to(bundle).as_posix(): path for path in bundle.rglob("*") if path.is_file()}
        if "NyxCore.exe" not in payload or any(f"_internal/{name}" not in payload for name in files):
            raise ValueError("The frozen runtime or one of the required resources is missing.")
        package = stage / ZIP_NAME
        write_archive(package, {**payload, **common, "INSTALL_WINDOWS.md": sources["INSTALL_WINDOWS.md"]}, provenance)
        destination.mkdir(parents=True, exist_ok=True)
        for archive in (package, source_zip):
            shutil.copyfile(archive, destination / archive.name)
            (destination / f"{archive.name}.sha256").write_text(
                f"{digest(archive)}  {archive.name}\n", encoding="ascii", newline="\n",
            )
        shutil.copyfile(inventory_path, destination / inventory_path.name)
        save_json(destination / "WINDOWS_BUILD.json", provenance)
    return destination / ZIP_NAME, destination / SOURCE_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/windows-rc")
    parser.add_argument("--working-tree", action="store_true", help="Label uncommitted inputs as an internal test build.")
    args = parser.parse_args()
    for package in build(args.out.resolve(), args.working_tree):
        print(f"{digest(package)}  {package} ({package.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
