"""Build a source-based review-only preview from the current Git commit."""

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
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("nyxcore", "pyproject.toml", "README.md", "LICENSE", "INSTALL_PREVIEW.md", "web", "demo/create_demo_library.py")


def tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(f"{name} is required to build the preview and must be on PATH.")
    return executable


def output(*command: str, cwd: Path = ROOT) -> bytes:
    return subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE).stdout


def frontend_files(dist: Path) -> list[str]:
    index = dist / "index.html"
    if not index.is_file():
        raise ValueError(f"Production frontend is missing {index}.")
    manifest = json.loads((dist / ".vite/manifest.json").read_bytes())
    entry = manifest.get("index.html", {})
    if not entry.get("isEntry") or not entry.get("file", "").endswith(".js") or not entry.get("css"):
        raise ValueError("Vite did not build an index entry with JavaScript and CSS.")
    names = {asset["file"] for asset in manifest.values()}
    for asset in manifest.values():
        names.update(asset.get("css", ()))
        names.update(asset.get("assets", ()))
    for name in sorted(names):
        path = dist / name
        if not path.resolve().is_relative_to(dist / "assets") or not path.is_file():
            raise ValueError(f"Production frontend asset is missing or outside assets/: {name}")
    return ["index.html", *sorted(names)]


def demo_files(demo: Path) -> dict[str, Path]:
    manifest = demo / "nyxcore_demo_manifest.json"
    data = json.loads(manifest.read_bytes())
    duplicate = data["exact_duplicate_copy"]
    tracks = {track["relative_path"] for track in data["tracks"]} | {duplicate["copy"]}
    if len(tracks) != 7 or duplicate["source"] not in tracks:
        raise ValueError("The synthetic demo must contain the seven expected tracks.")
    for name in sorted(tracks):
        path = demo / name
        if not path.resolve().is_relative_to(demo) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Synthetic audio is missing or invalid: {name}")
    if (demo / duplicate["source"]).read_bytes() != (demo / duplicate["copy"]).read_bytes():
        raise ValueError("The synthetic exact-duplicate tracks differ.")
    return {f"demo/generated/sample-library/{name}": demo / name for name in tracks | {manifest.name}}


def frontend_licenses(web: Path, npm: str) -> dict[str, Path]:
    packages = output(npm, "ls", "--omit=dev", "--all", "--parseable", cwd=web).decode("utf-8").splitlines()
    paths = {Path(path) for path in packages} - {web}
    paths.add(web / "node_modules/vite")  # Vite's module preload helper is emitted in the bundle.
    licenses = {}
    for package in sorted(paths):
        metadata = json.loads((package / "package.json").read_bytes())
        for path in package.iterdir():
            if path.is_file() and path.stem.upper() in {"LICENSE", "LICENCE", "COPYING", "NOTICE"}:
                licenses[f"licenses/{metadata['name']}@{metadata['version']}/{path.name}"] = path
    return licenses


def write_archive(archive: Path, files: dict[str, Path], provenance: dict) -> None:
    members = {**files, "PREVIEW_BUILD.json": None}
    with ZipFile(archive, "w") as bundle:
        for name, source in sorted(members.items()):
            entry = ZipInfo(f"{archive.stem}/{name}")
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            content = (
                (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
                if source is None else source.read_bytes()
            )
            bundle.writestr(entry, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(archive) as bundle:
        damaged = bundle.testzip()
        if damaged:
            raise ValueError(f"ZIP integrity check failed for {damaged}.")


def build_preview(destination: Path) -> tuple[Path, str]:
    git = tool("git")
    node = tool("node")
    npm = tool("npm.cmd" if os.name == "nt" else "npm")
    ffmpeg = tool("ffmpeg")
    commit = output(git, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    archive_name = f"nyxcore-review-preview-{commit}.zip"
    with TemporaryDirectory(prefix="nyxcore-preview-") as temporary:
        stage = Path(temporary).resolve()
        snapshot = output(
            git, "-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "archive", "--format=zip", commit, *SOURCE_PATHS,
        )
        with ZipFile(io.BytesIO(snapshot)) as source:
            source.extractall(stage)

        web = stage / "web"
        environment = os.environ.copy()
        environment.pop("NODE_ENV", None)
        subprocess.run((npm, "ci", "--engine-strict", "--include=dev"), cwd=web, env=environment, check=True)
        subprocess.run((npm, "run", "build", "--", "--manifest"), cwd=web, env=environment, check=True)
        assets = frontend_files(web / "dist")
        demo = stage / "demo/generated/sample-library"
        subprocess.run((sys.executable, "-B", "demo/create_demo_library.py", str(demo)), cwd=stage, check=True)

        files = {
            "README.md": stage / "README.md",
            "LICENSE": stage / "LICENSE",
            "INSTALL_PREVIEW.md": stage / "INSTALL_PREVIEW.md",
            "pyproject.toml": stage / "pyproject.toml",
            "nyxcore/resources/default.yaml": stage / "nyxcore/resources/default.yaml",
        }
        files.update({path.relative_to(stage).as_posix(): path for path in (stage / "nyxcore").rglob("*.py")})
        files.update({f"web/dist/{name}": web / "dist" / name for name in assets})
        files.update(demo_files(demo))
        files.update(frontend_licenses(web, npm))
        provenance = {
            "format_version": 1,
            "source_commit": commit,
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "node": output(node, "--version").decode("utf-8").strip(),
            "npm": output(npm, "--version").decode("utf-8").strip(),
            "ffmpeg": output(ffmpeg, "-version").decode("utf-8").splitlines()[0],
            "npm_lock_sha256": hashlib.sha256((web / "package-lock.json").read_bytes()).hexdigest(),
            "frontend_files": assets,
        }
        temporary_archive = stage / archive_name
        write_archive(temporary_archive, files, provenance)
        digest = hashlib.sha256(temporary_archive.read_bytes()).hexdigest()
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / archive_name
        shutil.copyfile(temporary_archive, archive)
        archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii", newline="\n")
    return archive, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/preview", help="Directory for the ZIP and checksum")
    args = parser.parse_args()
    try:
        archive, digest = build_preview(args.out.resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Preview build failed. {exc}\n")
    print(f"Preview archive: {archive}\nSHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
