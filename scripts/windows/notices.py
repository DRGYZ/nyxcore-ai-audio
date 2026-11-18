"""Inventory bundled packages and collect notices and matched upstream source."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

RUNTIME_ROOTS = ("fastapi", "aiohttp", "pillow", "mutagen", "pywebview", "pyyaml", "uvicorn")
FROZEN_BUILD_INPUTS = ("pyinstaller", "pyinstaller-hooks-contrib")
REQUIRED_SOURCE_DISTRIBUTIONS = ("mutagen", *FROZEN_BUILD_INPUTS)
GPL_NOTICES = {
    "licenses/GPL-3.0.txt": "scripts/windows/GPL-3.0.txt",
    "licenses/GPL-NOTICE.md": "scripts/windows/GPL-NOTICE.md",
}
NOTICE = re.compile(r"^(licen[cs]e|copying|copyright|notice)(?:$|[._-])", re.I)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def fetch(url: str, path: Path, expected: str) -> Path:
    path.parent.mkdir(exist_ok=True, parents=True)
    with urlopen(url, timeout=45) as response, path.open("wb") as destination:
        for chunk in iter(lambda: response.read(128 * 1024), b""):
            destination.write(chunk)
    if digest(path) != expected:
        raise ValueError(f"Upstream checksum mismatch for {url}")
    return path


def runtime_packages() -> dict[str, metadata.Distribution]:
    pending = set(RUNTIME_ROOTS)
    packages = {}
    while pending:
        name = pending.pop()
        distribution = metadata.distribution(name)
        packages[name] = distribution
        for expression in distribution.requires or ():
            requirement = Requirement(expression)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                dependency = canonicalize_name(requirement.name)
                if dependency not in packages:
                    pending.add(dependency)
    return packages


def source_distribution(name: str, version: str, stage: Path) -> tuple[dict, Path | None]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urlopen(url, timeout=45) as response:
        release = json.load(response)
    record = {"version": version, "release": url.removesuffix("/json").replace("/pypi/", "/project/")}
    artifact = next((item for item in release["urls"] if item["packagetype"] == "sdist"), None)
    if not artifact:
        record["source_status"] = "PyPI has no sdist; obtain and verify upstream source before publication."
        return record, None
    record.update(url=artifact["url"], sha256=artifact["digests"]["sha256"], file=artifact["filename"])
    return record, fetch(record["url"], stage / record["file"], record["sha256"])


def archive_notices(archive: Path, target: Path) -> list[Path]:
    copies = []
    if archive.suffix == ".zip":
        with ZipFile(archive) as source:
            contents = [(PurePosixPath(name).name, source.read(name)) for name in source.namelist()
                        if len(PurePosixPath(name).parts) == 2 and NOTICE.match(PurePosixPath(name).name)]
    else:
        with tarfile.open(archive) as source:
            contents = [(PurePosixPath(member.name).name, source.extractfile(member).read())
                        for member in source.getmembers() if member.isfile()
                        and len(PurePosixPath(member.name).parts) == 2 and NOTICE.match(PurePosixPath(member.name).name)]
    for name, content in contents:
        target.mkdir(parents=True, exist_ok=True)
        path = target / name
        path.write_bytes(content)
        copies.append(path)
    return copies


def rust_inventory(source: Path, out: Path) -> tuple[list[dict], dict[str, Path], dict[str, Path]]:
    with tarfile.open(source) as archive:
        lock = next(member for member in archive.getmembers() if member.name.endswith("/Cargo.lock"))
        packages = tomllib.loads(archive.extractfile(lock).read().decode("utf-8"))["package"]

    def collect(package: dict) -> tuple[dict, Path, list[Path]]:
        if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
            raise ValueError(f"Unexpected Rust source registry for {package['name']}.")
        name, version = package["name"], package["version"]
        filename = f"{name}-{version}.crate"
        url = f"https://static.crates.io/crates/{name}/{filename}"
        checksum = package["checksum"]
        archive = fetch(url, out / "rust" / filename, checksum)
        found = archive_notices(archive, out / "rust-notices" / f"{name}-{version}")
        return {"name": name, "version": version, "url": url, "sha256": checksum,
                "notices": [path.name for path in found]}, archive, found

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(collect, (package for package in packages if "source" in package)))
    return (
        [record for record, _, _ in results],
        {f"upstream/rust/{path.name}": path for _, path, _ in results},
        {f"licenses/pydantic-core-crates/{path.parent.name}/{path.name}": path
         for _, _, notices in results for path in notices},
    )


def native_inventory(bundle: Path, packages: dict[str, metadata.Distribution], runtime: dict) -> dict:
    base = Path(sys.base_prefix)
    internals = bundle / "_internal"
    checksums = {
        digest(file): runtime["python_build"]
        for directory in (base, base / "DLLs")
        for file in directory.iterdir() if file.is_file() and file.suffix.lower() in {".dll", ".pyd"}
    }
    for name, package in packages.items():
        for entry in package.files or ():
            if entry.suffix.lower() in {".dll", ".pyd"}:
                path = Path(package.locate_file(entry))
                checksums[digest(path)] = f"{name}=={package.version}"
    for filename, expected in runtime["webview2_sdk"].items():
        path = internals / "webview" / filename
        if digest(path) != expected:
            raise ValueError(f"The bundled WebView2 SDK file is missing or changed: {filename}")
    inventory = {"NyxCore.exe": {"sha256": digest(bundle / "NyxCore.exe"), "origin": "PyInstaller windowed bootloader"}}
    for file in sorted(internals.rglob("*")):
        if file.suffix.lower() not in {".dll", ".pyd", ".exe"}:
            continue
        checksum = digest(file)
        origin = checksums.get(checksum)
        if origin is None:
            raise ValueError(f"Unclassified native binary in the bundle: {file.relative_to(bundle)}")
        inventory[file.relative_to(bundle).as_posix()] = {"sha256": checksum, "origin": origin}
    return inventory


def collect_notices(stage: Path, bundle: Path, runtime: dict, frontend_lock: Path) -> tuple[dict, dict, dict]:
    runtime_distributions = runtime_packages()
    packages = {
        **runtime_distributions,
        **{name: metadata.distribution(name) for name in FROZEN_BUILD_INPUTS},
    }
    upstream = stage / "_upstream"
    licenses = {
        **{name: stage / path for name, path in GPL_NOTICES.items()},
        "licenses/CPython/LICENSE.txt": Path(sys.base_prefix) / "LICENSE.txt",
    }
    if not licenses["licenses/CPython/LICENSE.txt"].is_file():
        raise ValueError("The pinned CPython license is missing.")
    sdk = runtime["webview2_sdk_version"]
    package_name = f"microsoft.web.webview2.{sdk}.nupkg"
    package = fetch(
        f"https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/{sdk}/{package_name}",
        upstream / package_name, runtime["webview2_sdk_nupkg_sha256"],
    )
    with ZipFile(package) as sdk_files:
        for filename, expected in runtime["webview2_sdk"].items():
            member = (filename.removeprefix("lib/") if filename.startswith("lib/runtimes/")
                      else filename.replace("lib/", "lib/net462/", 1))
            if hashlib.sha256(sdk_files.read(member)).hexdigest() != expected:
                raise ValueError(f"WebView2 interop file differs from the pinned Microsoft SDK: {filename}")
        for filename in ("LICENSE.txt", "NOTICE.txt"):
            path = upstream / ("WebView2-" + filename)
            path.write_bytes(sdk_files.read(filename))
            licenses[f"licenses/WebView2-SDK/{filename}"] = path
    sources, python = {}, {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: source_distribution(item[0], item[1].version, upstream / "python"),
                                sorted(packages.items())))
    for (name, package), (record, source) in zip(sorted(packages.items()), results, strict=True):
        distribution_notices = [
            Path(package.locate_file(entry)) for entry in package.files or ()
            if entry.parts[0].endswith(".dist-info") and ("licenses" in entry.parts or NOTICE.match(entry.name))
        ]
        if source:
            sources[f"upstream/python/{source.name}"] = source
            if not distribution_notices:
                distribution_notices = archive_notices(source, upstream / "python-notices" / name)
        python[name] = {**record, "license": package.metadata.get("License-Expression") or
                        package.metadata.get("License"), "notices": [path.name for path in distribution_notices]}
        for path in distribution_notices:
            licenses[f"licenses/python/{name}/{path.name}"] = path
        if name in REQUIRED_SOURCE_DISTRIBUTIONS and not (source and distribution_notices):
            raise ValueError(f"The exact source and copying terms for {name} are required.")

    core_source = sources[f"upstream/python/{python['pydantic-core']['file']}"]
    crates, crate_sources, crate_notices = rust_inventory(core_source, upstream)
    licenses.update(crate_notices)
    sources.update(crate_sources)
    for entry in packages["pydantic-core"].files or ():
        if "sboms" in entry.parts:
            sources[f"upstream/pydantic-core/{entry.name}"] = Path(packages["pydantic-core"].locate_file(entry))
    supplemental = json.loads((stage / "scripts/windows/notices.json").read_bytes())
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = pool.map(lambda item: fetch(item[1]["url"], upstream / "notices" / item[0], item[1]["sha256"]),
                         supplemental.items())
        licenses.update((f"licenses/{name}", path) for name, path in zip(supplemental, paths, strict=True))
        runtime_sources = runtime["source_archives"]
        paths = pool.map(lambda item: fetch(item[1]["url"], upstream / "native" / item[0], item[1]["sha256"]),
                         runtime_sources.items())
        sources.update((f"upstream/native/{name}", path) for name, path in zip(runtime_sources, paths, strict=True))
    packages_lock = json.loads(frontend_lock.read_bytes())["packages"]
    frontend = {
        name.removeprefix("node_modules/"): {key: package.get(key) for key in ("version", "resolved", "integrity", "license")}
        for name, package in packages_lock.items() if name and (not package.get("dev") or name == "node_modules/vite")
    }
    return licenses, sources, {
        "runtime": runtime,
        "python": python,
        "pydantic_core_locked_rust_sources": crates,
        "supplemental_notices": supplemental,
        "frontend": frontend,
        "native_files": native_inventory(bundle, runtime_distributions, runtime),
    }
