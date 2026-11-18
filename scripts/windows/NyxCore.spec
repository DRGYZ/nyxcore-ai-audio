from pathlib import Path
import json
from importlib import metadata

from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parents[1]
resources = root / "_desktop_payload"
runtime = json.loads((root / "scripts/windows/runtime.json").read_bytes())
webview = Path(metadata.distribution("pywebview").locate_file("webview"))

sdk_files = [
    (str(webview / name), (Path("webview") / name).parent.as_posix())
    for name in runtime["webview2_sdk"]
]
unused_arches = ("win-arm64", "win-x86")
layout_markers = [
    (str(root / "scripts/windows/x64-only.txt"), f"webview/lib/runtimes/{arch}/native")
    for arch in unused_arches
]

analysis = Analysis(
    [str(root / "scripts/windows/entrypoint.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(resources), "."),
        *sdk_files,
        *layout_markers,
        *collect_data_files("webview", includes=["js/**"]),
        *collect_data_files("pythonnet", includes=["runtime/**"]),
        *collect_data_files("clr_loader", includes=["ffi/dlls/amd64/**"]),
    ],
    hiddenimports=[
        "anyio._backends._asyncio",
        "clr",
        "uvicorn.lifespan.on",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "webview.platforms.edgechromium",
        "webview.platforms.winforms",
    ],
    excludes=[
        "imageio_ffmpeg", "nyxcore.cli", "openai", "sqlmodel",
        "torch", "torchaudio", "essentia", "pytest", "setuptools", "pkg_resources",
        "webview.platforms.cef", "webview.platforms.mshtml", "webview.platforms.qt",
        "webview.platforms.gtk", "webview.platforms.cocoa", "webview.platforms.android",
    ],
)


def shipped(entry):
    name = entry[0].replace("\\", "/").lower()
    basename = name.rsplit("/", 1)[-1]
    if basename.endswith(".dll") and (
        basename.startswith(("api-ms-win-", "vcruntime", "msvcp140", "vcomp140"))
        or basename in {"ucrtbase.dll", "concrt140.dll"}
    ):
        return False
    if name.startswith("webview/lib/"):
        return name in {f"webview/{item.lower()}" for item in runtime["webview2_sdk"]} or (
            basename == "x64-only.txt" and any(f"/{arch}/native/" in name for arch in unused_arches)
        )
    return not name.startswith("clr_loader/ffi/dlls/x86/")


program = EXE(
    PYZ(analysis.pure),
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="NyxCore",
    console=False,
    upx=False,
    icon=str(root / "nyxcore/resources/desktop.ico"),
)
distribution = COLLECT(
    program,
    [item for item in analysis.binaries if shipped(item)],
    [item for item in analysis.datas if shipped(item)],
    upx=False,
    name="NyxCore",
)
