# Windows build and distribution

Desktop 01–05 are complete for personal use. The portable build at commit `80943c2` passed independent Windows 11 VM acceptance after installation of the documented Visual C++ runtime. Physical USB disconnect and reconnect was unverified because the VM had no USB passthrough.

## Build

The frozen app uses PyInstaller in one-folder, windowed mode. The binary ZIP contains the x64 executable, Python and the runtime packages, the compiled web frontend, the desktop resources, and the seven-track synthetic demo. The Python and JavaScript versions are pinned by `scripts/windows/requirements.txt` and `web/package-lock.json`. FFmpeg comes from the pinned build-only `imageio-ffmpeg` wheel.

Use Windows x64, Git and Node 22. Download the Windows Python archive identified in `scripts/windows/runtime.json`, check its SHA-256, extract it, and create a venv with that `python.exe`.

```powershell
& <pinned-python.exe> -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r scripts/windows/requirements.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe scripts/build_windows_candidate.py
& .\.venv\Scripts\python.exe scripts/smoke_windows_candidate.py dist/windows-rc/NyxCore-Windows-x64.zip
```

The builder reads a clean Git commit and writes both ZIPs, checksum files, build provenance, and the dependency inventory to `dist/windows-rc/`. Build timestamps and machine-dependent executable details can change the ZIP hashes. A local development build can use `--working-tree`, which labels provenance as uncommitted. The binary manifest records its exact matching source ZIP and SHA-256. Use [the clean Windows acceptance procedure](WINDOWS_ACCEPTANCE.md) for the independent guest checks.

## Runtime and redistribution

NyxCore's first-party source code is MIT-licensed under the included `LICENSE`. Bundled dependencies and third-party assets retain their own terms. The dependency inventory and notices identify those components, including the fonts under SIL Open Font License 1.1. Downloaded cover art remains subject to the rights of its respective owners. [Cover Art Archive](https://coverartarchive.org/)

Both ZIPs carry the complete GPLv3 text and a notice for the proposed GPLv3 distribution path. The frozen Python process imports Mutagen 1.48.1 under GPL-2.0-or-later and ships OpenSSL 3.5.8 under Apache-2.0. The notice preserves NyxCore's MIT terms and the PyInstaller exception. The companion contains the project, lockfiles and build scripts, Python source distributions including Mutagen and the freezer inputs, locked Rust sources, and the pinned native runtime source inputs and build recipes. Archive verification checks those supplied inputs against their inventory digests. The WebView2 frontend is separately built JavaScript executed in the renderer and uses the loopback HTTP API. Its project source, npm lockfile, and dependency notices ship with the pair. [Mutagen](https://pypi.org/project/mutagen/1.48.1/) · [Apache-2.0 compatibility](https://www.gnu.org/licenses/license-list.html#apache2) · [PyInstaller license](https://pyinstaller.org/en/stable/license.html)

### Redistribution decision record

**Public Windows binary distribution is out of scope.** The unsigned executable is for personal use. There is no planned installer, commercial launch, or end-user support commitment. The owner must explicitly revisit this decision before public binary distribution is considered. The redistribution questions below remain unresolved.

If distribution is revisited, confirm the scope of the covered combination and provide access to matching preferred source alongside the binary at no extra charge. A source mirror may be used with equivalent access and clear directions next to the binary. [GPLv3 §§1 and 6](https://www.gnu.org/licenses/gpl-3.0.html#section6)

| Component | Evidence and governing terms |
| --- | --- |
| WebView2 SDK 1.0.3856.49 | `Microsoft.Web.WebView2.Core.dll`, `Microsoft.Web.WebView2.WinForms.dll`, and x64 `WebView2Loader.dll` match the upstream NuGet binaries. Both ZIPs include the SDK license and notice. The guest supplies the Evergreen Runtime. |
| External Windows runtime | Windows supplies UCRT and .NET Framework WinForms. The Visual C++ v14 DLLs are installed separately if needed. These dependencies are outside the binary ZIP. Their normal delivery with Windows or the supported toolchain and their role providing platform interfaces are relevant to the GPLv3 System Libraries criteria. |
| CRT code embedded in the pinned Astral CPython 3.12.14 build | Matching PDBs show that all 19 shipped CPython `.pyd` modules plus `python312.dll`, `libcrypto-3-x64.dll`, `libssl-3-x64.dll`, and `sqlite3.dll` contain startup objects from Visual Studio Enterprise 2022 x64 `MSVCRT.lib` 14.44.35207. `libffi-8.dll` lacks matching symbols in the pinned runtime archive. CPython's Windows notice and Visual Studio's distributable-code terms govern the embedded Microsoft code. The notice requires protection of Microsoft notices, platform and use restrictions, and protective terms for downstream distributors and users. |

The CRT startup library is a normal MSVC toolchain input and enables the Windows C runtime. Those facts support considering the System Libraries exception. Qualified review must establish which grant covers the exact linked objects through to NyxCore, whether embedded startup code satisfies GPLv3 §1, and whether downstream assent can be scoped to Microsoft's code consistently with GPLv3 §§10 and 12. WebView2 SDK redistribution follows its own license. [Exact WebView2 SDK](https://www.nuget.org/packages/Microsoft.Web.WebView2/1.0.3856.49) · [Pinned Astral build](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901) · [Microsoft CRT startup library](https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features) · [CPython Windows notice](https://github.com/python/cpython/blob/v3.12.14/PC/crtlicense.txt) · [Visual Studio Enterprise terms](https://visualstudio.microsoft.com/license-terms/vs2022-ga-proenterprise/) · [GPLv3 §§10 and 12](https://www.gnu.org/licenses/gpl-3.0.html#section10)

## Archive upload

If the project has a committed root `LICENSE`, the builder includes it beside the executable and at the root of the source ZIP, as well as in the source snapshot. The build manifest records the exact SHA-256. Ordinary CI runs produce internal candidates and upload receipts and checksums only.

Keep `upload_archives` off and the upload approval variables unset. The workflow's archive upload gate requires a manual run on `main`, explicit redistribution clearance in `NYXCORE_WINDOWS_RC_UPLOAD`, and the approved committed license digest in `NYXCORE_WINDOWS_RC_APPROVED_LICENSE_SHA256`. The upload step verifies both ZIP checksums, the matching source snapshot and commit, and the approved license bytes in both ZIPs.
