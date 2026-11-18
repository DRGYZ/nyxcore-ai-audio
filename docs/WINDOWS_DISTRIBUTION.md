# Windows release candidate

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

The builder reads a clean Git commit and writes both ZIPs, checksum files, build provenance, and the dependency inventory to `dist/windows-rc/`. Build timestamps and machine-dependent executable details can change the ZIP hashes. A local development build can use `--working-tree`, which labels provenance as uncommitted. The binary manifest records its exact matching source ZIP and SHA-256.

## Runtime and redistribution

NyxCore's first-party source code is MIT-licensed under the included `LICENSE`. Bundled dependencies and third-party assets retain their own terms. The dependency inventory and notices identify those components, including the fonts under SIL Open Font License 1.1. Downloaded cover art remains subject to the rights of its respective owners. [Cover Art Archive](https://coverartarchive.org/)

- The app uses the operating system's WebView2 Evergreen Runtime, Windows UCRT, .NET Framework WinForms, and separately provided Microsoft Visual C++ v14 runtime. The bundle includes the x64 WebView2 SDK loader and two managed assemblies used by pywebview. It omits the Evergreen installer, legacy renderer binaries, and the VC runtime DLLs. [Microsoft WebView2 distribution documentation](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution) · [Exact SDK package](https://www.nuget.org/packages/Microsoft.Web.WebView2/1.0.3856.49)
- The frozen executable bundles and imports Mutagen 1.48.1 in-process. For distribution, treat the covered combination as a GPL work and retain the MIT notice for NyxCore's own code. Mutagen permits GPL version 2 or later. Review the GPLv3 distribution path for the combination, since Apache-2.0 components are incompatible with GPLv2 where they form part of the same covered work. Retain upstream copyright notices and provide the applicable GPL text and complete corresponding source. PyInstaller's bootloader exception leaves dependency obligations in effect. [Mutagen release](https://pypi.org/project/mutagen/1.48.1/) · [GNU guidance on imported GPL modules](https://www.gnu.org/licenses/gpl-faq.html#IfInterpreterIsGPL) · [Apache-2.0 compatibility](https://www.gnu.org/licenses/license-list.html#apache2) · [PyInstaller license](https://pyinstaller.org/en/stable/license.html)
- The source companion contains the candidate's project snapshot and build scripts, the exact Mutagen source, and matched Python and Rust archives. The frontend dependency inventory and pinned CPython build sources are recorded separately. Before publishing, confirm which linked or bundled components fall within the covered combination and supply their exact preferred sources and build inputs where required. Provide equivalent access to the complete corresponding source alongside the binary download at no additional charge under the GPLv3 network distribution option. [GPLv3 sections 1 and 6](https://www.gnu.org/licenses/gpl-3.0.html#section6)
- The pinned Astral CPython 3.12.14 archive's `LICENSE.txt` includes additional Windows binary conditions. It identifies Microsoft code embedded by the linker in the Windows executables, DLLs, and extensions, and requires downstream recipients to accept protective terms for that code. Excluding the standalone VC redistributable DLLs leaves those embedded-code conditions in place. Before release, establish the downstream redistribution grant applicable to the exact shipped binaries and confirm that its pass-through terms can be limited to separately covered Microsoft code without imposing additional restrictions on the GPL-covered combination. [Pinned Python build](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901) · [Microsoft redistribution terms](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files) · [GPLv3 sections 10 and 12](https://www.gnu.org/licenses/gpl-3.0.html#section10)
- The candidate is unsigned. Independent clean Windows 11 acceptance remains a release gate. Ordinary GitHub Actions runs build and test the candidate and publish validation receipts.

## Archive upload

If the project has a committed root `LICENSE`, the builder includes it beside the executable and at the root of the source ZIP, as well as in the source snapshot. The build manifest records the exact SHA-256. Ordinary CI runs produce internal candidates and upload receipts and checksums only.

Archive upload requires a manual workflow run on `main` with `upload_archives` selected. Set `NYXCORE_WINDOWS_RC_APPROVED_LICENSE_SHA256` to the digest of the approved committed license bytes. Set `NYXCORE_WINDOWS_RC_UPLOAD` to `true` only after separately clearing redistribution, corresponding-source delivery, and clean-machine acceptance. The upload step requires verification of both ZIP checksums, the matching source snapshot and commit, and the approved license bytes in both ZIPs.
