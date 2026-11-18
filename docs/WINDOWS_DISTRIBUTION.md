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

- The app uses the operating system's WebView2 Evergreen Runtime, Windows UCRT, .NET Framework WinForms, and separately provided Microsoft Visual C++ v14 runtime. The bundle includes the x64 WebView2 SDK loader and two managed assemblies used by pywebview. It omits the Evergreen installer, legacy renderer binaries, and the VC runtime DLLs. [Microsoft WebView2 distribution documentation](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution) · [Exact SDK package](https://www.nuget.org/packages/Microsoft.Web.WebView2/1.0.3856.49)
- The inventory identifies Python runtime packages, the npm lockfile inputs, the pinned CPython build, and the native binary hashes. The ZIPs include upstream notices and exact Mutagen source in the source companion. Mutagen is GPL-2.0-or-later. The PyInstaller bootloader exception leaves dependency obligations in effect. [Mutagen release](https://pypi.org/project/mutagen/1.48.1/) · [PyInstaller license](https://pyinstaller.org/en/stable/license.html)
- First-party license approval is pending. NYC-12's uncommitted license did not change the repository's licensing status. All three bundled WebView2 interop files match the pinned Microsoft NuGet package by SHA-256. The bundle includes its license and notice. The source companion supports review of corresponding-source requirements. Public binary distribution requires an approved first-party license, finalized source delivery, and resolution of the pinned CPython build's terms for embedded Microsoft code. [GPLv3 corresponding-source terms](https://www.gnu.org/licenses/gpl-3.0.html#section6)
- The candidate is unsigned. Independent clean Windows 11 acceptance remains a release gate. GitHub Actions builds and tests the candidate and publishes a validation receipt while binary and source upload approval is pending.

## Archive upload

If the project has a committed root `LICENSE`, the builder includes it beside the executable and at the root of the source ZIP, as well as in the source snapshot. The build manifest records the exact SHA-256. Ordinary CI runs produce internal candidates and upload receipts and checksums only.

Archive upload requires a manual workflow run on `main` with `upload_archives` selected. Set `NYXCORE_WINDOWS_RC_APPROVED_LICENSE_SHA256` to the digest of the approved committed license bytes. Set `NYXCORE_WINDOWS_RC_UPLOAD` to `true` only after separately clearing redistribution, corresponding-source delivery, and clean-machine acceptance. The upload step requires verification of both ZIP checksums, the matching source snapshot and commit, and the approved license bytes in both ZIPs.
