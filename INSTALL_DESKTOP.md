# NyxCore Desktop prototype

The prototype runs on Windows 11 with Python 3.12 and the [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/). The source setup uses Node 22 for the frontend build and FFmpeg to generate the synthetic demo. The desktop window launches the local engine itself.

## Set up the source checkout

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\python.exe demo/create_demo_library.py
Push-Location web
npm ci --engine-strict
npm run build
Pop-Location
```

Double-click `.venv\Scripts\nyxcore-desktop.exe` in Explorer, or create a shortcut to it. Pip supplies this console-free entry point for the source checkout. It opens the demo and stores its workspace in `%LOCALAPPDATA%\NyxCore\Desktop\demo-workspace`.

Use `--display 2` to open on a second display, with the primary display numbered first. To inspect a different library, specify an existing folder and a separate workspace.

```powershell
.\.venv\Scripts\nyxcore-desktop.exe --music "D:\Music Copy" --out "D:\NyxCore Review Workspace"
```

The desktop window is review-only. It can analyze music, preview plans, and keep review state in the workspace. Applying plans, undoing audio changes, and restoring audio files are disabled. The backend binds to a Windows-assigned loopback port and closes with the window. Explicit MusicBrainz reference links open in your system browser.

If startup fails, check the error in the window and the log at `%LOCALAPPDATA%\NyxCore\Desktop\desktop.log`. Rebuild the frontend after changing its source. For the browser-based workflow, use [INSTALL_PREVIEW.md](INSTALL_PREVIEW.md).
