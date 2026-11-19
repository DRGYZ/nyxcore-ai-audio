# NyxCore Desktop

The approved desktop identity is **Pressed & Cataloged**. The Windows application and browser workflow share one React frontend. See the [design rules](docs/DESKTOP_DESIGN.md) and [desktop experience acceptance report](docs/DESKTOP_EXPERIENCE_ACCEPTANCE.md).

NyxCore Desktop runs on Windows 11 with Python 3.12 and the [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/). The source setup uses Node 22 for the frontend build and FFmpeg to generate the synthetic demo. The desktop window launches the local engine itself.

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

Double-click `.venv\Scripts\nyxcore-desktop.exe` in Explorer, or create a shortcut to it. Pip supplies this console-free entry point for the source checkout.

Select **Choose Music Folder**, pick an existing folder in the Windows dialog, and confirm the displayed path with **Open Library**. You can navigate while NyxCore analyzes the folder. **Explore Demo Library** opens synthetic sample content. Use **Change Library** to switch collections.

NyxCore stores its last confirmed selection in the versioned `settings.json` under `%LOCALAPPDATA%\NyxCore\Desktop`. Each folder uses a workspace in `libraries` identified by a hash of its resolved, Windows-normalized path. The synthetic demo uses `demo-workspace`. Reopening the same path restores its review workspace. Renaming or moving a library gives it a new workspace. Reconnect a missing drive at its previous letter and select **Retry Library** to reuse its existing workspace.

Source launcher flags take precedence for the current launch. Opening a folder through the app records a selection for future launches.

| Flags | Behavior |
| --- | --- |
| None | Reopen the remembered library, or show the welcome screen. |
| `--music "D:\Music Copy"` | Open the selected folder with its managed workspace. |
| `--music "D:\Music Copy" --out "D:\NyxCore Review Workspace"` | Use an existing custom workspace or create one at the specified path. |
| `--demo` | Open the synthetic library with its demo workspace. |
| `--out "D:\Demo Review"` or `--demo --out "D:\Demo Review"` | Open the synthetic library with the supplied workspace. |

Use `--display 2` to open on a second display, with the primary display numbered first. With custom workspaces, reuse the same flags when you want to reopen that review history.

The desktop window is review-only. It can analyze music, preview plans, and keep review state in the workspace. Applying plans, undoing audio changes, and restoring audio files are disabled. In **Library Health**, use **Export JSON** to save the current report through the Windows Save dialog. Choose an existing folder outside the music library and NyxCore workspace. The backend binds to a Windows-assigned loopback port and closes with the window. Explicit MusicBrainz reference links open in your system browser.

If startup fails, check the error in the window and the log at `%LOCALAPPDATA%\NyxCore\Desktop\desktop.log`. Rebuild the frontend after changing its source. For the browser-based workflow, use [INSTALL_PREVIEW.md](INSTALL_PREVIEW.md).
