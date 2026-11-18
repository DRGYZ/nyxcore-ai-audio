# NyxCore for Windows 11

NyxCore is a portable x64 desktop application for reviewing a local music library. The unsigned executable is built for personal use. For the source-based setup, see [NyxCore Desktop](INSTALL_DESKTOP.md).

## Open the portable app

1. [Build the Windows ZIP from source](docs/WINDOWS_DISTRIBUTION.md#build) to produce `NyxCore-Windows-x64.zip` and its `.sha256` file. Open PowerShell in that folder and run `Get-FileHash -Algorithm SHA256 .\NyxCore-Windows-x64.zip`. Compare the full hash with the checksum file.
2. Extract the entire ZIP to a local folder. Keep `_internal` beside `NyxCore.exe`.
3. Double-click **NyxCore.exe**. Choose a music folder, check the shown path, and select **Open Library**. You can also explore the synthetic demo.

The app uses the system WebView2 Evergreen Runtime, .NET Framework 4.8 or later, and the x64 Microsoft Visual C++ v14 runtime. It includes Python and the compiled web interface. Node.js, npm, FFmpeg, and developer tools are used only during the build. If the machine is missing a system component, ask your administrator where applicable or obtain it from Microsoft. [WebView2 Evergreen](https://developer.microsoft.com/microsoft-edge/webview2/#download) · [.NET Framework](https://learn.microsoft.com/dotnet/framework/install/on-windows-and-server) · [Visual C++ runtime](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)

Windows may identify the executable as unsigned or show a SmartScreen reputation warning. Follow your organization's security policy and use an approved build if Windows blocks this candidate.

## Library state and removal

The desktop uses `%LOCALAPPDATA%\NyxCore\Desktop` for settings, workspaces, logs, and other local state. Each library keeps its own review workspace. If a folder is unavailable, reconnect the drive at the same letter and select **Retry Library**, or choose another folder.

NyxCore is review-only. It can analyze audio, preview plans, and save review state. Audio apply, restore, and undo are disabled. Health JSON exports go to the location you choose in the Save dialog outside your music library, NyxCore workspace, and application directory.

To remove the portable app, close NyxCore and delete its extracted folder. If you also intend to remove saved settings and review state, back up anything you need and remove its `%LOCALAPPDATA%\NyxCore\Desktop` folder separately.

See the [distribution and release status](docs/WINDOWS_DISTRIBUTION.md) in the source checkout, or open `WINDOWS_DISTRIBUTION.md` beside the executable.
