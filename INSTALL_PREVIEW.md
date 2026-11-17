# Install the NyxCore Review-Only Preview

Use Python 3.12 on Windows 11. Installation downloads Python dependencies. The ZIP includes the compiled web interface and seven synthetic tracks, so running it requires neither Node.js nor FFmpeg.

1. Sign into GitHub and open the latest successful main-branch run in [NyxCore CI](https://github.com/DRGYZ/nyxcore-ai-audio/actions/workflows/ci.yml?query=branch%3Amain). Download the `nyxcore-review-preview-<commit>` artifact from **Artifacts** and extract the GitHub download. It contains the preview ZIP and its `.sha256` checksum. Preview artifacts expire after 30 days.
2. In PowerShell, open the folder containing the preview ZIP and checksum. Replace `<commit>` with the commit in the filename.

```powershell
$zip = Get-Item -LiteralPath '.\nyxcore-review-preview-<commit>.zip'
$expected = ((Get-Content -Raw -LiteralPath "$($zip.FullName).sha256").Trim() -split '\s+')[0]
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip.FullName).Hash -ne $expected) {
    throw 'Preview checksum mismatch. Download the artifact again.'
}
Expand-Archive -LiteralPath $zip.FullName -DestinationPath '.\NyxCore Preview'
Set-Location -LiteralPath (Join-Path '.\NyxCore Preview' $zip.BaseName)
```

3. Create a virtual environment and install the included project in editable mode. Keep the extracted folder in place because the launcher reads the bundled assets from it.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[web]'
.\.venv\Scripts\python.exe -m nyxcore.launch --demo --out data/demo-reports
```

NyxCore prints its local URL and opens your browser when ready. Check **Active library** in Overview. In **Review Inbox**, choose the exact duplicate finding for **Blue Hour**, select **Generate Plan**, and inspect which copy the plan would keep.

The first analysis can take time on a large library. Select **Refresh Library** after adding, removing, or editing files. Browsing uses the last successful analysis. A failed refresh leaves those results visible and lets you retry.

To switch libraries or stop the service, press **Ctrl+C** in the launcher terminal. Restart with an existing library and a separate, non-overlapping workspace. Use a different workspace for each library.

```powershell
.\.venv\Scripts\python.exe -m nyxcore.launch --music 'D:\Music Library' --out 'C:\NyxCore\my-library-reports'
```

If port 8000 is occupied, add `--port 8001` or choose another free port. Use `--no-browser` to open the printed URL yourself. If the launcher reports missing files, verify the checksum and extract a fresh copy.

This experimental preview runs locally in review-only mode. The server rejects audio apply, restore, and undo requests. Triage and saved playlists can write local workspace state. Power-loss and filesystem crash durability remain unverified for NyxCore's experimental mutation tools.
