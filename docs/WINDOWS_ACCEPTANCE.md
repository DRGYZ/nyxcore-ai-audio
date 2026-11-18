# Clean Windows 11 acceptance

Use a fresh x64 Windows 11 Sandbox or disposable VM with a normal desktop session. Record the Windows edition and build, WebView2 and VC++ runtime availability, display scale, candidate commit, and both ZIP SHA-256 values. Confirm the guest has no development checkout or Python or Node installation. If it has missing prerequisites, record the initial result and install only the prerequisites listed in `INSTALL_WINDOWS.md` before retrying. Use a disposable guest account and a small, authorized copy of real audio. Include two libraries with distinct tracks and one with a review finding.

Copy both ZIPs and checksum files into the guest. Verify their SHA-256 values with `Get-FileHash -Algorithm SHA256` and check that the binary's `PREVIEW_BUILD.json` names and hashes the source ZIP. Extract the binary ZIP to a guest-local folder. Launch `NyxCore.exe` from Explorer without flags.

| Check | Pass evidence |
| --- | --- |
| First run and real analysis | Welcome appears. **Choose Music Folder** opens the native picker. Select library A, open it, verify its track count and a known track, then navigate Overview, Review, and Health. |
| Persistence | Mark a review finding **Seen** and record the item. Close and reopen the app. Confirm library A and the saved review state return. |
| Library switching | Open library B, then the synthetic demo, then library A. Confirm each has its own count or identifying track and that A retains its review state. |
| Health export | Use **Export JSON**, cancel once, then choose a guest-local destination outside either library. Verify the saved file parses with `Get-Content -Raw <path> \| ConvertFrom-Json` and describes the selected library. |
| Recovery | With the app closed, temporarily rename disposable library A. Relaunch, verify the missing-library screen, restore the folder name, and select **Retry Library**. Record this as a folder-availability test. |
| Duplicate launch and shutdown | Launch the executable again while A is open. Confirm a single app window remains and the log records the duplicate. Close the window. Confirm the NyxCore process exits and the previous engine port no longer listens. |
| 150% DPI and taskbar | Set the guest display to 150% in Windows settings. Reopen the app and verify the picker, Health export, window sizing and controls. Pin and relaunch the app. Verify the running window groups with the pinned icon and appears once in Alt+Tab. If the guest cannot set scale or reproduce normal shell behavior, record the affected check as **Unverified**. |
| Physical USB | On a clean physical computer or a VM with real USB passthrough, open a disposable library stored on the device. Disconnect and reconnect the device at its original drive letter and verify recovery with **Retry Library**. Without hardware and passthrough where applicable, record **Unverified**. |

Before and after these checks, record the relative path and SHA-256 of every file in each disposable library. The sets must match after any temporarily missing folder is restored. In Windows PowerShell, run the following for each library and compare the saved lines with `Compare-Object`. Keep the output outside the library.

```powershell
$library = (Resolve-Path 'C:\Acceptance\Library A').Path
Get-ChildItem -LiteralPath $library -Recurse -File | ForEach-Object {
    '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName).Hash,
        $_.FullName.Substring($library.Length + 1)
} | Sort-Object
```

## Review-only checks

Run these probes on the clean guest while the disposable library is open. Use the latest loopback address from `%LOCALAPPDATA%\NyxCore\Desktop\desktop.log`. Confirm `GET /api/status` reports `review_only: true` and the intended library. A valid empty organization preview supplies the ID for the last two checks. Expect HTTP 403 for each of the six POST requests, then compare the library manifests.

```powershell
$log = Join-Path $env:LOCALAPPDATA 'NyxCore\Desktop\desktop.log'
$ready = Select-String -LiteralPath $log -Pattern 'Local engine ready at (http://127\.0\.0\.1:\d+)' |
    Select-Object -Last 1
if (-not $ready) { throw 'NyxCore engine address was not found.' }
$origin = $ready.Matches[0].Groups[1].Value
$status = Invoke-RestMethod "$origin/api/status" -TimeoutSec 20
if (-not $status.review_only) { throw 'The app is outside review-only mode.' }
$status.music_path
$preview = Invoke-RestMethod "$origin/api/organization/preview" -Method Post `
    -ContentType 'application/json' -Body '{}' -TimeoutSec 20
if (-not $preview.id -or $preview.apply_enabled -ne $false) { throw 'Expected a review-only preview.' }

$probes = [ordered]@{
    '/api/review/plan/apply' = '{"plan_report":{}}'
    '/api/history/fixture/restore' = '{}'
    '/api/history/fixture/undo' = '{}'
    '/api/metadata/plan/apply' = '{}'
    "/api/organization/preview/$($preview.id)/review" = '{"paths":["fixture.flac"]}'
    "/api/organization/preview/$($preview.id)/apply" = '{"paths":["fixture.flac"]}'
}
foreach ($probe in $probes.GetEnumerator()) {
    try {
        $reply = Invoke-WebRequest "$origin$($probe.Key)" -UseBasicParsing -Method Post `
            -ContentType 'application/json' -Body $probe.Value -TimeoutSec 20 -ErrorAction Stop
        $code = [int]$reply.StatusCode
    } catch {
        if (-not $_.Exception.Response) { throw }
        $code = [int]$_.Exception.Response.StatusCode
    }
    [pscustomobject]@{ Endpoint = $probe.Key; Status = $code; Passed = $code -eq 403 }
}
```

Record **Passed**, **Failed**, or **Unverified** for each check, with the observed result or blocker. Capture the first-run screen and folder picker, analyzed library, persistence after restart, native Save dialog, missing-folder recovery, and 150% taskbar state. Save the screenshots, results, HTTP statuses and both file manifests outside the guest before disposing of it. [Windows Sandbox documentation](https://learn.microsoft.com/windows/security/application-security/application-isolation/windows-sandbox/)
