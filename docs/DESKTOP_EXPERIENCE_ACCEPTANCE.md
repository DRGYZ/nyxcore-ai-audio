# NyxCore Desktop experience acceptance

Tested on Windows 11 with the installed source launcher, pywebview 6.2.1, and WebView2 on the second display at 96 DPI. The libraries were disposable and contained synthetic tracks. Tests used a separate per-user desktop state directory.

## Native checks

- Fresh launch showed the welcome screen. The Windows picker accepted a folder with spaces and Unicode, showed its full path, and waited for confirmation. Canceling the picker left the active library in place. An empty folder opened and reported that no supported audio was found.
- Restart restored the selected path and a staged metadata edit. The explicit demo opened seven tracks in a separate workspace. Its reports and staged metadata were independent of the 3,069-track fixture. The previous loopback listener stopped after switching.
- Navigation remained responsive during first analysis. Refresh retained the complete prior report and prevented repeat refreshes. A switch requested during refresh reported that the library was busy. Removing the test folder caused Refresh Library to retain stale results and offer retry. Restoring it recovered the analysis.
- An unavailable saved folder showed recovery controls at launch. Restoring the folder and selecting **Retry Library** reconnected its original workspace. Corrupt settings showed folder selection again. A missing frontend build produced a native startup error without opening a loopback listener. The singleton prevented a second window. Closing during active analysis completed shutdown and the next launch recovered.
- Reloading the Health route preserved the desktop controls. The native Save dialog wrote exactly the current report, and canceling returned without a save message. Browser Health still downloaded JSON.
- The window maximized, minimized, and restored on display 2. The Alt+Tab gesture was exercised. WinForms constrained undersized bounds to 740×560. The 840×620 layout displayed the sidebar, library identity, analysis status, and change action. The desktop capture excludes the invisible window frame.

## Review-only and local controls

- All six existing guarded mutation endpoints returned HTTP 403 in the native session. The library fixture retained its original paths and SHA-256 hashes.
- Desktop controls require a per-process random window credential attached by WebView2 to requests at the active loopback origin. POST requests also require that exact Origin. Direct loopback requests without the credential returned 403. The existing navigation guards and disabled WebView2 host objects, web messages, and downloads remain in force.
- Library paths and export destinations come from native dialogs. The folder is validated again on confirmation. Export writes the current completed Health report as JSON and rejects destinations inside the active library or NyxCore state. Tests simulated write failures and confirmed existing output was kept.

## Observed performance

Times are from individual local runs, with tuning and runtime version held constant within each run. The Desktop 01 observations used a different fixture and measurement setup.

| Measurement | Observation |
| --- | --- |
| First run | Window visible in 0.94 s. Welcome document loaded in 3.31 s. |
| Repeat launches with the 3,069-track library | Window visible in 0.45–0.58 s. Document loaded in 0.97–2.48 s. The first report arrived 2.72–2.83 s after page navigation started. |
| Initial switch to the 3,069-track library | Populated Overview in about 3.2 s from confirmation. |
| Navigation during first analysis | Library Health in 99 ms and Overview in 38 ms. |
| Archive Search | Route opened in 22 ms. A specific track search returned in 126 ms. |
| Refresh | 3.29 s with previous results visible. |
| Idle sample | 599 MiB combined working set across seven processes and 7.14% of one logical core over five seconds. The working-set sum can count shared memory more than once. |
| Close during analysis | 2.66 s from the native close gesture to the clean shutdown log. |

## Automated validation and evidence

The Python suite passed 207 tests and 12 subtests with 44 skips. All 67 frontend tests and the production frontend build passed. The browser download was also exercised in Chrome.

Native screenshots are retained in the isolated worktree under `tests/_runtime_desktop_experience/screenshots/`. They cover welcome, selected folder, analysis in progress, populated Overview, Change Library, missing-library recovery, the Windows Save dialog, an 840×620 window, and the empty-library state.

True Windows 150% display scaling, physical USB removal, and taskbar grouping remain unverified. Folder removal and recovery were reproduced by renaming the disposable folder. Permission-denied handling was tested with a simulated Windows access failure. The native test windows used their production defaults outside the explicit compact geometry check.
