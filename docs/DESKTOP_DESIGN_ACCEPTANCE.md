# Desktop 01 design integration acceptance

Date: 18 November 2026. Branch: `codex/desktop01-creative-redesign`. Base: `8304155b5de83e8846e33ac591e96ff50c8a0768`.

## Result

Argon's Pressed & Cataloged implementation is integrated as the Desktop 01 identity in the existing redesign worktree. The warm palette, editorial typography, collection sleeves, shared components, and dense evidence views are retained. There is one React application and one production frontend build.

The original Argon source changes were preserved before refinement, and all 14 original design screenshots are retained unchanged. No push, merge, branch replacement, backend mutation-engine change, journal change, or recovery-contract change was made.

## Confirmed corrections

| Finding | Correction |
| --- | --- |
| Search suggested synthetic demo artists for every library | Removed the hardcoded shortcuts and their empty-state instruction. Search requests and URL state are unchanged. |
| The search-submit control suppressed its keyboard outline without a replacement | Added an amber focus ring. Notification links received matching visible focus states. |
| Flex-styled folder summaries hid the native disclosure marker | Added explicit open/closed cues while retaining native `details` keyboard behavior. |
| Folder leaves did not distinguish Ready from Unchanged | Retained proposal records in the existing grouping and displayed their status. Collision and Needs Review entries remain excluded by the existing folder API. |
| Two-file duplicate comparisons reserved an empty third column | Use two columns for two copies and retain three-column density for larger groups. |
| Native title/startup surfaces retained the previous palette | Match the approved background, caption, and foreground colors without changing window chrome or lifecycle. |
| Health export claimed success after the native download was canceled | Pass the existing desktop mode to Health; disable native export with a visible browser-availability explanation. Browser export still downloads real report JSON. This was an existing Desktop 01 limitation, not an API regression introduced by Argon. |

No lost API contract, React state transition, selection safeguard, or mutation guard was found in the reviewed Argon changes. Most source changes outside the corrections above are presentation, typography, spacing, and component composition.

## Functionality checked

| Area | Evidence |
| --- | --- |
| Overview | Real seven-track and 69-track analyses, counts, review navigation, independent scrolling, and refresh status. |
| Archive Search | Real ranked matches, query URL, keyboard submit, empty query and no-match states, metadata-editor links; no demo recommendations. |
| Duplicates | Exact/likely tabs, expanded full paths, preference reasons, reported pair comparisons, manual-review restrictions, and evidence-to-Inbox navigation. |
| Library Health | Actual counts and issue expansion, affected-track editor, disabled native export, successful browser export of the 69-track report. |
| Review Inbox | Type filtering, selected exact-duplicate evidence, generated action-plan preview, and absence of Apply in review-only mode. |
| Metadata | Current/proposed values, workspace staging, focus trap, Escape, trigger-focus restoration, and actual metadata plan with before/after values and no Apply control. |
| MusicBrainz and covers | Live public recording search, recording/release selection, no-cover state for a release, release-group fallback, real cover selection and successfully decoded thumbnails. Provider selection was discarded without staging it to audio. Existing frontend tests cover additional provider, source-binding, stale-selection, and error cases. |
| Organization | Real 69-track disposable library: 58 Ready, 6 Needs Review, 4 Collision, 1 Unchanged. All filters, 50-row pagination, folder exclusion rules, Ready/Unchanged labels, keyboard collapse/expand, and no move-selection/apply controls. |
| History | Synthetic completed and in-progress journal records, URL-backed selection, recovery inspection instructions, blocked reversal, and connected-API error/retry state. No audio moves were performed to create History evidence. |

History fixture records use the existing journal enums. An initially malformed fixture produced a visible connected-API error; correcting the fixture restored History through Retry. That fixture error was not treated as a product regression.

## Windows acceptance

Native captures were taken from the actual pywebview/Windows WebView2 application and include its Windows title bar. The normal installed GUI entry point was also launched from a different working directory, and it loaded this checkout's rebuilt frontend. Its initial window is 1320×860 nominal pixels at 96 DPI.

The default window, 1440×900, 1280×800, and 840×620 were exercised. The measured WebView client areas exclude Windows borders and caption. All eight navigation destinations passed page/main horizontal-overflow checks at the three explicit sizes. Wide evidence tables retain local scrolling. Compact metadata actions remain visible while the dialog body scrolls. Modal Tab/Shift+Tab trapping, Escape and focus restoration, folder keyboard activation, header focus, scroll navigation, Alt+Tab, minimize, maximize, restore, and close were exercised.

An immediate capture after restoring a minimized window briefly showed a blank surface. A later observation showed the intact document without reloading or changing code; a persistent rendering defect was not reproduced.

Both available displays reported 96 DPI. Windows Settings did not expose a targetable window for changing display scaling in this session. The 150% capture uses WebView2 `ZoomFactor=1.5` and `devicePixelRatio=1.5` on the actual native window. All eight destinations passed overflow checks at that scale. It is **content scaling**, not proof of Windows 150% per-monitor DPI behavior. Zoom was restored afterward.

See the [native screenshot index](../screenshots/desktop-integration/README.md).

## Validation and safety

- Node 22.19.0: all **67 frontend tests across 11 files passed**. The organization test now covers Ready and Unchanged leaves together while excluding collisions.
- Node 22.19.0: `npm run build` passed TypeScript project checking and the Vite production build. Final JS: 411.76 kB, 117.84 kB gzip; CSS: 56.09 kB, 16.56 kB gzip. The existing Browserslist age notice remains non-blocking.
- **15 focused backend/launcher tests passed**: `tests/test_desktop.py` and `tests/test_launch.py`. Loopback tests needed execution outside the filesystem/network sandbox.
- The existing Vite workflow started and served `/`, `/src/main.tsx`, and the changed Health module with HTTP 200. The temporary server was stopped.
- A separate browser acceptance run downloaded and parsed the real 69-track Health report, exercised a no-match search, and recorded no page errors or page overflow.
- The running native app reported `review_only=true` and organization `apply_enabled=false`.
- Direct requests to all six existing guarded routes returned **403**: review-plan apply, metadata-plan apply, History restore, History undo, organization selection review, and organization apply. Metadata/organization previews remained available.
- The original synthetic library's nine files and the populated fixture's 71 files retained their exact paths and SHA-256 hashes. The fixture contains 69 audio tracks. Staging and provider caching used separate workspaces.
- Native close shut down Uvicorn and released its loopback listener. The normal GUI executable was finally closed through native window controls as well.
- The original Desktop 01 worktree stayed clean at its existing commit. `main`, `v0.3.0`, and unrelated worktrees were not changed by this task. The isolated worktree's 13 screenshot deletions were retained.

The first native acceptance run bypassed only the production single-instance mutex in a disposable driver, because an earlier windowless process held it. Once that process was verified to be a review-only synthetic demo, it was stopped and the normal GUI executable was tested directly. No singleton or lifecycle code was changed in the product.

## Performance

| Measurement | Established Desktop 01 reference | Integration observations |
| --- | --- | --- |
| Window shown | 0.58 s | Three measured runs: 0.500, 0.468, 0.485 s; normal GUI executable: 0.469 s |
| Overview usable | 1.43 s | Three measured runs: 1.109, 1.093, 1.125 s |
| Page navigation | 74–124 ms | 21 samples across seven destinations: 28–51 ms, median 33 ms |

A contemporaneous unchanged Desktop 01 control showed its window in 0.531 s and Overview in 1.234 s. The initial integration run was slower: 0.672 s window, 3.250 s usable, with the local engine ready at 2.922 s. Subsequent runs did not reproduce that delay; no backend or lifecycle optimization was introduced. These are small local samples. Historical navigation instrumentation is not known to match the current click/heading/two-animation-frame measurement, so the numbers do not establish a speedup guarantee.

## Focused visual audit

Scores are implementation-review judgments, not accessibility certification.

| Dimension | /10 | Observed basis |
| --- | ---: | --- |
| Color consistency | 9 | Warm palette across all nine reviewed workflows and native startup/caption. |
| Typography | 8 | Display/body/editorial hierarchy; technical mono is reserved for evidence. |
| Spacing | 9 | Shared panel/control rhythm; dense views retain clear grouping. |
| Component consistency | 8 | Existing primitives are reused; legacy semantic tone aliases remain internal. |
| Responsive behavior | 8 | Native size matrix and content zoom pass; true 150% OS DPI remains unverified. |
| Dark treatment | 9 | Complete warm dark identity, including dialogs and error states. |
| Motion | 9 | Restrained state transitions and existing reduced-motion handling. |
| Keyboard/accessibility | 8 | Visible focus, native disclosures, modal trap/restoration; no comprehensive screen-reader certification. |
| Density | 8 | Two-copy comparison uses available width; long tables and History use contained scrolling. |
| Interaction polish | 8 | Loading, empty, error, selection, staging and preview states verified; native export feedback corrected. |

## Files and local setup

Source changes comprise `nyxcore/desktop.py`, `web/index.html`, `web/src/index.css`, `web/tailwind.config.ts`, and these UI files:

```text
App.tsx
ListPages.tsx
__tests__/organization.test.tsx
duplicates/DuplicateComparison.tsx
feedback.tsx
history/HistoryDetailPanel.tsx
metadata/MetadataEditor.tsx
metadata/MetadataIssueList.tsx
metadata/MusicInfo.tsx
pages/DashboardPage.tsx
pages/DuplicatesPage.tsx
pages/HealthPage.tsx
pages/HistoryPage.tsx
pages/MetadataPage.tsx
pages/OrganizationPage.tsx
pages/ReviewPage.tsx
pages/SearchPage.tsx
primitives.tsx
review/DuplicateDecisionEvidence.tsx
review/PlanReportModal.tsx
review/ReviewDetailPanel.tsx
shell.tsx
```

Documentation, original Argon PNGs, native acceptance JPGs, and a compact machine-readable evidence summary accompany the source changes. The local disposable drivers, generated audio, environments, and workspaces are ignored rather than shipped as product code.

The local `.venv/Scripts/nyxcore-desktop.exe` resolves the editable package to this checkout. For this session, that environment reuses the preserved Desktop 01 dependency directory through an ignored `.pth` file; the baseline environment was not modified. A fresh independent setup uses the ordinary `pip install -e ".[desktop]"` procedure in `INSTALL_DESKTOP.md`.

## Remaining limits and next acceptance

True Windows 150% display DPI still requires manual acceptance. General embedded-artwork thumbnails require a new backend endpoint and remain out of scope. Native JSON export remains unavailable under the existing download policy. Startup numbers are reference observations, not performance guarantees. Desktop 02 onboarding, packaging, installers, updates, mutation/recovery changes, and deployment remain outside this integration.

Next: review the native captures, then check true 150% Windows display DPI when a controllable display-settings surface is available. Continue on this branch; do not resume a competing theme or merge/push without a separate request.
