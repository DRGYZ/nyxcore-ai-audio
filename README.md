# NyxCore

NyxCore is a completed personal Windows desktop project for local music-library analysis and review. It finds duplicates and metadata issues, keeps review state in a separate workspace for each library, and previews changes while leaving audio files untouched.

The portable Windows build passed acceptance in an independent Windows 11 VM. The executable is for personal use, with no official Windows binary or installer distribution. Developers can [run the desktop app from source](INSTALL_DESKTOP.md). NyxCore's first-party code is [MIT-licensed](LICENSE), and third-party components retain their own licenses.

Everything runs against local files and local state. NyxCore does not require cloud accounts, remote media servers, or hosted databases.

Latest tagged release: `0.3.0`

![NyxCore Review Inbox](screenshots/02_review_inbox_selected_1440.png)
*NyxCore Review Inbox showing findings and generated plan previews.*

---

## Core Workflow

The desktop app and browser preview are review-only. The experimental mutation tools follow a review-first lifecycle:

```
scan
  → detect duplicate / health findings
  → review
  → generate explicit action plan
  → safely apply selected operations
  → inspect history
  → reverse when supported
```

1. **Scan**: Indexes audio files (`.mp3`, `.flac`, `.ogg`, `.m4a`, `.wav`, `.aif`, `.opus`), reading metadata tags and file characteristics using pure Python (`mutagen`).
2. **Detect Findings**: Runs duplicate analysis (exact SHA-256 and likely fuzzy metadata/duration matches) and library health checks (missing tags, placeholder values, low bitrates, missing artwork).
3. **Review**: Aggregates findings into a unified, prioritizeable review queue. Findings can be triaged without altering audio files.
4. **Plan**: Compiles approved review items into a serialized JSON action plan with explicit source paths, destination paths, field modifications, and cryptographic content fingerprints.
5. **Apply**: Executes selected plan operations under a library-scoped single-writer lock with durable intent journaling. Duplicate files are relocated to `.nyxcore_quarantine` rather than deleted.
6. **Inspect History**: Every applied batch is recorded in a local history ledger with execution timestamps, per-operation statuses, and rollback fingerprints.
7. **Reverse**: Supported operations (such as metadata updates and quarantine moves) can be undone or restored after verifying that files have not been modified externally.

---

## Approved Public Surfaces

The web interface exposes six approved public views:

| Route | Surface | Purpose |
|---|---|---|
| `/` | **Overview** | Library summary metrics, finding category distributions, recent activity, and quick access. |
| `/search` | **Archive Search** | Unicode-normalized, read-only search across filenames, tags, genres, and directory paths. |
| `/review` | **Review Inbox** | Central triage and plan execution workbench with expandable finding cards and plan previews. |
| `/duplicates` | **Duplicates** | Dedicated exact and likely duplicate clusters with track comparisons and review links. |
| `/health` | **Library Health** | Detailed integrity diagnostics: missing metadata, placeholder tags, artwork gaps, low bitrates. |
| `/history` | **History** | Chronological audit log of applied operation batches with reversal inspection and undo actions. |

> **Scope Note**: A saved-playlists query backend exists in the codebase for experimental research, but its web UI route (`/playlists`) is deferred and intentionally unexposed in this release.

---

## Safety Architecture & Limitations

NyxCore is designed around defensive, local-first safety primitives:

- **Explicit User Review**: No autonomous or background mutations. Every file modification or relocation requires an explicit action plan generated from human review.
- **Serialized Action Plans**: Plans are written to `review_plan.json` with strict operation schemas, preflight validation, and content fingerprints. Plans exceeding 50 operations are flagged for mandatory manual review.
- **No-Clobber Quarantine**: Duplicate removals never delete files directly. Non-preferred copies are moved into `.nyxcore_quarantine/` with collision-safe naming. Normal scans ignore quarantine folders.
- **Intent Journaling** for apply and reversal records each intent before changing an audio file. Interrupted operations can be reconciled when the paths, fingerprints, and recorded file identity support the same conclusion.
- **Single-Writer Locking** coordinates NyxCore writers through the library's `.nyxcore_mutation.lock`. An abandoned lock requires operator review.
- **Recovery Inspection** with `recover-review-action --action inspect` reports ambiguous states for manual review. Older interrupted moves without recorded file identity need manual review when a destination exists.
- **Reversal Verification** checks the live file's SHA-256 fingerprint. Metadata restoration also verifies the backup and a staged copy before atomically replacing the live file.
- **Metadata Backups**: Pre-edit metadata states are preserved in `.nyxcore_backups/` before in-place tag updates.

### Important Limitations

> [!WARNING]
> **No ACID Guarantees**: NyxCore operates directly on local filesystems. It does not provide distributed transactions, two-phase commits, or atomic directory swapping.
>
> **Concurrent filesystem changes** can occur after a path or fingerprint check and before the next filesystem call. The library lock coordinates NyxCore writers. It cannot coordinate other programs that edit files or replace paths. Keep other writers away from disposable mutation fixtures.
>
> **Disposable-library validation** covered CLI, API, browser apply and reversal, and interrupted recovery on Ubuntu 24.04 under WSL2 on ext4. Native Windows 11 on NTFS with Python 3.12 also passed quarantine, metadata restore, lock exclusion, and controlled interruptions after journal intent, hard-link creation, and reversal before result persistence. Abrupt native process termination, power loss, and filesystem crash durability remain unverified. An interruption during metadata staging can leave a temporary file for manual inspection. Older builds should not resume batches written with pending reversal intent.
>
> **Experimental use** on real music libraries is limited to read-only review and plan preview. Use synthetic or disposable libraries for mutation testing.

---

## Screenshot Showcase

### Overview
![NyxCore Overview](screenshots/01_overview_1440.png)
*Overview dashboard displaying library scan status, finding counts, and direct navigation.*

### Duplicates Analysis
![NyxCore Duplicates](screenshots/03_duplicates_1440.png)
*Duplicates view showing exact SHA-256 and likely duplicate groups with comparison metrics.*

### Library Health
![NyxCore Library Health](screenshots/04_library_health_1440.png)
*Health diagnostic report detailing missing tags, placeholder titles, artwork coverage, and bitrate distribution.*

### Operation History & Reversal
![NyxCore History](screenshots/05_history_1440.png)
*Audit history displaying executed batches, affected paths, and reversible action controls.*

### Action Plan Review Modal
![NyxCore Action Plan Review Modal](screenshots/08_plan_modal_1440.png)
*Focus-trapped dialog displaying generated action plan operations, individual operation selection checkboxes, and preflight safety checks.*

### Tablet & Responsive Support
![NyxCore Tablet Review](screenshots/09_review_inbox_tablet_768.png)
*Responsive layout verified across tablet and mobile viewports with accessible focus styling.*

*For the complete gallery of release screenshots and responsive verification captures, see [`screenshots/README.md`](screenshots/README.md).*

---

## Requirements

- **Python**: `3.11` or `3.12` are tested. The package minimum is `3.11`.
- **Node.js for frontend builds and development**: Use Node.js 22.13 or later within 22.x and npm. The launcher uses the compiled assets at runtime.
- **Audio Metadata Support**: Pure Python via `mutagen` (`.mp3`, `.flac`, `.ogg`, `.m4a`, `.wav`, `.aif`, `.opus`). No external media binaries are required for core indexing, health reporting, duplicate matching, or action plans.
- **Optional External Tools**:
  - `ffmpeg`: **Only** required if you want to generate synthetic audio test files with `demo/create_demo_library.py` or run optional CLAP audio transcoding. It is not required for the core scanner or web application.

---

## Source-Based Review Preview

The review preview uses an editable Python 3.12 install from the extracted ZIP. It includes the compiled frontend and seven synthetic tracks, so it can run without Node.js, Vite, or FFmpeg.

The launcher analyzes the selected library on first use. Overview, Duplicates, Review Inbox, and Archive Search share the latest analysis in memory. Select **Refresh Library** after changing files. A refresh performs a full scan and keeps the previous successful results available if it fails. Restarting the launcher or requesting a different analysis profile rebuilds the analysis. If NyxCore detects changes during a scan, let the transfers or edits finish and try again. Triage is saved independently in the workspace. A finding marked resolved can return as Seen if it remains after a fresh analysis.

Open a successful main-branch [GitHub Actions CI run](https://github.com/DRGYZ/nyxcore-ai-audio/actions/workflows/ci.yml?query=branch%3Amain) and download `nyxcore-review-preview-<commit>` from its **Artifacts** section. GitHub sign-in is required and artifacts expire after 30 days. The download contains the ZIP and its SHA-256 checksum. Follow the [preview installation guide](INSTALL_PREVIEW.md).

### Build a preview from source

From a clean checkout of the commit to package, use Python 3.12, Git, FFmpeg, and Node.js 22 with npm on PATH.

```bash
python scripts/build_preview.py
```

The script builds from Git HEAD in a temporary directory and writes the ZIP and checksum to `dist/preview/`. It installs from the npm lockfile and generates the bundled demo with FFmpeg. `PREVIEW_BUILD.json` records the source commit, lockfile hash, build tool versions, and expected frontend files. File ordering, ZIP timestamps, and ZIP permissions are normalized. Compare checksums with the same toolchain because frontend, audio encoding, and compression can vary across tool versions and platforms.

## Source Launcher on Native Windows

Use Windows 11, Python 3.12, and Node.js 22 (22.13 or later within 22.x). Put FFmpeg on PATH to generate the synthetic library. These PowerShell commands set up the launcher in a source checkout.

```powershell
git clone https://github.com/DRGYZ/nyxcore-ai-audio.git
Set-Location nyxcore-ai-audio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[web]'
Push-Location web
npm.cmd ci --engine-strict
npm.cmd run build
Pop-Location
```

Generate the demo once, then launch it.

```powershell
.\.venv\Scripts\python.exe demo/create_demo_library.py
.\.venv\Scripts\python.exe -m nyxcore.launch --demo --out data/demo-reports
```

NyxCore prints the configured library, workspace, and local URL, then opens the browser when ready. Check **Active library** in Overview. In **Review Inbox**, select the exact duplicate finding for **Blue Hour**, click **Generate Plan**, and inspect which copy would be kept and which would be moved.

To switch libraries, press **Ctrl+C** in the launcher terminal and restart with an explicit library and a separate workspace for that library.

```powershell
.\.venv\Scripts\python.exe -m nyxcore.launch --music 'D:\Music Library' --out 'C:\NyxCore\my-library-reports'
```

The launcher binds only to `127.0.0.1` and runs in **review-only mode** for every library. Search, findings, plan previews, and history inspection are available. Triage decisions, saved playlists, and staged metadata proposals may write local state to the workspace. The API refuses apply, restore, and undo requests. Mutations remain experimental.

Open a track from a metadata issue in Library Health, a metadata finding in Review Inbox, or an Archive Search result. Inspect the embedded values, then stage replacements for title, artist, or album. Filename suggestions are guesses, and NyxCore leaves missing album suggestions unavailable. Open **Review staged metadata** to select up to 50 files, set a shared artist or album, and generate a before-and-after preview. Staged proposals persist in `metadata_drafts.json` in the workspace. After a library refresh or server restart, reopen stale drafts and stage them against the current tags.

Only the development workflow below can apply metadata repairs. Configure an explicit disposable library root and a separate workspace. The editor supports writes to readable MP3, FLAC, and M4A files. Clearing tags is unavailable. Every applied file has a full backup in the workspace. Use **History** to reverse verified edits and inspect any operation that needs recovery.

### Library Organization preview

Open **Library Organization** and choose **Build preview** to see proposed `Artist / Album / Track.ext` paths from embedded tags. The planner prefers an explicit album artist and uses a track artist only when the observed album credits agree. Files with missing tags, uncertain compilation credits, unsafe paths, or collisions stay outside the ready folder tree. Use **Files** to inspect the reason and open the Metadata Editor for supported fields. Album artist and track or disc numbers may need a separate tag editor.

Building a preview only reads the configured library. It uses no metadata providers and keeps the latest plan in memory. Ready indicates that a proposed path passed the structural checks. Review recording identity separately and rebuild after changing tags or files.

On Linux, organization uses readable mount IDs from `/proc/self/fdinfo`. A configured library may be a mount point, and mounts beneath it are excluded from organization.

The development server can apply selected organization moves when `NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT` is set to the exact absolute path of the disposable library configured in `NYXCORE_WEB_MUSIC_DIR`. Use a separate workspace. Select up to 50 Ready files, review the exact paths, then confirm the batch. History can undo applied moves to their exact original paths. An occupied path blocks undo. Directories created by a move remain in place after undo.

If a move is interrupted, inspect the batch with `recover-review-action --action inspect` using the same library and workspace. The journal identifies moves that can be finalized or aborted. Keep the disposable-root opt-in set for recovery that changes state, and resolve any abandoned mutation lock first. Close other applications that edit or move files in the fixture while this workflow runs. The packaged launcher offers the organization preview in review-only mode.

### Find Music Info and front covers

In the metadata editor, open **Find Music Info**, check the query, and choose **Search MusicBrainz**. Title and artist are sent to MusicBrainz. Album, duration, and filename clues are used locally to compare the results. Cover lookup sends the selected MusicBrainz release or release-group identifier to Cover Art Archive. NyxCore sends no audio, paths, hashes, workspace identifiers, or unrelated tracks. Manual editing and the rest of library review work offline.

Compare the recording candidates, then choose the release whose album or artwork belongs on your file. A recording may appear on multiple releases. The match score describes how the returned result compares with the available clues. Select the suggested tags or a front cover before staging. You can edit the proposed fields and keep an existing cover unless you explicitly approve its replacement.

The workspace cache is limited to 64 MiB and 160 entries. Search and artwork-index results expire after seven days and cached images after 14 days. Approved artwork is a locally decoded and normalized JPEG. NyxCore accepts single-frame JPEG and PNG inputs up to 6 MiB and 16 megapixels. If cached artwork is missing, expired, or changed, select the cover again. A server restart also requires a fresh inspection and re-selection or removal of the previous online source from the draft.

The source and portable launchers allow lookup and previews in review-only mode. Apply and undo are available for disposable development libraries. Front-cover editing uses native ID3, FLAC picture blocks, and M4A cover atoms. Files with multiple front covers require manual inspection. Other picture types in MP3 and FLAC are preserved.

MusicBrainz offers its public API for noncommercial use. Commercial users should consult its service plans. MusicBrainz core metadata is CC0. Cover Art Archive images retain the rights of their respective copyright owners. [MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API) · [MusicBrainz data licenses](https://musicbrainz.org/doc/About/Data_License) · [Cover Art Archive](https://coverartarchive.org/)

- If the port is occupied, append `--port 8001` or choose another free port. Leave the existing service running.
- If the browser does not open, use the printed URL. Add `--no-browser` to open it yourself.
- If the build is missing or incomplete, repeat the commands in `web/`. After building, the launcher runs without Node.js or Vite. Keep `web/dist/` next to the Python source in the checkout.
- If the demo is missing, run the generator from the repository root. Choose an existing music folder and a writable workspace that does not overlap it.

For Ubuntu 24.04 under WSL2, follow the [WSL2 Setup Guide](INSTALL_WSL.md).

---

## Vite and FastAPI Development Workflow

The separate Vite and FastAPI workflow is for development. Use disposable libraries with it. The [WSL2 Setup Guide](INSTALL_WSL.md) covers system prerequisites. Run the commands below in Bash inside the Linux environment.

### 1. Set Up Python

```bash
git clone https://github.com/DRGYZ/nyxcore-ai-audio.git
cd nyxcore-ai-audio

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[web,dev]"
```

### 2. Generate the Demo and Reports

With `ffmpeg` on PATH, generate the synthetic fixture. If you already generated it at this path, skip the first command.

```bash
python demo/create_demo_library.py
python -m nyxcore.cli scan demo/generated/sample-library --out data/demo-reports
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/demo-reports
python -m nyxcore.cli health demo/generated/sample-library --out data/demo-reports
python -m nyxcore.cli review demo/generated/sample-library --out data/demo-reports
```

The fixture has seven tracks, one exact duplicate group, and one likely duplicate group. The CLI writes reports and local state under `data/demo-reports/` without changing audio files. The API computes findings from the configured library and reads local review state; it does not load the CLI reports as snapshots.

### 3. Launch the Web API

From the repository root with the virtual environment active:

```bash
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/demo-reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000
```

Open [API status](http://127.0.0.1:8000/api/status) and check that `music_path` ends in `demo/generated/sample-library`.

### 4. Launch the React Web UI and Inspect a Plan

In a second terminal in the same environment, from the repository root:

```bash
cd web
npm ci --engine-strict
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

1. Open [Overview](http://127.0.0.1:5173/) and check that **Active library** points at the synthetic fixture and **Library Files** shows seven files. If the backend is still starting, use **Check Connection** when it is ready. If port `5173` is occupied, use the [verified WSL alternate port](INSTALL_WSL.md#vite-cannot-bind-port-5173).
2. In **Archive Search**, search for `Blue Hour` to find both copies of the demo track.
3. Open **Review Inbox** from Overview and select the exact duplicate finding for **Blue Hour**. Click **Generate Plan**. The preview identifies the copy to keep and the source and destination of each proposed move. Use **Full path** to inspect the original absolute path, then **Close** the preview.

To inspect another library, stop the backend, change `NYXCORE_WEB_MUSIC_DIR` and `NYXCORE_WEB_OUT_DIR` to its library and reports folders, and restart the backend. Refresh Overview to verify the active library before reviewing findings.

Browsing, searching, opening a finding, and generating or closing a plan preview do not change audio files. Triage controls can save local review state. Stop at the preview for this demo. Applying plans, restoring or reversing batches, taking over locks, and recovery writes are outside this demo.

If the UI shows **Local API is not connected**, check [API status](http://127.0.0.1:8000/api/status), start the backend if needed, then click **Check Connection**. For localhost or port conflicts, see the [WSL2 troubleshooting steps](INSTALL_WSL.md#8-troubleshooting).

---

## CLI Reference

NyxCore provides a unified CLI under `python -m nyxcore.cli`:

### Core Inspection & Review Commands

```bash
# Index library and output scan summary
python -m nyxcore.cli scan <music-dir> --out data/reports

# Detect exact and likely duplicates
python -m nyxcore.cli duplicates <music-dir> --out data/reports

# Run metadata, quality, and artwork health checks
python -m nyxcore.cli health <music-dir> --out data/reports

# Build review queue
python -m nyxcore.cli review <music-dir> --out data/reports

# Generate action plan for a specific review item
python -m nyxcore.cli review-plan <music-dir> --out data/reports --item-id <item-id>

# View operation history
python -m nyxcore.cli history --out data/reports

# Inspect an interrupted batch
python -m nyxcore.cli recover-review-action <batch-id> --action inspect --music <music-dir> --out data/reports
```

### Mutation Commands (Outside the Demo)

`apply-review-plan` and `restore-review-action` (also called `undo-review-action`) can move audio files or write tags. Use the CLI's `--help` for their options.

### Note on Legacy / Experimental Mutation Commands

Earlier experimental CLI verbs (`apply`, `rename --apply`, `rename-undo`, `apply-ai`, `apply-judge`) are intentionally **disabled** in this release. All supported mutations must flow through the inspected `review` → `review-plan` → `apply-review-plan` pipeline to ensure locking, journaling, and verification.

---

## Repository Structure

```
nyxcore-ai-audio/
├── .github/workflows/       # CI test automation
├── demo/                    # Synthetic audio fixture generator & docs
├── docs/assets/screenshots/ # Screenshot documentation pointer
├── nyxcore/                 # Python backend package
│   ├── core/                # Scanner, track model, hashing, locking
│   ├── duplicates/          # Exact SHA-256 and fuzzy tag/duration matcher
│   ├── health/              # Missing metadata, artwork, bitrate checkers
│   ├── review_queue/        # Review item generation and triage state
│   ├── action_plan/         # Action plan generator, validator, journal, executor
│   ├── search/              # Read-only Unicode-normalized archive search
│   └── webapi/              # FastAPI REST endpoints
├── screenshots/             # Release screenshots and responsive verifications
├── tests/                   # Backend pytest test suite (142 tests)
└── web/                     # React 18 + TypeScript + Vite frontend
    └── src/
        ├── test/            # Frontend test setup and render utilities
        └── ui/              # Components, pages, hooks, state
            └── __tests__/   # Frontend Vitest test suite (12 tests)
```

---

## Testing & Validation

Run backend tests:

```bash
pytest
```

Run frontend test suite:

```bash
cd web
npm test
```

Run frontend production build:

```bash
cd web
npm run build
```

---

## WSL2 Setup

For running NyxCore inside Windows Subsystem for Linux (WSL2), refer to the dedicated [WSL2 Setup Guide](INSTALL_WSL.md).

---

## License

NyxCore's first-party source code is available under the MIT License. Third-party dependencies, bundled fonts, and externally sourced assets retain their own licenses and attribution requirements. [MIT License](LICENSE)

The frozen Windows candidate includes Mutagen under GPL-2.0-or-later. Distribution of the combined executable has separate GPL and runtime obligations. [Windows distribution notes](docs/WINDOWS_DISTRIBUTION.md)
