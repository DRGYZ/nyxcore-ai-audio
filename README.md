# NyxCore

NyxCore is a local-first music library review and cleanup toolkit for people who manage folders of audio files, not streaming playlists. It scans a library, finds duplicates and metadata problems, builds a review queue, generates explicit action plans, records reversible history, and supports saved playlists from natural-language queries through a CLI, a FastAPI backend, and a local React UI.

Everything runs against local files and local state. NyxCore does not require a cloud account, a remote media server, or a hosted database.

Current release version: `0.2.0`

## Why This Exists

Music folders accumulate drift:

- exact duplicate files across imports, downloads, and archive folders
- likely duplicates across transcodes and rename variants
- missing or placeholder metadata
- weak artwork coverage
- low-quality files mixed into otherwise clean collections
- manual cleanup steps with poor reversibility

NyxCore exists to turn that mess into a review-first workflow instead of a pile of one-off scripts.

## Who It Is For

- collectors maintaining a local archive
- DJs and curators cleaning mixed-source folders
- developers building local music-library tooling
- anyone who wants a demoable local-first metadata and duplicate-review stack

## Main Surfaces

- CLI: scan, duplicates, health, review, plans, history, rename, analysis, tagging, and playlists
- API: FastAPI layer in `nyxcore.webapi` that exposes the same service modules to the frontend
- Web UI: React + Vite app in `web/` for Mission Control, Review Inbox, Saved Playlists, History, Duplicates, and Health

## Start Here

If you want the fastest safe first run, use the built-in demo fixture generator instead of pointing NyxCore at a real library immediately.

### 1-Minute Quickstart

```bash
python -m venv .venv
# Windows PowerShell
# .venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[web]"

python demo/create_demo_library.py --force
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/reports
python -m nyxcore.cli health demo/generated/sample-library --out data/reports
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
```

Then start the API and frontend:

```bash
# Bash / zsh
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload
```

```powershell
$env:NYXCORE_WEB_MUSIC_DIR = (Resolve-Path "demo/generated/sample-library").Path
$env:NYXCORE_WEB_OUT_DIR = (Resolve-Path "data/reports").Path
uvicorn nyxcore.webapi.app:app --reload
```

```bash
cd web
npm install
npm run dev
```

Open:

- API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## Demo Library

NyxCore ships a documented fixture path instead of committing binary audio to Git.

Generator:

- `demo/create_demo_library.py`

Fixture docs:

- `demo/README.md`

Default generated output:

- `demo/generated/sample-library/`

The generated library is intentionally tiny and designed to surface:

- one exact duplicate group
- one likely duplicate group
- missing metadata
- weak or placeholder metadata
- missing artwork
- low-bitrate audio
- saved-playlist candidates for a query like `ambient focus instrumental`

## Install

Requirements:

- Python 3.11+
- `ffmpeg` on `PATH`
- Node.js 18+ for the web frontend

Base install:

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[audio-analysis]"
pip install -e ".[clap]"
pip install -e ".[web]"
pip install -e ".[dev]"
pip install -e ".[audio-analysis,clap,web,dev]"
```

## Local Run Guide

### CLI-Only Workflow

```bash
python -m nyxcore.cli scan <music-dir> --out data/reports
python -m nyxcore.cli duplicates <music-dir> --out data/reports
python -m nyxcore.cli health <music-dir> --out data/reports
python -m nyxcore.cli review <music-dir> --out data/reports
```

### Web API

NyxCore's web API defaults to:

- music path: `music`
- output path: `data/reports`

For the demo library or any non-default folder, set:

- `NYXCORE_WEB_MUSIC_DIR`
- `NYXCORE_WEB_OUT_DIR`

Example:

```bash
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload
```

### Frontend

```bash
cd web
npm install
npm run dev
```

The frontend talks to `http://127.0.0.1:8000/api` by default.

### CLI + Web First-Run Path

Use this sequence when you want the UI to show live data immediately instead of mock fallback:

1. Generate the demo library with `python demo/create_demo_library.py --force`
2. Run `duplicates`, `health`, and `review` against `demo/generated/sample-library`
3. Start the API with `NYXCORE_WEB_MUSIC_DIR` pointing to the generated demo library
4. Start the frontend with `npm run dev`
5. Open the web UI and verify that the header shows live API mode, not mock fallback

## 5-Minute Guided Demo

### Step 1: Generate a safe sample library

```bash
python demo/create_demo_library.py --force
```

What you should see:

- a small library under `demo/generated/sample-library`
- a generated manifest and README inside that folder

### Step 2: Build the core reports

```bash
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/reports
python -m nyxcore.cli health demo/generated/sample-library --out data/reports
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
```

What you should see:

- exact duplicates around `Blue Hour`
- a likely duplicate FLAC vs MP3 pair for `Night Drift Ambient`
- missing metadata and placeholder metadata findings from `imports/legacy`
- artwork coverage problems and low-bitrate warnings

### Step 3: Create a saved playlist example

```bash
python -m nyxcore.cli save-playlist demo/generated/sample-library --out data/reports --name "Ambient Focus" --query "ambient focus instrumental"
python -m nyxcore.cli list-playlists --out data/reports
```

What you should see:

- a saved playlist entry with a deterministic playlist id
- tracks ranked from the demo library's ambient/focus candidates

### Step 4: Launch the API and web UI

```bash
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload
```

```bash
cd web
npm install
npm run dev
```

What you should see:

- Dashboard with live report counts
- Review Inbox populated from the generated demo data
- Saved Playlists showing the `Ambient Focus` result
- Health and Duplicates pages populated from the same library

### Step 5: Populate history with one safe batch

Generate a plan from an exact duplicate item:

```bash
python -m nyxcore.cli review-plan demo/generated/sample-library --out data/reports --item-id <exact-duplicate-item-id>
python -m nyxcore.cli apply-review-plan data/reports/review_plan.json --out data/reports
python -m nyxcore.cli history --out data/reports
```

How to get the item id:

- open `data/reports/review.json`
- copy the `item_id` for an `exact_duplicate_group`

What you should see:

- one reversible history batch in `history`
- the History page populated once the API is refreshed

## Outputs and Local State

When you use `--out data/reports`, NyxCore writes report outputs and persisted state under that root.

Common reports:

- `scan.json`
- `scan.md`
- `duplicates.json`
- `duplicates.md`
- `health.json`
- `health.md`
- `review.json`
- `review.md`
- `review_plan.json`
- `review_apply.json`
- `playlist_query.json`
- `playlist_query.md`

Persisted state:

- `review_state.json`
- `review_history.json`
- `library_state.json`
- `saved_playlists/saved_playlists.json`
- `saved_playlists/playlists/<playlist-id>/latest_result.json`
- `saved_playlists/playlists/<playlist-id>/latest_tracks.json`
- `saved_playlists/playlists/<playlist-id>/latest.m3u`

## Project Structure

Core runtime:

- `nyxcore/core`: scanner, track models, JSONL utilities, low-level helpers
- `nyxcore/duplicates`: duplicate analysis
- `nyxcore/health`: metadata, quality, naming, and duplicate-impact reporting
- `nyxcore/review_queue`: review item generation and triage state
- `nyxcore/action_plan`: plan generation, apply, quarantine, and ledger/history handling
- `nyxcore/saved_playlists`: saved-playlist definitions and refresh tracking
- `nyxcore/playlist_query`: natural-language playlist ranking
- `nyxcore/webapi`: FastAPI layer
- `web/src`: React frontend
- `tests`: unit and smoke coverage for backend workflows
- `demo`: demo-library generator and fixture notes
- `docs/assets/screenshots`: screenshot location and capture checklist

## CLI Command Families

Current command families include:

- `scan`
- `duplicates`
- `health`
- `review`
- `review-plan`
- `apply-review-plan`
- `history`
- `show-history`
- `restore-review-action`
- `undo-review-action`
- `playlist`
- `save-playlist`
- `list-playlists`
- `show-playlist`
- `rename-playlist`
- `edit-playlist`
- `delete-playlist`
- `refresh-playlist`
- `refresh-all-playlists`
- `normalize`
- `apply`
- `rename`
- `rename-undo`
- `analyze`
- `judge`
- `apply-judge`
- `apply-ai`

Legacy note:

- `python -m nyxcore.cli playlists ...` still exists for compatibility
- it is the older bucketed M3U export workflow
- prefer `playlist` plus saved-playlist commands for the current workflow

## Web UI

Primary routes:

- `/` Mission Control
- `/review` Review Inbox
- `/playlists` Saved Playlists
- `/history` Operation History
- `/duplicates` Duplicates
- `/health` Health

The UI uses live FastAPI responses when available and falls back to local mock data for development. Mutation actions remain disabled in fallback mode.

## API Surface

Route groups:

- status: `GET /api/status`
- reports: `GET /api/duplicates`, `GET /api/health`, `GET /api/review`
- review mutations: `POST /api/review/state`, `POST /api/review/plan`, `POST /api/review/plan/apply`
- saved playlists: `GET /api/playlists`
- history: `GET /api/history`, `POST /api/history/{batch_id}/restore`, `POST /api/history/{batch_id}/undo`

The API is intentionally thin. It uses the same service modules as the CLI.

## Configuration

NyxCore ships a packaged default YAML config in `nyxcore/resources/default.yaml`.

Built-in profiles:

- `default`
- `collector`
- `dj`
- `casual`
- `archivist`

Commands that depend on tuning typically accept `--config` and `--profile`, including `duplicates`, `health`, `review`, `playlist`, and `watch`.

## Screenshots and Showcase Assets

Store repo-local showcase assets here:

- `docs/assets/screenshots/README.md`

Preferred captures:

- Dashboard
- Review Inbox
- History
- Saved Playlists

If screenshots are not in the repo yet, use the checklist in that folder before publishing the README or portfolio page.

## Safety Model

- review-first reporting by default
- explicit apply step for action plans
- backups for supported mutation paths
- no silent overwrite on restore paths
- local persisted state for review, history, incremental refresh, and saved playlists

Current compatibility note:

- `restore-review-action` and `undo-review-action` currently use the same safe history-batch reversal path
- both names remain available for compatibility

## Development Notes

- frontend source: `web/src`
- backend API source: `nyxcore/webapi`
- demo fixture generator: `demo/create_demo_library.py`
- cleanup validation entry used during the repo polish work: `python -m unittest tests.test_web_api`

## WSL

For Windows + WSL2 setup, see [INSTALL_WSL.md](INSTALL_WSL.md).
