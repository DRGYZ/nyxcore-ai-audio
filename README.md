# 🧠🎧 NyxCore

Local-first AI-powered music intelligence engine.  
Clean metadata. Analyze audio. Write structured AI tags. Generate smart playlists.

## 📦 Installation

Requirements:

- Python 3.11+ (3.12 recommended)
- `ffmpeg` available on PATH
- Dependencies include `aiohttp`, `sqlmodel`, and `PyYAML`

Setup:

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

## 🚀 Overview

NyxCore is a local CLI tool that transforms a messy MP3 library into a structured, AI-enhanced music system.

It combines:

- 🔎 Metadata normalization
- 🎼 Signal-based audio analysis (Essentia)
- 🤖 Semantic AI tagging (CLAP)
- 🏷 Safe ID3 TXXX frame writing
- 📂 Smart playlist generation
- 💾 Persistent caching


## ⚡ Quickstart

```bash
python -m nyxcore.cli scan music --out data/reports
python -m nyxcore.cli analyze music --backend hybrid --out data/reports
python -m nyxcore.cli judge music --analysis data/reports/analysis_preview.jsonl --out data/reports --concurrency 10
python -m nyxcore.cli apply-judge music --in data/reports/judge_preview.jsonl --backup-dir data/backups --dry-run
python -m nyxcore.cli playlists music --from-cache --out data/playlists
```

## 🏗 Architecture

NyxCore uses a modular backend system:

### 1️⃣ Essentia (Signal Analysis)

Extracts:

- `energy_0_10`
- `bpm`

Based on digital signal processing (DSP), with objective acoustic measurements.

### 2️⃣ CLAP (AI Semantic Model)

Generates:

- `tags` (mood labels)
- `genre_top`

Uses pretrained contrastive audio-text embeddings and matches audio against descriptive prompts.

### 3️⃣ Hybrid Backend

Combines both:

- Physical features (Essentia)
- Semantic understanding (CLAP)

## 🧩 Features

### Phase 1 — Scan

Recursive MP3 scanning with:

- Missing metadata detection
- Cover art detection
- Bitrate checks
- Safe read-only mode

```bash
python -m nyxcore.cli scan music --out data/reports
```

### Phase 2 — Normalize → Review → Apply

#### Normalize

Generates preview of proposed metadata changes.

```bash
python -m nyxcore.cli normalize music --out data/reports --strategy smart
```

Outputs:

- `normalize_preview.jsonl`
- `normalize_preview.csv`
- `normalize_preview.md`

#### Apply (Safe ID3 Write)

Writes only selected fields with backups.

```bash
python -m nyxcore.cli apply music \
  --in data/reports/normalize_preview.jsonl \
  --fields album \
  --backup-dir data/backups
```

### Phase 3 — AI Audio Analysis

#### Analyze (Hybrid AI)

```bash
python -m nyxcore.cli analyze music \
  --out data/reports \
  --backend hybrid
```

Outputs:

- `analysis_preview.jsonl`
- `analysis_summary.md`
- SQLite cache (persistent)

#### Apply AI Tags (Safe)

Writes only custom ID3 frames:

- `TXXX:NYX_ENERGY`
- `TXXX:NYX_BPM`
- `TXXX:NYX_TAGS`
- `TXXX:NYX_GENRE_TOP`

```bash
python -m nyxcore.cli apply-ai music \
  --in data/reports/analysis_preview.jsonl \
  --fields energy,bpm,tags,genre \
  --backup-dir data/backups
```

Safety guarantees:

- Does NOT modify title
- Does NOT modify artist
- Does NOT modify album
- Does NOT modify cover art

### Phase 3.5 — LLM Judge (DeepSeek-V3.2)

Uses an OpenAI-compatible DeepSeek endpoint to refine mood tags and genre from hybrid analysis.

Required environment variables:

```bash
# default base URL is https://api.deepseek.com if not set
# both are accepted:
# - https://api.deepseek.com
# - https://api.deepseek.com/v1
export DEEPSEEK_API_KEY="your_key_here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
# backward compatible:
# export NYX_DEEPSEEK_API_KEY="your_key_here"
# export NYX_DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
# backward compatible:
# $env:NYX_DEEPSEEK_API_KEY="your_key_here"
# $env:NYX_DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Model mapping:

- `deepseek-chat` = DeepSeek-V3.2 non-thinking
- `deepseek-reasoner` = DeepSeek-V3.2 thinking

Judge preview (cached in `data/cache/judge.sqlite`):

```bash
python -m nyxcore.cli judge music \
  --analysis data/reports/analysis_preview.jsonl \
  --out data/reports \
  --provider deepseek \
  --model deepseek-chat \
  --concurrency 10 \
  --limit 50
```

Concurrency notes:

- `--concurrency` default is `10`
- allowed range is `1-20`

Apply judge tags safely (NYX_* TXXX only):

```bash
python -m nyxcore.cli apply-judge music \
  --in data/reports/judge_preview.jsonl \
  --backup-dir data/backups \
  --limit 50 \
  --dry-run
```

Recommended full flow:

```bash
python -m nyxcore.cli analyze music --backend hybrid --out data/reports
python -m nyxcore.cli judge music --analysis data/reports/analysis_preview.jsonl --out data/reports --provider deepseek --model deepseek-chat
python -m nyxcore.cli apply-judge music --in data/reports/judge_preview.jsonl --backup-dir data/backups
python -m nyxcore.cli playlists music --from-cache --out data/playlists
```

Quick judge smoke test:

```bash
export DEEPSEEK_API_KEY=...
python -m nyxcore.cli judge music --analysis data/reports/analysis_preview.jsonl --out data/reports --model deepseek-chat --limit 5 --force
```

### Phase 3.6 — Smart Rename (Preview → Apply → Undo)

Smart filename cleanup for `.mp3` files only.
Folders are preserved; only filenames are changed.

Safe rename workflow:

1. Deterministic dry-run (no LLM), limited sample:

```bash
python -m nyxcore.cli rename music --dry-run --no-llm --limit 50
```

2. Dry-run with optional LLM refinement:

```bash
python -m nyxcore.cli rename music --dry-run --limit 50 --concurrency 10
```

3. Apply full rename and write map to reports folder:

```bash
python -m nyxcore.cli rename music --apply --out data/reports --concurrency 10
```

4. Undo from saved map:

```bash
python -m nyxcore.cli rename-undo --map data/reports/rename_map.jsonl --dry-run
python -m nyxcore.cli rename-undo --map data/reports/rename_map.jsonl
```

Rename safety notes:

- Default mode is preview (`--dry-run`)
- Windows-safe filename sanitization is applied
- Collision handling appends ` - 2`, ` - 3`, etc.
- No directory moves; rename stays within the same folder
- Writes a rename map JSONL into the reports folder

## 🗒 Release Notes

### 2026-02-20

- Refactored judge logic into `JudgeService` with cleaner CLI orchestration.
- Added deterministic local conflict scoring with debug-only `conflicts_llm`.
- Hardened judge reason cleanup and reduced noisy BPM mismatch wording.
- Added async concurrent DeepSeek judge requests with semaphore control.
- Introduced YAML config loading and dependency injection for judge prompts/rules.
- Added smart rename preview/apply/undo workflow with optional LLM refinement.

## 🎶 Smart Playlists

Generate playlists from AI data:

```bash
python -m nyxcore.cli playlists music --from-cache --out data/playlists
```

Examples:

- `energy_8_10.m3u`
- `bpm_120_140.m3u`
- `mood_chill.m3u`
- `mood_dark.m3u`

## 🛡 Safety Design

- Read-only scanning by default
- Preview-first workflow
- Confidence filtering
- Field-level write control
- Backup directory support
- No destructive overwrites
- AI metadata isolated in `NYX_*` custom frames

## ⚙️ Tech Stack

- Python 3.12
- Typer + Rich (CLI UX)
- Mutagen (ID3 handling)
- Essentia (audio DSP)
- LAION CLAP (audio-text model)
- SQLite caching
- FFmpeg fallback decoding

## 🧠 Why This Project Matters

NyxCore demonstrates:

- Multi-backend architecture
- AI inference pipelines
- Audio signal processing
- Safe metadata engineering
- Local-first system design
- Cache-aware computation
- Real-world data normalization

It bridges:

Low-level audio processing + modern AI embeddings + robust CLI tooling.

## 🔮 Future Vision

Planned expansions:

- Prompt-optimized genre detection
- Confidence-aware tagging
- Parallel analysis
- GPU acceleration
- Web dashboard (NeuroDesk integration)
- Context-aware music modes (Focus, Gym, Night)

## 🧪 Status

- ✔ Phase 1 — Scan
- ✔ Phase 2 — Normalize & Apply
- ✔ Phase 3 — Hybrid AI + Tag Writing
- ✔ Playlist generation
- 🔜 UX layer & dashboard integration

## 👤 Author

Built as a local-first AI systems experiment combining:

- Audio intelligence
- Metadata engineering
- Modular backend architecture
