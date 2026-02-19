🎧 NyxCore

Local-first AI-powered music intelligence engine.
Clean metadata. Analyze audio. Write structured AI tags. Generate smart playlists.

 Overview

NyxCore is a local CLI tool that transforms a messy MP3 library into a structured, AI-enhanced music system.

It combines:

 Metadata normalization

 Signal-based audio analysis (Essentia)

 Semantic AI tagging (CLAP)

 Safe ID3 TXXX frame writing

 Smart playlist generation

 Persistent caching

Everything runs locally.
No cloud. No streaming APIs. No data leaving your machine.

 Architecture

NyxCore uses a modular backend system:

1️⃣ Essentia (Signal Analysis)

Extracts:

energy_0_10

bpm

Based on digital signal processing (DSP)

Objective acoustic measurements

2️⃣ CLAP (AI Semantic Model)

Generates:

tags (mood labels)

genre_top

Uses pretrained contrastive audio-text embeddings

Matches audio against descriptive prompts

3️⃣ Hybrid Backend

Combines both:

Physical features (Essentia)

Semantic understanding (CLAP)

🧩 Features
Phase 1 — Scan

Recursive MP3 scanning with:

Missing metadata detection

Cover art detection

Bitrate checks

Safe read-only mode

python -m nyxcore.cli scan music --out data/reports

Phase 2 — Normalize → Review → Apply
Normalize

Generates preview of proposed metadata changes.

python -m nyxcore.cli normalize music --out data/reports --strategy smart


Outputs:

normalize_preview.jsonl

normalize_preview.csv

normalize_preview.md

Apply (Safe ID3 Write)

Writes only selected fields with backups.

python -m nyxcore.cli apply music \
  --in data/reports/normalize_preview.jsonl \
  --fields album \
  --backup-dir data/backups

Phase 3 — AI Audio Analysis
Analyze (Hybrid AI)
python -m nyxcore.cli analyze music \
  --out data/reports \
  --backend hybrid


Outputs:

analysis_preview.jsonl

analysis_summary.md

SQLite cache (persistent)

Apply AI Tags (Safe)

Writes only custom ID3 frames:

TXXX:NYX_ENERGY

TXXX:NYX_BPM

TXXX:NYX_TAGS

TXXX:NYX_GENRE_TOP

python -m nyxcore.cli apply-ai music \
  --in data/reports/analysis_preview.jsonl \
  --fields energy,bpm,tags,genre \
  --backup-dir data/backups


Safety guarantees:

Does NOT modify title

Does NOT modify artist

Does NOT modify album

Does NOT modify cover art

 Smart Playlists

Generate playlists from AI data:

python -m nyxcore.cli playlists music --from-cache --out data/playlists


Examples:

energy_8_10.m3u

bpm_120_140.m3u

mood_chill.m3u

mood_dark.m3u

🛡 Safety Design

Read-only scanning by default

Preview-first workflow

Confidence filtering

Field-level write control

Backup directory support

No destructive overwrites

AI metadata isolated in NYX_* custom frames

⚙️ Tech Stack

Python 3.12

Typer + Rich (CLI UX)

Mutagen (ID3 handling)

Essentia (audio DSP)

LAION CLAP (audio-text model)

SQLite caching

FFmpeg fallback decoding

 Why This Project Matters

NyxCore demonstrates:

Multi-backend architecture

AI inference pipelines

Audio signal processing

Safe metadata engineering

Local-first system design

Cache-aware computation

Real-world data normalization

It bridges:

Low-level audio processing + Modern AI embeddings + Robust CLI tooling.

🔮 Future Vision

Planned expansions:

Prompt-optimized genre detection

Confidence-aware tagging

Parallel analysis

GPU acceleration

Web dashboard (NeuroDesk integration)

Context-aware music modes (Focus, Gym, Night)

 Status

✔ Phase 1 — Scan
✔ Phase 2 — Normalize & Apply
✔ Phase 3 — Hybrid AI + Tag Writing
✔ Playlist generation
🔜 UX layer & dashboard integration

👤 Author

Built as a local-first AI systems experiment combining:

Audio intelligence

Metadata engineering

Modular backend architecture
