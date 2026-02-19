# nyxcore

Local-first music library auditor for MP3 collections.

## MVP (Step 1)

- Recursively scans a folder (default: `./music`)
- Finds all `.mp3` files
- Reads ID3 tags using `mutagen` (read-only)
- Writes:
  - `scan.json` (machine readable)
  - `scan.md` (human readable)

Corrupted or unreadable files are captured with warnings and do not crash the scan.

## Requirements

- Python 3.11+

## Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e .
```

## Usage

Run a scan:

```bash
python -m nyxcore.cli scan music --out data/reports
```

If `music` is omitted, the command defaults to `./music`.

Normalize preview (read-only):

```bash
python -m nyxcore.cli normalize music --out data/reports --strategy smart
```

Conservative artist hygiene preview:

```bash
python -m nyxcore.cli normalize music --out data/reports --strategy artist_hygiene
```

Apply plan (dry run):

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.7 --dry-run
```

Apply changes (writes title/artist/album tags only):

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.7
```

Optional backups before write:

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.7 --backup-dir data/backups
```

Album-only conservative apply:

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.6 --fields album --backup-dir data/backups
```

Staged test:

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.6 --fields album --limit 50 --backup-dir data/backups
```

Artist-only hygiene dry-run:

```bash
python -m nyxcore.cli apply music --in data/reports/normalize_preview.jsonl --min-confidence 0.9 --fields artist --backup-dir data/backups --limit 50 --dry-run
```

Phase 3 analyze (local-first with cache):

```bash
python -m nyxcore.cli analyze music --out data/reports --backend essentia --limit 0
```

Phase 3 CLAP/HYBRID (WSL/Linux):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install laion-clap
python -m nyxcore.cli analyze music --backend clap
python -m nyxcore.cli analyze music --backend hybrid
```

Phase 3 apply AI tags (NYX_* TXXX only):

```bash
python -m nyxcore.cli apply-ai music --in data/reports/analysis_preview.jsonl --fields energy,bpm,tags,genre --backup-dir data/backups --limit 50 --dry-run
```

Apply AI tags (real write):

```bash
python -m nyxcore.cli apply-ai music --in data/reports/analysis_preview.jsonl --fields energy,bpm,tags,genre
```

Phase 3 playlists:

```bash
python -m nyxcore.cli playlists music --from-cache --out data/playlists
```

## Output

`data/reports/scan.json` contains:

- `path`
- `file_size_bytes`
- `mtime_iso`
- `tags`: `title`, `artist`, `album`, `albumartist`, `tracknumber`, `date`, `genre`
- `has_cover_art`
- `duration_seconds`
- `warnings` (examples: `missing_title`, `missing_artist`, `missing_album`, `missing_cover_art`, `duration_unavailable`, `bitrate_unavailable`, `low_bitrate`, `filename_youtube_noise`, `filename_brackets_noise`, `filename_feat_pattern`, `read_error`, `tag_parse_error`, `possible_duplicate`)

`data/reports/scan.md` contains:

- Total tracks scanned
- Missing tag counts (title/artist/album)
- Cover art present vs missing
- Top 15 artists
- Top 15 albums
- First 30 problematic tracks
- "What to fix next" summary

Normalization preview outputs:

- `data/reports/normalize_preview.jsonl`
- `data/reports/normalize_preview.csv`
- `data/reports/normalize_preview.md`

Apply outputs:

- `data/reports/apply_plan.md` (for `--dry-run`)
- `data/reports/apply_log.jsonl` (when writing)

AI analysis outputs:

- `data/reports/analysis_preview.jsonl`
- `data/reports/analysis_summary.md`
- `data/reports/apply_ai_log.jsonl`
- `data/cache/analysis.sqlite`
- `.m3u` playlists under `data/playlists`

## Example CLI summary

```text
            Scan Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric            ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total tracks      │  1024 │
│ Missing title     │    31 │
│ Missing artist    │    18 │
│ Missing album     │    54 │
│ Cover art present │   873 │
│ Cover art missing │   151 │
└───────────────────┴───────┘
Wrote: data/reports/scan.json
Wrote: data/reports/scan.md
```

## Notes

- Uses `pathlib` for cross-platform path handling (Windows and Linux).
- The scanner never edits audio files.
