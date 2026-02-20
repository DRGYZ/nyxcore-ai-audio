from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.backends.dummy_backend import DummyBackend
from nyxcore.audio.backends.essentia_backend import EssentiaBackend
from nyxcore.audio.cache import AnalysisCache
from nyxcore.audio.models import AnalysisResult
from nyxcore.config import DEFAULT_CONFIG_PATH, load_config
from nyxcore.core.scanner import scan_music_folder
from nyxcore.core.track import TrackRecord
from nyxcore.core.jsonl import read_jsonl, write_jsonl
from nyxcore.core.utils import ensure_out_dir
from nyxcore.judge.service import JudgeService
from nyxcore.llm.cache import JudgeCache
from nyxcore.llm.deepseek_client import chat_json_async
from nyxcore.llm.models import JudgeResult
from nyxcore.normalize.parser import NormalizePreviewRecord, build_normalize_preview
from nyxcore.normalize.rules import is_missing
from nyxcore.rename.service import (
    apply_rename,
    iter_mp3_files,
    propose_rename_for_file,
    undo_rename,
)
from nyxcore.tagging.ai_writer import (
    get_existing_nyx_fields,
    get_existing_nyx_judge_fields,
    write_ai_txxx,
    write_judge_txxx,
)
from nyxcore.tagging.writer import backup_file, write_tags

app = typer.Typer(help="nyxcore - local-first music library auditor")
console = Console()


@app.callback()
def main() -> None:
    """nyxcore CLI entrypoint."""


def _write_scan_json(out_dir: Path, source: Path, records: list[TrackRecord], stats: dict) -> Path:
    output_path = out_dir / "scan.json"
    payload = {
        "scanned_at": datetime.now(tz=UTC).isoformat(),
        "source": str(source),
        "tracks": [r.to_dict() for r in records],
        "stats": stats,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _markdown_top_list(rows: list[tuple[str, int]], title: str) -> list[str]:
    lines = [f"## {title}", "", "| Name | Count |", "| --- | ---: |"]
    if not rows:
        lines.append("| _None_ | 0 |")
    else:
        for name, count in rows:
            safe_name = name.replace("|", "\\|")
            lines.append(f"| {safe_name} | {count} |")
    lines.append("")
    return lines


def _write_scan_md(out_dir: Path, stats: dict) -> Path:
    output_path = out_dir / "scan.md"
    lines: list[str] = [
        "# nyxcore scan report",
        "",
        "## Summary",
        "",
        f"- Total tracks scanned: **{stats['total_tracks']}**",
        f"- Missing title: **{stats['missing_title']}**",
        f"- Missing artist: **{stats['missing_artist']}**",
        f"- Missing album: **{stats['missing_album']}**",
        f"- Cover art present: **{stats['cover_art_present']}**",
        f"- Cover art missing: **{stats['cover_art_missing']}**",
        "",
    ]

    lines.extend(_markdown_top_list(stats["top_artists"], "Top 15 artists by count"))
    lines.extend(_markdown_top_list(stats["top_albums"], "Top 15 albums by count"))
    lines.extend(["## First 30 problematic tracks", "", "| Path | Warnings |", "| --- | --- |"])

    problems = stats.get("problematic_tracks_preview", [])
    if not problems:
        lines.append("| _None_ | _None_ |")
    else:
        for item in problems:
            path = str(item["path"]).replace("|", "\\|")
            warnings = ", ".join(item["warnings"])
            lines.append(f"| `{path}` | `{warnings}` |")
    lines.append("")
    lines.extend(
        [
            "## What to fix next",
            "",
            "- Fill missing `title`, `artist`, and `album` tags for tracks flagged above.",
            "- Add cover art for albums with missing artwork to improve library browsing.",
            "- Re-scan after updates and compare `scan.json` snapshots to track cleanup progress.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _summary_table(stats: dict) -> Table:
    table = Table(title="Scan Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Total tracks", str(stats["total_tracks"]))
    table.add_row("Missing title", str(stats["missing_title"]))
    table.add_row("Missing artist", str(stats["missing_artist"]))
    table.add_row("Missing album", str(stats["missing_album"]))
    table.add_row("Cover art present", str(stats["cover_art_present"]))
    table.add_row("Cover art missing", str(stats["cover_art_missing"]))
    return table


def _write_normalize_jsonl(out_dir: Path, records: list[NormalizePreviewRecord]) -> Path:
    path = out_dir / "normalize_preview.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return path


def _write_normalize_csv(out_dir: Path, records: list[NormalizePreviewRecord]) -> Path:
    path = out_dir / "normalize_preview.csv"
    fieldnames = [
        "path",
        "current_title",
        "current_artist",
        "current_album",
        "proposed_title",
        "proposed_artist",
        "proposed_album",
        "reasons",
        "confidence",
        "would_change",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = rec.to_dict()
            row["reasons"] = ",".join(rec.reasons)
            writer.writerow(row)
    return path


def _write_normalize_md(out_dir: Path, records: list[NormalizePreviewRecord]) -> Path:
    path = out_dir / "normalize_preview.md"
    total = len(records)
    changed = [r for r in records if r.would_change]
    safe = [r for r in changed if r.confidence >= 0.7]
    lines = [
        "# nyxcore normalize preview",
        "",
        "## Summary",
        "",
        f"- Total tracks evaluated: **{total}**",
        f"- Tracks with proposed changes: **{len(changed)}**",
        f"- Proposed changes with confidence >= 0.7: **{len(safe)}**",
        "",
        "## First 50 proposed changes",
        "",
        "| Path | Current | Proposed | Confidence | Reasons |",
        "| --- | --- | --- | ---: | --- |",
    ]

    if not changed:
        lines.append("| _None_ | _None_ | _None_ | 0.0 | _None_ |")
    else:
        for rec in changed[:50]:
            current = (
                f"title={rec.current_title or 'UNKNOWN'}; "
                f"artist={rec.current_artist or 'UNKNOWN'}; "
                f"album={rec.current_album or 'UNKNOWN'}"
            )
            proposed = (
                f"title={rec.proposed_title or 'UNKNOWN'}; "
                f"artist={rec.proposed_artist or 'UNKNOWN'}; "
                f"album={rec.proposed_album or 'UNKNOWN'}"
            )
            reasons = ", ".join(rec.reasons) if rec.reasons else "none"
            lines.append(
                f"| `{rec.path.replace('|', '\\|')}` | `{current}` | `{proposed}` | {rec.confidence:.2f} | `{reasons}` |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalize_summary_table(records: list[NormalizePreviewRecord]) -> Table:
    changed = [r for r in records if r.would_change]
    safe = [r for r in changed if r.confidence >= 0.7]
    table = Table(title="Normalize Preview Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Tracks evaluated", str(len(records)))
    table.add_row("Would change", str(len(changed)))
    table.add_row("Would change (>=0.7)", str(len(safe)))
    return table


def _read_preview_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _resolve_track_path(track_path: str, music_root: Path) -> Path:
    original = Path(track_path)
    if original.is_absolute():
        return original
    direct = Path.cwd() / original
    if direct.exists():
        return direct
    try:
        rel = original.relative_to(music_root.name)
        candidate = music_root / rel
        if candidate.exists():
            return candidate
    except ValueError:
        candidate = music_root / original
        if candidate.exists():
            return candidate
    return direct


def _write_apply_plan(path: Path, selected: list[dict], min_confidence: float) -> Path:
    lines = [
        "# nyxcore apply plan",
        "",
        "## Summary",
        "",
        f"- Planned updates: **{len(selected)}**",
        f"- Min confidence threshold: **{min_confidence:.2f}**",
        "",
        "## First 100 updates",
        "",
        "| Path | Proposed title | Proposed artist | Proposed album | Confidence |",
        "| --- | --- | --- | --- | ---: |",
    ]
    if not selected:
        lines.append("| _None_ | _None_ | _None_ | _None_ | 0.0 |")
    else:
        for rec in selected[:100]:
            lines.append(
                f"| `{str(rec['path']).replace('|', '\\|')}` | `{rec.get('proposed_title') or 'UNKNOWN'}` | "
                f"`{rec.get('proposed_artist') or 'UNKNOWN'}` | `{rec.get('proposed_album') or 'UNKNOWN'}` | "
                f"{float(rec.get('confidence', 0.0)):.2f} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _backend_for_name(name: str) -> AudioBackend:
    normalized = name.strip().lower()
    if normalized == "dummy":
        return DummyBackend()
    if normalized == "essentia":
        return EssentiaBackend()
    if normalized == "clap":
        from nyxcore.audio.backends.clap_backend import ClapBackend

        return ClapBackend()
    if normalized == "hybrid":
        from nyxcore.audio.backends.hybrid_backend import HybridBackend

        return HybridBackend()
    raise typer.BadParameter(f"Unknown backend: {name}. Allowed: essentia, dummy, clap, hybrid")


def _analysis_preview_jsonl_path(out_dir: Path) -> Path:
    return out_dir / "analysis_preview.jsonl"


def _write_analysis_preview_jsonl(out_dir: Path, rows: list[dict]) -> Path:
    path = _analysis_preview_jsonl_path(out_dir)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_analysis_summary_md(
    out_dir: Path,
    *,
    backend: str,
    total_tracks: int,
    analyzed_tracks: int,
    cache_hits: int,
    cache_misses: int,
    rows: list[dict],
) -> Path:
    path = out_dir / "analysis_summary.md"
    genres = Counter()
    tags = Counter()
    for row in rows:
        genre = row.get("genre_top")
        if genre:
            genres[str(genre)] += 1
        for tag in row.get("tags", []):
            tags[str(tag)] += 1
    lines = [
        "# nyxcore analysis summary",
        "",
        "## Summary",
        "",
        f"- Backend: **{backend}**",
        f"- Total tracks discovered: **{total_tracks}**",
        f"- Tracks processed: **{analyzed_tracks}**",
        f"- Cache hits: **{cache_hits}**",
        f"- Cache misses: **{cache_misses}**",
        "",
        "## Top tags",
        "",
    ]
    if not tags:
        lines.append("- _None_")
    else:
        for tag, count in tags.most_common(10):
            lines.append(f"- {tag}: {count}")
    lines.extend(["", "## Top genres", ""])
    if not genres:
        lines.append("- _None_")
    else:
        for genre, count in genres.most_common(10):
            lines.append(f"- {genre}: {count}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_judge_summary_md(
    out_dir: Path,
    *,
    provider: str,
    model: str,
    total_rows: int,
    cache_hits: int,
    cache_misses: int,
    failures: int,
    avg_total_tokens: float | None,
) -> Path:
    path = out_dir / "judge_summary.md"
    lines = [
        "# nyxcore judge summary",
        "",
        "## Summary",
        "",
        f"- Provider: **{provider}**",
        f"- Model: **{model}**",
        f"- Total rows: **{total_rows}**",
        f"- Cache hits: **{cache_hits}**",
        f"- Cache misses: **{cache_misses}**",
        f"- Failures: **{failures}**",
    ]
    if avg_total_tokens is not None:
        lines.append(f"- Avg tokens (total): **{avg_total_tokens:.2f}**")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalize_judge_write_fields(fields_csv: str | None) -> list[str]:
    allowed = {"tags", "genre", "conf", "judge", "reason"}
    if fields_csv is None or fields_csv.strip() == "":
        return ["tags", "genre", "conf", "judge", "reason"]
    out: list[str] = []
    for part in fields_csv.split(","):
        field = part.strip().lower()
        if not field:
            continue
        if field not in allowed:
            raise typer.BadParameter(
                f"Invalid --fields value: {field}. Allowed CSV values: {','.join(sorted(allowed))}"
            )
        if field not in out:
            out.append(field)
    return out or ["tags", "genre", "conf", "judge", "reason"]


def _has_judge_field_value(field: str, row: dict) -> bool:
    if field == "tags":
        tags = row.get("tags")
        return isinstance(tags, list) and any(str(t).strip() for t in tags)
    if field == "genre":
        genre = row.get("genre_top")
        return genre is not None and str(genre).strip() != ""
    if field == "conf":
        return row.get("confidence") is not None
    if field == "judge":
        v = row.get("judge_model")
        return v is not None and str(v).strip() != ""
    if field == "reason":
        v = row.get("reason")
        return v is not None and str(v).strip() != ""
    return False


def _normalize_ai_fields(fields_csv: str | None) -> list[str]:
    allowed = {"energy", "bpm", "tags", "genre"}
    if fields_csv is None or fields_csv.strip() == "":
        return ["energy", "bpm", "tags", "genre"]
    out: list[str] = []
    for part in fields_csv.split(","):
        field = part.strip().lower()
        if not field:
            continue
        if field not in allowed:
            allowed_text = ",".join(sorted(allowed))
            raise typer.BadParameter(f"Invalid --fields value: {field}. Allowed CSV values: {allowed_text}")
        if field not in out:
            out.append(field)
    if not out:
        return ["energy", "bpm", "tags", "genre"]
    return out


def _has_ai_field_value(field: str, row: dict) -> bool:
    if field == "energy":
        return row.get("energy_0_10") is not None
    if field == "bpm":
        return row.get("bpm") is not None
    if field == "tags":
        tags = row.get("tags")
        if not isinstance(tags, list):
            return False
        return any(str(tag).strip() for tag in tags)
    if field == "genre":
        genre = row.get("genre_top")
        return genre is not None and str(genre).strip() != ""
    return False


def _relative_for_playlist(track_path: Path, music_root: Path) -> str:
    track_abs = track_path.resolve()
    music_abs = music_root.resolve()
    try:
        rel = track_abs.relative_to(music_abs)
        return rel.as_posix()
    except ValueError:
        return track_path.name


def _normalize_fields(fields: list[str] | None) -> list[str]:
    allowed = {"title", "artist", "album"}
    if fields is None:
        return ["title", "artist", "album"]

    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        value = str(field).strip().lower()
        if value not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise typer.BadParameter(f"Invalid --fields value: {field}. Allowed values: {allowed_text}")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _safe_console_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


@app.command()
def scan(
    music: Path = typer.Argument(Path("./music"), help="Folder to scan recursively"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for reports"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Scanning MP3 files", total=1)

        def on_progress(done: int, total: int, _: Path) -> None:
            if progress.tasks[0].total != total:
                progress.update(task, total=total)
            progress.update(task, completed=done)

        records, stats = scan_music_folder(music, on_progress=on_progress)
        if not records:
            progress.update(task, total=1, completed=1)

    json_path = _write_scan_json(out, music, records, stats)
    md_path = _write_scan_md(out, stats)
    console.print(_summary_table(stats))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command()
def normalize(
    music: Path = typer.Argument(Path("./music"), help="Folder to scan recursively"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for normalize reports"),
    strategy: str = typer.Option("smart", "--strategy", help="Album strategy"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)
    records = build_normalize_preview(music, strategy=strategy)

    jsonl_path = _write_normalize_jsonl(out, records)
    csv_path = _write_normalize_csv(out, records)
    md_path = _write_normalize_md(out, records)

    console.print(_normalize_summary_table(records))
    console.print(f"[green]Wrote:[/green] {jsonl_path}")
    console.print(f"[green]Wrote:[/green] {csv_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command()
def apply(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    in_: Path = typer.Option(..., "--in", help="Path to normalize_preview.jsonl"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", help="Minimum confidence"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write tags; only plan"),
    backup_dir: Path | None = typer.Option(None, "--backup-dir", help="Optional backup directory"),
    fields: list[str] | None = typer.Option(
        None,
        "--fields",
        help="Fields to write. Allowed: title, artist, album. Can be repeated.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Limit number of files to process (for safe staged apply).",
    ),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if not in_.exists():
        raise typer.BadParameter(f"Input preview file does not exist: {in_}")
    if limit is not None and limit < 0:
        raise typer.BadParameter("--limit must be >= 0")

    selected_fields = _normalize_fields(fields)

    input_rows = _read_preview_jsonl(in_)
    skipped_no_change = 0
    skipped_low_confidence = 0
    selected: list[dict] = []
    for row in input_rows:
        if not bool(row.get("would_change")):
            skipped_no_change += 1
            continue
        if float(row.get("confidence", 0.0)) < min_confidence:
            skipped_low_confidence += 1
            continue
        selected.append(row)

    total_selected = len(selected)
    dry_run_selected = selected[:limit] if limit is not None else selected
    total_limited = max(0, total_selected - len(dry_run_selected)) if limit is not None else 0
    reports_dir = in_.parent
    ensure_out_dir(reports_dir)
    plan_path = reports_dir / "apply_plan.md"

    if dry_run:
        _write_apply_plan(plan_path, dry_run_selected, min_confidence)
        table = Table(title="Apply Dry Run Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        table.add_row("Preview rows", str(len(input_rows)))
        table.add_row("total_selected", str(total_selected))
        table.add_row("total_skipped_low_confidence", str(skipped_low_confidence))
        table.add_row("total_skipped_no_change", str(skipped_no_change))
        if limit is not None:
            table.add_row("total_limited", str(total_limited))
        table.add_row("Min confidence", f"{min_confidence:.2f}")
        table.add_row("Fields", ", ".join(selected_fields))
        console.print(table)
        console.print(f"[green]Wrote:[/green] {plan_path}")
        return

    log_path = reports_dir / "apply_log.jsonl"
    success = 0
    failed = 0
    attempted = 0

    for row in selected:
        if limit is not None and success >= limit:
            break
        target = _resolve_track_path(str(row.get("path", "")), music)
        log_row = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "path": str(target),
            "status": "ok",
            "confidence": float(row.get("confidence", 0.0)),
            "fields": selected_fields,
        }
        try:
            if backup_dir is not None:
                backup_target = backup_file(target, backup_dir)
                log_row["backup"] = str(backup_target)

            title = row.get("proposed_title")
            artist = row.get("proposed_artist")
            album = row.get("proposed_album")
            write_tags(
                target,
                title=None if is_missing(title) else str(title),
                artist=None if is_missing(artist) else str(artist),
                album=None if is_missing(album) else str(album),
                fields=selected_fields,
            )
            success += 1
        except Exception as exc:
            log_row["status"] = "error"
            log_row["error"] = str(exc)
            failed += 1
        attempted += 1
        _append_jsonl(log_path, log_row)

    table = Table(title="Apply Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Preview rows", str(len(input_rows)))
    table.add_row("Selected for apply", str(total_selected))
    table.add_row("Skipped no change", str(skipped_no_change))
    table.add_row("Skipped low confidence", str(skipped_low_confidence))
    if limit is not None:
        table.add_row("Limit", str(limit))
        table.add_row("Limited (not attempted)", str(max(0, total_selected - attempted)))
    table.add_row("Fields", ", ".join(selected_fields))
    table.add_row("Succeeded", str(success))
    table.add_row("Failed", str(failed))
    table.add_row("Log file", str(log_path))
    console.print(table)
    console.print(f"[green]Wrote:[/green] {log_path}")


@app.command("rename")
def rename_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Reports folder for rename map"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview only by default"),
    limit: int = typer.Option(0, "--limit", help="Limit number of files to evaluate (0 means all)"),
    concurrency: int = typer.Option(10, "--concurrency", help="Concurrent LLM rename requests (1-20)"),
    force: bool = typer.Option(False, "--force", help="Force LLM refinement even when deterministic output is clean"),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Allow optional DeepSeek refinement for messy names"),
    model: str = typer.Option("deepseek-chat", "--model", help="DeepSeek model for optional LLM cleanup"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if limit < 0:
        raise typer.BadParameter("--limit must be >= 0")
    if concurrency < 1 or concurrency > 20:
        raise typer.BadParameter("--concurrency must be between 1 and 20")

    ensure_out_dir(out)
    files = sorted(iter_mp3_files(music))
    if limit > 0:
        files = files[:limit]

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("NYX_DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("NYX_DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if llm and not api_key:
        console.print("[yellow]LLM disabled:[/yellow] no DEEPSEEK_API_KEY found; using deterministic rename only.")
        llm = False

    async def _run() -> list:
        sem = asyncio.Semaphore(concurrency)
        results: list = []
        if llm:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=45.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [
                    asyncio.create_task(
                        propose_rename_for_file(
                            p,
                            use_llm=llm,
                            force=force,
                            sem=sem,
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            max_retries=3,
                            session=session,
                        )
                    )
                    for p in files
                ]
                for fut in asyncio.as_completed(tasks):
                    results.append(await fut)
        else:
            tasks = [
                asyncio.create_task(
                    propose_rename_for_file(
                        p,
                        use_llm=False,
                        force=force,
                        sem=sem,
                        api_key=None,
                        base_url=base_url,
                        model=model,
                        max_retries=3,
                        session=None,
                    )
                )
                for p in files
            ]
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
        return sorted(results, key=lambda r: str(r.old_path))

    results = asyncio.run(_run())
    changed = [r for r in results if r.changed]

    table = Table(title="Rename Preview")
    table.add_column("Old", style="cyan")
    table.add_column("New", style="magenta")
    shown = 0
    for res in changed[:80]:
        table.add_row(_safe_console_text(str(res.old_path)), _safe_console_text(str(res.new_path)))
        shown += 1
    if shown == 0:
        table.add_row("_No changes_", "_No changes_")
    console.print(table)

    rename_map_path = out / "rename_map.jsonl"
    map_rows = [
        {
            "old_path": str(r.old_path),
            "new_path": str(r.new_path),
            "ts": r.ts,
            "rule_notes": r.rule_notes,
            "llm_used": r.llm_used,
        }
        for r in changed
    ]

    if dry_run:
        console.print(f"[green]Preview complete.[/green] Candidates: {len(changed)}")
        return

    applied = 0
    failed = 0
    for res in changed:
        try:
            apply_rename(res)
            applied += 1
        except Exception:
            failed += 1

    write_jsonl(rename_map_path, map_rows)
    summary = Table(title="Rename Apply Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="magenta")
    summary.add_row("Scanned", str(len(results)))
    summary.add_row("Changed", str(len(changed)))
    summary.add_row("Applied", str(applied))
    summary.add_row("Failed", str(failed))
    summary.add_row("Map", str(rename_map_path))
    console.print(summary)
    console.print(f"[green]Wrote:[/green] {rename_map_path}")


@app.command("rename-undo")
def rename_undo(
    map_path: Path = typer.Option(Path("data/reports/rename_map.jsonl"), "--map", help="Path to rename map"),
    limit: int = typer.Option(0, "--limit", help="Limit number of undo operations (0 means all)"),
    force: bool = typer.Option(False, "--force", help="Force undo when old_path already exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview undo actions only"),
) -> None:
    if not map_path.exists():
        raise typer.BadParameter(f"Rename map does not exist: {map_path}")
    if limit < 0:
        raise typer.BadParameter("--limit must be >= 0")

    rows = list(read_jsonl(map_path))
    rows = list(reversed(rows))
    if limit > 0:
        rows = rows[:limit]

    table = Table(title="Rename Undo Preview")
    table.add_column("Current", style="cyan")
    table.add_column("Restore To", style="magenta")
    for row in rows[:80]:
        table.add_row(
            _safe_console_text(str(row.get("new_path", ""))),
            _safe_console_text(str(row.get("old_path", ""))),
        )
    if not rows:
        table.add_row("_No entries_", "_No entries_")
    console.print(table)

    if dry_run:
        return

    restored = 0
    skipped = 0
    failed = 0
    for row in rows:
        old_path = Path(str(row.get("old_path", "")))
        new_path = Path(str(row.get("new_path", "")))
        try:
            ok, status = undo_rename(old_path, new_path, force=force)
            if ok:
                restored += 1
            else:
                skipped += 1
                _ = status
        except Exception:
            failed += 1

    summary = Table(title="Rename Undo Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="magenta")
    summary.add_row("Entries", str(len(rows)))
    summary.add_row("Restored", str(restored))
    summary.add_row("Skipped", str(skipped))
    summary.add_row("Failed", str(failed))
    console.print(summary)


@app.command("analyze")
def analyze(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for analysis reports"),
    backend: str = typer.Option("essentia", "--backend", help="Analysis backend: essentia, dummy, clap, or hybrid"),
    limit: int = typer.Option(0, "--limit", help="Limit number of tracks to analyze (0 means all)"),
    force: bool = typer.Option(False, "--force", help="Ignore cache and recompute analysis"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if limit < 0:
        raise typer.BadParameter("--limit must be >= 0")

    try:
        backend_impl = _backend_for_name(backend)
    except RuntimeError as exc:
        console.print(Panel(str(exc), title="Backend Error", border_style="red"))
        raise typer.Exit(code=1) from exc

    ensure_out_dir(out)
    cache = AnalysisCache(Path("data/cache/analysis.sqlite"))

    records, _ = scan_music_folder(music)
    if limit > 0:
        records = records[:limit]

    rows: list[dict] = []
    cache_hits = 0
    cache_misses = 0
    clap_failed_count = 0
    essentia_failed_count = 0
    partial_results = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    try:
        with progress:
            task = progress.add_task("Analyzing tracks", total=max(1, len(records)))
            for idx, rec in enumerate(records, start=1):
                track_path = _resolve_track_path(rec.path, music)
                cached: AnalysisResult | None = None
                if not force:
                    cached = cache.get(
                        path=rec.path,
                        file_size_bytes=rec.file_size_bytes,
                        mtime_iso=rec.mtime_iso,
                        backend=backend_impl.name,
                    )

                if cached is not None:
                    result = cached
                    cache_hits += 1
                else:
                    try:
                        result = backend_impl.analyze_track(track_path)
                        cache.set(
                            path=rec.path,
                            file_size_bytes=rec.file_size_bytes,
                            mtime_iso=rec.mtime_iso,
                            result=result,
                        )
                        cache_misses += 1
                    except Exception as exc:
                        rows.append(
                            {
                                "path": rec.path,
                                "file_size_bytes": rec.file_size_bytes,
                                "mtime_iso": rec.mtime_iso,
                                "energy_0_10": None,
                                "bpm": None,
                                "tags": [],
                                "genre_top": None,
                                "backend": backend_impl.name,
                                "created_at_iso": datetime.now(tz=UTC).isoformat(),
                                "confidence": None,
                                "error": str(exc),
                                "errors": [str(exc)],
                            }
                        )
                        progress.update(task, completed=idx)
                        continue

                rows.append(
                    {
                        "path": rec.path,
                        "file_size_bytes": rec.file_size_bytes,
                        "mtime_iso": rec.mtime_iso,
                        "energy_0_10": result.energy_0_10,
                        "bpm": result.bpm,
                        "tags": result.tags,
                        "genre_top": result.genre_top,
                        "backend": result.backend,
                        "created_at_iso": result.created_at_iso,
                        "confidence": result.confidence,
                        "errors": result.errors,
                    }
                )
                if result.errors:
                    partial_results += 1
                    has_clap_error = False
                    has_essentia_error = False
                    for err in result.errors:
                        lower = err.lower()
                        if lower.startswith("clap"):
                            has_clap_error = True
                        if lower.startswith("essentia"):
                            has_essentia_error = True
                    if has_clap_error:
                        clap_failed_count += 1
                    if has_essentia_error:
                        essentia_failed_count += 1
                progress.update(task, completed=idx)
    finally:
        cache.close()

    preview_path = _write_analysis_preview_jsonl(out, rows)
    summary_path = _write_analysis_summary_md(
        out,
        backend=backend_impl.name,
        total_tracks=len(records),
        analyzed_tracks=len(rows),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        rows=rows,
    )

    table = Table(title="Analyze Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Backend", backend_impl.name)
    table.add_row("Tracks processed", str(len(rows)))
    table.add_row("Cache hits", str(cache_hits))
    table.add_row("Cache misses", str(cache_misses))
    if backend_impl.name == "hybrid":
        table.add_row("Partial results", str(partial_results))
        table.add_row("clap_failed_count", str(clap_failed_count))
        table.add_row("essentia_failed_count", str(essentia_failed_count))
    console.print(table)
    console.print(f"[green]Wrote:[/green] {preview_path}")
    console.print(f"[green]Wrote:[/green] {summary_path}")


@app.command("judge")
def judge(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    analysis: Path = typer.Option(..., "--analysis", help="Path to analysis_preview.jsonl"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for judge reports"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to YAML config"),
    provider: str = typer.Option("deepseek", "--provider", help="LLM provider"),
    model: str = typer.Option(
        "deepseek-chat",
        "--model",
        help="LLM model (recommended: deepseek-chat or deepseek-reasoner)",
    ),
    limit: int = typer.Option(0, "--limit", help="Limit number of rows to process (0 means all)"),
    concurrency: int | None = typer.Option(None, "--concurrency", help="Concurrent LLM requests (1-20)"),
    force: bool = typer.Option(False, "--force", help="Ignore judge cache and re-request LLM"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if not analysis.exists():
        raise typer.BadParameter(f"Analysis file does not exist: {analysis}")
    if provider.lower() != "deepseek":
        raise typer.BadParameter("Only provider=deepseek is currently supported")
    if model not in {"deepseek-chat", "deepseek-reasoner"}:
        raise typer.BadParameter("Use model deepseek-chat or deepseek-reasoner for DeepSeek judge")
    if limit < 0:
        raise typer.BadParameter("--limit must be >= 0")
    if not config.exists():
        raise typer.BadParameter(f"Config file does not exist: {config}")

    try:
        app_config = load_config(config)
    except Exception as exc:
        raise typer.BadParameter(f"Failed to load config: {exc}") from exc

    if concurrency is None:
        concurrency = app_config.judge.concurrency_default
    if concurrency < 1 or concurrency > 20:
        raise typer.BadParameter("--concurrency must be between 1 and 20")

    ensure_out_dir(out)
    input_rows = list(read_jsonl(analysis))
    if limit > 0:
        input_rows = input_rows[:limit]
    judge_service = JudgeService(
        config=app_config.judge,
        llm_client=chat_json_async,
    )
    judge_prompt_version = app_config.judge.prompt_version

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("NYX_DEEPSEEK_API_KEY")
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("NYX_DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    if not api_key:
        raise typer.BadParameter(
            "DEEPSEEK_API_KEY is required for judge command (NYX_DEEPSEEK_API_KEY is also accepted)"
        )

    cache = JudgeCache(Path("data/cache/judge.sqlite"))
    rows: list[dict] = []
    cache_hits = 0
    cache_misses = 0
    failures = 0
    total_tokens_list: list[int] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    async def _run_async() -> list[dict]:
        import aiohttp

        nonlocal cache_hits, cache_misses, failures, total_tokens_list
        system_prompt = judge_service.build_system_prompt()
        sem = asyncio.Semaphore(concurrency)
        ordered_rows: list[dict | None] = [None] * len(input_rows)
        timeout = aiohttp.ClientTimeout(total=45.0)

        async def _process_one(index: int, row: dict, session: aiohttp.ClientSession) -> tuple[int, dict]:
            nonlocal cache_hits, cache_misses, failures
            path = str(row.get("path", ""))
            size = int(row.get("file_size_bytes", 0) or 0)
            mtime = str(row.get("mtime_iso", ""))

            cached = None
            if not force:
                cached = await asyncio.to_thread(
                    cache.get,
                    path=path,
                    file_size_bytes=size,
                    mtime_iso=mtime,
                    model=model,
                    prompt_version=judge_prompt_version,
                )

            if cached is not None:
                cache_hits += 1
                result = cached
                source_genre = judge_service.source_genre_from_row(row)
                eval_genre = source_genre or judge_service.canonicalize_genre(result.genre_top)
                conflicts_local, bpm_note, _filename_signal = judge_service.compute_conflicts_local(row, eval_genre)
                conflicts_llm = None
            else:
                cache_misses += 1
                try:
                    async with sem:
                        response = await judge_service.call_llm_async(
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=judge_service.build_user_prompt(row),
                            temperature=0.0,
                            max_retries=3,
                            session=session,
                        )
                    sanitized = judge_service.sanitize_llm_response(response.data, row)
                    tags = sanitized["tags"]
                    genre_top = sanitized["genre_top"]
                    confidence = sanitized["confidence"]
                    reason = sanitized["reason"]
                    bpm_note = sanitized["bpm_note"]
                    conflicts_local = sanitized["conflicts_local"]
                    conflicts_llm = sanitized["conflicts_llm"]
                    result = JudgeResult(
                        tags=tags,
                        genre_top=genre_top,
                        confidence=confidence,
                        reason=reason,
                        provider=provider,
                        model=model,
                        prompt_version=judge_prompt_version,
                        usage_prompt_tokens=response.usage.get("prompt_tokens"),
                        usage_completion_tokens=response.usage.get("completion_tokens"),
                        usage_total_tokens=response.usage.get("total_tokens"),
                    )
                except Exception as exc:
                    failures += 1
                    source_genre = judge_service.source_genre_from_row(row)
                    conflicts_local, bpm_note, _filename_signal = judge_service.compute_conflicts_local(row, source_genre)
                    conflicts_llm = None
                    result = JudgeResult(
                        tags=[],
                        genre_top=None,
                        confidence=None,
                        reason="",
                        provider=provider,
                        model=model,
                        prompt_version=judge_prompt_version,
                        errors=[f"judge_error: {exc}"],
                    )
                await asyncio.to_thread(
                    cache.set,
                    path=path,
                    file_size_bytes=size,
                    mtime_iso=mtime,
                    model=model,
                    prompt_version=judge_prompt_version,
                    result=result,
                )

            if result.usage_total_tokens is not None:
                total_tokens_list.append(int(result.usage_total_tokens))

            row_out = {
                "path": path,
                "file_size_bytes": size,
                "mtime_iso": mtime,
                "source_backend": row.get("backend"),
                "source_energy_0_10": row.get("energy_0_10"),
                "source_bpm": row.get("bpm"),
                "source_tags": row.get("tags", []),
                "source_genre_top": row.get("genre_top"),
                "tags": result.tags,
                "genre_top": result.genre_top,
                "confidence": result.confidence,
                "reason": result.reason,
                "bpm_note": bpm_note,
                "conflicts": conflicts_local,
                "conflicts_local": conflicts_local,
                "conflicts_llm": conflicts_llm,
                "judge_provider": result.provider,
                "judge_model": result.model,
                "prompt_version": result.prompt_version,
                "created_at_iso": result.created_at_iso,
                "errors": result.errors,
                "usage_prompt_tokens": result.usage_prompt_tokens,
                "usage_completion_tokens": result.usage_completion_tokens,
                "usage_total_tokens": result.usage_total_tokens,
            }
            return index, row_out

        with progress:
            task = progress.add_task("LLM judging tracks", total=max(1, len(input_rows)))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [asyncio.create_task(_process_one(i, row, session)) for i, row in enumerate(input_rows)]
                for fut in asyncio.as_completed(tasks):
                    idx, row_out = await fut
                    ordered_rows[idx] = row_out
                    progress.update(task, advance=1)

        return [row for row in ordered_rows if row is not None]

    try:
        rows = asyncio.run(_run_async())
    finally:
        cache.close()

    preview_path = out / "judge_preview.jsonl"
    write_jsonl(preview_path, rows)
    avg_tokens = None if not total_tokens_list else sum(total_tokens_list) / len(total_tokens_list)
    summary_path = _write_judge_summary_md(
        out,
        provider=provider,
        model=model,
        total_rows=len(rows),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        failures=failures,
        avg_total_tokens=avg_tokens,
    )

    table = Table(title="Judge Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Provider", provider)
    table.add_row("Model", model)
    table.add_row("Rows", str(len(rows)))
    table.add_row("Cache hits", str(cache_hits))
    table.add_row("Cache misses", str(cache_misses))
    table.add_row("Failures", str(failures))
    if avg_tokens is not None:
        table.add_row("Avg total tokens", f"{avg_tokens:.2f}")
    console.print(table)
    console.print(f"[green]Wrote:[/green] {preview_path}")
    console.print(f"[green]Wrote:[/green] {summary_path}")


@app.command("apply-ai")
def apply_ai(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    in_: Path = typer.Option(..., "--in", help="Path to analysis_preview.jsonl"),
    fields: str | None = typer.Option(
        None,
        "--fields",
        help="CSV fields to write: energy,bpm,tags,genre (default all)",
    ),
    min_confidence: float | None = typer.Option(None, "--min-confidence", help="Optional minimum confidence"),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of files to process"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write tags"),
    backup_dir: Path | None = typer.Option(None, "--backup-dir", help="Optional backup directory"),
    force: bool = typer.Option(False, "--force", help="Write even when NYX_* TXXX fields already exist"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if not in_.exists():
        raise typer.BadParameter(f"Input preview file does not exist: {in_}")
    if limit is not None and limit < 0:
        raise typer.BadParameter("--limit must be >= 0")

    selected_fields = _normalize_ai_fields(fields)
    input_rows = list(read_jsonl(in_))
    reports_dir = in_.parent
    ensure_out_dir(reports_dir)
    log_path = reports_dir / "apply_ai_log.jsonl"

    selected: list[dict] = []
    skipped_confidence = 0
    for row in input_rows:
        conf = row.get("confidence")
        if min_confidence is not None and conf is not None and float(conf) < min_confidence:
            skipped_confidence += 1
            continue
        selected.append(row)

    succeeded = 0
    failed = 0
    skipped_existing = 0
    skipped_empty_field = 0
    checked = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Applying AI tags", total=max(1, len(selected)))
        for idx, row in enumerate(selected, start=1):
            if limit is not None and succeeded >= limit:
                break

            target = _resolve_track_path(str(row.get("path", "")), music)
            log_row = {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "path": str(target),
                "fields": selected_fields,
                "status": "ok",
                "dry_run": dry_run,
            }
            try:
                existing = get_existing_nyx_fields(target)
                empty_fields = [field for field in selected_fields if not _has_ai_field_value(field, row)]
                if empty_fields:
                    skipped_empty_field += 1
                    log_row["empty_fields"] = empty_fields

                effective = [field for field in selected_fields if field not in empty_fields]
                if not force:
                    effective = [field for field in effective if field not in existing]

                if not effective:
                    if empty_fields:
                        log_row["status"] = "skipped_empty_field"
                    else:
                        skipped_existing += 1
                        log_row["status"] = "skipped_existing"
                    _append_jsonl(log_path, log_row)
                    progress.update(task, completed=idx)
                    continue

                if dry_run:
                    log_row["status"] = "dry_run"
                    log_row["would_write_fields"] = effective
                    succeeded += 1
                else:
                    if backup_dir is not None:
                        backup_target = backup_file(target, backup_dir)
                        log_row["backup"] = str(backup_target)

                    written, skipped = write_ai_txxx(
                        target,
                        energy=None
                        if row.get("energy_0_10") is None
                        else float(row.get("energy_0_10")),
                        bpm=None if row.get("bpm") is None else float(row.get("bpm")),
                        tags=[str(t) for t in row.get("tags", [])],
                        genre_top=None if row.get("genre_top") is None else str(row.get("genre_top")),
                        fields=effective,
                        force=force,
                    )
                    log_row["written_fields"] = written
                    log_row["skipped_existing_fields"] = skipped
                    succeeded += 1
                checked += 1
            except Exception as exc:
                log_row["status"] = "error"
                log_row["error"] = str(exc)
                failed += 1
            _append_jsonl(log_path, log_row)
            progress.update(task, completed=idx)

    table = Table(title="Apply-AI Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Preview rows", str(len(input_rows)))
    table.add_row("Selected", str(len(selected)))
    table.add_row("Skipped by confidence", str(skipped_confidence))
    table.add_row("Skipped existing", str(skipped_existing))
    table.add_row("Skipped empty field", str(skipped_empty_field))
    table.add_row("Processed", str(checked))
    table.add_row("Succeeded", str(succeeded))
    table.add_row("Failed", str(failed))
    table.add_row("Fields", ",".join(selected_fields))
    table.add_row("Log file", str(log_path))
    console.print(table)
    console.print(f"[green]Wrote:[/green] {log_path}")


@app.command("apply-judge")
def apply_judge(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    in_: Path = typer.Option(..., "--in", help="Path to judge_preview.jsonl"),
    fields: str | None = typer.Option(
        None,
        "--fields",
        help="CSV fields to write: tags,genre,conf,judge,reason (default all)",
    ),
    min_confidence: float | None = typer.Option(None, "--min-confidence", help="Optional minimum confidence"),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of files to process"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write tags"),
    backup_dir: Path | None = typer.Option(None, "--backup-dir", help="Optional backup directory"),
    force: bool = typer.Option(False, "--force", help="Write even when NYX_* fields already exist"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if not in_.exists():
        raise typer.BadParameter(f"Input judge preview file does not exist: {in_}")
    if limit is not None and limit < 0:
        raise typer.BadParameter("--limit must be >= 0")

    selected_fields = _normalize_judge_write_fields(fields)
    input_rows = list(read_jsonl(in_))
    reports_dir = in_.parent
    ensure_out_dir(reports_dir)
    log_path = reports_dir / "apply_judge_log.jsonl"

    selected: list[dict] = []
    skipped_confidence = 0
    for row in input_rows:
        conf = row.get("confidence")
        if min_confidence is not None and conf is not None and float(conf) < min_confidence:
            skipped_confidence += 1
            continue
        selected.append(row)

    succeeded = 0
    failed = 0
    skipped_existing = 0
    skipped_empty_field = 0
    checked = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Applying judge tags", total=max(1, len(selected)))
        for idx, row in enumerate(selected, start=1):
            if limit is not None and succeeded >= limit:
                break
            target = _resolve_track_path(str(row.get("path", "")), music)
            log_row = {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "path": str(target),
                "fields": selected_fields,
                "status": "ok",
                "dry_run": dry_run,
            }
            try:
                existing = get_existing_nyx_judge_fields(target)
                empty_fields = [field for field in selected_fields if not _has_judge_field_value(field, row)]
                if empty_fields:
                    skipped_empty_field += 1
                    log_row["empty_fields"] = empty_fields

                effective = [field for field in selected_fields if field not in empty_fields]
                if not force:
                    effective = [field for field in effective if field not in existing]

                if not effective:
                    if empty_fields:
                        log_row["status"] = "skipped_empty_field"
                    else:
                        skipped_existing += 1
                        log_row["status"] = "skipped_existing"
                    _append_jsonl(log_path, log_row)
                    progress.update(task, completed=idx)
                    continue

                if dry_run:
                    log_row["status"] = "dry_run"
                    log_row["would_write_fields"] = effective
                    succeeded += 1
                else:
                    if backup_dir is not None:
                        backup_target = backup_file(target, backup_dir)
                        log_row["backup"] = str(backup_target)

                    written, skipped = write_judge_txxx(
                        target,
                        tags=[str(t) for t in row.get("tags", [])],
                        genre_top=None if row.get("genre_top") is None else str(row.get("genre_top")),
                        conf=None if row.get("confidence") is None else float(row.get("confidence")),
                        judge=None if row.get("judge_model") is None else str(row.get("judge_model")),
                        reason=None if row.get("reason") is None else str(row.get("reason")),
                        fields=effective,
                        force=force,
                    )
                    log_row["written_fields"] = written
                    log_row["skipped_existing_fields"] = skipped
                    succeeded += 1
                checked += 1
            except Exception as exc:
                log_row["status"] = "error"
                log_row["error"] = str(exc)
                failed += 1
            _append_jsonl(log_path, log_row)
            progress.update(task, completed=idx)

    table = Table(title="Apply-Judge Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Preview rows", str(len(input_rows)))
    table.add_row("Selected", str(len(selected)))
    table.add_row("Skipped by confidence", str(skipped_confidence))
    table.add_row("Skipped existing", str(skipped_existing))
    table.add_row("Skipped empty field", str(skipped_empty_field))
    table.add_row("Processed", str(checked))
    table.add_row("Succeeded", str(succeeded))
    table.add_row("Failed", str(failed))
    table.add_row("Fields", ",".join(selected_fields))
    table.add_row("Log file", str(log_path))
    console.print(table)
    console.print(f"[green]Wrote:[/green] {log_path}")


@app.command()
def playlists(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    from_cache: bool = typer.Option(False, "--from-cache", help="Read from sqlite cache"),
    out: Path = typer.Option(Path("data/playlists"), "--out", help="Output playlist folder"),
    in_: Path = typer.Option(
        Path("data/reports/analysis_preview.jsonl"),
        "--in",
        help="Input analysis preview jsonl (used if --from-cache is not set)",
    ),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)

    rows: list[dict]
    if from_cache:
        cache = AnalysisCache(Path("data/cache/analysis.sqlite"))
        try:
            rows = cache.rows()
        finally:
            cache.close()
    else:
        if not in_.exists():
            raise typer.BadParameter(f"Input analysis file does not exist: {in_}")
        rows = list(read_jsonl(in_))

    buckets: dict[str, list[str]] = {
        "energy_8_10.m3u": [],
        "energy_5_7.m3u": [],
        "energy_0_4.m3u": [],
        "bpm_120_140.m3u": [],
        "mood_dark.m3u": [],
        "mood_hypnotic.m3u": [],
    }

    for row in rows:
        energy = row.get("energy_0_10")
        bpm = row.get("bpm")
        tags = [str(t).lower() for t in row.get("tags", [])]
        track = _resolve_track_path(str(row.get("path", "")), music)
        rel = _relative_for_playlist(track, music)

        if energy is not None:
            e = float(energy)
            if e >= 8.0:
                buckets["energy_8_10.m3u"].append(rel)
            elif e >= 5.0:
                buckets["energy_5_7.m3u"].append(rel)
            else:
                buckets["energy_0_4.m3u"].append(rel)

        if bpm is not None:
            b = float(bpm)
            if 120.0 <= b <= 140.0:
                buckets["bpm_120_140.m3u"].append(rel)

        if "dark" in tags:
            buckets["mood_dark.m3u"].append(rel)
        if "hypnotic" in tags:
            buckets["mood_hypnotic.m3u"].append(rel)

    for name, entries in buckets.items():
        playlist_path = out / name
        lines = ["#EXTM3U"] + entries
        playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = Table(title="Playlists Summary")
    table.add_column("Playlist", style="cyan")
    table.add_column("Tracks", justify="right", style="magenta")
    for name, entries in buckets.items():
        table.add_row(name, str(len(entries)))
    console.print(table)
    console.print(f"[green]Wrote playlists to:[/green] {out}")


@app.command("debug-clap")
def debug_clap() -> None:
    import sys

    from nyxcore.audio.backends.clap_backend import clap_import_diagnostics

    report = clap_import_diagnostics()
    console.print(f"python_executable: {sys.executable}")
    console.print(f"torch_ok: {report['torch_ok']}")
    console.print(f"torch_version: {report['torch_version']}")
    if report.get("torch_error"):
        console.print(f"torch_error: {report['torch_error']}")
    console.print(f"torchaudio_ok: {report['torchaudio_ok']}")
    console.print(f"torchaudio_version: {report['torchaudio_version']}")
    if report.get("torchaudio_error"):
        console.print(f"torchaudio_error: {report['torchaudio_error']}")

    attempts = report.get("clap_attempts", [])
    for name, ok, detail in attempts:
        status = "success" if ok else "failure"
        console.print(f"{name}: {status}")
        if not ok:
            console.print(f"  error: {detail}")


if __name__ == "__main__":
    app()
