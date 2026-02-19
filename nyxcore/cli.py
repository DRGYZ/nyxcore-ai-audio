from __future__ import annotations

import csv
import json
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
from nyxcore.core.scanner import scan_music_folder
from nyxcore.core.track import TrackRecord
from nyxcore.core.utils import ensure_out_dir
from nyxcore.normalize.parser import NormalizePreviewRecord, build_normalize_preview
from nyxcore.normalize.rules import is_missing
from nyxcore.tagging.ai_writer import get_existing_nyx_fields, write_ai_txxx
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


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


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
    input_rows = _read_jsonl(in_)
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
        rows = _read_jsonl(in_)

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
