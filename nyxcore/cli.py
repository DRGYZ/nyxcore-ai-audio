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

from nyxcore.action_plan.service import (
    ActionPlanReport,
    AppliedPlanResult,
    apply_action_plan_report,
    build_action_plan_report,
)
from nyxcore.action_plan.ledger import (
    OperationBatch,
    OperationLedger,
    append_operation_batch,
    find_batch,
    load_operation_ledger,
    save_operation_ledger,
    undo_operation_batch,
)
from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.backends.dummy_backend import DummyBackend
from nyxcore.audio.cache import AnalysisCache
from nyxcore.audio.models import AnalysisResult
from nyxcore.config import NyxConfig, load_config
from nyxcore.core.scanner import scan_music_folder
from nyxcore.core.track import TrackRecord
from nyxcore.core.jsonl import read_jsonl, write_jsonl
from nyxcore.core.utils import compute_stats, ensure_out_dir
from nyxcore.duplicates.service import DuplicateAnalysisReport
from nyxcore.health.service import HealthReport
from nyxcore.incremental.service import ChangeSet, RefreshSummary, refresh_incremental_state, watch_incremental_state
from nyxcore.judge.service import JudgeService
from nyxcore.llm.cache import JudgeCache
from nyxcore.llm.deepseek_client import chat_json_async
from nyxcore.llm.models import JudgeResult
from nyxcore.normalize.parser import NormalizePreviewRecord, build_normalize_preview
from nyxcore.normalize.rules import is_missing
from nyxcore.playlist_query.service import PlaylistReport, build_playlist_report
from nyxcore.report_pipeline import build_duplicate_health_reports, build_duplicate_report, build_review_pipeline
from nyxcore.review_queue.service import REVIEW_ITEM_TYPES, ReviewQueueReport
from nyxcore.review_queue.state import (
    ReviewStateStore,
    apply_review_action,
    load_review_state,
    save_review_state,
)
from nyxcore.saved_playlists.service import (
    SavedPlaylistDefinition,
    SavedPlaylistLatestResult,
    create_saved_playlist_definition,
    delete_saved_playlist_definition,
    edit_saved_playlist_definition,
    export_saved_playlist_json,
    export_saved_playlist_m3u,
    load_saved_playlist_store,
    read_saved_playlist_latest_result,
    rename_saved_playlist_definition,
    refresh_saved_playlist,
    save_saved_playlist_definition,
)
from nyxcore.rename.service import (
    apply_rename,
    iter_library_audio_files,
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

app = typer.Typer(help="nyxcore - local-first music library review and cleanup toolkit")
console = Console()


@app.callback()
def main() -> None:
    """nyxcore CLI entrypoint."""


def _write_scan_json(
    out_dir: Path,
    source: Path,
    records: list[TrackRecord],
    stats: dict,
    *,
    refresh_summary: RefreshSummary | None = None,
) -> Path:
    output_path = out_dir / "scan.json"
    payload = {
        "scanned_at": datetime.now(tz=UTC).isoformat(),
        "source": str(source),
        "tracks": [r.to_dict() for r in records],
        "stats": stats,
    }
    if refresh_summary is not None:
        payload["refresh"] = refresh_summary.to_dict()
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


def _write_scan_md(out_dir: Path, stats: dict, *, refresh_summary: RefreshSummary | None = None) -> Path:
    output_path = out_dir / "scan.md"
    lines: list[str] = ["# nyxcore scan report", "", "## Summary", ""]
    if refresh_summary is not None:
        lines.extend(
            [
                f"- Refresh mode: **{refresh_summary.mode}**",
                f"- Added files: **{len(refresh_summary.changes.added_files)}**",
                f"- Modified files: **{len(refresh_summary.changes.modified_files)}**",
                f"- Removed files: **{len(refresh_summary.changes.removed_files)}**",
                f"- Unchanged files reused: **{len(refresh_summary.changes.unchanged_files)}**",
                "",
            ]
        )
    lines.extend(
        [
            f"- Total tracks scanned: **{stats['total_tracks']}**",
            f"- Missing title: **{stats['missing_title']}**",
            f"- Missing artist: **{stats['missing_artist']}**",
            f"- Missing album: **{stats['missing_album']}**",
            f"- Cover art present: **{stats['cover_art_present']}**",
            f"- Cover art missing: **{stats['cover_art_missing']}**",
            "",
        ]
    )

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


def _write_duplicates_json(
    out_dir: Path,
    source: Path,
    report: DuplicateAnalysisReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "duplicates.json"
    payload = {
        "source": str(source),
        "summary": report.summary.to_dict(),
        "exact_duplicates": [group.to_dict() for group in report.exact_duplicates],
        "likely_duplicates": [group.to_dict() for group in report.likely_duplicates],
    }
    if refresh_summary is not None:
        payload["refresh"] = refresh_summary.to_dict()
    if config_meta is not None:
        payload["config"] = config_meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_duplicates_md(
    out_dir: Path,
    report: DuplicateAnalysisReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "duplicates.md"
    lines = ["# nyxcore duplicate analysis", "", "## Summary", ""]
    if refresh_summary is not None:
        lines.extend(
            [
                f"- Refresh mode: **{refresh_summary.mode}**",
                f"- Added files: **{len(refresh_summary.changes.added_files)}**",
                f"- Modified files: **{len(refresh_summary.changes.modified_files)}**",
                f"- Removed files: **{len(refresh_summary.changes.removed_files)}**",
                "",
            ]
        )
    if config_meta is not None:
        lines.extend([f"- Active profile: **{config_meta['active_profile']}**", ""])
    lines.extend(
        [
            f"- Total tracks analyzed: **{report.summary.total_tracks}**",
            f"- Exact duplicate groups: **{report.summary.exact_group_count}**",
            f"- Exact duplicate files: **{report.summary.exact_duplicate_file_count}**",
            f"- Likely duplicate groups: **{report.summary.likely_group_count}**",
            f"- Likely duplicate files: **{report.summary.likely_duplicate_file_count}**",
            "",
            "## Exact duplicates",
            "",
        ]
    )
    if not report.exact_duplicates:
        lines.append("- _None_")
    else:
        for group in report.exact_duplicates[:20]:
            lines.append(f"### {group.group_id}")
            lines.append("")
            lines.append(f"- Preferred: `{group.preferred.path}`")
            lines.append(f"- Reason: `{', '.join(group.preferred.reasons)}`")
            lines.append(f"- Hash: `{group.content_hash}`")
            lines.append("")
            for item in group.files:
                lines.append(f"- `{item.path}`")
            lines.append("")
    lines.extend(["## Likely duplicates", ""])
    if not report.likely_duplicates:
        lines.append("- _None_")
    else:
        for group in report.likely_duplicates[:20]:
            lines.append(f"### {group.group_id}")
            lines.append("")
            lines.append(f"- Confidence: **{group.confidence:.2f}**")
            lines.append(f"- Preferred: `{group.preferred.path}`")
            lines.append(f"- Group reasons: `{', '.join(group.reasons)}`")
            lines.append("")
            for item in group.files:
                lines.append(f"- `{item.path}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _duplicates_summary_table(report: DuplicateAnalysisReport) -> Table:
    table = Table(title="Duplicate Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Tracks analyzed", str(report.summary.total_tracks))
    table.add_row("Exact groups", str(report.summary.exact_group_count))
    table.add_row("Exact duplicate files", str(report.summary.exact_duplicate_file_count))
    table.add_row("Likely groups", str(report.summary.likely_group_count))
    table.add_row("Likely duplicate files", str(report.summary.likely_duplicate_file_count))
    return table


def _write_health_json(
    out_dir: Path,
    source: Path,
    report: HealthReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "health.json"
    payload = {
        "source": str(source),
        **report.to_dict(),
    }
    if refresh_summary is not None:
        payload["refresh"] = refresh_summary.to_dict()
    if config_meta is not None:
        payload["config"] = config_meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_health_md(
    out_dir: Path,
    report: HealthReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "health.md"
    lines = ["# nyxcore library health report", "", "## Overview", ""]
    if refresh_summary is not None:
        lines.extend(
            [
                f"- Refresh mode: **{refresh_summary.mode}**",
                f"- Added files: **{len(refresh_summary.changes.added_files)}**",
                f"- Modified files: **{len(refresh_summary.changes.modified_files)}**",
                f"- Removed files: **{len(refresh_summary.changes.removed_files)}**",
                f"- Unchanged files reused: **{len(refresh_summary.changes.unchanged_files)}**",
                "",
            ]
        )
    if config_meta is not None:
        lines.extend([f"- Active profile: **{config_meta['active_profile']}**", ""])
    lines.extend(
        [
            f"- Total audio files: **{report.overview.total_audio_files}**",
            f"- Total folders touched: **{report.overview.total_folders_touched}**",
            f"- Total library size (bytes): **{report.overview.total_library_size_bytes}**",
            f"- Exact duplicate groups: **{report.overview.duplicate_exact_groups}**",
            f"- Likely duplicate groups: **{report.overview.duplicate_likely_groups}**",
            "",
            "## Metadata",
            "",
            f"- Missing title: **{report.metadata.missing_title.count}**",
            f"- Missing artist: **{report.metadata.missing_artist.count}**",
            f"- Missing album: **{report.metadata.missing_album.count}**",
            f"- Placeholder metadata: **{report.metadata.placeholder_metadata.count}**",
            "",
            "## Artwork",
            "",
            f"- With artwork: **{report.artwork.with_artwork}**",
            f"- Without artwork: **{report.artwork.without_artwork}**",
            f"- Coverage: **{report.artwork.coverage_percent:.2f}%**",
            "",
            "## Quality",
            "",
            f"- Low bitrate files: **{report.quality.low_bitrate_files.count}**",
            f"- Duration outliers: **{report.quality.duration_outliers.count}**",
            f"- Unreadable or unparseable files: **{report.quality.unreadable_or_unparseable_files.count}**",
            "",
            "## Duplicates",
            "",
            f"- Files involved in duplicates: **{report.duplicates.total_files_in_duplicates}**",
            f"- Reclaimable exact-duplicate bytes: **{report.duplicates.reclaimable_bytes_exact}**",
            "",
            "## Priority Recommendations",
            "",
        ]
    )
    if not report.priorities.recommended_actions:
        lines.append("- _No urgent issues detected_")
    else:
        for action in report.priorities.recommended_actions:
            lines.append(f"- {action}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _health_summary_table(report: HealthReport) -> Table:
    table = Table(title="Health Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Audio files", str(report.overview.total_audio_files))
    table.add_row("Folders", str(report.overview.total_folders_touched))
    table.add_row("Exact duplicate groups", str(report.duplicates.exact_duplicate_groups))
    table.add_row("Likely duplicate groups", str(report.duplicates.likely_duplicate_groups))
    table.add_row("Missing artist", str(report.metadata.missing_artist.count))
    table.add_row("Placeholder metadata", str(report.metadata.placeholder_metadata.count))
    table.add_row("Low bitrate files", str(report.quality.low_bitrate_files.count))
    table.add_row("Artwork coverage", f"{report.artwork.coverage_percent:.2f}%")
    return table


def _write_review_json(
    out_dir: Path,
    source: Path,
    report: ReviewQueueReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "review.json"
    payload = {"source": str(source), **report.to_dict()}
    if refresh_summary is not None:
        payload["refresh"] = refresh_summary.to_dict()
    if config_meta is not None:
        payload["config"] = config_meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_review_md(
    out_dir: Path,
    report: ReviewQueueReport,
    *,
    refresh_summary: RefreshSummary | None = None,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "review.md"
    lines = ["# nyxcore review queue", "", "## Summary", ""]
    if refresh_summary is not None:
        lines.extend(
            [
                f"- Refresh mode: **{refresh_summary.mode}**",
                f"- Added files: **{len(refresh_summary.changes.added_files)}**",
                f"- Modified files: **{len(refresh_summary.changes.modified_files)}**",
                f"- Removed files: **{len(refresh_summary.changes.removed_files)}**",
                f"- Unchanged files reused: **{len(refresh_summary.changes.unchanged_files)}**",
                "",
            ]
        )
    if config_meta is not None:
        lines.extend([f"- Active profile: **{config_meta['active_profile']}**", ""])
    lines.extend(
        [
            f"- Queue items: **{report.summary.total_items}**",
            f"- Files referenced: **{report.summary.total_files_referenced}**",
            f"- Generation mode: **{report.metadata.generation_mode}**",
            "",
            "## Counts By Type",
            "",
            "| Type | Count |",
            "| --- | ---: |",
        ]
    )
    if not report.summary.counts_by_type:
        lines.append("| _None_ | 0 |")
    else:
        for item_type, count in report.summary.counts_by_type.items():
            lines.append(f"| `{item_type}` | {count} |")
    lines.extend(["", "## Counts By State", "", "| State | Count |", "| --- | ---: |"])
    if not report.summary.counts_by_state:
        lines.append("| _None_ | 0 |")
    else:
        for state_name, count in report.summary.counts_by_state.items():
            lines.append(f"| `{state_name}` | {count} |")
    lines.extend(["", "## Review Items", ""])
    if not report.items:
        lines.append("- _No items matched the current filters_")
    else:
        for item in report.items:
            lines.append(f"### {item.item_id} · {item.priority_band.upper()} · {item.priority_score:.1f}")
            lines.append("")
            lines.append(f"- Type: `{item.item_type}`")
            lines.append(f"- State: `{item.review_status}`")
            if item.state_updated_at is not None:
                lines.append(f"- State updated: `{item.state_updated_at}`")
            if item.snooze_until is not None:
                lines.append(f"- Snooze until: `{item.snooze_until}`")
            lines.append(f"- Summary: {item.summary}")
            lines.append(f"- Reason: {item.reason_summary}")
            if item.preferred_path is not None:
                lines.append(f"- Preferred copy: `{item.preferred_path}`")
            if item.reclaimable_bytes is not None:
                lines.append(f"- Reclaimable bytes: **{item.reclaimable_bytes}**")
            if item.confidence is not None:
                lines.append(f"- Confidence: **{item.confidence:.3f}**")
            if item.folder is not None:
                lines.append(f"- Folder: `{item.folder}`")
            if item.sample_paths:
                lines.append("- Samples:")
                for sample_path in item.sample_paths:
                    lines.append(f"  - `{sample_path}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _review_summary_table(report: ReviewQueueReport) -> Table:
    table = Table(title="Review Queue Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Queue items", str(report.summary.total_items))
    table.add_row("Files referenced", str(report.summary.total_files_referenced))
    table.add_row("High priority", str(report.summary.counts_by_priority_band.get("high", 0)))
    table.add_row("Medium priority", str(report.summary.counts_by_priority_band.get("medium", 0)))
    table.add_row("Low priority", str(report.summary.counts_by_priority_band.get("low", 0)))
    table.add_row("New", str(report.summary.counts_by_state.get("new", 0)))
    table.add_row("Seen", str(report.summary.counts_by_state.get("seen", 0)))
    return table


def _write_action_plan_json(
    out_dir: Path,
    source: Path,
    report: ActionPlanReport,
    *,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "review_plan.json"
    payload = {"source": str(source), **report.to_dict()}
    if config_meta is not None:
        payload["config"] = config_meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_action_plan_md(
    out_dir: Path,
    report: ActionPlanReport,
    *,
    config_meta: dict | None = None,
) -> Path:
    path = out_dir / "review_plan.md"
    lines = ["# nyxcore review action plan", "", "## Summary", ""]
    if config_meta is not None:
        lines.extend([f"- Active profile: **{config_meta['active_profile']}**", ""])
    lines.extend(
        [
            f"- Requested review items: **{report.summary.requested_item_count}**",
            f"- Plans generated: **{report.summary.generated_plan_count}**",
            f"- Apply-capable plans: **{report.summary.apply_supported_plan_count}**",
            f"- Unsupported items: **{report.summary.unsupported_item_count}**",
            "",
            "## Plans",
            "",
        ]
    )
    if not report.plans:
        lines.append("- _No plans generated_")
    else:
        for plan in report.plans:
            lines.append(f"### {plan.plan_id}")
            lines.append("")
            lines.append(f"- Type: `{plan.action_type}`")
            lines.append(f"- Safety: `{plan.safety_level}`")
            lines.append(f"- Apply supported: `{plan.apply_supported}`")
            lines.append(f"- Confidence: `{plan.confidence:.3f}`")
            lines.append(f"- Source review items: `{plan.source_review_item_ids}`")
            if plan.reasons:
                lines.append(f"- Reasons: `{', '.join(plan.reasons)}`")
            if plan.notes:
                lines.append(f"- Notes: `{'; '.join(plan.notes)}`")
            if plan.proposed_operations:
                lines.append("- Operations:")
                for operation in plan.proposed_operations:
                    summary = operation.path or operation.destination_path or operation.operation_type
                    lines.append(f"  - `{operation.operation_type}` -> `{summary}`")
            lines.append("")
    if report.unsupported_items:
        lines.extend(["## Unsupported", ""])
        for item in report.unsupported_items:
            lines.append(f"- `{item.source_review_item_id}`: {item.reason}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_applied_plan_json(out_dir: Path, results: list[AppliedPlanResult]) -> Path:
    path = out_dir / "review_apply.json"
    payload = {
        "applied_at": datetime.now(tz=UTC).isoformat(),
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_applied_plan_md(out_dir: Path, results: list[AppliedPlanResult]) -> Path:
    path = out_dir / "review_apply.md"
    lines = ["# nyxcore applied review plan", "", "## Results", ""]
    if not results:
        lines.append("- _No plan results_")
    else:
        for result in results:
            lines.append(f"### {result.plan_id}")
            lines.append("")
            lines.append(f"- Action type: `{result.action_type}`")
            lines.append(f"- Status: `{result.status}`")
            if result.resolved_review_item_ids:
                lines.append(f"- Resolved review items: `{result.resolved_review_item_ids}`")
            if result.operation_results:
                lines.append("- Operations:")
                for operation in result.operation_results:
                    summary = operation.path or operation.destination_path or operation.operation_type
                    lines.append(f"  - `{operation.status}` `{operation.operation_type}` -> `{summary}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _action_plan_summary_table(report: ActionPlanReport) -> Table:
    table = Table(title="Action Plan Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Requested items", str(report.summary.requested_item_count))
    table.add_row("Plans generated", str(report.summary.generated_plan_count))
    table.add_row("Apply-capable plans", str(report.summary.apply_supported_plan_count))
    table.add_row("Unsupported items", str(report.summary.unsupported_item_count))
    return table


def _write_history_json(out_dir: Path, ledger: OperationLedger) -> Path:
    path = out_dir / "review_history_snapshot.json"
    path.write_text(json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _history_table(ledger: OperationLedger) -> Table:
    table = Table(title="Review History")
    table.add_column("Batch ID", style="cyan")
    table.add_column("Applied At")
    table.add_column("Plans", justify="right", style="magenta")
    table.add_column("Ops", justify="right", style="magenta")
    table.add_column("Reversible", justify="right", style="magenta")
    for batch in sorted(ledger.batches, key=lambda item: item.applied_at):
        reversible = sum(1 for operation in batch.operations if operation.reversible)
        table.add_row(batch.batch_id, batch.applied_at, str(len(batch.source_plan_ids)), str(len(batch.operations)), str(reversible))
    if not ledger.batches:
        table.add_row("_None_", "_None_", "0", "0", "0")
    return table


def _history_detail_table(batch: OperationBatch) -> Table:
    table = Table(title=f"History Detail: {batch.batch_id}")
    table.add_column("Operation", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Undo")
    table.add_column("Path")
    for operation in batch.operations:
        table.add_row(
            operation.operation_id,
            operation.operation_type,
            operation.status,
            operation.undo_status,
            operation.current_path or operation.original_path or "",
        )
    if not batch.operations:
        table.add_row("_None_", "_None_", "_None_", "_None_", "_None_")
    return table


def _write_playlist_json(out_dir: Path, source: Path, report: PlaylistReport, *, config_meta: dict | None = None) -> Path:
    path = out_dir / "playlist_query.json"
    payload = {"source": str(source), **report.to_dict()}
    if config_meta is not None:
        payload["config"] = config_meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_playlist_md(out_dir: Path, report: PlaylistReport, *, config_meta: dict | None = None) -> Path:
    path = out_dir / "playlist_query.md"
    lines = [
        "# nyxcore playlist query",
        "",
        "## Query",
        "",
        f"- Original query: **{report.original_query}**",
        f"- Tracks selected: **{report.summary.track_count}**",
        f"- Estimated total duration (seconds): **{report.summary.estimated_total_duration_seconds:.3f}**",
    ]
    if config_meta is not None:
        lines.append(f"- Active profile: **{config_meta['active_profile']}**")
    if report.summary.average_bpm is not None:
        lines.append(f"- Average BPM: **{report.summary.average_bpm:.2f}**")
    if report.summary.average_energy_0_10 is not None:
        lines.append(f"- Average energy: **{report.summary.average_energy_0_10:.2f}**")
    lines.extend(["", "## Parsed Intent", ""])
    parsed = report.parsed_query.to_dict()
    for key in ("moods", "genres", "keywords", "negative_keywords", "cultural_hints", "instrumental_preference"):
        value = parsed.get(key)
        if value:
            lines.append(f"- {key}: `{value}`")
    if parsed.get("bpm_min") is not None or parsed.get("bpm_max") is not None:
        lines.append(f"- bpm_range: `{parsed.get('bpm_min')}..{parsed.get('bpm_max')}`")
    if parsed.get("max_duration_seconds") is not None:
        lines.append(f"- max_duration_seconds: `{parsed.get('max_duration_seconds')}`")
    if report.unsupported_request_aspects:
        lines.extend(["", "## Unsupported / Partial Aspects", ""])
        for item in report.unsupported_request_aspects:
            lines.append(f"- {item}")
    lines.extend(["", "## Ranked Tracks", "", "| Score | Path | Reasons |", "| ---: | --- | --- |"])
    if not report.ranked_tracks:
        lines.append("| 0.0 | _None_ | _None_ |")
    else:
        for track in report.ranked_tracks:
            lines.append(
                f"| {track.score:.3f} | `{track.path.replace('|', '\\|')}` | `{', '.join(track.reasons) or 'none'}` |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _playlist_summary_table(report: PlaylistReport) -> Table:
    table = Table(title="Playlist Query Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Tracks selected", str(report.summary.track_count))
    table.add_row("Total duration (sec)", f"{report.summary.estimated_total_duration_seconds:.3f}")
    table.add_row("Unsupported aspects", str(len(report.unsupported_request_aspects)))
    if report.summary.average_bpm is not None:
        table.add_row("Avg BPM", f"{report.summary.average_bpm:.2f}")
    if report.summary.average_energy_0_10 is not None:
        table.add_row("Avg energy", f"{report.summary.average_energy_0_10:.2f}")
    return table


def _saved_playlist_table(definitions: list[SavedPlaylistDefinition]) -> Table:
    table = Table(title="Saved Playlists")
    table.add_column("Playlist ID", style="cyan")
    table.add_column("Name")
    table.add_column("Profile")
    table.add_column("Last Refresh")
    table.add_column("Track Count", justify="right")
    for definition in definitions:
        table.add_row(
            definition.playlist_id,
            definition.name,
            definition.profile,
            definition.last_refreshed_at or "",
            str(definition.last_refresh_summary.get("track_count", 0)),
        )
    if not definitions:
        table.add_row("_None_", "_None_", "_None_", "_None_", "0")
    return table


def _saved_playlist_detail_table(definition: SavedPlaylistDefinition, latest: SavedPlaylistLatestResult | None) -> Table:
    table = Table(title=f"Saved Playlist: {definition.playlist_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Name", definition.name)
    table.add_row("Query", definition.query)
    table.add_row("Profile", definition.profile)
    table.add_row("Max tracks", "" if definition.max_tracks is None else str(definition.max_tracks))
    table.add_row("Min score", "" if definition.min_score is None else str(definition.min_score))
    table.add_row("Created", definition.created_at)
    table.add_row("Updated", definition.updated_at or "")
    table.add_row("Last refresh", definition.last_refreshed_at or "")
    if latest is not None:
        table.add_row("Latest track count", str(latest.summary.get("track_count", 0)))
        table.add_row("Latest refresh mode", latest.refresh_mode)
        table.add_row("Tracks added", str(len(latest.refresh_diff.get("tracks_added", []))))
        table.add_row("Tracks removed", str(len(latest.refresh_diff.get("tracks_removed", []))))
        table.add_row("Track count delta", str(latest.refresh_diff.get("track_count_delta", 0)))
        duration_delta = float(latest.refresh_diff.get("estimated_duration_delta_seconds", 0.0))
        table.add_row("Duration delta (sec)", f"{duration_delta:.3f}")
        table.add_row("Rank changes", str(len(latest.refresh_diff.get("rank_changes", []))))
    return table


def _default_state_path(out_dir: Path) -> Path:
    return out_dir / "library_state.json"


def _default_review_state_path(out_dir: Path) -> Path:
    return out_dir / "review_state.json"


def _default_history_path(out_dir: Path) -> Path:
    return out_dir / "review_history.json"


def _default_saved_playlists_root(out_dir: Path) -> Path:
    return out_dir / "saved_playlists"


def _saved_playlist_diff_table(latest: SavedPlaylistLatestResult | None) -> Table:
    table = Table(title="Saved Playlist Refresh Diff")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    if latest is None:
        table.add_row("Status", "No refresh result saved yet")
        return table
    refresh_diff = latest.refresh_diff
    table.add_row("Has previous result", "yes" if refresh_diff.get("has_previous_result") else "no")
    table.add_row("Tracks added", str(len(refresh_diff.get("tracks_added", []))))
    table.add_row("Tracks removed", str(len(refresh_diff.get("tracks_removed", []))))
    table.add_row("Track count delta", str(refresh_diff.get("track_count_delta", 0)))
    table.add_row(
        "Duration delta (sec)",
        f"{float(refresh_diff.get('estimated_duration_delta_seconds', 0.0)):.3f}",
    )
    rank_changes = refresh_diff.get("rank_changes", [])
    table.add_row("Rank changes", str(len(rank_changes)))
    if rank_changes:
        top_change = rank_changes[0]
        table.add_row(
            "Top rank move",
            f"{top_change['path']} ({top_change['old_rank']} -> {top_change['new_rank']})",
        )
    return table


def _full_refresh_summary(records: list[TrackRecord]) -> RefreshSummary:
    current_paths = sorted(record.path for record in records)
    return RefreshSummary(
        mode="full",
        changes=ChangeSet(
            added_files=current_paths,
            modified_files=[],
            removed_files=[],
            unchanged_files=[],
        ),
        rescanned_files=len(records),
    )


def _load_library_records(
    music: Path,
    *,
    incremental: bool,
    state_path: Path,
) -> tuple[list[TrackRecord], RefreshSummary]:
    if incremental:
        refreshed = refresh_incremental_state(music, state_path)
        return refreshed.records, refreshed.summary
    records, _stats = scan_music_folder(music)
    return records, _full_refresh_summary(records)


def _build_review_report(
    music: Path,
    records: list[TrackRecord],
    *,
    refresh_summary: RefreshSummary,
    app_config: NyxConfig,
    review_state_store: ReviewStateStore,
    max_items: int | None,
    min_priority: str | None,
    include_types: set[str] | None,
    exclude_types: set[str] | None,
    include_ignored: bool,
    include_snoozed: bool,
    include_resolved: bool,
    only_unresolved: bool,
) -> ReviewQueueReport:
    return build_review_pipeline(
        music,
        records,
        app_config=app_config,
        review_state=review_state_store,
        generation_mode=refresh_summary.mode,
        max_items=max_items,
        min_priority_band=min_priority,
        include_types=include_types,
        exclude_types=exclude_types,
        include_ignored=include_ignored,
        include_snoozed=include_snoozed,
        include_resolved=include_resolved,
        only_unresolved=only_unresolved,
    ).review_report


def _refresh_saved_playlist_definition(
    music: Path,
    definition: SavedPlaylistDefinition,
    *,
    store_root: Path,
    app_config: NyxConfig,
    incremental: bool,
    state_path: Path,
    analysis_cache: Path,
    profile_override: str | None = None,
    max_tracks_override: int | None = None,
    min_score_override: float | None = None,
) -> tuple[SavedPlaylistLatestResult, RefreshSummary]:
    records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    latest = refresh_saved_playlist(
        store_root,
        definition,
        records=records,
        refresh_summary=refresh_summary,
        app_config=app_config,
        analysis_cache_path=analysis_cache if analysis_cache.exists() else None,
        profile_override=profile_override,
        max_tracks_override=max_tracks_override,
        min_score_override=min_score_override,
    )
    return latest, refresh_summary


def _review_states_table(review_state: ReviewStateStore) -> Table:
    table = Table(title="Review State Store")
    table.add_column("Item ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Snooze Until")
    table.add_column("Updated")
    entries = sorted(review_state.items.values(), key=lambda item: item.item_id)
    if not entries:
        table.add_row("_None_", "_None_", "_None_", "_None_", "_None_")
        return table
    for entry in entries:
        table.add_row(
            entry.item_id,
            entry.status,
            entry.item_type or "",
            entry.snooze_until or "",
            entry.updated_at,
        )
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
        from nyxcore.audio.backends.essentia_backend import EssentiaBackend

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


def _load_app_config(config_path: Path | None, profile: str | None) -> NyxConfig:
    try:
        return load_config(config_path, profile=profile)
    except Exception as exc:
        raise typer.BadParameter(f"Failed to load config: {exc}") from exc


def _config_meta(config: NyxConfig, *sections: str) -> dict:
    meta = {"active_profile": config.profile}
    thresholds: dict[str, dict] = {}
    for section in sections:
        value = getattr(config, section, None)
        if value is not None and hasattr(value, "model_dump"):
            thresholds[section] = value.model_dump()
    if thresholds:
        meta["thresholds"] = thresholds
    return meta


@app.command()
def scan(
    music: Path = typer.Argument(Path("./music"), help="Folder to scan recursively"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for reports"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)
    state_path = state or _default_state_path(out)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Scanning MP3 files", total=1)
        records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
        stats = compute_stats(records)
        if not records:
            progress.update(task, total=1, completed=1)
        else:
            progress.update(task, total=max(1, len(records)), completed=max(1, refresh_summary.rescanned_files))

    json_path = _write_scan_json(out, music, records, stats, refresh_summary=refresh_summary)
    md_path = _write_scan_md(out, stats, refresh_summary=refresh_summary)
    console.print(_summary_table(stats))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("duplicates")
def duplicates_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for duplicate reports"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    state_path = state or _default_state_path(out)
    records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    report = build_duplicate_report(records, app_config=app_config)
    meta = _config_meta(app_config, "duplicates")
    json_path = _write_duplicates_json(out, music, report, refresh_summary=refresh_summary, config_meta=meta)
    md_path = _write_duplicates_md(out, report, refresh_summary=refresh_summary, config_meta=meta)
    console.print(_duplicates_summary_table(report))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("health")
def health_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for health reports"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    state_path = state or _default_state_path(out)
    records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    report = build_duplicate_health_reports(music, records, app_config=app_config).health_report
    meta = _config_meta(app_config, "health", "duplicates")
    json_path = _write_health_json(out, music, report, refresh_summary=refresh_summary, config_meta=meta)
    md_path = _write_health_md(out, report, refresh_summary=refresh_summary, config_meta=meta)
    console.print(_health_summary_table(report))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("playlist")
def playlist_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    query: str = typer.Option(..., "--query", help="Natural-language playlist query"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for playlist query reports"),
    max_tracks: int | None = typer.Option(None, "--max-tracks", help="Maximum ranked tracks to keep"),
    min_score: float | None = typer.Option(None, "--min-score", help="Minimum score required for inclusion"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    analysis_cache: Path = typer.Option(
        Path("data/cache/analysis.sqlite"),
        "--analysis-cache",
        help="Optional analysis cache path for BPM/energy/tag enrichment",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    """Run the current natural-language playlist query workflow."""
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if max_tracks is not None and max_tracks < 1:
        raise typer.BadParameter("--max-tracks must be >= 1")

    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    state_path = state or _default_state_path(out)
    records, _refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    report = build_playlist_report(
        records,
        query=query,
        settings=app_config.playlist,
        max_tracks=max_tracks,
        min_score=min_score,
        analysis_cache_path=analysis_cache if analysis_cache.exists() else None,
    )
    meta = _config_meta(app_config, "playlist")
    json_path = _write_playlist_json(out, music, report, config_meta=meta)
    md_path = _write_playlist_md(out, report, config_meta=meta)
    console.print(_playlist_summary_table(report))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("save-playlist")
def save_playlist_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    name: str = typer.Option(..., "--name", help="Human-readable saved playlist name"),
    query: str = typer.Option(..., "--query", help="Natural-language playlist query"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
    max_tracks: int | None = typer.Option(None, "--max-tracks", help="Optional default max tracks for this saved playlist"),
    min_score: float | None = typer.Option(None, "--min-score", help="Optional default minimum score for this saved playlist"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    analysis_cache: Path = typer.Option(Path("data/cache/analysis.sqlite"), "--analysis-cache", help="Optional analysis cache path"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
    export_m3u: bool = typer.Option(False, "--export-m3u", help="Export the refreshed saved playlist as M3U"),
    export_json_tracks: bool = typer.Option(False, "--export-json-tracks", help="Export the refreshed saved playlist as JSON track list"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if max_tracks is not None and max_tracks < 1:
        raise typer.BadParameter("--max-tracks must be >= 1")
    ensure_out_dir(out)
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    app_config = _load_app_config(config, profile)
    definition = create_saved_playlist_definition(
        name=name,
        query=query,
        profile=app_config.profile,
        max_tracks=max_tracks,
        min_score=min_score,
    )
    store.playlists[definition.playlist_id] = definition
    state_path = state or _default_state_path(out)
    latest, _refresh_summary = _refresh_saved_playlist_definition(
        music,
        definition,
        store_root=store_root,
        app_config=app_config,
        incremental=incremental,
        state_path=state_path,
        analysis_cache=analysis_cache,
    )
    save_saved_playlist_definition(store_root, store)
    if export_m3u:
        export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
    if export_json_tracks:
        export_saved_playlist_json(store_root, definition.playlist_id, latest)
    console.print(_saved_playlist_detail_table(definition, latest))
    console.print(_saved_playlist_diff_table(latest))


@app.command("list-playlists")
def list_playlists_cmd(
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
) -> None:
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    console.print(_saved_playlist_table(sorted(store.playlists.values(), key=lambda item: item.playlist_id)))


@app.command("show-playlist")
def show_playlist_cmd(
    playlist_id: str = typer.Argument(..., help="Saved playlist id"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
) -> None:
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    definition = store.playlists.get(playlist_id)
    if definition is None:
        raise typer.BadParameter(f"Saved playlist not found: {playlist_id}")
    latest = read_saved_playlist_latest_result(store_root, playlist_id)
    console.print(_saved_playlist_detail_table(definition, latest))
    console.print(_saved_playlist_diff_table(latest))


@app.command("rename-playlist")
def rename_playlist_cmd(
    playlist_id: str = typer.Argument(..., help="Saved playlist id"),
    name: str = typer.Option(..., "--name", help="New human-readable playlist name"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
) -> None:
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    definition = store.playlists.get(playlist_id)
    if definition is None:
        raise typer.BadParameter(f"Saved playlist not found: {playlist_id}")
    rename_saved_playlist_definition(definition, name=name)
    save_saved_playlist_definition(store_root, store)
    latest = read_saved_playlist_latest_result(store_root, playlist_id)
    console.print(_saved_playlist_detail_table(definition, latest))


@app.command("edit-playlist")
def edit_playlist_cmd(
    playlist_id: str = typer.Argument(..., help="Saved playlist id"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
    query: str | None = typer.Option(None, "--query", help="Updated natural-language query"),
    max_tracks: int | None = typer.Option(None, "--max-tracks", help="Updated default max tracks"),
    clear_max_tracks: bool = typer.Option(False, "--clear-max-tracks", help="Clear the saved max-tracks override"),
    min_score: float | None = typer.Option(None, "--min-score", help="Updated default minimum score"),
    clear_min_score: bool = typer.Option(False, "--clear-min-score", help="Clear the saved min-score override"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Updated saved profile"),
) -> None:
    if max_tracks is not None and max_tracks < 1:
        raise typer.BadParameter("--max-tracks must be >= 1")
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    definition = store.playlists.get(playlist_id)
    if definition is None:
        raise typer.BadParameter(f"Saved playlist not found: {playlist_id}")
    resolved_profile = None
    if profile is not None:
        resolved_profile = _load_app_config(config, profile).profile
    edit_saved_playlist_definition(
        definition,
        query=query,
        max_tracks=max_tracks,
        min_score=min_score,
        profile=resolved_profile,
        clear_max_tracks=clear_max_tracks,
        clear_min_score=clear_min_score,
    )
    save_saved_playlist_definition(store_root, store)
    latest = read_saved_playlist_latest_result(store_root, playlist_id)
    console.print(_saved_playlist_detail_table(definition, latest))
    console.print(_saved_playlist_diff_table(latest))


@app.command("delete-playlist")
def delete_playlist_cmd(
    playlist_id: str = typer.Argument(..., help="Saved playlist id"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion of the saved playlist definition and latest result"),
) -> None:
    if not yes:
        raise typer.BadParameter("Deletion requires --yes")
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    deleted = delete_saved_playlist_definition(store_root, store, playlist_id)
    if deleted is None:
        raise typer.BadParameter(f"Saved playlist not found: {playlist_id}")
    save_saved_playlist_definition(store_root, store)
    console.print(f"[green]Deleted saved playlist:[/green] {playlist_id}")


@app.command("refresh-playlist")
def refresh_playlist_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    playlist_id: str = typer.Argument(..., help="Saved playlist id"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    analysis_cache: Path = typer.Option(Path("data/cache/analysis.sqlite"), "--analysis-cache", help="Optional analysis cache path"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Override the saved playlist profile for this refresh"),
    max_tracks: int | None = typer.Option(None, "--max-tracks", help="Override max tracks for this refresh"),
    min_score: float | None = typer.Option(None, "--min-score", help="Override minimum score for this refresh"),
    export_m3u: bool = typer.Option(False, "--export-m3u", help="Export the refreshed saved playlist as M3U"),
    export_json_tracks: bool = typer.Option(False, "--export-json-tracks", help="Export the refreshed saved playlist as JSON track list"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    definition = store.playlists.get(playlist_id)
    if definition is None:
        raise typer.BadParameter(f"Saved playlist not found: {playlist_id}")
    app_config = _load_app_config(config, profile or definition.profile)
    state_path = state or _default_state_path(out)
    latest, _refresh_summary = _refresh_saved_playlist_definition(
        music,
        definition,
        store_root=store_root,
        app_config=app_config,
        incremental=incremental,
        state_path=state_path,
        analysis_cache=analysis_cache,
        profile_override=app_config.profile,
        max_tracks_override=max_tracks,
        min_score_override=min_score,
    )
    save_saved_playlist_definition(store_root, store)
    if export_m3u:
        export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
    if export_json_tracks:
        export_saved_playlist_json(store_root, definition.playlist_id, latest)
    console.print(_saved_playlist_detail_table(definition, latest))
    console.print(_saved_playlist_diff_table(latest))


@app.command("refresh-all-playlists")
def refresh_all_playlists_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output root for saved playlists"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    analysis_cache: Path = typer.Option(Path("data/cache/analysis.sqlite"), "--analysis-cache", help="Optional analysis cache path"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    export_m3u: bool = typer.Option(False, "--export-m3u", help="Export refreshed playlists as M3U"),
    export_json_tracks: bool = typer.Option(False, "--export-json-tracks", help="Export refreshed playlists as JSON track lists"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    store_root = _default_saved_playlists_root(out)
    store = load_saved_playlist_store(store_root)
    state_path = state or _default_state_path(out)
    rows: list[tuple[SavedPlaylistDefinition, SavedPlaylistLatestResult]] = []
    for definition in sorted(store.playlists.values(), key=lambda item: item.playlist_id):
        app_config = _load_app_config(config, definition.profile)
        latest, _refresh_summary = _refresh_saved_playlist_definition(
            music,
            definition,
            store_root=store_root,
            app_config=app_config,
            incremental=incremental,
            state_path=state_path,
            analysis_cache=analysis_cache,
            profile_override=definition.profile,
        )
        if export_m3u:
            export_saved_playlist_m3u(store_root, definition.playlist_id, latest)
        if export_json_tracks:
            export_saved_playlist_json(store_root, definition.playlist_id, latest)
        rows.append((definition, latest))
    save_saved_playlist_definition(store_root, store)
    console.print(_saved_playlist_table([definition for definition, _latest in rows]))


@app.command("review")
def review_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for review queue reports"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    review_state: Path | None = typer.Option(None, "--review-state", help="Path to review triage state file"),
    max_items: int | None = typer.Option(None, "--max-items", help="Maximum review items to keep"),
    min_priority: str | None = typer.Option(None, "--min-priority", help="Minimum priority band: low, medium, high"),
    include_type: list[str] | None = typer.Option(None, "--include-type", help="Repeat to keep only specific item types"),
    exclude_type: list[str] | None = typer.Option(None, "--exclude-type", help="Repeat to hide specific item types"),
    include_ignored: bool = typer.Option(False, "--include-ignored", help="Include ignored items in the review output"),
    include_snoozed: bool = typer.Option(False, "--include-snoozed", help="Include active snoozed items in the review output"),
    include_resolved: bool = typer.Option(False, "--include-resolved", help="Include resolved items in the review output"),
    only_unresolved: bool = typer.Option(False, "--only-unresolved", help="Show only new or seen items"),
    mark_seen: list[str] | None = typer.Option(None, "--mark-seen", help="Repeat to mark one or more review items as seen"),
    ignore_ids: list[str] | None = typer.Option(None, "--ignore", help="Repeat to ignore one or more review items"),
    snooze_ids: list[str] | None = typer.Option(None, "--snooze", help="Repeat to snooze one or more review items"),
    resolve_ids: list[str] | None = typer.Option(None, "--resolve", help="Repeat to mark one or more review items resolved"),
    days: int | None = typer.Option(None, "--days", help="Snooze duration in days; used with --snooze"),
    list_states: bool = typer.Option(False, "--list-states", help="Print the persisted review state store"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if max_items is not None and max_items < 1:
        raise typer.BadParameter("--max-items must be >= 1")
    if min_priority is not None and min_priority not in {"low", "medium", "high"}:
        raise typer.BadParameter("--min-priority must be one of: low, medium, high")
    include_types = set(include_type or [])
    exclude_types = set(exclude_type or [])
    unknown_types = sorted((include_types | exclude_types) - REVIEW_ITEM_TYPES)
    if unknown_types:
        raise typer.BadParameter(f"Unknown review item type(s): {', '.join(unknown_types)}")
    actions = {
        "seen": list(mark_seen or []),
        "ignored": list(ignore_ids or []),
        "snoozed": list(snooze_ids or []),
        "resolved": list(resolve_ids or []),
    }
    active_actions = [name for name, ids in actions.items() if ids]
    if len(active_actions) > 1:
        raise typer.BadParameter("Use only one review-state action at a time")
    if snooze_ids and (days is None or days < 1):
        raise typer.BadParameter("--snooze requires --days >= 1")
    if days is not None and not snooze_ids:
        raise typer.BadParameter("--days is only valid with --snooze")

    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    state_path = state or _default_state_path(out)
    review_state_path = review_state or _default_review_state_path(out)
    review_state_store = load_review_state(review_state_path)
    records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    full_report = _build_review_report(
        music,
        records,
        refresh_summary=refresh_summary,
        app_config=app_config,
        review_state_store=review_state_store,
        max_items=None,
        min_priority=None,
        include_types=None,
        exclude_types=None,
        include_ignored=True,
        include_snoozed=True,
        include_resolved=True,
        only_unresolved=False,
    )
    if active_actions:
        action_name = active_actions[0]
        target_ids = actions[action_name]
        known_ids = {item.item_id for item in full_report.items} | set(review_state_store.items)
        missing_ids = sorted(item_id for item_id in target_ids if item_id not in known_ids)
        if missing_ids:
            raise typer.BadParameter(f"Unknown review item id(s): {', '.join(missing_ids)}")
        item_type_by_id = {item.item_id: item.item_type for item in full_report.items}
        summary_by_id = {item.item_id: item.summary for item in full_report.items}
        apply_review_action(
            review_state_store,
            item_ids=target_ids,
            status=action_name,
            days=days,
            item_type_by_id=item_type_by_id,
            summary_by_id=summary_by_id,
        )
        save_review_state(review_state_path, review_state_store)
        full_report = _build_review_report(
            music,
            records,
            refresh_summary=refresh_summary,
            app_config=app_config,
            review_state_store=review_state_store,
            max_items=None,
            min_priority=None,
            include_types=None,
            exclude_types=None,
            include_ignored=True,
            include_snoozed=True,
            include_resolved=True,
            only_unresolved=False,
        )
    else:
        save_review_state(review_state_path, review_state_store)

    if list_states:
        console.print(_review_states_table(review_state_store))

    report = _build_review_report(
        music,
        records,
        refresh_summary=refresh_summary,
        app_config=app_config,
        review_state_store=review_state_store,
        max_items=max_items,
        min_priority=min_priority,
        include_types=include_types or None,
        exclude_types=exclude_types or None,
        include_ignored=include_ignored,
        include_snoozed=include_snoozed,
        include_resolved=include_resolved,
        only_unresolved=only_unresolved,
    )
    meta = _config_meta(app_config, "review", "health", "duplicates")
    json_path = _write_review_json(out, music, report, refresh_summary=refresh_summary, config_meta=meta)
    md_path = _write_review_md(out, report, refresh_summary=refresh_summary, config_meta=meta)
    console.print(_review_summary_table(report))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("review-plan")
def review_plan_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    item_id: list[str] | None = typer.Option(None, "--item-id", help="Repeat to select one or more review item ids"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for review plan reports"),
    incremental: bool = typer.Option(False, "--incremental", help="Reuse cached scan records for unchanged files"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    review_state: Path | None = typer.Option(None, "--review-state", help="Path to review triage state file"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    selected_ids = list(item_id or [])
    if not selected_ids:
        raise typer.BadParameter("Provide at least one --item-id")

    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    state_path = state or _default_state_path(out)
    review_state_path = review_state or _default_review_state_path(out)
    review_state_store = load_review_state(review_state_path)
    records, refresh_summary = _load_library_records(music, incremental=incremental, state_path=state_path)
    review_report = _build_review_report(
        music,
        records,
        refresh_summary=refresh_summary,
        app_config=app_config,
        review_state_store=review_state_store,
        max_items=None,
        min_priority=None,
        include_types=None,
        exclude_types=None,
        include_ignored=True,
        include_snoozed=True,
        include_resolved=True,
        only_unresolved=False,
    )
    plan_report = build_action_plan_report(music, records, review_report, source_review_item_ids=selected_ids)
    meta = _config_meta(app_config, "review", "health", "duplicates")
    json_path = _write_action_plan_json(out, music, plan_report, config_meta=meta)
    md_path = _write_action_plan_md(out, plan_report, config_meta=meta)
    console.print(_action_plan_summary_table(plan_report))
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("apply-review-plan")
def apply_review_plan_cmd(
    plan: Path = typer.Argument(..., help="Path to review_plan.json"),
    out: Path | None = typer.Option(None, "--out", help="Output folder for apply result reports"),
    review_state: Path | None = typer.Option(None, "--review-state", help="Path to review triage state file"),
    history: Path | None = typer.Option(None, "--history", help="Path to applied review history ledger"),
    backup_dir: Path | None = typer.Option(None, "--backup-dir", help="Optional backup directory for low-risk file writes"),
) -> None:
    if not plan.exists():
        raise typer.BadParameter(f"Plan file does not exist: {plan}")
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    report = ActionPlanReport.from_dict(plan_payload)
    output_dir = out or plan.parent
    ensure_out_dir(output_dir)
    review_state_path = review_state or _default_review_state_path(output_dir)
    history_path = history or _default_history_path(output_dir)
    review_state_store = load_review_state(review_state_path)
    results = apply_action_plan_report(report, review_state=review_state_store, backup_dir=backup_dir)
    save_review_state(review_state_path, review_state_store)
    ledger = load_operation_ledger(history_path)
    batch = append_operation_batch(ledger, plan_report=report, results=results)
    save_operation_ledger(history_path, ledger)
    json_path = _write_applied_plan_json(output_dir, results)
    md_path = _write_applied_plan_md(output_dir, results)
    table = Table(title="Applied Review Plan Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Plans", str(len(results)))
    table.add_row("Successful", str(sum(1 for item in results if item.status == "ok")))
    table.add_row("Partial failure", str(sum(1 for item in results if item.status == "partial_failure")))
    table.add_row("Skipped", str(sum(1 for item in results if item.status == "skipped")))
    table.add_row("History batch", batch.batch_id)
    console.print(table)
    console.print(f"[green]Wrote:[/green] {json_path}")
    console.print(f"[green]Wrote:[/green] {md_path}")


@app.command("history")
def history_cmd(
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder containing review history"),
    history: Path | None = typer.Option(None, "--history", help="Path to applied review history ledger"),
) -> None:
    history_path = history or _default_history_path(out)
    ledger = load_operation_ledger(history_path)
    console.print(_history_table(ledger))


@app.command("show-history")
def show_history_cmd(
    batch_id: str = typer.Argument(..., help="History batch id"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder containing review history"),
    history: Path | None = typer.Option(None, "--history", help="Path to applied review history ledger"),
) -> None:
    history_path = history or _default_history_path(out)
    ledger = load_operation_ledger(history_path)
    batch = find_batch(ledger, batch_id)
    if batch is None:
        raise typer.BadParameter(f"History batch not found: {batch_id}")
    console.print(_history_detail_table(batch))


def _restore_or_undo_history_batch(
    *,
    batch_id: str,
    out: Path,
    history: Path | None,
    review_state: Path | None,
    alternate_restore_dir: Path | None,
    target_path: str | None,
) -> None:
    history_path = history or _default_history_path(out)
    review_state_path = review_state or _default_review_state_path(out)
    ledger = load_operation_ledger(history_path)
    batch = find_batch(ledger, batch_id)
    if batch is None:
        raise typer.BadParameter(f"History batch not found: {batch_id}")
    review_state_store = load_review_state(review_state_path)
    undo_operation_batch(
        batch,
        review_state=review_state_store,
        alternate_restore_dir=alternate_restore_dir,
        target_path=target_path,
    )
    save_review_state(review_state_path, review_state_store)
    save_operation_ledger(history_path, ledger)
    console.print(_history_detail_table(batch))


@app.command("restore-review-action")
def restore_review_action_cmd(
    batch_id: str = typer.Argument(..., help="History batch id to restore"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder containing review history"),
    history: Path | None = typer.Option(None, "--history", help="Path to applied review history ledger"),
    review_state: Path | None = typer.Option(None, "--review-state", help="Path to review triage state file"),
    alternate_restore_dir: Path | None = typer.Option(None, "--alternate-restore-dir", help="Optional alternate restore directory when original path is occupied"),
    target_path: str | None = typer.Option(None, "--target-path", help="Restore only one recorded path from the batch"),
) -> None:
    """Restore a recorded history batch using the reversible operation ledger."""
    _restore_or_undo_history_batch(
        batch_id=batch_id,
        out=out,
        history=history,
        review_state=review_state,
        alternate_restore_dir=alternate_restore_dir,
        target_path=target_path,
    )


@app.command("undo-review-action")
def undo_review_action_cmd(
    batch_id: str = typer.Argument(..., help="History batch id to undo"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder containing review history"),
    history: Path | None = typer.Option(None, "--history", help="Path to applied review history ledger"),
    review_state: Path | None = typer.Option(None, "--review-state", help="Path to review triage state file"),
    alternate_restore_dir: Path | None = typer.Option(None, "--alternate-restore-dir", help="Optional alternate restore directory when original path is occupied"),
    target_path: str | None = typer.Option(None, "--target-path", help="Undo only one recorded path from the batch"),
) -> None:
    """Compatibility alias for the same history-batch reversal path."""
    _restore_or_undo_history_batch(
        batch_id=batch_id,
        out=out,
        history=history,
        review_state=review_state,
        alternate_restore_dir=alternate_restore_dir,
        target_path=target_path,
    )


@app.command("watch")
def watch_cmd(
    music: Path = typer.Argument(Path("./music"), help="Music root folder"),
    out: Path = typer.Option(Path("data/reports"), "--out", help="Output folder for refreshed reports"),
    state: Path | None = typer.Option(None, "--state", help="Path to incremental scan state file"),
    interval: float | None = typer.Option(None, "--interval", help="Polling interval in seconds"),
    once: bool = typer.Option(False, "--once", help="Run a single incremental refresh and exit"),
    cycles: int | None = typer.Option(None, "--cycles", help="Optional number of poll cycles to run"),
    config: Path | None = typer.Option(None, "--config", help="Path to YAML config"),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
) -> None:
    if not music.exists() or not music.is_dir():
        raise typer.BadParameter(f"Music directory does not exist: {music}")
    if cycles is not None and cycles < 1:
        raise typer.BadParameter("--cycles must be >= 1")

    ensure_out_dir(out)
    app_config = _load_app_config(config, profile)
    interval = app_config.watch.default_interval_seconds if interval is None else interval
    if interval < 0.2:
        raise typer.BadParameter("--interval must be >= 0.2 seconds")
    state_path = state or _default_state_path(out)
    max_cycles = 1 if once else cycles

    def _on_refresh(result: RefreshSummary, records: list[TrackRecord]) -> None:
        review_state_path = _default_review_state_path(out)
        review_state_store = load_review_state(review_state_path)
        pipeline = build_review_pipeline(
            music,
            records,
            app_config=app_config,
            review_state=review_state_store,
            generation_mode=result.mode,
        )
        scan_stats = compute_stats(records)
        _write_scan_json(out, music, records, scan_stats, refresh_summary=result)
        _write_scan_md(out, scan_stats, refresh_summary=result)
        duplicate_meta = _config_meta(app_config, "duplicates")
        health_meta = _config_meta(app_config, "health", "duplicates")
        review_meta = _config_meta(app_config, "review", "health", "duplicates")
        _write_duplicates_json(out, music, pipeline.duplicate_report, refresh_summary=result, config_meta=duplicate_meta)
        _write_duplicates_md(out, pipeline.duplicate_report, refresh_summary=result, config_meta=duplicate_meta)
        _write_health_json(out, music, pipeline.health_report, refresh_summary=result, config_meta=health_meta)
        _write_health_md(out, pipeline.health_report, refresh_summary=result, config_meta=health_meta)
        save_review_state(review_state_path, review_state_store)
        _write_review_json(out, music, pipeline.review_report, refresh_summary=result, config_meta=review_meta)
        _write_review_md(out, pipeline.review_report, refresh_summary=result, config_meta=review_meta)
        console.print(
            f"[cyan]refresh={result.mode}[/cyan] added={len(result.changes.added_files)} "
            f"modified={len(result.changes.modified_files)} removed={len(result.changes.removed_files)} "
            f"unchanged={len(result.changes.unchanged_files)} rescanned={result.rescanned_files}"
        )

    def _callback(result_obj) -> None:
        _on_refresh(result_obj.summary, result_obj.records)

    watch_incremental_state(
        music,
        state_path,
        interval_seconds=interval,
        max_cycles=max_cycles,
        on_refresh=_callback,
    )


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
    files = iter_library_audio_files(music)
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
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to YAML config (defaults to the packaged nyxcore config)",
    ),
    profile: str | None = typer.Option(None, "--profile", help="Built-in tuning profile"),
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

    app_config = _load_app_config(config, profile)

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


def _select_preview_rows_by_confidence(input_rows: list[dict], *, min_confidence: float | None) -> tuple[list[dict], int]:
    selected: list[dict] = []
    skipped_confidence = 0
    for row in input_rows:
        confidence = row.get("confidence")
        if min_confidence is not None and confidence is not None and float(confidence) < min_confidence:
            skipped_confidence += 1
            continue
        selected.append(row)
    return selected, skipped_confidence


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
    selected, skipped_confidence = _select_preview_rows_by_confidence(input_rows, min_confidence=min_confidence)

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
    selected, skipped_confidence = _select_preview_rows_by_confidence(input_rows, min_confidence=min_confidence)

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
    """Legacy bucketed M3U export. Prefer `playlist` and saved-playlist commands for current workflows."""
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
