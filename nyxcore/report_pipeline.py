from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nyxcore.config import NyxConfig
from nyxcore.core.track import TrackRecord
from nyxcore.duplicates.service import DuplicateAnalysisReport, analyze_duplicates
from nyxcore.health.service import HealthReport, build_health_report
from nyxcore.review_queue.service import ReviewQueueReport, build_review_queue
from nyxcore.review_queue.state import ReviewStateStore


@dataclass(frozen=True)
class DuplicateHealthReports:
    duplicate_report: DuplicateAnalysisReport
    health_report: HealthReport


@dataclass(frozen=True)
class ReviewPipelineReports:
    duplicate_report: DuplicateAnalysisReport
    health_report: HealthReport
    review_report: ReviewQueueReport


def build_duplicate_report(records: list[TrackRecord], *, app_config: NyxConfig) -> DuplicateAnalysisReport:
    return analyze_duplicates(records, settings=app_config.duplicates)


def build_duplicate_health_reports(
    music: Path,
    records: list[TrackRecord],
    *,
    app_config: NyxConfig,
    duplicate_report: DuplicateAnalysisReport | None = None,
) -> DuplicateHealthReports:
    resolved_duplicate_report = duplicate_report or build_duplicate_report(records, app_config=app_config)
    health_report = build_health_report(
        music,
        records,
        settings=app_config.health,
        duplicate_report=resolved_duplicate_report,
    )
    return DuplicateHealthReports(
        duplicate_report=resolved_duplicate_report,
        health_report=health_report,
    )


def build_review_pipeline(
    music: Path,
    records: list[TrackRecord],
    *,
    app_config: NyxConfig,
    review_state: ReviewStateStore,
    generation_mode: str,
    max_items: int | None = None,
    min_priority_band: str | None = None,
    include_types: set[str] | None = None,
    exclude_types: set[str] | None = None,
    include_ignored: bool = False,
    include_snoozed: bool = False,
    include_resolved: bool = False,
    only_unresolved: bool = False,
) -> ReviewPipelineReports:
    reports = build_duplicate_health_reports(music, records, app_config=app_config)
    review_report = build_review_queue(
        records,
        health_report=reports.health_report,
        duplicate_report=reports.duplicate_report,
        review_settings=app_config.review,
        health_settings=app_config.health,
        active_profile=app_config.profile,
        generation_mode=generation_mode,
        max_items=max_items,
        min_priority_band=min_priority_band,
        include_types=include_types,
        exclude_types=exclude_types,
        review_state=review_state,
        include_ignored=include_ignored,
        include_snoozed=include_snoozed,
        include_resolved=include_resolved,
        only_unresolved=only_unresolved,
    )
    return ReviewPipelineReports(
        duplicate_report=reports.duplicate_report,
        health_report=reports.health_report,
        review_report=review_report,
    )
