from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nyxcore.config import HealthConfig, ReviewConfig
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.duplicates.service import DuplicateAnalysisReport, ExactDuplicateGroup, LikelyDuplicateGroup
from nyxcore.health.service import HealthReport
from nyxcore.review_queue.state import ReviewStateStore, normalize_review_state

REVIEW_ITEM_TYPES = {
    "exact_duplicate_group",
    "likely_duplicate_group",
    "missing_metadata",
    "weak_or_placeholder_metadata",
    "artwork_missing",
    "low_quality_audio",
    "folder_hotspot",
}

PRIORITY_BAND_ORDER = {"high": 0, "medium": 1, "low": 2}


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(text.split())


def _is_placeholder(value: str | None, settings: HealthConfig) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    if normalized in {_normalize_text(item) for item in settings.placeholder_values}:
        return True
    return bool(re.compile(settings.placeholder_track_pattern, re.IGNORECASE).match(normalized))


def _sample_paths(paths: set[str] | list[str], limit: int) -> list[str]:
    return sorted(paths)[:limit]


def _stable_item_id(item_type: str, *parts: object) -> str:
    payload = json.dumps([item_type, *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{item_type}-{digest}"


def _priority_band(score: float, settings: ReviewConfig) -> str:
    if score >= settings.high_priority_score:
        return "high"
    if score >= settings.medium_priority_score:
        return "medium"
    return "low"


def _filter_items(
    items: list["ReviewQueueItem"],
    *,
    max_items: int | None,
    min_priority_band: str | None,
    include_types: set[str] | None,
    exclude_types: set[str] | None,
    include_ignored: bool,
    include_snoozed: bool,
    include_resolved: bool,
    only_unresolved: bool,
) -> list["ReviewQueueItem"]:
    filtered = items
    if only_unresolved:
        filtered = [item for item in filtered if item.review_status in {"new", "seen"}]
    else:
        if not include_ignored:
            filtered = [item for item in filtered if item.review_status != "ignored"]
        if not include_snoozed:
            filtered = [item for item in filtered if item.review_status != "snoozed"]
        if not include_resolved:
            filtered = [item for item in filtered if item.review_status != "resolved"]
    if min_priority_band is not None:
        min_rank = PRIORITY_BAND_ORDER[min_priority_band]
        filtered = [item for item in filtered if PRIORITY_BAND_ORDER[item.priority_band] <= min_rank]
    if include_types:
        filtered = [item for item in filtered if item.item_type in include_types]
    if exclude_types:
        filtered = [item for item in filtered if item.item_type not in exclude_types]
    if max_items is not None:
        filtered = filtered[:max_items]
    return filtered


@dataclass(slots=True)
class ReviewQueueItem:
    item_id: str
    item_type: str
    priority_score: float
    priority_band: str
    summary: str
    reason_summary: str
    review_status: str = "new"
    state_updated_at: str | None = None
    snooze_until: str | None = None
    affected_paths: list[str] = field(default_factory=list)
    sample_paths: list[str] = field(default_factory=list)
    folder: str | None = None
    preferred_path: str | None = None
    reclaimable_bytes: int | None = None
    confidence: float | None = None
    file_count: int = 0
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ReviewQueueSummary:
    total_items: int
    counts_by_type: dict[str, int]
    counts_by_priority_band: dict[str, int]
    counts_by_state: dict[str, int]
    total_files_referenced: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ReviewQueueMetadata:
    active_profile: str
    generation_mode: str
    min_priority_band: str | None
    include_types: list[str]
    exclude_types: list[str]
    include_ignored: bool
    include_snoozed: bool
    include_resolved: bool
    only_unresolved: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ReviewQueueReport:
    summary: ReviewQueueSummary
    metadata: ReviewQueueMetadata
    items: list[ReviewQueueItem]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(),
            "metadata": self.metadata.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }


class ReviewQueueBuilder:
    def __init__(
        self,
        *,
        review_settings: ReviewConfig | None = None,
        health_settings: HealthConfig | None = None,
    ) -> None:
        self.review_settings = review_settings or ReviewConfig()
        self.health_settings = health_settings or HealthConfig()

    def build(
        self,
        records: list[TrackRecord],
        *,
        health_report: HealthReport,
        duplicate_report: DuplicateAnalysisReport,
        active_profile: str = "default",
        generation_mode: str = "full",
        max_items: int | None = None,
        min_priority_band: str | None = None,
        include_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
        review_state: ReviewStateStore | None = None,
        include_ignored: bool = False,
        include_snoozed: bool = False,
        include_resolved: bool = False,
        only_unresolved: bool = False,
    ) -> ReviewQueueReport:
        items: list[ReviewQueueItem] = []
        items.extend(self._exact_duplicate_items(duplicate_report))
        items.extend(self._likely_duplicate_items(duplicate_report))
        items.extend(self._metadata_items(records))
        items.extend(self._artwork_items(records, health_report))
        items.extend(self._quality_items(records, health_report))
        items.extend(self._folder_hotspot_items(health_report))
        state_store = review_state or ReviewStateStore()
        normalize_review_state(state_store, active_item_ids={item.item_id for item in items})
        for item in items:
            entry = state_store.items.get(item.item_id)
            if entry is not None:
                item.review_status = entry.status
                item.state_updated_at = entry.updated_at
                item.snooze_until = entry.snooze_until
        items = sorted(items, key=self._sort_key)
        items = _filter_items(
            items,
            max_items=max_items,
            min_priority_band=min_priority_band,
            include_types=include_types,
            exclude_types=exclude_types,
            include_ignored=include_ignored,
            include_snoozed=include_snoozed,
            include_resolved=include_resolved,
            only_unresolved=only_unresolved,
        )

        counts_by_type = Counter(item.item_type for item in items)
        counts_by_priority_band = Counter(item.priority_band for item in items)
        counts_by_state = Counter(item.review_status for item in items)
        referenced_paths = {path for item in items for path in item.affected_paths}
        summary = ReviewQueueSummary(
            total_items=len(items),
            counts_by_type={key: counts_by_type[key] for key in sorted(counts_by_type)},
            counts_by_priority_band={
                key: counts_by_priority_band[key]
                for key in ("high", "medium", "low")
                if counts_by_priority_band[key] > 0
            },
            counts_by_state={key: counts_by_state[key] for key in ("new", "seen", "snoozed", "ignored", "resolved") if counts_by_state[key] > 0},
            total_files_referenced=len(referenced_paths),
        )
        metadata = ReviewQueueMetadata(
            active_profile=active_profile,
            generation_mode=generation_mode,
            min_priority_band=min_priority_band,
            include_types=sorted(include_types or []),
            exclude_types=sorted(exclude_types or []),
            include_ignored=include_ignored,
            include_snoozed=include_snoozed,
            include_resolved=include_resolved,
            only_unresolved=only_unresolved,
        )
        return ReviewQueueReport(summary=summary, metadata=metadata, items=items)

    def _sort_key(self, item: ReviewQueueItem) -> tuple:
        return (-item.priority_score, PRIORITY_BAND_ORDER[item.priority_band], item.item_type, item.item_id)

    def _exact_duplicate_items(self, report: DuplicateAnalysisReport) -> list[ReviewQueueItem]:
        items: list[ReviewQueueItem] = []
        for group in report.exact_duplicates:
            reclaimable_bytes = sum(item.file_size_bytes for item in group.files if item.path != group.preferred.path)
            score = (
                self.review_settings.exact_duplicate_base_score
                + max(0, len(group.files) - 1) * self.review_settings.files_affected_weight
                + min(24.0, (reclaimable_bytes / self.review_settings.reclaimable_bytes_unit) * 6.0)
            )
            score = round(score, 3)
            reasons = [
                f"{len(group.files)} confirmed copies",
                f"{reclaimable_bytes} reclaimable bytes",
            ]
            items.append(
                ReviewQueueItem(
                    item_id=_stable_item_id(
                        "exact_duplicate_group",
                        sorted(item.path for item in group.files),
                    ),
                    item_type="exact_duplicate_group",
                    priority_score=score,
                    priority_band=_priority_band(score, self.review_settings),
                    review_status="new",
                    state_updated_at=None,
                    snooze_until=None,
                    summary=(
                        f"Review {len(group.files)} exact duplicates; keep "
                        f"{Path(group.preferred.path).name} as the preferred copy"
                    ),
                    reason_summary=", ".join(reasons),
                    affected_paths=[item.path for item in group.files],
                    sample_paths=_sample_paths([item.path for item in group.files], self.review_settings.sample_limit),
                    preferred_path=group.preferred.path,
                    reclaimable_bytes=reclaimable_bytes,
                    file_count=len(group.files),
                    details={
                        "source_group_id": group.group_id,
                        "content_hash": group.content_hash,
                        "preferred_reasons": list(group.preferred.reasons),
                        "group_reasons": list(group.reasons),
                    },
                )
            )
        return items

    def _likely_duplicate_items(self, report: DuplicateAnalysisReport) -> list[ReviewQueueItem]:
        items: list[ReviewQueueItem] = []
        for group in report.likely_duplicates:
            score = (
                self.review_settings.likely_duplicate_base_score
                + group.confidence * self.review_settings.likely_confidence_weight
                + max(0, len(group.files) - 1) * self.review_settings.files_affected_weight
            )
            score = round(score, 3)
            items.append(
                ReviewQueueItem(
                    item_id=_stable_item_id(
                        "likely_duplicate_group",
                        sorted(item.path for item in group.files),
                    ),
                    item_type="likely_duplicate_group",
                    priority_score=score,
                    priority_band=_priority_band(score, self.review_settings),
                    review_status="new",
                    state_updated_at=None,
                    snooze_until=None,
                    summary=(
                        f"Review likely duplicate set of {len(group.files)} files; "
                        f"preferred copy is {Path(group.preferred.path).name}"
                    ),
                    reason_summary=", ".join(group.reasons[: self.review_settings.sample_limit]) or "high-confidence likely duplicate",
                    affected_paths=[item.path for item in group.files],
                    sample_paths=_sample_paths([item.path for item in group.files], self.review_settings.sample_limit),
                    preferred_path=group.preferred.path,
                    confidence=group.confidence,
                    file_count=len(group.files),
                    details={
                        "source_group_id": group.group_id,
                        "relationship_count": len(group.relationships),
                        "preferred_reasons": list(group.preferred.reasons),
                    },
                )
            )
        return items

    def _metadata_items(self, records: list[TrackRecord]) -> list[ReviewQueueItem]:
        missing_title: set[str] = set()
        missing_artist: set[str] = set()
        missing_album: set[str] = set()
        placeholder: set[str] = set()
        for record in records:
            if WarningCode.missing_title in record.warnings:
                missing_title.add(record.path)
            if WarningCode.missing_artist in record.warnings:
                missing_artist.add(record.path)
            if WarningCode.missing_album in record.warnings:
                missing_album.add(record.path)
            if any(_is_placeholder(record.tags.get(field), self.health_settings) for field in ("title", "artist", "album")):
                placeholder.add(record.path)

        items: list[ReviewQueueItem] = []
        impacted = missing_title | missing_artist | missing_album
        if impacted:
            score = (
                self.review_settings.missing_metadata_base_score
                + len(missing_artist) * self.review_settings.missing_artist_weight
                + len(missing_title) * self.review_settings.missing_title_weight
                + len(missing_album) * self.review_settings.missing_album_weight
                + len(impacted) * self.review_settings.files_affected_weight
            )
            score = round(min(score, 99.0), 3)
            items.append(
                ReviewQueueItem(
                    item_id=_stable_item_id("missing_metadata", sorted(impacted)),
                    item_type="missing_metadata",
                    priority_score=score,
                    priority_band=_priority_band(score, self.review_settings),
                    review_status="new",
                    state_updated_at=None,
                    snooze_until=None,
                    summary=(
                        f"Fix missing core metadata on {len(impacted)} files "
                        f"(artist: {len(missing_artist)}, title: {len(missing_title)}, album: {len(missing_album)})"
                    ),
                    reason_summary="Missing title/artist/album reduces searchability and playlist quality",
                    affected_paths=sorted(impacted),
                    sample_paths=_sample_paths(impacted, self.review_settings.sample_limit),
                    file_count=len(impacted),
                    details={
                        "missing_title_count": len(missing_title),
                        "missing_artist_count": len(missing_artist),
                        "missing_album_count": len(missing_album),
                    },
                )
            )
        if placeholder:
            score = (
                self.review_settings.weak_metadata_base_score
                + len(placeholder) * self.review_settings.files_affected_weight
            )
            score = round(min(score, 99.0), 3)
            items.append(
                ReviewQueueItem(
                    item_id=_stable_item_id("weak_or_placeholder_metadata", sorted(placeholder)),
                    item_type="weak_or_placeholder_metadata",
                    priority_score=score,
                    priority_band=_priority_band(score, self.review_settings),
                    review_status="new",
                    state_updated_at=None,
                    snooze_until=None,
                    summary=f"Replace weak or placeholder metadata on {len(placeholder)} files",
                    reason_summary="Placeholder tags like unknown or track 01 reduce library quality",
                    affected_paths=sorted(placeholder),
                    sample_paths=_sample_paths(placeholder, self.review_settings.sample_limit),
                    file_count=len(placeholder),
                )
            )
        return items

    def _artwork_items(self, records: list[TrackRecord], health_report: HealthReport) -> list[ReviewQueueItem]:
        missing_art = {record.path for record in records if not record.has_cover_art}
        if not missing_art:
            return []
        score = (
            self.review_settings.artwork_missing_base_score
            + len(missing_art) * self.review_settings.files_affected_weight
            + (100.0 - health_report.artwork.coverage_percent) * self.review_settings.artwork_gap_weight
        )
        score = round(min(score, 99.0), 3)
        return [
            ReviewQueueItem(
                item_id=_stable_item_id("artwork_missing", sorted(missing_art)),
                item_type="artwork_missing",
                priority_score=score,
                priority_band=_priority_band(score, self.review_settings),
                review_status="new",
                state_updated_at=None,
                snooze_until=None,
                summary=(
                    f"Add artwork for {len(missing_art)} files; current coverage is "
                    f"{health_report.artwork.coverage_percent:.2f}%"
                ),
                reason_summary="Artwork improves browsing and release-level consistency",
                affected_paths=sorted(missing_art),
                sample_paths=_sample_paths(missing_art, self.review_settings.sample_limit),
                file_count=len(missing_art),
                details={"coverage_percent": health_report.artwork.coverage_percent},
            )
        ]

    def _quality_items(self, records: list[TrackRecord], health_report: HealthReport) -> list[ReviewQueueItem]:
        low_quality: set[str] = set()
        low_bitrate_count = 0
        unreadable_count = 0
        duration_outlier_count = health_report.quality.duration_outliers.count
        for record in records:
            if WarningCode.low_bitrate in record.warnings:
                low_quality.add(record.path)
                low_bitrate_count += 1
            if WarningCode.read_error in record.warnings or WarningCode.tag_parse_error in record.warnings:
                low_quality.add(record.path)
                unreadable_count += 1
        if not low_quality and duration_outlier_count == 0:
            return []
        score = (
            self.review_settings.low_quality_base_score
            + low_bitrate_count * self.review_settings.low_bitrate_weight
            + unreadable_count * self.review_settings.unreadable_weight
            + len(low_quality) * self.review_settings.files_affected_weight
        )
        score = round(min(score, 99.0), 3)
        return [
            ReviewQueueItem(
                item_id=_stable_item_id("low_quality_audio", sorted(low_quality)),
                item_type="low_quality_audio",
                priority_score=score,
                priority_band=_priority_band(score, self.review_settings),
                review_status="new",
                state_updated_at=None,
                snooze_until=None,
                summary=(
                    f"Review {len(low_quality)} low-quality or unreadable files "
                    f"({low_bitrate_count} low bitrate, {unreadable_count} unreadable)"
                ),
                reason_summary="Low bitrate and unreadable files are conservative quality risks",
                affected_paths=sorted(low_quality),
                sample_paths=_sample_paths(low_quality, self.review_settings.sample_limit),
                file_count=len(low_quality),
                details={
                    "low_bitrate_count": low_bitrate_count,
                    "unreadable_count": unreadable_count,
                    "duration_outlier_count": duration_outlier_count,
                },
            )
        ]

    def _folder_hotspot_items(self, health_report: HealthReport) -> list[ReviewQueueItem]:
        items: list[ReviewQueueItem] = []
        for folder_issue in health_report.naming.high_issue_folders:
            score = (
                self.review_settings.folder_hotspot_base_score
                + folder_issue.issue_count * self.review_settings.folder_issue_weight
            )
            score = round(min(score, 99.0), 3)
            items.append(
                ReviewQueueItem(
                    item_id=_stable_item_id("folder_hotspot", folder_issue.folder),
                    item_type="folder_hotspot",
                    priority_score=score,
                    priority_band=_priority_band(score, self.review_settings),
                    review_status="new",
                    state_updated_at=None,
                    snooze_until=None,
                    summary=f"Inspect folder {folder_issue.folder} with {folder_issue.issue_count} concentrated issues",
                    reason_summary="Multiple issues cluster in this folder",
                    affected_paths=[],
                    sample_paths=[],
                    folder=folder_issue.folder,
                    file_count=folder_issue.issue_count,
                    details={"issue_count": folder_issue.issue_count},
                )
            )
        return items


def build_review_queue(
    records: list[TrackRecord],
    *,
    health_report: HealthReport,
    duplicate_report: DuplicateAnalysisReport,
    review_settings: ReviewConfig | None = None,
    health_settings: HealthConfig | None = None,
    active_profile: str = "default",
    generation_mode: str = "full",
    max_items: int | None = None,
    min_priority_band: str | None = None,
    include_types: set[str] | None = None,
    exclude_types: set[str] | None = None,
    review_state: ReviewStateStore | None = None,
    include_ignored: bool = False,
    include_snoozed: bool = False,
    include_resolved: bool = False,
    only_unresolved: bool = False,
) -> ReviewQueueReport:
    if min_priority_band is not None and min_priority_band not in PRIORITY_BAND_ORDER:
        raise ValueError(f"Unknown priority band: {min_priority_band}")
    invalid_types = (include_types or set()) | (exclude_types or set())
    unknown_types = sorted(item for item in invalid_types if item not in REVIEW_ITEM_TYPES)
    if unknown_types:
        raise ValueError(f"Unknown review item type(s): {', '.join(unknown_types)}")
    return ReviewQueueBuilder(review_settings=review_settings, health_settings=health_settings).build(
        records,
        health_report=health_report,
        duplicate_report=duplicate_report,
        active_profile=active_profile,
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
