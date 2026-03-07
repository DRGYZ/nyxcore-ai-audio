from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mutagen import File as MutagenFile

from nyxcore.config import HealthConfig
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.duplicates.service import DuplicateAnalysisReport, LOSSLESS_EXTENSIONS, analyze_duplicates

FILENAME_SPLIT_RE = re.compile(r"\s+-\s+")


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(cleaned.split())


def _sample_paths(paths: list[str], limit: int) -> list[str]:
    return sorted(paths)[:limit]


def _folder_key(path: Path, root: Path) -> str:
    try:
        rel = path.parent.relative_to(root)
        return "." if str(rel) == "." else rel.as_posix()
    except ValueError:
        return str(path.parent)


def _is_placeholder(value: str | None, settings: HealthConfig) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    if normalized in {_normalize_text(item) for item in settings.placeholder_values}:
        return True
    return bool(re.compile(settings.placeholder_track_pattern, re.IGNORECASE).match(normalized))


def _parse_filename_artist_title(stem: str) -> tuple[str | None, str | None]:
    parts = FILENAME_SPLIT_RE.split(stem, maxsplit=1)
    if len(parts) != 2:
        return None, None
    left = _normalize_text(parts[0])
    right = _normalize_text(parts[1])
    if not left or not right:
        return None, None
    return left, right


@dataclass(slots=True)
class IssueCount:
    count: int
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OverviewSection:
    total_audio_files: int
    total_folders_touched: int
    format_breakdown: dict[str, int]
    total_library_size_bytes: int
    duplicate_exact_groups: int
    duplicate_likely_groups: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MetadataSection:
    missing_title: IssueCount
    missing_artist: IssueCount
    missing_album: IssueCount
    placeholder_metadata: IssueCount
    suspicious_title_artist_swaps: IssueCount

    def to_dict(self) -> dict:
        return {
            "missing_title": self.missing_title.to_dict(),
            "missing_artist": self.missing_artist.to_dict(),
            "missing_album": self.missing_album.to_dict(),
            "placeholder_metadata": self.placeholder_metadata.to_dict(),
            "suspicious_title_artist_swaps": self.suspicious_title_artist_swaps.to_dict(),
        }


@dataclass(slots=True)
class ArtworkSection:
    with_artwork: int
    without_artwork: int
    coverage_percent: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class QualitySection:
    bitrate_buckets: dict[str, int]
    low_bitrate_files: IssueCount
    lossless_files: int
    lossy_files: int
    duration_outliers: IssueCount
    unreadable_or_unparseable_files: IssueCount

    def to_dict(self) -> dict:
        return {
            "bitrate_buckets": dict(self.bitrate_buckets),
            "low_bitrate_files": self.low_bitrate_files.to_dict(),
            "lossless_files": self.lossless_files,
            "lossy_files": self.lossy_files,
            "duration_outliers": self.duration_outliers.to_dict(),
            "unreadable_or_unparseable_files": self.unreadable_or_unparseable_files.to_dict(),
        }


@dataclass(slots=True)
class FolderIssue:
    folder: str
    issue_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class NamingSection:
    noisy_filenames: IssueCount
    long_filenames: IssueCount
    filename_metadata_disagreement: IssueCount
    high_issue_folders: list[FolderIssue]

    def to_dict(self) -> dict:
        return {
            "noisy_filenames": self.noisy_filenames.to_dict(),
            "long_filenames": self.long_filenames.to_dict(),
            "filename_metadata_disagreement": self.filename_metadata_disagreement.to_dict(),
            "high_issue_folders": [item.to_dict() for item in self.high_issue_folders],
        }


@dataclass(slots=True)
class DuplicatesSection:
    exact_duplicate_groups: int
    likely_duplicate_groups: int
    files_in_exact_duplicates: int
    files_in_likely_duplicates: int
    total_files_in_duplicates: int
    reclaimable_bytes_exact: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RecommendationItem:
    category: str
    count: int
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PrioritySection:
    top_issue_categories: list[RecommendationItem]
    top_problematic_folders: list[FolderIssue]
    recommended_actions: list[str]

    def to_dict(self) -> dict:
        return {
            "top_issue_categories": [item.to_dict() for item in self.top_issue_categories],
            "top_problematic_folders": [item.to_dict() for item in self.top_problematic_folders],
            "recommended_actions": list(self.recommended_actions),
        }


@dataclass(slots=True)
class HealthReport:
    overview: OverviewSection
    metadata: MetadataSection
    artwork: ArtworkSection
    quality: QualitySection
    naming: NamingSection
    duplicates: DuplicatesSection
    priorities: PrioritySection

    def to_dict(self) -> dict:
        return {
            "overview": self.overview.to_dict(),
            "metadata": self.metadata.to_dict(),
            "artwork": self.artwork.to_dict(),
            "quality": self.quality.to_dict(),
            "naming": self.naming.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "priorities": self.priorities.to_dict(),
        }


class CollectionHealthAnalyzer:
    def analyze(
        self,
        root: Path,
        records: list[TrackRecord],
        *,
        settings: HealthConfig | None = None,
        duplicate_report: DuplicateAnalysisReport | None = None,
    ) -> HealthReport:
        settings = settings or HealthConfig()
        if duplicate_report is None:
            duplicate_report = analyze_duplicates(records)

        folder_counter: Counter[str] = Counter()
        format_counter: Counter[str] = Counter()
        total_size = 0

        missing_title: list[str] = []
        missing_artist: list[str] = []
        missing_album: list[str] = []
        placeholder_metadata: list[str] = []
        suspicious_swaps: list[str] = []
        low_bitrate: list[str] = []
        duration_outliers: list[str] = []
        unreadable: list[str] = []
        noisy_filenames: list[str] = []
        long_filenames: list[str] = []
        filename_disagreement: list[str] = []
        folder_issue_counter: Counter[str] = Counter()

        bitrate_buckets = {
            "unknown": 0,
            "<128k": 0,
            "128k-191k": 0,
            "192k-255k": 0,
            ">=256k": 0,
        }
        artwork_with = 0
        artwork_without = 0
        lossless = 0
        lossy = 0

        for record in sorted(records, key=lambda item: item.path):
            path = Path(record.path)
            folder = _folder_key(path, root)
            folder_counter[folder] += 1
            format_counter[path.suffix.lower()] += 1
            total_size += int(record.file_size_bytes)
            if path.suffix.lower() in LOSSLESS_EXTENSIONS:
                lossless += 1
            else:
                lossy += 1

            issue_hits = 0
            if WarningCode.missing_title in record.warnings:
                missing_title.append(record.path)
                issue_hits += 1
            if WarningCode.missing_artist in record.warnings:
                missing_artist.append(record.path)
                issue_hits += 1
            if WarningCode.missing_album in record.warnings:
                missing_album.append(record.path)
                issue_hits += 1
            if any(_is_placeholder(record.tags.get(field), settings) for field in ("title", "artist", "album")):
                placeholder_metadata.append(record.path)
                issue_hits += 1
            if self._is_suspicious_swap(path, record):
                suspicious_swaps.append(record.path)
                issue_hits += 1

            if record.has_cover_art:
                artwork_with += 1
            else:
                artwork_without += 1

            bitrate_buckets[self._bitrate_bucket(path, settings)] += 1
            if WarningCode.low_bitrate in record.warnings:
                low_bitrate.append(record.path)
                issue_hits += 1
            if record.duration_seconds is not None and (
                record.duration_seconds < settings.short_duration_seconds
                or record.duration_seconds > settings.long_duration_seconds
            ):
                duration_outliers.append(record.path)
                issue_hits += 1
            if WarningCode.read_error in record.warnings or WarningCode.tag_parse_error in record.warnings:
                unreadable.append(record.path)
                issue_hits += 1
            if any(
                warning in record.warnings
                for warning in (
                    WarningCode.filename_youtube_noise,
                    WarningCode.filename_brackets_noise,
                    WarningCode.filename_feat_pattern,
                )
            ):
                noisy_filenames.append(record.path)
                issue_hits += 1
            if len(path.name) > settings.long_filename_threshold:
                long_filenames.append(record.path)
                issue_hits += 1
            if self._has_filename_metadata_disagreement(path, record):
                filename_disagreement.append(record.path)
                issue_hits += 1
            if issue_hits > 0:
                folder_issue_counter[folder] += issue_hits

        coverage = 0.0 if not records else round((artwork_with / len(records)) * 100.0, 2)
        duplicates_section = self._build_duplicates_section(duplicate_report)
        high_issue_folders = [
            FolderIssue(folder=folder, issue_count=count)
            for folder, count in sorted(folder_issue_counter.items(), key=lambda item: (-item[1], item[0]))[
                : settings.sample_limit
            ]
        ]
        priority_section = self._build_priorities(
            duplicates_section=duplicates_section,
            missing_artist_count=len(missing_artist),
            placeholder_count=len(placeholder_metadata),
            noisy_filename_count=len(noisy_filenames),
            low_bitrate_count=len(low_bitrate),
            unreadable_count=len(unreadable),
            high_issue_folders=high_issue_folders,
            sample_limit=settings.sample_limit,
        )

        return HealthReport(
            overview=OverviewSection(
                total_audio_files=len(records),
                total_folders_touched=len(folder_counter),
                format_breakdown=dict(sorted(format_counter.items())),
                total_library_size_bytes=total_size,
                duplicate_exact_groups=duplicate_report.summary.exact_group_count,
                duplicate_likely_groups=duplicate_report.summary.likely_group_count,
            ),
            metadata=MetadataSection(
                missing_title=IssueCount(len(missing_title), _sample_paths(missing_title, settings.sample_limit)),
                missing_artist=IssueCount(len(missing_artist), _sample_paths(missing_artist, settings.sample_limit)),
                missing_album=IssueCount(len(missing_album), _sample_paths(missing_album, settings.sample_limit)),
                placeholder_metadata=IssueCount(
                    len(placeholder_metadata),
                    _sample_paths(placeholder_metadata, settings.sample_limit),
                ),
                suspicious_title_artist_swaps=IssueCount(
                    len(suspicious_swaps),
                    _sample_paths(suspicious_swaps, settings.sample_limit),
                ),
            ),
            artwork=ArtworkSection(
                with_artwork=artwork_with,
                without_artwork=artwork_without,
                coverage_percent=coverage,
            ),
            quality=QualitySection(
                bitrate_buckets=bitrate_buckets,
                low_bitrate_files=IssueCount(len(low_bitrate), _sample_paths(low_bitrate, settings.sample_limit)),
                lossless_files=lossless,
                lossy_files=lossy,
                duration_outliers=IssueCount(
                    len(duration_outliers),
                    _sample_paths(duration_outliers, settings.sample_limit),
                ),
                unreadable_or_unparseable_files=IssueCount(
                    len(unreadable),
                    _sample_paths(unreadable, settings.sample_limit),
                ),
            ),
            naming=NamingSection(
                noisy_filenames=IssueCount(
                    len(noisy_filenames),
                    _sample_paths(noisy_filenames, settings.sample_limit),
                ),
                long_filenames=IssueCount(len(long_filenames), _sample_paths(long_filenames, settings.sample_limit)),
                filename_metadata_disagreement=IssueCount(
                    len(filename_disagreement),
                    _sample_paths(filename_disagreement, settings.sample_limit),
                ),
                high_issue_folders=high_issue_folders,
            ),
            duplicates=duplicates_section,
            priorities=priority_section,
        )

    def _bitrate_bucket(self, path: Path, settings: HealthConfig) -> str:
        bitrate = self._load_bitrate(path)
        if bitrate is None:
            return "unknown"
        threshold = settings.low_bitrate_threshold_bps
        if bitrate < threshold:
            return "<128k"
        if bitrate < 192_000:
            return "128k-191k"
        if bitrate < 256_000:
            return "192k-255k"
        return ">=256k"

    def _load_bitrate(self, path: Path) -> int | None:
        try:
            audio = MutagenFile(path)
            info = getattr(audio, "info", None)
            bitrate = getattr(info, "bitrate", None) if info is not None else None
            if bitrate is None:
                return None
            return int(bitrate)
        except Exception:
            return None

    def _is_suspicious_swap(self, path: Path, record: TrackRecord) -> bool:
        parsed_artist, parsed_title = _parse_filename_artist_title(path.stem)
        if parsed_artist is None or parsed_title is None:
            return False
        title = _normalize_text(record.tags.get("title"))
        artist = _normalize_text(record.tags.get("artist"))
        return bool(title and artist and title == parsed_artist and artist == parsed_title)

    def _has_filename_metadata_disagreement(self, path: Path, record: TrackRecord) -> bool:
        parsed_artist, parsed_title = _parse_filename_artist_title(path.stem)
        if parsed_artist is None or parsed_title is None:
            return False
        title = _normalize_text(record.tags.get("title"))
        artist = _normalize_text(record.tags.get("artist"))
        if not title or not artist:
            return False
        return title != parsed_title and artist != parsed_artist

    def _build_duplicates_section(self, report: DuplicateAnalysisReport) -> DuplicatesSection:
        exact_files = sum(len(group.files) for group in report.exact_duplicates)
        likely_files = sum(len(group.files) for group in report.likely_duplicates)
        all_dup_paths = {
            item.path for group in report.exact_duplicates for item in group.files
        } | {
            item.path for group in report.likely_duplicates for item in group.files
        }
        reclaimable = 0
        for group in report.exact_duplicates:
            preferred_path = group.preferred.path
            reclaimable += sum(item.file_size_bytes for item in group.files if item.path != preferred_path)
        return DuplicatesSection(
            exact_duplicate_groups=report.summary.exact_group_count,
            likely_duplicate_groups=report.summary.likely_group_count,
            files_in_exact_duplicates=exact_files,
            files_in_likely_duplicates=likely_files,
            total_files_in_duplicates=len(all_dup_paths),
            reclaimable_bytes_exact=reclaimable,
        )

    def _build_priorities(
        self,
        *,
        duplicates_section: DuplicatesSection,
        missing_artist_count: int,
        placeholder_count: int,
        noisy_filename_count: int,
        low_bitrate_count: int,
        unreadable_count: int,
        high_issue_folders: list[FolderIssue],
        sample_limit: int,
    ) -> PrioritySection:
        categories = [
            ("exact_duplicates", duplicates_section.exact_duplicate_groups, "Review exact duplicates first"),
            ("missing_artist", missing_artist_count, "Fix missing artist tags"),
            ("placeholder_metadata", placeholder_count, "Replace placeholder metadata values"),
            ("noisy_filenames", noisy_filename_count, "Normalize noisy filenames"),
            ("low_bitrate", low_bitrate_count, "Review low-bitrate files for better sources"),
            ("unreadable_files", unreadable_count, "Investigate unreadable or tag-unparseable files"),
        ]
        top_issue_categories = [
            RecommendationItem(category=category, count=count, action=action)
            for category, count, action in sorted(categories, key=lambda item: (-item[1], item[0]))
            if count > 0
        ][:sample_limit]

        actions: list[str] = []
        for item in top_issue_categories:
            if item.category == "missing_artist" and high_issue_folders:
                actions.append(f"{item.action} in {high_issue_folders[0].folder}")
            else:
                actions.append(item.action)
        if high_issue_folders:
            actions.append(f"Inspect folder {high_issue_folders[0].folder} for concentrated issues")
        return PrioritySection(
            top_issue_categories=top_issue_categories,
            top_problematic_folders=high_issue_folders,
            recommended_actions=list(dict.fromkeys(actions))[:sample_limit],
        )


def build_health_report(
    root: Path,
    records: list[TrackRecord],
    *,
    settings: HealthConfig | None = None,
    duplicate_report: DuplicateAnalysisReport | None = None,
) -> HealthReport:
    return CollectionHealthAnalyzer().analyze(root, records, settings=settings, duplicate_report=duplicate_report)
