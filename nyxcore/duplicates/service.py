from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from mutagen import File as MutagenFile

from nyxcore.config import DuplicateConfig
from nyxcore.core.track import TrackRecord

LOSSLESS_EXTENSIONS = {".flac", ".wav", ".aiff", ".aif"}
PARTIAL_HASH_BYTES = 65_536


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(text.split())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _duration_score(a: float | None, b: float | None, settings: DuplicateConfig) -> tuple[float, str | None]:
    if a is None or b is None:
        return 0.0, None
    delta = abs(float(a) - float(b))
    if delta <= settings.duration_close_seconds:
        return 0.15, f"duration_delta={delta:.2f}s"
    if delta <= settings.duration_near_seconds:
        return 0.08, f"duration_delta={delta:.2f}s"
    return 0.0, None


def _metadata_richness(tags: dict[str, str | None]) -> int:
    return sum(1 for key in ("title", "artist", "album", "albumartist", "genre", "date", "tracknumber") if tags.get(key))


def _partial_hash(path: Path, *, chunk_size: int = PARTIAL_HASH_BYTES) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as f:
        if size <= chunk_size * 2:
            digest.update(f.read())
        else:
            digest.update(f.read(chunk_size))
            f.seek(-chunk_size, 2)
            digest.update(f.read(chunk_size))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _full_hash(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_bitrate_bps(path: Path) -> int | None:
    try:
        audio = MutagenFile(path)
        info = getattr(audio, "info", None)
        bitrate = getattr(info, "bitrate", None) if info is not None else None
        if bitrate is None:
            return None
        return int(bitrate)
    except Exception:
        return None


@dataclass(slots=True)
class DuplicateTrackInfo:
    path: str
    file_size_bytes: int
    extension: str
    duration_seconds: float | None
    bitrate_bps: int | None
    title: str | None
    artist: str | None
    album: str | None
    has_cover_art: bool
    metadata_fields_present: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PreferredCopyRecommendation:
    path: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ExactDuplicateGroup:
    group_id: str
    content_hash: str
    files: list[DuplicateTrackInfo]
    preferred: PreferredCopyRecommendation
    reasons: list[str] = field(default_factory=lambda: ["full_hash_match"])

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "content_hash": self.content_hash,
            "files": [item.to_dict() for item in self.files],
            "preferred": self.preferred.to_dict(),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class LikelyDuplicateRelationship:
    paths: list[str]
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class LikelyDuplicateGroup:
    group_id: str
    confidence: float
    files: list[DuplicateTrackInfo]
    relationships: list[LikelyDuplicateRelationship]
    preferred: PreferredCopyRecommendation
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "confidence": self.confidence,
            "files": [item.to_dict() for item in self.files],
            "relationships": [item.to_dict() for item in self.relationships],
            "preferred": self.preferred.to_dict(),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class DuplicateSummary:
    total_tracks: int
    exact_group_count: int
    exact_duplicate_file_count: int
    likely_group_count: int
    likely_duplicate_file_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DuplicateAnalysisReport:
    summary: DuplicateSummary
    exact_duplicates: list[ExactDuplicateGroup]
    likely_duplicates: list[LikelyDuplicateGroup]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(),
            "exact_duplicates": [group.to_dict() for group in self.exact_duplicates],
            "likely_duplicates": [group.to_dict() for group in self.likely_duplicates],
        }


@dataclass(slots=True)
class _DuplicateCandidate:
    record: TrackRecord
    path: Path
    size: int
    extension: str
    stem_normalized: str
    title_normalized: str
    artist_normalized: str
    album_normalized: str
    duration_seconds: float | None
    bitrate_bps: int | None
    has_cover_art: bool
    metadata_fields_present: int

    def to_track_info(self) -> DuplicateTrackInfo:
        return DuplicateTrackInfo(
            path=str(self.path),
            file_size_bytes=self.size,
            extension=self.extension,
            duration_seconds=self.duration_seconds,
            bitrate_bps=self.bitrate_bps,
            title=self.record.tags.get("title"),
            artist=self.record.tags.get("artist"),
            album=self.record.tags.get("album"),
            has_cover_art=self.has_cover_art,
            metadata_fields_present=self.metadata_fields_present,
        )


class DuplicateAnalyzer:
    def __init__(self, settings: DuplicateConfig | None = None) -> None:
        self.settings = settings or DuplicateConfig()

    def build_candidates(self, records: list[TrackRecord]) -> list[_DuplicateCandidate]:
        candidates: list[_DuplicateCandidate] = []
        for record in sorted(records, key=lambda item: item.path):
            path = Path(record.path)
            candidates.append(
                _DuplicateCandidate(
                    record=record,
                    path=path,
                    size=int(record.file_size_bytes),
                    extension=path.suffix.lower(),
                    stem_normalized=_normalize_text(path.stem),
                    title_normalized=_normalize_text(record.tags.get("title")),
                    artist_normalized=_normalize_text(record.tags.get("artist")),
                    album_normalized=_normalize_text(record.tags.get("album")),
                    duration_seconds=record.duration_seconds,
                    bitrate_bps=_load_bitrate_bps(path),
                    has_cover_art=record.has_cover_art,
                    metadata_fields_present=_metadata_richness(record.tags),
                )
            )
        return candidates

    def analyze(self, records: list[TrackRecord]) -> DuplicateAnalysisReport:
        candidates = self.build_candidates(records)
        exact_groups, exact_paths = self._find_exact_duplicates(candidates)
        likely_groups = self._find_likely_duplicates(candidates, excluded_paths=exact_paths)
        summary = DuplicateSummary(
            total_tracks=len(candidates),
            exact_group_count=len(exact_groups),
            exact_duplicate_file_count=sum(len(group.files) for group in exact_groups),
            likely_group_count=len(likely_groups),
            likely_duplicate_file_count=sum(len(group.files) for group in likely_groups),
        )
        return DuplicateAnalysisReport(
            summary=summary,
            exact_duplicates=exact_groups,
            likely_duplicates=likely_groups,
        )

    def _find_exact_duplicates(
        self,
        candidates: list[_DuplicateCandidate],
    ) -> tuple[list[ExactDuplicateGroup], set[str]]:
        by_size: dict[int, list[_DuplicateCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_size[candidate.size].append(candidate)

        groups: list[ExactDuplicateGroup] = []
        grouped_paths: set[str] = set()
        group_index = 1

        for size in sorted(by_size):
            same_size = by_size[size]
            if len(same_size) < 2:
                continue

            by_partial: dict[str, list[_DuplicateCandidate]] = defaultdict(list)
            for candidate in same_size:
                by_partial[_partial_hash(candidate.path)].append(candidate)

            for partial_key in sorted(by_partial):
                partial_candidates = by_partial[partial_key]
                if len(partial_candidates) < 2:
                    continue

                by_full: dict[str, list[_DuplicateCandidate]] = defaultdict(list)
                for candidate in partial_candidates:
                    by_full[_full_hash(candidate.path)].append(candidate)

                for content_hash in sorted(by_full):
                    exact = sorted(by_full[content_hash], key=lambda item: str(item.path))
                    if len(exact) < 2:
                        continue
                    preferred = self._recommend_preferred(exact)
                    groups.append(
                        ExactDuplicateGroup(
                            group_id=f"exact-{group_index:03d}",
                            content_hash=content_hash,
                            files=[candidate.to_track_info() for candidate in exact],
                            preferred=preferred,
                        )
                    )
                    grouped_paths.update(str(candidate.path) for candidate in exact)
                    group_index += 1

        return groups, grouped_paths

    def _find_likely_duplicates(
        self,
        candidates: list[_DuplicateCandidate],
        *,
        excluded_paths: set[str],
    ) -> list[LikelyDuplicateGroup]:
        active = [candidate for candidate in candidates if str(candidate.path) not in excluded_paths]
        buckets: dict[str, list[_DuplicateCandidate]] = defaultdict(list)
        for candidate in active:
            keys: set[str] = set()
            if candidate.title_normalized and candidate.artist_normalized:
                keys.add(f"ta:{candidate.artist_normalized}|{candidate.title_normalized}")
            if candidate.title_normalized:
                keys.add(f"t:{candidate.title_normalized}")
            if candidate.stem_normalized and candidate.artist_normalized:
                keys.add(f"sa:{candidate.artist_normalized}|{candidate.stem_normalized}")
            for key in keys:
                buckets[key].append(candidate)

        edges: dict[str, list[LikelyDuplicateRelationship]] = defaultdict(list)
        seen_pairs: set[tuple[str, str]] = set()
        for key in sorted(buckets):
            bucket = sorted(buckets[key], key=lambda item: str(item.path))
            if len(bucket) < 2:
                continue
            for idx, left in enumerate(bucket):
                for right in bucket[idx + 1 :]:
                    pair_key = tuple(sorted((str(left.path), str(right.path))))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    confidence, reasons = self._score_likely_pair(left, right)
                    if confidence < self.settings.likely_duplicate_threshold:
                        continue
                    relationship = LikelyDuplicateRelationship(
                        paths=[pair_key[0], pair_key[1]],
                        confidence=confidence,
                        reasons=reasons,
                    )
                    edges[pair_key[0]].append(relationship)
                    edges[pair_key[1]].append(relationship)

        groups: list[LikelyDuplicateGroup] = []
        visited: set[str] = set()
        group_index = 1
        path_to_candidate = {str(candidate.path): candidate for candidate in active}
        for path in sorted(edges):
            if path in visited:
                continue
            stack = [path]
            component_paths: set[str] = set()
            component_relationships: dict[tuple[str, str], LikelyDuplicateRelationship] = {}
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_paths.add(current)
                for relationship in edges.get(current, []):
                    rel_key = tuple(relationship.paths)
                    component_relationships[rel_key] = relationship
                    for next_path in relationship.paths:
                        if next_path not in visited:
                            stack.append(next_path)

            if len(component_paths) < 2:
                continue
            members = sorted((path_to_candidate[item] for item in component_paths), key=lambda item: str(item.path))
            relationships = sorted(
                component_relationships.values(),
                key=lambda item: (item.paths[0], item.paths[1]),
            )
            preferred = self._recommend_preferred(members)
            group_reasons = sorted({reason for rel in relationships for reason in rel.reasons})
            confidence = round(min(rel.confidence for rel in relationships), 3)
            groups.append(
                LikelyDuplicateGroup(
                    group_id=f"likely-{group_index:03d}",
                    confidence=confidence,
                    files=[member.to_track_info() for member in members],
                    relationships=relationships,
                    preferred=preferred,
                    reasons=group_reasons,
                )
            )
            group_index += 1

        return groups

    def _score_likely_pair(self, left: _DuplicateCandidate, right: _DuplicateCandidate) -> tuple[float, list[str]]:
        title_similarity = _similarity(left.title_normalized, right.title_normalized)
        artist_similarity = _similarity(left.artist_normalized, right.artist_normalized)
        album_similarity = _similarity(left.album_normalized, right.album_normalized)
        stem_similarity = _similarity(left.stem_normalized, right.stem_normalized)
        duration_points, duration_reason = _duration_score(left.duration_seconds, right.duration_seconds, self.settings)

        score = 0.0
        reasons: list[str] = []

        title_strong = title_similarity >= self.settings.title_strong_similarity
        title_good = title_similarity >= self.settings.title_good_similarity
        artist_strong = artist_similarity >= self.settings.artist_strong_similarity
        artist_good = artist_similarity >= self.settings.artist_good_similarity
        stem_good = stem_similarity >= self.settings.stem_good_similarity

        if title_strong:
            score += 0.40
            reasons.append("title_match_strong")
        elif title_good:
            score += 0.30
            reasons.append("title_match_good")

        if artist_strong:
            score += 0.25
            reasons.append("artist_match_strong")
        elif artist_good:
            score += 0.18
            reasons.append("artist_match_good")

        if album_similarity >= 0.92 and left.album_normalized and right.album_normalized:
            score += 0.10
            reasons.append("album_match")

        if stem_good:
            score += 0.10
            reasons.append("filename_match")

        if duration_points > 0:
            score += duration_points
            if duration_reason is not None:
                reasons.append(duration_reason)

        bitrate_delta = None
        if left.bitrate_bps is not None and right.bitrate_bps is not None:
            bitrate_delta = abs(left.bitrate_bps - right.bitrate_bps)
            if bitrate_delta <= 32_000:
                score += 0.03
                reasons.append("bitrate_close")

        size_ratio = min(left.size, right.size) / max(left.size, right.size) if min(left.size, right.size) > 0 else 0.0
        if size_ratio >= 0.85:
            score += 0.03
            reasons.append("file_size_close")

        if not (
            (title_good and artist_good)
            or (title_strong and duration_points > 0 and stem_good)
            or (title_strong and artist_strong and duration_points > 0)
        ):
            return 0.0, []

        return round(min(score, 0.99), 3), reasons

    def _recommend_preferred(self, candidates: list[_DuplicateCandidate]) -> PreferredCopyRecommendation:
        ranked = sorted(candidates, key=lambda item: self._sort_preference_key(item))
        preferred = ranked[0]
        reasons: list[str] = []

        if (
            self.settings.prefer_lossless_formats
            and preferred.extension in LOSSLESS_EXTENSIONS
            and any(item.extension not in LOSSLESS_EXTENSIONS for item in candidates)
        ):
            reasons.append("preferred_lossless_format")

        bitrate_values = [item.bitrate_bps for item in candidates if item.bitrate_bps is not None]
        if (
            self.settings.prefer_higher_bitrate
            and preferred.bitrate_bps is not None
            and bitrate_values
            and preferred.bitrate_bps == max(bitrate_values)
        ):
            if len(set(bitrate_values)) > 1:
                reasons.append("highest_bitrate")

        metadata_values = [item.metadata_fields_present for item in candidates]
        if (
            self.settings.prefer_richer_metadata
            and preferred.metadata_fields_present == max(metadata_values)
            and len(set(metadata_values)) > 1
        ):
            reasons.append("richest_metadata")

        if self.settings.prefer_cover_art and preferred.has_cover_art and any(not item.has_cover_art for item in candidates):
            reasons.append("has_cover_art")

        size_values = [item.size for item in candidates]
        if self.settings.prefer_larger_file and preferred.size == max(size_values) and len(set(size_values)) > 1:
            reasons.append("largest_file_size")

        if not reasons:
            reasons.append("deterministic_tie_break")

        return PreferredCopyRecommendation(path=str(preferred.path), reasons=reasons)

    def _preference_key(self, candidate: _DuplicateCandidate) -> tuple:
        lossless_rank = 1 if candidate.extension in LOSSLESS_EXTENSIONS else 0
        bitrate_rank = candidate.bitrate_bps or 0
        metadata_rank = candidate.metadata_fields_present
        cover_rank = 1 if candidate.has_cover_art else 0
        size_rank = candidate.size
        path_depth_rank = len(candidate.path.parts)
        return (
            -lossless_rank,
            -(1 if candidate.bitrate_bps is not None else 0),
            -bitrate_rank,
            -metadata_rank,
            -cover_rank,
            -size_rank,
            path_depth_rank,
            str(candidate.path),
        )

    def _sort_preference_key(self, candidate: _DuplicateCandidate) -> tuple:
        return self._preference_key(candidate)


def analyze_duplicates(records: list[TrackRecord], *, settings: DuplicateConfig | None = None) -> DuplicateAnalysisReport:
    return DuplicateAnalyzer(settings=settings).analyze(records)
