from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nyxcore.audio.cache import AnalysisCache
from nyxcore.config import PlaylistConfig
from nyxcore.core.track import TrackRecord

MOOD_KEYWORDS = {
    "chill": ("chill", "calm", "relaxed", "lofi", "ambient"),
    "dark": ("dark", "mysterious", "tense", "aggressive"),
    "melancholic": ("melancholic", "sad", "emotional"),
    "cinematic": ("cinematic", "epic", "soundtrack"),
    "focus": ("focus", "ambient", "lofi", "instrumental"),
    "gym": ("gym", "energetic", "aggressive"),
    "uplifting": ("uplifting", "positive"),
    "night": ("night", "night-drive", "dark"),
}

GENRE_KEYWORDS = {
    "hip hop": ("hip hop", "hiphop", "rap"),
    "electronic": ("electronic", "edm"),
    "ambient": ("ambient",),
    "techno": ("techno",),
    "house": ("house",),
    "phonk": ("phonk",),
    "soundtrack": ("soundtrack", "score", "ost"),
    "arabic": ("arabic", "arab",),
}

ENERGY_HINTS = {
    "low": (0.0, 4.0),
    "chill": (0.0, 4.5),
    "focus": (0.0, 4.5),
    "mid": (3.5, 6.5),
    "high": (7.0, 10.0),
    "energetic": (7.0, 10.0),
    "gym": (7.0, 10.0),
}

INSTRUMENTAL_HINTS = {"instrumental", "no vocals", "without vocals", "score", "soundtrack"}
VOCAL_NEGATIVE_HINTS = {"vocals", "vocal", "singer", "feat", "featuring"}
BPM_RANGE_RE = re.compile(r"(\d{2,3})\s*(?:to|-)\s*(\d{2,3})\s*bpm|\baround\s+(\d{2,3})\s*bpm", re.IGNORECASE)
DURATION_UNDER_RE = re.compile(r"(?:under|below|less than)\s+(\d+)\s*(?:minutes|min)", re.IGNORECASE)
DURATION_OVER_RE = re.compile(r"(?:over|above|more than)\s+(\d+)\s*(?:minutes|min)", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(no|not|without)\s+([a-z][a-z\- ]+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9\-]+", re.IGNORECASE)


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(text.split())


def _tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


@dataclass(slots=True)
class ParsedPlaylistQuery:
    original_query: str
    moods: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    bpm_min: float | None = None
    bpm_max: float | None = None
    energy_min: float | None = None
    energy_max: float | None = None
    max_duration_seconds: float | None = None
    min_duration_seconds: float | None = None
    instrumental_preference: str | None = None
    cultural_hints: list[str] = field(default_factory=list)
    unsupported_aspects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PlaylistTrack:
    path: str
    score: float
    reasons: list[str]
    title: str | None
    artist: str | None
    album: str | None
    duration_seconds: float | None
    bpm: float | None
    energy_0_10: float | None
    tags: list[str]
    genre_top: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PlaylistSummary:
    track_count: int
    estimated_total_duration_seconds: float
    average_bpm: float | None
    average_energy_0_10: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PlaylistReport:
    original_query: str
    parsed_query: ParsedPlaylistQuery
    ranked_tracks: list[PlaylistTrack]
    unsupported_request_aspects: list[str]
    summary: PlaylistSummary

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "parsed_query": self.parsed_query.to_dict(),
            "ranked_tracks": [track.to_dict() for track in self.ranked_tracks],
            "unsupported_request_aspects": list(self.unsupported_request_aspects),
            "summary": self.summary.to_dict(),
        }


@dataclass(slots=True)
class _TrackFeatures:
    record: TrackRecord
    normalized_text: str
    tags: list[str]
    genre_top: str | None
    bpm: float | None
    energy_0_10: float | None


def parse_playlist_query(query: str, *, settings: PlaylistConfig | None = None) -> ParsedPlaylistQuery:
    settings = settings or PlaylistConfig()
    normalized = _normalize_text(query)
    tokens = _tokenize(query)
    moods: set[str] = set()
    genres: set[str] = set()
    cultural_hints: set[str] = set()
    keywords: set[str] = set()
    negative_keywords: set[str] = set()
    unsupported: list[str] = []
    bpm_min = bpm_max = None
    energy_min = energy_max = None
    max_duration = None
    min_duration = None
    instrumental_preference: str | None = None

    for mood, terms in MOOD_KEYWORDS.items():
        if any(term in normalized for term in terms):
            moods.add(mood)
            if mood in ENERGY_HINTS and energy_min is None and energy_max is None:
                energy_min, energy_max = ENERGY_HINTS[mood]

    for genre, terms in GENRE_KEYWORDS.items():
        if any(term in normalized for term in terms):
            genres.add(genre)
            if genre == "arabic":
                cultural_hints.add("arabic")

    for label, rng in ENERGY_HINTS.items():
        if label in normalized:
            energy_min, energy_max = rng

    bpm_match = BPM_RANGE_RE.search(query)
    if bpm_match:
        if bpm_match.group(1) and bpm_match.group(2):
            bpm_min = float(bpm_match.group(1))
            bpm_max = float(bpm_match.group(2))
        elif bpm_match.group(3):
            center = float(bpm_match.group(3))
            bpm_min = center - settings.bpm_window_around
            bpm_max = center + settings.bpm_window_around

    under_match = DURATION_UNDER_RE.search(query)
    if under_match:
        max_duration = float(under_match.group(1)) * 60.0
    over_match = DURATION_OVER_RE.search(query)
    if over_match:
        min_duration = float(over_match.group(1)) * 60.0

    if "no vocals" in normalized or "without vocals" in normalized or "instrumental" in normalized:
        instrumental_preference = "prefer_instrumental"
    elif "vocals" in normalized:
        instrumental_preference = "prefer_vocals"

    for neg, tail in NEGATION_RE.findall(query):
        tail_text = _normalize_text(tail).split()
        if tail_text:
            negative_keywords.add(tail_text[0])

    stop = {"with", "and", "for", "the", "music", "tracks", "track", "mostly", "around", "under", "over", "minutes"}
    for token in tokens:
        if token in stop or token.isdigit():
            continue
        if token in {"vocals", "vocal"} and instrumental_preference == "prefer_instrumental":
            continue
        keywords.add(token)

    if instrumental_preference is not None:
        unsupported.append("instrumental_vocal_detection_is_heuristic")
    if cultural_hints:
        unsupported.append("language_cultural_matching_depends_on_metadata_keywords")
    if bpm_min is not None or bpm_max is not None:
        pass
    else:
        if "tempo" in normalized or "mid tempo" in normalized:
            unsupported.append("tempo_terms_without_numeric_bpm_use_soft_keyword_matching")

    return ParsedPlaylistQuery(
        original_query=query,
        moods=sorted(moods),
        genres=sorted(genres),
        keywords=sorted(keywords),
        negative_keywords=sorted(negative_keywords),
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        energy_min=energy_min,
        energy_max=energy_max,
        max_duration_seconds=max_duration,
        min_duration_seconds=min_duration,
        instrumental_preference=instrumental_preference,
        cultural_hints=sorted(cultural_hints),
        unsupported_aspects=list(dict.fromkeys(unsupported)),
    )


def _load_analysis_map(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cache = AnalysisCache(cache_path)
    try:
        rows = cache.rows()
    finally:
        cache.close()
    best_by_path: dict[str, dict] = {}
    for row in rows:
        current = best_by_path.get(str(row["path"]))
        score = 0 if row.get("confidence") is None else float(row.get("confidence") or 0.0)
        if current is None:
            best_by_path[str(row["path"])] = row
            continue
        current_score = 0 if current.get("confidence") is None else float(current.get("confidence") or 0.0)
        if score > current_score:
            best_by_path[str(row["path"])] = row
    return best_by_path


def _make_features(record: TrackRecord, analysis_row: dict | None) -> _TrackFeatures:
    text_parts = [
        record.tags.get("title") or "",
        record.tags.get("artist") or "",
        record.tags.get("album") or "",
        Path(record.path).stem,
    ]
    tags = [str(item).lower() for item in (analysis_row or {}).get("tags", [])]
    genre_top = None if analysis_row is None else analysis_row.get("genre_top")
    return _TrackFeatures(
        record=record,
        normalized_text=_normalize_text(" ".join(text_parts + tags + ([str(genre_top)] if genre_top else []))),
        tags=tags,
        genre_top=None if genre_top is None else str(genre_top).lower(),
        bpm=None if analysis_row is None or analysis_row.get("bpm") is None else float(analysis_row["bpm"]),
        energy_0_10=None
        if analysis_row is None or analysis_row.get("energy_0_10") is None
        else float(analysis_row["energy_0_10"]),
    )


def _score_track(parsed: ParsedPlaylistQuery, features: _TrackFeatures, settings: PlaylistConfig) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    text = features.normalized_text

    for mood in parsed.moods:
        terms = MOOD_KEYWORDS.get(mood, ())
        if any(term in text for term in terms):
            score += settings.mood_match_weight
            reasons.append(f"mood:{mood}")

    for genre in parsed.genres:
        if genre in text:
            score += settings.genre_match_weight
            reasons.append(f"genre:{genre}")

    for hint in parsed.cultural_hints:
        if hint in text:
            score += settings.cultural_match_weight
            reasons.append(f"culture:{hint}")

    if parsed.bpm_min is not None and parsed.bpm_max is not None:
        if features.bpm is None:
            score -= 0.4
        elif parsed.bpm_min <= features.bpm <= parsed.bpm_max:
            score += settings.bpm_match_weight
            reasons.append(f"bpm:{features.bpm:.0f}")
        else:
            distance = min(abs(features.bpm - parsed.bpm_min), abs(features.bpm - parsed.bpm_max))
            score -= min(2.0, distance / 20.0)

    if parsed.energy_min is not None and parsed.energy_max is not None:
        if features.energy_0_10 is None:
            score -= 0.4
        elif parsed.energy_min <= features.energy_0_10 <= parsed.energy_max:
            score += settings.energy_match_weight
            reasons.append(f"energy:{features.energy_0_10:.1f}")
        else:
            distance = min(abs(features.energy_0_10 - parsed.energy_min), abs(features.energy_0_10 - parsed.energy_max))
            score -= min(1.5, distance / 3.0)

    if parsed.max_duration_seconds is not None:
        if features.record.duration_seconds is None:
            score -= 0.2
        elif features.record.duration_seconds <= parsed.max_duration_seconds:
            score += settings.duration_cap_bonus
            reasons.append("duration_cap_ok")
        else:
            score -= 2.5

    if parsed.min_duration_seconds is not None:
        if features.record.duration_seconds is None:
            score -= 0.2
        elif features.record.duration_seconds >= parsed.min_duration_seconds:
            score += settings.duration_floor_bonus
            reasons.append("duration_floor_ok")
        else:
            score -= 2.0

    if parsed.instrumental_preference == "prefer_instrumental":
        if any(term in text for term in INSTRUMENTAL_HINTS):
            score += settings.instrumental_hint_weight
            reasons.append("instrumental_hint")
        if any(term in text for term in VOCAL_NEGATIVE_HINTS):
            score -= settings.vocal_penalty_weight
    elif parsed.instrumental_preference == "prefer_vocals":
        if any(term in text for term in VOCAL_NEGATIVE_HINTS):
            score += 0.8
            reasons.append("vocal_hint")

    for keyword in parsed.keywords:
        if keyword in text:
            score += settings.keyword_match_weight
            reasons.append(f"keyword:{keyword}")

    for keyword in parsed.negative_keywords:
        if keyword in text:
            score -= settings.negative_keyword_penalty

    if features.record.has_cover_art:
        score += settings.cover_art_bonus

    deduped_reasons = list(dict.fromkeys(reasons))[: settings.max_reason_count]
    return round(score, 3), deduped_reasons


def build_playlist_report(
    records: list[TrackRecord],
    *,
    query: str,
    settings: PlaylistConfig | None = None,
    max_tracks: int | None = None,
    min_score: float | None = None,
    analysis_cache_path: Path | None = None,
) -> PlaylistReport:
    settings = settings or PlaylistConfig()
    parsed = parse_playlist_query(query, settings=settings)
    analysis_map = {} if analysis_cache_path is None else _load_analysis_map(analysis_cache_path)
    max_tracks = settings.default_max_tracks if max_tracks is None else max_tracks
    min_score = settings.default_min_score if min_score is None else min_score

    ranked: list[PlaylistTrack] = []
    unsupported = list(parsed.unsupported_aspects)
    found_bpm = False
    found_energy = False
    for record in sorted(records, key=lambda item: item.path):
        row = analysis_map.get(record.path)
        features = _make_features(record, row)
        found_bpm = found_bpm or features.bpm is not None
        found_energy = found_energy or features.energy_0_10 is not None
        score, reasons = _score_track(parsed, features, settings)
        if score < min_score:
            continue
        ranked.append(
            PlaylistTrack(
                path=record.path,
                score=score,
                reasons=reasons,
                title=record.tags.get("title"),
                artist=record.tags.get("artist"),
                album=record.tags.get("album"),
                duration_seconds=record.duration_seconds,
                bpm=features.bpm,
                energy_0_10=features.energy_0_10,
                tags=list(features.tags),
                genre_top=features.genre_top,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.path))
    ranked = ranked[:max_tracks]

    if (parsed.bpm_min is not None or parsed.bpm_max is not None) and not found_bpm:
        unsupported.append("bpm_requested_but_not_available_for_current_library_subset")
    if (parsed.energy_min is not None or parsed.energy_max is not None) and not found_energy:
        unsupported.append("energy_requested_but_not_available_for_current_library_subset")

    total_duration = sum(track.duration_seconds or 0.0 for track in ranked)
    bpm_values = [track.bpm for track in ranked if track.bpm is not None]
    energy_values = [track.energy_0_10 for track in ranked if track.energy_0_10 is not None]
    summary = PlaylistSummary(
        track_count=len(ranked),
        estimated_total_duration_seconds=round(total_duration, 3),
        average_bpm=None if not bpm_values else round(sum(bpm_values) / len(bpm_values), 3),
        average_energy_0_10=None if not energy_values else round(sum(energy_values) / len(energy_values), 3),
    )
    return PlaylistReport(
        original_query=query,
        parsed_query=parsed,
        ranked_tracks=ranked,
        unsupported_request_aspects=list(dict.fromkeys(unsupported)),
        summary=summary,
    )
