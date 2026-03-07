from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PromptConfig(BaseModel):
    system: str
    user_rules: list[str] = Field(default_factory=list)


class BPMGenreBand(BaseModel):
    typical: list[tuple[float, float]] = Field(default_factory=list)
    tolerant: list[tuple[float, float]] = Field(default_factory=list)


class ReasonConfig(BaseModel):
    max_chars: int = 120


class JudgeConfig(BaseModel):
    prompt_version: str = "judge_v1_heuristics"
    prompts: PromptConfig
    moods: list[str]
    genres: list[str]
    genre_aliases: dict[str, str] = Field(default_factory=dict)
    filename_hints: dict[str, str] = Field(default_factory=dict)
    bpm_bands: dict[str, BPMGenreBand] = Field(default_factory=dict)
    reason: ReasonConfig = Field(default_factory=ReasonConfig)
    concurrency_default: int = 10

    @field_validator("concurrency_default")
    @classmethod
    def validate_concurrency(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("judge.concurrency_default must be between 1 and 20")
        return value


class HealthConfig(BaseModel):
    sample_limit: int = 5
    low_bitrate_threshold_bps: int = 128_000
    short_duration_seconds: float = 30.0
    long_duration_seconds: float = 900.0
    long_filename_threshold: int = 120
    placeholder_values: list[str] = Field(default_factory=lambda: ["unknown", "untitled", "track", "audio", "song"])
    placeholder_track_pattern: str = r"^track\s*0*\d+$"

    @field_validator("sample_limit", "long_filename_threshold")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("health thresholds must be >= 1")
        return value

    @field_validator("low_bitrate_threshold_bps")
    @classmethod
    def validate_bitrate(cls, value: int) -> int:
        if value < 1_000:
            raise ValueError("health.low_bitrate_threshold_bps must be >= 1000")
        return value

    @model_validator(mode="after")
    def validate_duration_range(self) -> "HealthConfig":
        if self.short_duration_seconds < 0 or self.long_duration_seconds <= self.short_duration_seconds:
            raise ValueError("health duration thresholds must satisfy 0 <= short < long")
        return self


class DuplicateConfig(BaseModel):
    likely_duplicate_threshold: float = 0.78
    duration_close_seconds: float = 1.5
    duration_near_seconds: float = 3.0
    title_strong_similarity: float = 0.96
    title_good_similarity: float = 0.90
    artist_strong_similarity: float = 0.96
    artist_good_similarity: float = 0.90
    stem_good_similarity: float = 0.92
    prefer_lossless_formats: bool = True
    prefer_higher_bitrate: bool = True
    prefer_richer_metadata: bool = True
    prefer_cover_art: bool = True
    prefer_larger_file: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> "DuplicateConfig":
        float_fields = {
            "likely_duplicate_threshold": self.likely_duplicate_threshold,
            "duration_close_seconds": self.duration_close_seconds,
            "duration_near_seconds": self.duration_near_seconds,
            "title_strong_similarity": self.title_strong_similarity,
            "title_good_similarity": self.title_good_similarity,
            "artist_strong_similarity": self.artist_strong_similarity,
            "artist_good_similarity": self.artist_good_similarity,
            "stem_good_similarity": self.stem_good_similarity,
        }
        for name, value in float_fields.items():
            if value < 0:
                raise ValueError(f"duplicates.{name} must be >= 0")
        if self.likely_duplicate_threshold > 1:
            raise ValueError("duplicates.likely_duplicate_threshold must be <= 1")
        if self.duration_near_seconds < self.duration_close_seconds:
            raise ValueError("duplicates.duration_near_seconds must be >= duration_close_seconds")
        return self


class PlaylistConfig(BaseModel):
    default_max_tracks: int = 25
    default_min_score: float = 0.0
    mood_match_weight: float = 2.0
    genre_match_weight: float = 2.2
    cultural_match_weight: float = 1.6
    bpm_match_weight: float = 2.0
    energy_match_weight: float = 1.8
    keyword_match_weight: float = 0.6
    instrumental_hint_weight: float = 1.8
    vocal_penalty_weight: float = 1.8
    negative_keyword_penalty: float = 1.6
    cover_art_bonus: float = 0.1
    duration_cap_bonus: float = 1.2
    duration_floor_bonus: float = 1.0
    max_reason_count: int = 4
    bpm_window_around: float = 5.0

    @field_validator("default_max_tracks", "max_reason_count")
    @classmethod
    def validate_playlist_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("playlist integer settings must be >= 1")
        return value


class WatchConfig(BaseModel):
    default_interval_seconds: float = 2.0

    @field_validator("default_interval_seconds")
    @classmethod
    def validate_interval(cls, value: float) -> float:
        if value < 0.2:
            raise ValueError("watch.default_interval_seconds must be >= 0.2")
        return value


class ReviewConfig(BaseModel):
    sample_limit: int = 5
    high_priority_score: float = 75.0
    medium_priority_score: float = 45.0
    exact_duplicate_base_score: float = 68.0
    likely_duplicate_base_score: float = 52.0
    missing_metadata_base_score: float = 56.0
    weak_metadata_base_score: float = 42.0
    artwork_missing_base_score: float = 34.0
    low_quality_base_score: float = 40.0
    folder_hotspot_base_score: float = 38.0
    reclaimable_bytes_unit: int = 50_000_000
    files_affected_weight: float = 3.0
    folder_issue_weight: float = 2.0
    missing_artist_weight: float = 4.0
    missing_title_weight: float = 3.0
    missing_album_weight: float = 2.0
    unreadable_weight: float = 6.0
    low_bitrate_weight: float = 3.0
    artwork_gap_weight: float = 0.2
    likely_confidence_weight: float = 20.0

    @field_validator("sample_limit", "reclaimable_bytes_unit")
    @classmethod
    def validate_positive_review_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("review integer settings must be >= 1")
        return value

    @field_validator(
        "high_priority_score",
        "medium_priority_score",
        "exact_duplicate_base_score",
        "likely_duplicate_base_score",
        "missing_metadata_base_score",
        "weak_metadata_base_score",
        "artwork_missing_base_score",
        "low_quality_base_score",
        "folder_hotspot_base_score",
        "files_affected_weight",
        "folder_issue_weight",
        "missing_artist_weight",
        "missing_title_weight",
        "missing_album_weight",
        "unreadable_weight",
        "low_bitrate_weight",
        "artwork_gap_weight",
        "likely_confidence_weight",
    )
    @classmethod
    def validate_non_negative_review_numbers(cls, value: float) -> float:
        if value < 0:
            raise ValueError("review numeric settings must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_priority_bands(self) -> "ReviewConfig":
        if self.medium_priority_score > self.high_priority_score:
            raise ValueError("review.medium_priority_score must be <= review.high_priority_score")
        return self


class NyxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    profile: str = "default"
    judge: JudgeConfig
    health: HealthConfig = Field(default_factory=HealthConfig)
    duplicates: DuplicateConfig = Field(default_factory=DuplicateConfig)
    playlist: PlaylistConfig = Field(default_factory=PlaylistConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)


DEFAULT_CONFIG_RESOURCE = resources.files("nyxcore").joinpath("resources/default.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path | None) -> dict:
    if path is None:
        return yaml.safe_load(DEFAULT_CONFIG_RESOURCE.read_text(encoding="utf-8")) or {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolved_raw_config(path: Path | None = None, *, profile: str | None = None) -> dict:
    package_raw = _read_yaml(None)
    user_raw = {} if path is None else _read_yaml(path)
    merged_profiles = _deep_merge(package_raw.get("profiles", {}), user_raw.get("profiles", {}))
    active_profile = profile or user_raw.get("profile") or package_raw.get("profile") or "default"
    if active_profile not in merged_profiles:
        raise ValueError(f"Unknown config profile: {active_profile}")

    package_base = {k: v for k, v in package_raw.items() if k not in {"profiles", "profile"}}
    explicit_overrides = {k: v for k, v in user_raw.items() if k not in {"profiles", "profile"}}
    resolved = _deep_merge(package_base, merged_profiles.get(active_profile, {}))
    resolved = _deep_merge(resolved, explicit_overrides)
    resolved["profile"] = active_profile
    return resolved


def load_config(path: Path | None = None, *, profile: str | None = None) -> NyxConfig:
    raw = _resolved_raw_config(path, profile=profile)
    return NyxConfig.model_validate(raw)
