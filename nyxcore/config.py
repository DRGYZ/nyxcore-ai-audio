from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class NyxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    judge: JudgeConfig


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(path: Path | None = None) -> NyxConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return NyxConfig.model_validate(raw)
