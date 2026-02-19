from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class JudgeResult:
    tags: list[str] = field(default_factory=list)
    genre_top: str | None = None
    confidence: float | None = None
    reason: str = ""
    provider: str = "deepseek"
    model: str = "deepseek-v3.2"
    prompt_version: str = "judge_v1"
    created_at_iso: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    errors: list[str] = field(default_factory=list)
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

