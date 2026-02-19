from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class AnalysisResult:
    energy_0_10: float
    bpm: float | None
    tags: list[str] = field(default_factory=list)
    genre_top: str | None = None
    backend: str = "unknown"
    created_at_iso: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    confidence: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
