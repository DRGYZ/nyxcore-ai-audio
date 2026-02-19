from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.backends.clap_backend import ClapBackend
from nyxcore.audio.backends.essentia_backend import EssentiaBackend
from nyxcore.audio.models import AnalysisResult


class HybridBackend(AudioBackend):
    name = "hybrid"

    def __init__(self) -> None:
        self.essentia = None
        self.clap = None
        self.init_errors: list[str] = []
        try:
            self.essentia = EssentiaBackend()
        except Exception as exc:
            self.init_errors.append(f"essentia_init: {exc}")
        try:
            self.clap = ClapBackend()
        except Exception as exc:
            self.init_errors.append(f"clap_init: {exc}")

    def analyze_track(self, path: Path) -> AnalysisResult:
        energy: float = 0.0
        bpm: float | None = None
        tags: list[str] = []
        genre_top: str | None = None
        confidence: float | None = None
        errors = list(self.init_errors)

        if self.essentia is not None:
            try:
                ess = self.essentia.analyze_track(path)
                energy = ess.energy_0_10
                bpm = ess.bpm
                errors.extend(ess.errors)
            except Exception as exc:
                errors.append(f"essentia: {exc}")
        else:
            errors.append("essentia: unavailable")

        if self.clap is not None:
            try:
                clap = self.clap.analyze_track(path)
                tags = clap.tags
                genre_top = clap.genre_top
                confidence = clap.confidence
                errors.extend(clap.errors)
            except Exception as exc:
                errors.append(f"clap: {exc}")
        else:
            errors.append("clap: unavailable")

        return AnalysisResult(
            energy_0_10=energy,
            bpm=bpm,
            tags=tags,
            genre_top=genre_top,
            backend=self.name,
            created_at_iso=datetime.now(tz=UTC).isoformat(),
            confidence=confidence,
            errors=errors,
        )
