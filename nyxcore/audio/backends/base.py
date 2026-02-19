from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nyxcore.audio.models import AnalysisResult


class AudioBackend(Protocol):
    name: str

    def analyze_track(self, path: Path) -> AnalysisResult:
        ...

