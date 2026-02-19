from __future__ import annotations

import hashlib
from pathlib import Path

from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.models import AnalysisResult

_DUMMY_TAGS = ["dark", "hypnotic", "night-drive", "calm", "focus", "uplifting", "cinematic"]
_DUMMY_GENRES = ["electronic", "hip-hop", "ambient", "rock", "pop"]


class DummyBackend(AudioBackend):
    name = "dummy"

    def analyze_track(self, path: Path) -> AnalysisResult:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)

        energy = round((seed % 101) / 10.0, 2)
        bpm = float(70 + (seed % 121))
        genre = _DUMMY_GENRES[seed % len(_DUMMY_GENRES)]

        tags: list[str] = []
        for idx in range(3):
            tags.append(_DUMMY_TAGS[(seed + idx * 3) % len(_DUMMY_TAGS)])

        return AnalysisResult(
            energy_0_10=energy,
            bpm=bpm,
            tags=tags,
            genre_top=genre,
            backend=self.name,
            confidence=0.85,
        )

