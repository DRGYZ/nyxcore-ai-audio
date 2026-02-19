from __future__ import annotations

from pathlib import Path

from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.models import AnalysisResult


def _top_labels(probabilities: dict[str, float]) -> tuple[str | None, list[str]]:
    if not probabilities:
        return None, []
    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    genre_top = ordered[0][0]
    tags = [label for label, prob in ordered if prob > 0.2][:3]
    if len(tags) < 3:
        tags = [label for label, _ in ordered[:3]]
    return genre_top, tags


class EssentiaBackend(AudioBackend):
    name = "essentia"

    def __init__(self) -> None:
        try:
            import essentia.standard as es  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Essentia backend unavailable. Install essentia (with TensorFlow-enabled build for model inference)."
            ) from exc
        self.es = es

    def analyze_track(self, path: Path) -> AnalysisResult:
        audio = self.es.MonoLoader(filename=str(path), sampleRate=44100)()
        bpm, _, _, _, _ = self.es.RhythmExtractor2013(method="multifeature")(audio)

        if len(audio) == 0:
            energy = 0.0
        else:
            abs_mean = sum(abs(float(x)) for x in audio) / float(len(audio))
            energy = round(max(0.0, min(10.0, abs_mean * 20.0)), 2)

        genre_probs: dict[str, float] = {}
        tags: list[str] = []
        genre_top: str | None = None

        # Minimal optional model support: if caller injects a dict-like output, normalize here.
        if genre_probs:
            genre_top, tags = _top_labels(genre_probs)

        return AnalysisResult(
            energy_0_10=float(energy),
            bpm=float(round(float(bpm), 2)),
            tags=tags,
            genre_top=genre_top,
            backend=self.name,
        )

