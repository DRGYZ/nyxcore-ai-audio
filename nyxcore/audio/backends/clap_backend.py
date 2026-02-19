from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from nyxcore.audio.backends.base import AudioBackend
from nyxcore.audio.models import AnalysisResult

MOOD_LABELS = [
    "dark",
    "hypnotic",
    "aggressive",
    "sad",
    "melancholic",
    "uplifting",
    "calm",
    "chill",
    "night-drive",
    "gym",
    "epic",
    "cinematic",
    "ambient",
    "romantic",
    "angry",
]

GENRE_LABELS = [
    "hip hop",
    "trap",
    "phonk",
    "electronic",
    "techno",
    "house",
    "drum and bass",
    "dubstep",
    "pop",
    "rock",
    "metal",
    "classical",
    "jazz",
    "lofi",
]

_CLAP_MODEL: Any = None
_MOOD_TEXT_EMBED: Any = None
_GENRE_TEXT_EMBED: Any = None


def _import_torch_stack():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CLAP backend unavailable: failed importing torch.\n"
            f"python_executable: {sys.executable}\n"
            f"first_error: {repr(exc)}\n"
            "Install in WSL:\n"
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            "pip install laion-clap"
        ) from exc

    try:
        import torchaudio  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CLAP backend unavailable: failed importing torchaudio.\n"
            f"python_executable: {sys.executable}\n"
            f"first_error: {repr(exc)}\n"
            "Install in WSL:\n"
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            "pip install laion-clap"
        ) from exc
    return torch, torchaudio


def _attempt_clap_imports() -> tuple[type[Any] | None, list[tuple[str, bool, str]]]:
    attempts: list[tuple[str, bool, str]] = []

    try:
        laion_clap = importlib.import_module("laion_clap")
        clap_module = getattr(laion_clap, "CLAP_Module")
        attempts.append(("import laion_clap", True, "ok"))
        return clap_module, attempts
    except Exception as exc:
        attempts.append(("import laion_clap", False, repr(exc)))

    try:
        mod = importlib.import_module("laion_clap")
        clap_module = getattr(mod, "CLAP_Module")
        attempts.append(("from laion_clap import CLAP_Module", True, "ok"))
        return clap_module, attempts
    except Exception as exc:
        attempts.append(("from laion_clap import CLAP_Module", False, repr(exc)))

    try:
        mod = importlib.import_module("laion_clap.clap_module")
        clap_module = getattr(mod, "CLAP_Module")
        attempts.append(("from laion_clap.clap_module import CLAP_Module", True, "ok"))
        return clap_module, attempts
    except Exception as exc:
        attempts.append(("from laion_clap.clap_module import CLAP_Module", False, repr(exc)))

    return None, attempts


def _import_clap_module():
    clap_module, attempts = _attempt_clap_imports()
    if clap_module is not None:
        return clap_module
    first_error = attempts[0][2] if attempts else "unknown"
    details = "\n".join(f"- {name}: {detail}" for name, _, detail in attempts)
    raise RuntimeError(
        "CLAP backend unavailable: failed importing laion-clap.\n"
        f"python_executable: {sys.executable}\n"
        f"first_error: {first_error}\n"
        "Try:\n"
        "pip show laion-clap\n"
        "Install in WSL:\n"
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        "pip install laion-clap\n"
        "Import attempts:\n"
        f"{details}"
    )


def clap_import_diagnostics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "torch_ok": False,
        "torchaudio_ok": False,
        "torch_version": None,
        "torchaudio_version": None,
        "torch_error": None,
        "torchaudio_error": None,
        "clap_attempts": [],
    }
    try:
        import torch  # type: ignore

        report["torch_ok"] = True
        report["torch_version"] = getattr(torch, "__version__", "unknown")
    except Exception as exc:
        report["torch_error"] = repr(exc)
    try:
        import torchaudio  # type: ignore

        report["torchaudio_ok"] = True
        report["torchaudio_version"] = getattr(torchaudio, "__version__", "unknown")
    except Exception as exc:
        report["torchaudio_error"] = repr(exc)
    _, attempts = _attempt_clap_imports()
    report["clap_attempts"] = attempts
    return report


def _lazy_import():
    torch, torchaudio = _import_torch_stack()
    clap_module = _import_clap_module()
    return clap_module, torch, torchaudio


def _init_model():
    clap_module, torch, _ = _lazy_import()
    device = torch.device("cpu")
    first_exc: Exception | None = None
    second_exc: Exception | None = None

    try:
        model = clap_module(enable_fusion=False, amodel="HTSAT-base", device=device)
        model.load_ckpt()
        return model
    except Exception as exc:
        first_exc = exc

    try:
        model = clap_module(enable_fusion=False, device=device)
        model.load_ckpt()
        return model
    except Exception as exc:
        second_exc = exc

    raise RuntimeError(
        "CLAP backend unavailable: checkpoint/model initialization failed.\n"
        f"first_attempt_error: {repr(first_exc)}\n"
        f"fallback_attempt_error: {repr(second_exc)}"
    )


def _get_model():
    global _CLAP_MODEL
    if _CLAP_MODEL is None:
        _CLAP_MODEL = _init_model()
    return _CLAP_MODEL


def _get_text_embeddings(model):
    global _MOOD_TEXT_EMBED, _GENRE_TEXT_EMBED
    if _MOOD_TEXT_EMBED is None:
        _MOOD_TEXT_EMBED = model.get_text_embedding(MOOD_LABELS, use_tensor=True)
    if _GENRE_TEXT_EMBED is None:
        _GENRE_TEXT_EMBED = model.get_text_embedding(GENRE_LABELS, use_tensor=True)
    return _MOOD_TEXT_EMBED, _GENRE_TEXT_EMBED


def _embed_audio_file(model, path: Path):
    try:
        return model.get_audio_embedding_from_filelist([str(path)], use_tensor=True)
    except Exception as original_exc:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.strip()}") from original_exc
            return model.get_audio_embedding_from_filelist([str(tmp_path)], use_tensor=True)
        finally:
            tmp_path.unlink(missing_ok=True)


def _to_scores(audio_embed, text_embed):
    _, torch, _ = _lazy_import()
    a = torch.nn.functional.normalize(audio_embed, dim=-1)
    b = torch.nn.functional.normalize(text_embed, dim=-1)
    return (a @ b.T).detach().cpu().numpy().reshape(-1)


def _top_labels(labels: list[str], scores, threshold: float = 0.25, top_k: int = 3) -> tuple[list[str], float | None]:
    pairs = sorted([(labels[i], float(scores[i])) for i in range(len(labels))], key=lambda kv: kv[1], reverse=True)
    selected = [label for label, score in pairs if score >= threshold][:top_k]
    if not selected:
        selected = [label for label, _ in pairs[:top_k]]
    confidence = pairs[0][1] if pairs else None
    return selected, confidence


class ClapBackend(AudioBackend):
    name = "clap"

    def __init__(self) -> None:
        self.model = _get_model()
        self.mood_text_embed, self.genre_text_embed = _get_text_embeddings(self.model)

    def analyze_track(self, path: Path) -> AnalysisResult:
        try:
            audio_embed = _embed_audio_file(self.model, path)
            mood_scores = _to_scores(audio_embed, self.mood_text_embed)
            genre_scores = _to_scores(audio_embed, self.genre_text_embed)

            tags, confidence = _top_labels(MOOD_LABELS, mood_scores, threshold=0.25, top_k=3)
            genre_pairs = sorted(
                [(GENRE_LABELS[i], float(genre_scores[i])) for i in range(len(GENRE_LABELS))],
                key=lambda kv: kv[1],
                reverse=True,
            )
            genre_top = genre_pairs[0][0] if genre_pairs else None
            return AnalysisResult(
                energy_0_10=0.0,
                bpm=None,
                tags=tags,
                genre_top=genre_top,
                backend=self.name,
                confidence=confidence,
            )
        except Exception as exc:
            return AnalysisResult(
                energy_0_10=0.0,
                bpm=None,
                tags=[],
                genre_top=None,
                backend=self.name,
                confidence=None,
                errors=[f"clap_track_error: {exc}"],
            )


def quick_verify(path: str) -> None:
    backend = ClapBackend()
    result = backend.analyze_track(Path(path))
    print(f"path={path}")
    print(f"tags={result.tags}")
    print(f"genre_top={result.genre_top}")
    print(f"confidence={result.confidence}")
    if result.errors:
        print(f"errors={result.errors}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python -m nyxcore.audio.backends.clap_backend <audio_file>")
        raise SystemExit(1)
    quick_verify(_sys.argv[1])

