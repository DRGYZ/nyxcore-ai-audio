from __future__ import annotations

import json
from pathlib import Path

from nyxcore.core.text import clean_reason_text, format_reason


class JudgeService:
    def __init__(self, prompt_version: str, moods: list[str], genres: list[str], llm_client=None):
        self.prompt_version = prompt_version
        self.moods = moods
        self.genres = genres
        self.llm_client = llm_client

    def call_llm(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ):
        if self.llm_client is None:
            raise RuntimeError("JudgeService.llm_client is not configured")
        return self.llm_client(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

    def build_system_prompt(self) -> str:
        return (
            "You are a strict music mood/genre judge.\n"
            "Return STRICT JSON with keys: tags, genre_top, tag_agreement, genre_decision, reason.\n"
            "Optional debug key: conflicts (0,1,2).\n"
            "tags must be 2-3 items max, unique, subset of allowed moods.\n"
            "genre_top must be null or subset of allowed genres.\n"
            "tag_agreement must be one of: low, medium, high.\n"
            "genre_decision must be one of: keep, drop.\n"
            "Do not output numeric confidence.\n"
            "reason must be <=120 chars.\n"
            "Never invent artist/title. Use only supplied evidence."
        )

    def build_user_prompt(self, row: dict) -> str:
        allowed_moods = ", ".join(self.moods)
        allowed_genres = ", ".join(self.genres)
        evidence = {
            "path": row.get("path"),
            "filename": Path(str(row.get("path", ""))).name,
            "energy_0_10": row.get("energy_0_10"),
            "bpm": row.get("bpm"),
            "clap_tags": row.get("tags", []),
            "clap_genre_top": row.get("genre_top"),
            "errors": row.get("errors", []),
        }
        return (
            "Allowed moods: [" + allowed_moods + "]\n"
            "Allowed genres: [" + allowed_genres + "]\n"
            "Rules:\n"
            "- use hybrid evidence (bpm/energy + clap tags/genre + filename path)\n"
            "- if genre ambiguous, set genre_top=null\n"
            "- 2-3 tags max, no duplicates\n"
            "- keep consistent with vocab\n\n"
            "- use phrase 'BPM atypical for genre' instead of strong mismatch claims\n"
            "- set tag_agreement based on filename/title cues + hybrid tag consistency\n"
            "- set genre_decision=keep only when genre evidence is coherent\n\n"
            "Evidence JSON:\n"
            + json.dumps(evidence, ensure_ascii=False)
        )

    def canonicalize_genre(self, genre: str | None) -> str | None:
        if genre is None:
            return None
        g = str(genre).strip().lower()
        if g == "":
            return None
        aliases = {
            "drum and bass": "drum and bass",
            "drum & bass": "drum and bass",
            "dnb": "drum and bass",
            "hip hop": "hip hop",
            "hiphop": "hip hop",
        }
        if g in aliases:
            return aliases[g]
        if g in self.genres:
            return g
        return None

    def strong_filename_genre_hint(self, path_str: str) -> str | None:
        text = Path(path_str).name.lower()
        if any(token in text for token in ("ost", "soundtrack", "theme")):
            return "soundtrack"
        if "2pac" in text:
            return "hip hop"
        return None

    def filename_genre_signal(self, path_str: str, genre: str | None) -> str:
        hint = self.strong_filename_genre_hint(path_str)
        if hint is None or genre is None:
            return "neutral"
        if self.canonicalize_genre(hint) == self.canonicalize_genre(genre):
            return "supports"
        return "contradicts"

    def bpm_note_for_genre(self, genre: str | None, bpm: float | None) -> str:
        g = self.canonicalize_genre(genre)
        if g is None:
            return "atypical"
        if bpm is None:
            return "atypical"
        b = float(bpm)
        if g == "drum and bass":
            if 160 <= b <= 180 or 150 <= b <= 190:
                return "ok"
            return "atypical"
        if g == "dubstep":
            if 135 <= b <= 150 or 130 <= b <= 155 or 65 <= b <= 77:
                return "ok"
            return "atypical"
        if g == "hip hop":
            if 70 <= b <= 105 or 60 <= b <= 115 or 140 <= b <= 170:
                return "ok"
            return "atypical"
        return "atypical"

    def source_genre_from_row(self, source_row: dict) -> str | None:
        source_genre_raw = source_row.get("source_genre_top")
        if source_genre_raw is None:
            source_genre_raw = source_row.get("genre_top")
        return self.canonicalize_genre(None if source_genre_raw is None else str(source_genre_raw))

    def source_bpm_from_row(self, source_row: dict) -> float | None:
        bpm_raw = source_row.get("source_bpm")
        if bpm_raw is None:
            bpm_raw = source_row.get("bpm")
        if bpm_raw is None:
            return None
        try:
            return float(bpm_raw)
        except (TypeError, ValueError):
            return None

    def compute_conflicts_local(self, source_row: dict, genre_for_eval: str | None) -> tuple[int, str, str]:
        bpm_note = self.bpm_note_for_genre(genre_for_eval, self.source_bpm_from_row(source_row))
        filename_signal = self.filename_genre_signal(str(source_row.get("path", "")), genre_for_eval)
        score = 0
        if bpm_note == "atypical":
            score += 1
        if filename_signal == "contradicts":
            score += 2
        if score <= 0:
            conflicts_local = 0
        elif score == 1:
            conflicts_local = 1
        else:
            conflicts_local = 2
        return conflicts_local, bpm_note, filename_signal

    def sanitize_llm_response(self, data: dict, source_row: dict) -> dict:
        tags_raw = data.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            for tag in tags_raw:
                t = str(tag).strip().lower()
                if t in self.moods and t not in tags:
                    tags.append(t)
                if len(tags) >= 3:
                    break
        genre_raw = data.get("genre_top")
        llm_genre = self.canonicalize_genre(None if genre_raw is None else str(genre_raw))
        tag_agreement = str(data.get("tag_agreement", "")).strip().lower()
        conflicts_raw = data.get("conflicts")
        conflicts_llm: int | None = None
        try:
            parsed_conflicts = int(conflicts_raw)
            if parsed_conflicts in {0, 1, 2}:
                conflicts_llm = parsed_conflicts
        except (TypeError, ValueError):
            conflicts_llm = None

        source_genre = self.source_genre_from_row(source_row)
        filename_hint = self.strong_filename_genre_hint(str(source_row.get("path", "")))
        genre_for_eval = source_genre or llm_genre or self.canonicalize_genre(filename_hint)
        conflicts_local, bpm_note, filename_signal = self.compute_conflicts_local(source_row, genre_for_eval)
        filename_supports_genre = filename_signal == "supports"
        filename_contradicts_genre = filename_signal == "contradicts"

        keep_by_policy = filename_supports_genre or (source_genre is not None and conflicts_local <= 1) or bpm_note == "ok"
        drop_by_policy = conflicts_local >= 2 and bpm_note == "atypical" and filename_contradicts_genre

        if drop_by_policy:
            final_genre = None
        elif source_genre is not None:
            final_genre = source_genre
        else:
            final_genre = llm_genre or self.canonicalize_genre(filename_hint)
        if keep_by_policy and not drop_by_policy and final_genre is None:
            final_genre = llm_genre or self.canonicalize_genre(filename_hint)

        conf = 0.55
        if tag_agreement == "high":
            conf += 0.10
        elif tag_agreement == "medium":
            conf += 0.05
        if conflicts_local == 0:
            conf += 0.05
        elif conflicts_local == 2:
            conf -= 0.05
        if final_genre is not None:
            conf += 0.05
        confidence: float | None = max(0.50, min(0.85, conf))

        if tag_agreement not in {"low", "medium", "high"} and conflicts_raw is None:
            conf_raw = data.get("confidence")
            if conf_raw is not None:
                try:
                    confidence = max(0.50, min(0.85, float(conf_raw)))
                except (TypeError, ValueError):
                    confidence = 0.55

        reason = clean_reason_text(str(data.get("reason", "")))
        if bpm_note == "ok":
            pieces = [p.strip() for p in reason.replace(";", ",").split(",")]
            pieces = [p for p in pieces if "bpm atypical" not in p.lower() and "bpm" not in p.lower()]
            reason = clean_reason_text(", ".join(pieces))
        if final_genre is None:
            fallback_reason = "Genre remains ambiguous from current evidence"
        elif bpm_note == "atypical":
            fallback_reason = f"Genre kept as {final_genre}; BPM atypical for genre but other evidence supports it"
        else:
            fallback_reason = f"Genre kept as {final_genre} from combined evidence"
        reason = format_reason(reason, fallback_reason)

        return {
            "tags": tags,
            "genre_top": final_genre,
            "confidence": confidence,
            "reason": reason,
            "bpm_note": bpm_note,
            "conflicts": conflicts_local,
            "conflicts_local": conflicts_local,
            "conflicts_llm": conflicts_llm,
        }
