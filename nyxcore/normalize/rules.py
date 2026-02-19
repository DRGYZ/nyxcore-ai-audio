from __future__ import annotations

import re

YOUTUBE_NOISE_PHRASES = [
    "official video",
    "official audio",
    "lyric video",
    "lyrics",
    "hd",
    "4k",
    "audio",
    "visualizer",
]

REMIX_ALBUM_HINTS = ("remix", "edit", "bootleg", "slowed", "reverb", "mix", "megamix", "extended")
NOISE_TOKEN_PATTERN = re.compile(
    r"\b(official|video|audio|lyrics?|lyric|visualizer|hd|4k|youtube|topic|music)\b",
    re.IGNORECASE,
)
YEAR_OR_RES_PATTERN = re.compile(r"^(?:\d{4}|[1248]k|[0-9]{3,4}p|hq|uhd|fhd)+$", re.IGNORECASE)
SEPARATOR_PATTERN = re.compile(r"\s[-–—]\s", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
BRACKET_SEGMENT_PATTERN = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")


def collapse_ws(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_missing(value: str | None) -> bool:
    if value is None:
        return True
    stripped = collapse_ws(value)
    return stripped == "" or stripped.upper() == "UNKNOWN"


def normalize_artist_name(artist: str | None) -> tuple[str | None, list[str]]:
    if artist is None:
        return None, []
    value = collapse_ws(artist)
    if not value:
        return None, []

    reasons: list[str] = []

    if value.lower().endswith(" - topic"):
        value = value[: -len(" - topic")].strip()
        reasons.append("artist_topic_removed")

    lower = value.lower()
    if lower.endswith(" music"):
        value = value[: -len(" music")].strip()
        reasons.append("artist_music_suffix_removed")
    elif lower.endswith("music") and " " not in value[:-5].strip() and value[:-5].strip():
        value = value[:-5].strip()
        reasons.append("artist_music_suffix_removed")

    value = collapse_ws(value)
    if not value:
        return None, reasons
    return value, reasons


def is_artist_hygiene_applicable(current_artist: str) -> bool:
    artist = collapse_ws(current_artist)
    if not artist:
        return False
    lower = artist.lower()
    return lower.endswith(" - topic") or lower.endswith("vevo") or lower.endswith(" music")


def clean_artist_hygiene(current_artist: str) -> tuple[str, list[str]]:
    value = collapse_ws(current_artist)
    reasons: list[str] = []

    if value.lower().endswith(" - topic"):
        value = value[: -len(" - topic")].strip()
        reasons.append("artist_topic_removed")

    if value.lower().endswith("vevo"):
        value = value[:-len("vevo")].strip()
        reasons.append("artist_vevo_removed")

    if value.lower().endswith(" music"):
        value = value[: -len(" music")].strip()
        reasons.append("artist_music_suffix_removed")

    value = collapse_ws(value)
    return value, reasons


def _is_noise_bracket(inner: str) -> bool:
    inner_clean = collapse_ws(inner).lower()
    if not inner_clean:
        return False
    if any(phrase in inner_clean for phrase in YOUTUBE_NOISE_PHRASES):
        return True
    tokens = re.split(r"[\s\-_]+", inner_clean)
    if not tokens:
        return False
    if all(NOISE_TOKEN_PATTERN.search(token) or YEAR_OR_RES_PATTERN.match(token) for token in tokens):
        return True
    return False


def cleaned_filename_stem(stem: str) -> tuple[str, list[str]]:
    working = stem.replace("_", " ")
    reasons: list[str] = []

    for phrase in YOUTUBE_NOISE_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        if pattern.search(working):
            working = pattern.sub(" ", working)
            if "filename_youtube_noise_removed" not in reasons:
                reasons.append("filename_youtube_noise_removed")

    def _bracket_replacer(match: re.Match[str]) -> str:
        inner = match.group(1)
        if _is_noise_bracket(inner):
            if "filename_brackets_noise_removed" not in reasons:
                reasons.append("filename_brackets_noise_removed")
            return " "
        return match.group(0)

    working = BRACKET_SEGMENT_PATTERN.sub(_bracket_replacer, working)
    return collapse_ws(working), reasons


def parse_artist_title_from_filename(stem: str) -> tuple[str | None, str | None, bool]:
    parts = SEPARATOR_PATTERN.split(stem, maxsplit=1)
    if len(parts) != 2:
        return None, None, False
    artist = collapse_ws(parts[0])
    title = collapse_ws(parts[1])
    if not artist or not title:
        return None, None, False
    return artist, title, True


def propose_album(current_album: str | None, stem: str, strategy: str) -> tuple[str | None, str | None]:
    if strategy != "smart":
        return current_album, None
    if not is_missing(current_album):
        return current_album, None

    lowered = stem.lower()
    if any(token in lowered for token in REMIX_ALBUM_HINTS):
        return "Remixes", "album_strategy_remixes"
    return "Singles", "album_strategy_singles"
