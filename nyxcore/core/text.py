from __future__ import annotations

import re

REASON_CONNECTOR_TAILS = {"but", "and", "or", "with", "because", "so", "which"}
DANGLING_REASON_FRAGMENTS = {"clap tags", "clap_tags", "genre evidence", "evidence inconsistent"}
REASON_TRAILING_TOKENS = {"genre", "evidence", "inconsistent", "clap", "tags"}
REASON_END_CONNECTORS = {"for", "but", "and", "or", "with", "because", "vs", "v", "to", "of"}
REASON_END_WEAK_TAILS = {"conflict", "contradictory", "ambiguous", "partial", "partially"}
REASON_BAD_END_PATTERNS = (
    r"(?i)\bbpm atypical for$",
    r"(?i)\bbpm atypical$",
    r"(?i)\bconflict\s*\($",
    r"(?i)^conflict(\s*\(|\b)",
    r"(?i)\bcontradictory$",
)


def _remove_conflict_segments(text: str) -> str:
    # Remove "conflict (...)" including broken/unclosed parenthesis variants.
    t = re.sub(r"(?i)\bconflict\s*\([^)]*\)", "", text)
    t = re.sub(r"(?i)\bconflict\s*\([^)]*$", "", t)
    t = re.sub(r"(?i)\bconflict\s*\([^,.;:!?)]*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _drop_unmatched_bracket_tail(text: str, open_ch: str, close_ch: str) -> str:
    if text.count(open_ch) > text.count(close_ch):
        idx = text.rfind(open_ch)
        if idx >= 0:
            return text[:idx].rstrip(" ,.;:!?-()[]")
    return text


def _normalize_reason_punctuation(text: str) -> str:
    t = re.sub(r"([,;])(\s*[,;])+", r"\1", text)
    t = re.sub(r"\s+([,;:])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t.strip(" ,.;:!?-()[]")


def clean_reason_text(text: str) -> str:
    r = _remove_conflict_segments(text.strip())
    for sep in (".", "!", "?"):
        if sep in r:
            r = r.split(sep, 1)[0]
            break
    r = r.strip()
    if len(r) > 120:
        clipped = r[:120]
        r = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    r = r.strip()
    while r:
        words = r.split()
        if not words:
            break
        if words[-1].lower().strip(".,;:!?") in REASON_CONNECTOR_TAILS:
            r = " ".join(words[:-1]).strip()
            continue
        break
    while r:
        lowered = r.lower().rstrip(" ,.;:!?")
        removed = False
        for frag in DANGLING_REASON_FRAGMENTS:
            if lowered.endswith(frag):
                r = r[: len(r) - len(frag)].rstrip(" ,.;:!?")
                removed = True
                break
        if not removed:
            break
    r = r.rstrip(" ,.;:!?")
    r = _drop_unmatched_bracket_tail(r, "(", ")")
    r = _drop_unmatched_bracket_tail(r, "[", "]")
    r = _normalize_reason_punctuation(r)
    while r:
        words = r.split()
        if not words:
            break
        tail = words[-1].lower().strip(".,;:!?-()")
        if tail in REASON_END_CONNECTORS or tail in REASON_END_WEAK_TAILS:
            r = " ".join(words[:-1]).rstrip(" ,.;:!?-()")
            continue
        break
    return r


def format_reason(reason: str, fallback: str, max_chars: int = 120) -> str:
    r = _remove_conflict_segments(reason or "")
    for sep in (".", "!", "?"):
        if sep in r:
            r = r.split(sep, 1)[0]
            break

    r = re.sub(r"(?i)\bclap[_\s]*tags?\b", "", r)
    r = re.sub(r"(?i)\bgenre\s+evidence\b", "", r)
    r = re.sub(r"(?i)\bevidence\s+inconsistent\b", "", r)
    r = re.sub(r"(?i)\bgenre\s+e\b", "", r)
    r = re.sub(r"\s+", " ", r).strip(" ,.;:!?")
    r = _drop_unmatched_bracket_tail(r, "(", ")")
    r = _drop_unmatched_bracket_tail(r, "[", "]")
    r = _normalize_reason_punctuation(r)

    r = clean_reason_text(r)
    if len(r) > max_chars:
        clipped = r[:max_chars]
        r = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    r = _drop_unmatched_bracket_tail(r, "(", ")")
    r = _drop_unmatched_bracket_tail(r, "[", "]")
    r = r.strip(" ,.;:!?-()")
    while r:
        words = r.split()
        if not words:
            break
        tail = words[-1].lower().strip(".,;:!?-()")
        if tail in REASON_CONNECTOR_TAILS:
            r = " ".join(words[:-1]).strip(" ,.;:!?-()")
            continue
        if tail in REASON_TRAILING_TOKENS:
            r = " ".join(words[:-1]).strip(" ,.;:!?-()")
            continue
        if tail in REASON_END_CONNECTORS or tail in REASON_END_WEAK_TAILS:
            r = " ".join(words[:-1]).strip(" ,.;:!?-()")
            continue
        break
    r = r.strip(" ,.;:!?-()")
    if r:
        for pattern in REASON_BAD_END_PATTERNS:
            if re.search(pattern, r):
                r = ""
                break
    if len(r) <= 12:
        r = ""
    if not r:
        r = clean_reason_text(fallback)
        r = _normalize_reason_punctuation(r)
    if len(r) < 12:
        r = "Genre kept from source; evidence mixed"
    return r
