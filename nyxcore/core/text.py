from __future__ import annotations

import re

REASON_CONNECTOR_TAILS = {"but", "and", "or", "with", "because", "so", "which"}
DANGLING_REASON_FRAGMENTS = {"clap tags", "clap_tags", "genre evidence", "evidence inconsistent"}
REASON_TRAILING_TOKENS = {"genre", "evidence", "inconsistent", "clap", "tags"}


def clean_reason_text(text: str) -> str:
    r = text.strip()
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
    return r


def format_reason(reason: str, fallback: str, max_chars: int = 120) -> str:
    r = reason or ""
    for sep in (".", "!", "?"):
        if sep in r:
            r = r.split(sep, 1)[0]
            break

    r = re.sub(r"(?i)\bclap[_\s]*tags?\b", "", r)
    r = re.sub(r"(?i)\bgenre\s+evidence\b", "", r)
    r = re.sub(r"(?i)\bevidence\s+inconsistent\b", "", r)
    r = re.sub(r"(?i)\bgenre\s+e\b", "", r)
    r = re.sub(r"\s+", " ", r).strip(" ,.;:!?")

    r = clean_reason_text(r)
    if len(r) > max_chars:
        clipped = r[:max_chars]
        r = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    r = r.strip(" ,.;:!?")
    while r:
        words = r.split()
        if not words:
            break
        if words[-1].lower().strip(".,;:!?") in REASON_CONNECTOR_TAILS:
            r = " ".join(words[:-1]).strip(" ,.;:!?")
            continue
        if words[-1].lower().strip(".,;:!?") in REASON_TRAILING_TOKENS:
            r = " ".join(words[:-1]).strip(" ,.;:!?")
            continue
        break
    if not r:
        r = clean_reason_text(fallback)
    return r
