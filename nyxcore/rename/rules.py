from __future__ import annotations

import re
from dataclasses import dataclass, field

MEANINGFUL_VERSION_TOKENS = {
    "slowed": "Slowed",
    "sped up": "Sped Up",
    "speed up": "Speed Up",
    "nightcore": "Nightcore",
    "remix": "Remix",
    "edit": "Edit",
    "live": "Live",
    "instrumental": "Instrumental",
    "ost": "OST",
}

JUNK_PATTERNS = [
    r"(?i)\bextended\b",
    r"(?i)\bbest\s+parts\b",
    r"(?i)\bbass\s+boosted\b",
    r"(?i)\bofficial\s+video\b",
    r"(?i)\bofficial\s+audio\b",
    r"(?i)\blyrics?\b",
    r"(?i)\blyric\s+video\b",
    r"(?i)\bhd\b",
    r"(?i)\b4k\b",
    r"(?i)\btiktok\b",
    r"(?i)\bfree\s+download\b",
    r"(?i)\bno\s+copyright\b",
]


@dataclass(slots=True)
class RenameProposal:
    new_base: str
    rule_notes: list[str] = field(default_factory=list)
    messy: bool = False


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _replace_separators(text: str) -> str:
    t = text.replace("_", " ")
    t = re.sub(r"\s*[-|]+\s*", " - ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _strip_edges(text: str) -> str:
    return text.strip(" -_.;,|[]{}")


def _remove_junk_tokens(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    out = text
    for pattern in JUNK_PATTERNS:
        new_out = re.sub(pattern, "", out)
        if new_out != out:
            notes.append("junk_token_removed")
            out = new_out
    return out, notes


def _normalize_bracket_segment(content: str) -> tuple[str | None, str | None]:
    c = _collapse_spaces(content.lower())
    if not c:
        return None, None
    for token, canonical in MEANINGFUL_VERSION_TOKENS.items():
        if token in c:
            return f"({canonical})", "version_kept"
    if any(re.search(pattern, c) for pattern in JUNK_PATTERNS):
        return None, "junk_bracket_removed"
    # likely uploader/channel noise in brackets
    return None, "bracket_noise_removed"


def _normalize_brackets(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    versions: list[str] = []

    def repl(m: re.Match[str]) -> str:
        segment = m.group(1) or m.group(2) or m.group(3) or ""
        normalized, note = _normalize_bracket_segment(segment)
        if note:
            notes.append(note)
        if normalized and normalized not in versions:
            versions.append(normalized)
        return " "

    stripped = re.sub(r"\(([^)]*)\)|\[([^\]]*)\]|\{([^}]*)\}", repl, text)
    stripped = _collapse_spaces(stripped)
    if versions:
        stripped = _collapse_spaces(stripped + " " + " ".join(versions))
    return stripped, notes


def _remove_leading_timestamp(text: str) -> tuple[str, list[str]]:
    updated = re.sub(r"^\d+[_\-\s]+", "", text)
    if updated != text:
        return updated, ["leading_timestamp_removed"]
    return text, []


def _looks_messy(text: str) -> bool:
    if not text:
        return True
    if len(text) > 95:
        return True
    if re.search(r"[-|]{2,}", text):
        return True
    if text.count("-") > 2:
        return True
    if len(text.split()) > 14:
        return True
    return False


def deterministic_cleanup(stem: str) -> RenameProposal:
    notes: list[str] = []
    out = _replace_separators(stem)
    out, ts_notes = _remove_leading_timestamp(out)
    notes.extend(ts_notes)

    out, br_notes = _normalize_brackets(out)
    notes.extend(br_notes)

    out, junk_notes = _remove_junk_tokens(out)
    notes.extend(junk_notes)

    out = re.sub(r"\s*-\s*-\s*", " - ", out)
    out = re.sub(r"\s*;\s*;\s*", "; ", out)
    out = _collapse_spaces(out)
    out = _strip_edges(out)
    out = _collapse_spaces(out)

    messy = _looks_messy(out)
    if out != stem:
        notes.append("deterministic_cleanup")
    return RenameProposal(new_base=out or _strip_edges(stem), rule_notes=notes, messy=messy)
