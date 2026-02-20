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
    raw = _collapse_spaces(content)
    c = raw.lower()
    if not c:
        return None, None
    if re.search(r"(?i)\b(feat\.?|ft\.?|featuring)\b", c):
        normalized = re.sub(r"(?i)\b(featuring|ft\.?)\b", "feat.", raw)
        normalized = _collapse_spaces(normalized)
        return f"({normalized})", "feat_kept"
    for token, canonical in MEANINGFUL_VERSION_TOKENS.items():
        if token in c:
            if token == "remix":
                prefix = re.sub(r"(?i)\bremix\b", "", raw).strip(" -_.;,")
                if prefix:
                    pretty = _collapse_spaces(" ".join(part.capitalize() for part in prefix.split()))
                    return f"({pretty} Remix)", "remix_normalized"
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
    updated = re.sub(r"^\d{7,}[_-]+", "", text)
    if updated != text:
        return updated, ["leading_timestamp_removed"]
    return text, []


def _normalize_remix_suffix(text: str) -> tuple[str, list[str]]:
    t = _collapse_spaces(text)
    notes: list[str] = []
    if re.search(r"(?i)\([^)]*remix\)\s*$", t):
        return t, notes
    if not re.search(r"(?i)\bremix\s*$", t):
        return t, notes

    pre = re.sub(r"(?i)\s+remix\s*$", "", t).rstrip(" -_.;,")
    if not pre:
        return t, notes

    prefix_main = ""
    tail = pre
    if " - " in pre:
        prefix_main, tail = pre.rsplit(" - ", 1)
        prefix_main = prefix_main.rstrip(" -_.;,")
        tail = tail.strip(" -_.;,")

    words = tail.split()
    if not words:
        return t, notes
    if len(words) <= 2:
        name_tokens = words
        core_tokens: list[str] = []
    else:
        name_tokens = [words[-1]]
        core_tokens = words[:-1]

    name = " ".join(name_tokens).strip(" -_.;,")
    if not name:
        return t, notes
    pretty = " ".join(part.capitalize() for part in name.split())
    core = " ".join(core_tokens).strip(" -_.;,")

    left_parts = [p for p in [prefix_main, core] if p]
    left = " - ".join(left_parts) if prefix_main and core else (" ".join(left_parts).strip() if left_parts else "")
    out = f"{left} ({pretty} Remix)" if left else f"({pretty} Remix)"
    notes.append("remix_suffix_normalized")
    return _collapse_spaces(out), notes


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
    out, remix_notes = _normalize_remix_suffix(out)
    notes.extend(remix_notes)
    out = _collapse_spaces(out)
    out = _strip_edges(out)
    out = _collapse_spaces(out)

    messy = _looks_messy(out)
    if out != stem:
        notes.append("deterministic_cleanup")
    return RenameProposal(new_base=out or _strip_edges(stem), rule_notes=notes, messy=messy)


def _self_test_rename_rules() -> None:
    assert deterministic_cleanup("5 Seconds of Summer - Teeth").new_base.startswith("5 Seconds")
    assert deterministic_cleanup("50 Cent - P.I.M.P.").new_base.startswith("50 Cent")
    assert "(feat. merees)" in deterministic_cleanup("Asenssia - Каждый раз (feat. merees)").new_base
    assert deterministic_cleanup("DOOM - Unholy Siege PANDORA REMIX").new_base.endswith("(Pandora Remix)")


if __name__ == "__main__":
    _self_test_rename_rules()
