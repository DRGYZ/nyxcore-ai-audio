from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nyxcore.llm.deepseek_client import chat_json_async
from nyxcore.rename.rules import RenameProposal, deterministic_cleanup


@dataclass(slots=True)
class RenameResult:
    old_path: Path
    new_path: Path
    ts: str
    rule_notes: str
    llm_used: bool
    changed: bool


def iter_mp3_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.mp3") if p.is_file()]


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE) if t.strip()}


def _same_script_family(a: str, b: str) -> bool:
    def has_cyrillic(s: str) -> bool:
        return bool(re.search(r"[\u0400-\u04FF]", s))

    return has_cyrillic(a) == has_cyrillic(b)


def _valid_llm_output(raw_stem: str, deterministic: str, candidate: str) -> bool:
    c = candidate.strip().strip(".")
    if not c:
        return False
    if c.lower().endswith(".mp3"):
        return False
    if not _same_script_family(raw_stem, c):
        return False

    source_tokens = _tokenize(raw_stem) | _tokenize(deterministic)
    out_tokens = _tokenize(c)
    allowed_extra = {"feat", "ft", "and"}
    for token in out_tokens:
        if token not in source_tokens and token not in allowed_extra:
            return False
    return True


def _resolve_collision(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    idx = 2
    while True:
        candidate = target.with_name(f"{stem} - {idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_rename_result(path: Path, new_base: str, notes: list[str], llm_used: bool) -> RenameResult:
    new_name = f"{new_base}.mp3"
    candidate = path.with_name(new_name)
    resolved = _resolve_collision(candidate) if candidate != path else candidate
    changed = resolved != path
    return RenameResult(
        old_path=path,
        new_path=resolved,
        ts=datetime.now(tz=UTC).isoformat(),
        rule_notes=",".join(dict.fromkeys(notes)),
        llm_used=llm_used,
        changed=changed,
    )


async def propose_rename_for_file(
    path: Path,
    *,
    use_llm: bool,
    force: bool,
    sem: asyncio.Semaphore,
    api_key: str | None,
    base_url: str,
    model: str,
    max_retries: int = 3,
    session=None,
) -> RenameResult:
    raw_stem = path.stem
    proposal: RenameProposal = deterministic_cleanup(raw_stem)
    llm_used = False
    notes = list(proposal.rule_notes)
    new_base = proposal.new_base

    should_use_llm = use_llm and (force or proposal.messy)
    if should_use_llm and api_key:
        llm_system = (
            "You clean MP3 filenames. Return STRICT JSON only: {\"new_base\":\"...\"}. "
            "Do not add information not present in source. Preserve original language and intent."
        )
        llm_user = (
            f"Original stem: {raw_stem}\n"
            f"Deterministic cleaned stem: {proposal.new_base}\n"
            "Return only JSON with key new_base (no extension)."
        )
        try:
            async with sem:
                resp = await chat_json_async(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    system_prompt=llm_system,
                    user_prompt=llm_user,
                    temperature=0.0,
                    max_retries=max_retries,
                    session=session,
                )
            cand = str(resp.data.get("new_base", "")).strip()
            if _valid_llm_output(raw_stem, proposal.new_base, cand):
                new_base = cand
                llm_used = True
                notes.append("llm_refine")
            else:
                notes.append("llm_rejected")
        except Exception:
            notes.append("llm_error")

    return build_rename_result(path, new_base, notes, llm_used)


def apply_rename(result: RenameResult) -> None:
    if not result.changed:
        return
    result.old_path.rename(result.new_path)


def undo_rename(old_path: Path, new_path: Path, *, force: bool) -> tuple[bool, str]:
    if not new_path.exists():
        return False, "missing_new_path"
    if old_path.exists():
        if not force:
            return False, "old_path_exists"
        old_path.unlink()
    new_path.rename(old_path)
    return True, "ok"
