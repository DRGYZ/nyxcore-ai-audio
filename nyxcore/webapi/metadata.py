from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Callable, Literal
from threading import Lock
from unicodedata import category

from fastapi import Depends, FastAPI, HTTPException, Query
from mutagen import File as MutagenFile, MutagenError
from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import EasyMP3
from mutagen.easymp4 import EasyMP4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nyxcore.action_plan.service import (
    MAX_AUTOMATIC_PLAN_OPERATIONS,
    ActionPlan,
    ActionPlanOperation,
    ActionPlanReport,
    ActionPlanSummary,
    execute_authorized_action_plan,
)
from nyxcore.config import HealthConfig
from nyxcore.core.atomic import atomic_write_text
from nyxcore.core.audio_files import SUPPORTED_AUDIO_EXTENSIONS
from nyxcore.core.filesystem import FileIdentity, file_snapshot
from nyxcore.core.text import normalize_match_text
from nyxcore.core.track import TrackRecord
from nyxcore.enrichment import Providers, WorkspaceCache
from nyxcore.health.service import is_placeholder, is_suspicious_swap
from nyxcore.normalize.rules import SEPARATOR_PATTERN, cleaned_filename_stem, collapse_ws, parse_artist_title_from_filename
from nyxcore.tagging.writer import ArtworkEdit, front_covers
from nyxcore.webapi.enrichment import ArtworkReference, Enrichment, Source, thumbnail
from nyxcore.webapi.schemas import ReviewPlanApplyResponse

Tag = Literal["title", "artist", "album"]
Issue = Literal["missing_title", "missing_artist", "missing_album", "placeholder_metadata", "suspicious_swaps"]
FIELDS: tuple[Tag, ...] = ("title", "artist", "album")
MAX_DRAFTS = MAX_AUTOMATIC_PLAN_OPERATIONS
MAX_DRAFT_BYTES = 2 * 1024 * 1024
TrackPath = Annotated[str, Field(min_length=1, max_length=4096)]
Proof = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TagText = Annotated[str, Field(min_length=1, max_length=512)]


class MetadataProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: TrackPath
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    inspection: Proof
    values: dict[Tag, TagText] = Field(max_length=3)
    source: Source | None = None
    source_fields: list[Tag] | None = Field(None, max_length=3)
    artwork: ArtworkReference | None = None

    @model_validator(mode="after")
    def validate_proposal(self):
        if not self.values and not self.artwork:
            raise ValueError("Choose at least one tag change or front cover.")
        if self.artwork and not self.source:
            raise ValueError("Artwork must identify its selected MusicBrainz release.")
        if self.source_fields:
            if self.source is None or len(set(self.source_fields)) != len(self.source_fields):
                raise ValueError("Provider fields must identify their selected MusicBrainz source.")
            suggested = self.source.tag_values()
            if any(field not in self.values or self.values[field] != suggested[field] for field in self.source_fields):
                raise ValueError("A MusicBrainz-suggested field must match its source. Label overrides as manual edits.")
        return self

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: dict[Tag, str]) -> dict[Tag, str]:
        for value in values.values():
            if (not value.strip() or value != value.strip() or value.upper() == "UNKNOWN"
                    or any(category(character) in {"Cc", "Cs"} for character in value)):
                raise ValueError("Enter a nonempty value without leading or trailing whitespace or control characters. Clearing tags and UNKNOWN are unsupported.")
        return values


class Drafts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[MetadataProposal] = Field(max_length=MAX_DRAFTS)


class DraftUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: Proof
    entries: list[MetadataProposal] = Field(max_length=MAX_DRAFTS)


class ProposalSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[MetadataProposal] = Field(min_length=1, max_length=MAX_DRAFTS)


class ConfirmedProposal(ProposalSelection):
    preview: Proof
    confirm: Literal[True]


@dataclass(frozen=True)
class Inspection:
    path: Path
    tags: dict[str, list[str]]
    readable: bool
    write_blocker: str | None
    snapshot: tuple[FileIdentity, str]
    duration_ms: int | None
    covers: list[bytes] | None


def _reparse_point(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _track_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise HTTPException(403, "Choose a track inside the configured library.")
    parts = path.relative_to(root).parts
    if not parts or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS or any(
        (path.drive and ":" in part) or part.lower() in {".nyxcore_quarantine", ".nyxcore_backups"} for part in parts
    ):
        raise HTTPException(403, "Choose an audio file from the active library.")
    current = root
    for part in parts:
        current = current / part
        if _reparse_point(current):
            raise HTTPException(403, "Linked paths cannot be edited. Choose the original track location.")
    if path.resolve(strict=True) != path:
        raise HTTPException(403, "Choose a track at its canonical library path.")
    return path


def _write_blocker(path: Path, audio: object) -> str | None:
    supported_type = {".mp3": EasyMP3, ".flac": FLAC, ".m4a": EasyMP4}.get(path.suffix.lower())
    if supported_type is None or not isinstance(audio, supported_type):
        return "Writing tags is available for compatible MP3, FLAC, and M4A files."
    if path.stat().st_nlink != 1:
        return "This file has multiple hard links. Use a separate copy before editing."
    if not os.access(path, os.W_OK):
        return "This file is read-only or your account cannot write to it. Check its permissions before staging."
    if isinstance(audio, EasyMP3):
        try:
            if ID3(path, translate=False, load_v1=False).version[:2] not in {(2, 3), (2, 4)}:
                return "This MP3 uses an unsupported ID3 version. Tag editing supports ID3v2.3 and ID3v2.4."
        except ID3NoHeaderError:
            pass
    return None


def _inspect(root: Path, value: str) -> Inspection:
    try:
        path = _track_path(root, value)
        before = file_snapshot(path)
        if before is None:
            raise ValueError("Choose a regular audio file.")
        tags = {field: [] for field in FIELDS}
        readable = False
        covers = None
        duration = None
        blocker = "Embedded metadata could not be read. Check the file before editing."
        try:
            audio = MutagenFile(path, easy=True)
            stored = getattr(audio, "tags", None)
            if stored is not None:
                for field in FIELDS:
                    values = stored.get(field)
                    if values is not None:
                        tags[field] = [str(item) for item in values] if isinstance(values, (list, tuple)) else [str(values)]
            if audio is not None:
                blocker = _write_blocker(path, audio)
                readable = True
                seconds = getattr(getattr(audio, "info", None), "length", None)
                if seconds and 0 < seconds <= 24 * 3600:
                    duration = max(1, round(seconds * 1000))
                if blocker is None:
                    covers = front_covers(path)
        except (MutagenError, KeyError, OSError, ValueError):
            pass
        if _track_path(root, value) != path or file_snapshot(path) != before:
            raise ValueError("The file changed during inspection. Reopen it and try again.")
        return Inspection(path, tags, readable, blocker, before, duration, covers)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(409, "The file is missing, unreadable, or changed. Refresh the library and inspect it again.") from exc


def _issues(record: TrackRecord, settings: HealthConfig) -> list[str]:
    issues = [f"missing_{field}" for field in FIELDS if not (record.tags.get(field) or "").strip()]
    if any(is_placeholder(record.tags.get(field), settings) for field in FIELDS):
        issues.append("placeholder_metadata")
    if is_suspicious_swap(Path(record.path), record):
        issues.append("suspicious_swaps")
    return issues


def _suggest(inspection: Inspection, settings: HealthConfig) -> tuple[list[str], dict, bool]:
    current = {field: next(iter(inspection.tags[field]), None) for field in FIELDS}
    record = TrackRecord(str(inspection.path), 0, "", current, False, None)
    issues = _issues(record, settings)
    stem, _ = cleaned_filename_stem(inspection.path.stem)
    artist, title, parsed = parse_artist_title_from_filename(stem)
    parsed = parsed and len(SEPARATOR_PATTERN.findall(stem)) == 1
    filename = {"artist": artist, "title": title}
    swap = (
        "suspicious_swaps" in issues
        and all(len(inspection.tags[field]) == 1 for field in ("artist", "title"))
        and normalize_match_text(current["artist"]) != normalize_match_text(current["title"])
    )
    suggestions = {}
    for field in FIELDS:
        value = current[field]
        guess = filename.get(field)
        if swap and field in filename:
            other = "artist" if field == "title" else "title"
            suggestions[field] = {"value": current[other], "source": "swap", "confidence": "matched",
                                  "evidence": "Both embedded values match the opposite filename components."}
        elif len(inspection.tags[field]) > 1:
            suggestions[field] = {"value": None, "source": "unavailable", "confidence": "ambiguous",
                                  "evidence": "This field has multiple embedded values. Review them before replacing."}
        elif not value or not value.strip() or is_placeholder(value, settings):
            if parsed and guess and not is_placeholder(guess, settings):
                suggestions[field] = {"value": guess, "source": "filename", "confidence": "tentative",
                                      "evidence": "Parsed from the filename. Check the identification yourself."}
            else:
                suggestions[field] = {"value": None, "source": "unavailable", "confidence": "ambiguous",
                                      "evidence": "Enter the album if you know it." if field == "album"
                                      else "The filename offers no clear replacement."}
        elif collapse_ws(value) != value:
            suggestions[field] = {"value": collapse_ws(value), "source": "formatting", "confidence": "high",
                                  "evidence": "Normalize whitespace in the embedded value."}
        else:
            suggestions[field] = {"value": value, "source": "embedded", "confidence": "existing",
                                  "evidence": "Current embedded value."}
    return issues, suggestions, swap


def install_metadata_routes(
    app: FastAPI,
    *,
    library: Callable[[], Path],
    workspace: Callable[[], Path],
    settings: Callable[[], HealthConfig],
    generation: Callable[[], str],
    records: Callable[[], list[TrackRecord]],
    require_mutations: Callable[[], None],
    invalidate: Callable[[], None],
    review_only: bool,
) -> None:
    key = secrets.token_bytes(32)
    draft_lock = Lock()

    def proof(*data: object) -> str:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    enrichment = Enrichment(Providers(WorkspaceCache(workspace)), proof)
    enrichment.mount(app)

    def inspection_proof(item: Inspection, version: str) -> str:
        return proof("inspection", str(item.path), item.snapshot, version)

    def draft_file() -> Path:
        path = workspace() / "metadata_drafts.json"
        if path.is_symlink() or (path.exists() and (_reparse_point(path) or not path.is_file())):
            raise HTTPException(409, "Metadata drafts must be a regular workspace file.")
        return path

    def load_drafts(path: Path) -> Drafts:
        if not path.exists():
            return Drafts(entries=[])
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_DRAFT_BYTES + 1)
            if len(data) > MAX_DRAFT_BYTES:
                raise ValueError("Draft file is too large")
            return Drafts.model_validate_json(data)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "The metadata draft file cannot be read. Check metadata_drafts.json in the workspace.") from exc

    def draft_response(drafts: Drafts) -> dict:
        data = drafts.model_dump(exclude_none=True)
        return {**data, "revision": proof("drafts", str(workspace()), data), "generation": generation()}

    def inspect_proposal(entry: MetadataProposal, version: str) -> tuple[Inspection, dict[str, str], ArtworkEdit | None]:
        if entry.generation != version:
            raise HTTPException(409, "Library analysis changed. Reopen the staged track and stage it again.")
        inspected = _inspect(library(), entry.path)
        if not hmac.compare_digest(entry.inspection, inspection_proof(inspected, version)):
            raise HTTPException(409, f"File changed since inspection: {inspected.path.name}. Reopen and stage it again.")
        if inspected.write_blocker:
            raise HTTPException(409, f"Metadata writing is unavailable for {inspected.path.name}. {inspected.write_blocker}")
        changes = {field: value for field, value in entry.values.items() if inspected.tags[field] != [value]}
        if entry.source:
            enrichment.check_source(entry.source)
        artwork = None
        if entry.artwork:
            if inspected.covers is None or len(inspected.covers) > 1:
                raise HTTPException(409, "This tag layout does not support safe front-cover editing.")
            if inspected.covers and not entry.artwork.replace_existing:
                raise HTTPException(409, "Explicitly approve replacing this track’s existing front cover.")
            assert entry.source is not None
            artwork = enrichment.cached_artwork(entry.source, entry.artwork)
        if not changes and artwork is None:
            raise HTTPException(409, f"No metadata changes for {inspected.path.name}.")
        return inspected, changes, artwork

    def propose(entries: list[MetadataProposal], *, preview: bool) -> tuple[
        ActionPlanReport, dict[str, FileIdentity], list[dict], dict[str, ArtworkEdit]
    ]:
        version = generation()
        operations = []
        identities = {}
        comparisons = []
        paths = set()
        artwork_edits = {}
        for entry in entries:
            item, changes, artwork = inspect_proposal(entry, version)
            if item.path in paths or item.snapshot[0] in identities.values():
                raise HTTPException(409, "A batch can include a physical file only once.")
            paths.add(item.path)
            operation_values = {**changes, "front_cover": artwork.digest} if artwork else changes
            operation_id = "metadata-" + proof("operation", str(item.path), item.snapshot, operation_values)[:24]
            identities[operation_id] = item.snapshot[0]
            if artwork:
                artwork_edits[operation_id] = artwork
            operations.append(ActionPlanOperation(
                operation_id=operation_id, operation_type="write_metadata", path=str(item.path),
                fields=list(operation_values), values=operation_values, apply_supported=True, expected_hash=item.snapshot[1],
            ))
            if preview:
                comparison = {"path": str(item.path), "before": item.tags, "changes": changes}
                if entry.source:
                    comparison["source"] = entry.source.model_dump(exclude={"token"})
                    comparison["source_fields"] = [field for field in entry.source_fields or [] if field in changes]
                if artwork:
                    comparison["artwork"] = {
                        "before_present": bool(item.covers),
                        "before": thumbnail(item.covers[0]) if item.covers else None,
                        "after": thumbnail(artwork.data),
                        "release_id": entry.artwork.release_id,
                        "digest": artwork.digest,
                    }
                comparisons.append(comparison)
        if generation() != version:
            raise HTTPException(409, "Library analysis changed during planning. Inspect the staged tracks again.")
        plan = ActionPlan(
            plan_id="metadata-repair-" + proof("plan", sorted(identities))[:24],
            source_review_item_ids=[], action_type="metadata_fix_plan", affected_files=sorted(map(str, paths)),
            proposed_operations=operations, confidence=1, safety_level="review_required",
            reasons=["User-approved tag replacements"], notes=[], apply_supported=True, resolves_review_items=False,
        )
        report = ActionPlanReport(datetime.now(UTC).isoformat(), [], [plan], [], ActionPlanSummary(0, 1, 0, 1))
        return report, identities, comparisons, artwork_edits

    @app.get("/api/metadata/issues")
    def metadata_issues(issue: Issue, offset: int = Query(0, ge=0)) -> dict:
        config = settings()
        matches = [item.path for item in records() if issue in _issues(item, config)]
        return {"paths": matches[offset:offset + 25], "total": len(matches)}

    @app.get("/api/metadata/track")
    def metadata_track(path: TrackPath) -> dict:
        version = generation()
        item = _inspect(library(), path)
        issues, suggestions, swap = _suggest(item, settings()) if item.readable else ([], {}, False)
        return {"path": str(item.path), "tags": item.tags, "supported": item.write_blocker is None,
                "readable": item.readable, "write_blocker": item.write_blocker, "issues": issues,
                "suggestions": suggestions, "swap": swap, "generation": version,
                "duration_ms": item.duration_ms, "cover_present": bool(item.covers),
                "artwork_supported": item.covers is not None and len(item.covers) <= 1,
                "inspection": inspection_proof(item, version)}

    @app.get("/api/metadata/drafts")
    def metadata_drafts() -> dict:
        with draft_lock:
            return draft_response(load_drafts(draft_file()))

    @app.post("/api/metadata/drafts")
    def update_metadata_drafts(request: DraftUpdate) -> dict:
        with draft_lock:
            path = draft_file()
            current = load_drafts(path)
            response = draft_response(current)
            if not hmac.compare_digest(request.revision, response["revision"]):
                raise HTTPException(409, "Staged edits changed in another tab. Reload the drafts and retry.")
            previous = {entry.path: entry for entry in current.entries}
            if len({Path(entry.path) for entry in request.entries}) != len(request.entries):
                raise HTTPException(409, "Stage only one edit per track.")
            for entry in request.entries:
                if previous.get(entry.path) != entry:
                    inspect_proposal(entry, response["generation"])
            updated = Drafts(entries=request.entries)
            data = updated.model_dump_json(indent=2, exclude_none=True)
            if len(data.encode()) > MAX_DRAFT_BYTES:
                raise HTTPException(413, "The staged edits exceed the workspace draft limit.")
            try:
                atomic_write_text(path, data + "\n")
            except OSError as exc:
                raise HTTPException(409, "The metadata draft could not be saved. Check workspace permissions.") from exc
            return draft_response(updated)

    @app.post("/api/metadata/plan")
    def metadata_plan(selection: ProposalSelection) -> dict:
        _, _, comparisons, _ = propose(selection.entries, preview=True)
        return {"preview": proof("preview", str(library()), selection.model_dump()), "files": comparisons,
                "review_only": review_only}

    @app.post("/api/metadata/plan/apply", response_model=ReviewPlanApplyResponse, dependencies=[Depends(require_mutations)])
    def apply_metadata_plan(request: ConfirmedProposal) -> ReviewPlanApplyResponse:
        root, out = library(), workspace()
        selection = ProposalSelection(entries=request.entries)
        expected = proof("preview", str(root), selection.model_dump())
        if not hmac.compare_digest(request.preview, expected):
            raise HTTPException(409, "The approved preview is stale or changed. Generate a new preview.")

        artwork_changes: dict[str, ArtworkEdit] = {}

        def authorize() -> tuple[ActionPlanReport, dict[str, FileIdentity]]:
            report, identities, _, verified = propose(request.entries, preview=False)
            artwork_changes.update(verified)
            return report, identities

        try:
            _, results, batch = execute_authorized_action_plan(
                root, authorize, review_state_path=out / "review_state.json",
                ledger_path=out / "review_history.json", backup_dir=out / "metadata_backups", workspace_root=out,
                artwork_changes=artwork_changes,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
        invalidate()
        return ReviewPlanApplyResponse(
            result_count=len(results), batch_id=batch.batch_id, results=[item.to_dict() for item in results],
        )
