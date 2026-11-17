"""Read-only path proposals built from the current library inventory and embedded tags."""
from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from nyxcore.config import HealthConfig
from nyxcore.core.audio_files import INTERNAL_AUDIO_DIRS, SUPPORTED_AUDIO_EXTENSIONS
from nyxcore.core.filesystem import is_linked_stat, scan_stat_key
from nyxcore.core.scanner import scan_audio_files
from nyxcore.core.text import normalize_match_text
from nyxcore.core.track import TrackRecord, WarningCode
from nyxcore.health.service import is_placeholder, is_suspicious_swap
from nyxcore.normalize.rules import is_missing
from nyxcore.rename.rules import MAX_FILENAME_LEN, sanitize_windows_filename

Status = Literal["ready", "needs_review", "collision", "unchanged"]
STATUS_VALUES = ("ready", "needs_review", "collision", "unchanged")
SCAN_BATCH_SIZE = 128
RESERVED = re.compile(r"(?i)(?:con|prn|aux|nul|conin\$|conout\$|(?:com|lpt)[1-9¹²³])\Z")
CHANNEL = re.compile(r"(?:\s+-\s+topic|vevo|\s+-\s+official\s+channel)$", re.I)
GUEST = re.compile(r"\b(?:ft\.?|feat\.?|featuring)\s", re.I)
VARIOUS = {"various", "various artists", "va", "v a"}
BAD_TAG = {"untitled", "unknown", "unknown album", "unknown album artist", "unknown artist", "unknown title", "no album"}


@dataclass(frozen=True, slots=True)
class Proposal:
    path: str
    source: str
    status: Status
    reasons: tuple[str, ...]
    destination: tuple[str, str, str] | None = None
    inspectable: bool = False


@dataclass(frozen=True, slots=True)
class Entry:
    path: Path
    parts: tuple[str, ...]
    info: os.stat_result | None
    blocked: str | None = None


def path_key(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part.rstrip(" .").upper().casefold()) for part in parts)


def _unsafe_text(value: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cs"} for char in value)


def _inventory(root: Path, info: os.stat_result) -> tuple[dict[tuple[str, ...], list[Entry]], list[Entry]]:
    occupied: dict[tuple[str, ...], list[Entry]] = defaultdict(list)
    candidates = []
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            children = list(folder.iterdir())
        except OSError:
            if folder == root:
                raise
            key = folder.relative_to(root).parts
            old = next(entry for entry in occupied[path_key(key)] if entry.parts == key)
            blocked = replace(old, blocked="This folder could not be read. Check its permissions.")
            occupied[path_key(key)].remove(old)
            occupied[path_key(key)].append(blocked)
            candidates.append(blocked)
            continue
        for path in children:
            parts = path.relative_to(root).parts
            audio = path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
            internal = path_key(parts)[-1] in INTERNAL_AUDIO_DIRS
            blocked = None
            try:
                node = path.lstat()
                if is_linked_stat(node):
                    blocked = "Linked paths are excluded. Inspect the original location."
                elif node.st_dev != info.st_dev:
                    blocked = "This location is on another filesystem."
                elif _unsafe_text(path.name) or (path.drive and ":" in path.name):
                    blocked = "The current path contains unsupported characters."
                elif audio and not stat.S_ISDIR(node.st_mode) and (not stat.S_ISREG(node.st_mode) or node.st_nlink != 1):
                    blocked = "Use a separate regular audio file before organizing this path."
            except OSError:
                node = None
                blocked = "The path could not be inspected. Check that it is available."
            entry = Entry(path, parts, node, blocked)
            occupied[path_key(parts)].append(entry)
            if internal:
                continue
            if blocked or audio and node is not None and stat.S_ISREG(node.st_mode):
                candidates.append(entry)
            elif node is not None and stat.S_ISDIR(node.st_mode):
                pending.append(path)
    return occupied, sorted(candidates, key=lambda item: (path_key(item.parts), item.parts))


def _tag(record: TrackRecord, field: str) -> str | None:
    values = record.organization_tags.get(field)
    if values is None:
        value = record.tags.get(field)
        return value.strip() if value else None
    distinct = {unicodedata.normalize("NFC", value.strip()): value.strip() for value in values}
    if len(distinct) > 1:
        raise ValueError(f"Conflicting embedded {field.replace('albumartist', 'album artist')} values.")
    return next(iter(distinct.values()), None)


def _unreliable(value: str | None, settings: HealthConfig) -> bool:
    return is_missing(value) or normalize_match_text(value) in BAD_TAG or is_placeholder(value, settings)


def _compilation(record: TrackRecord) -> bool:
    flags = {normalize_match_text(value) for value in record.organization_tags.get("compilation", ())}
    flags.discard("")
    yes, no = {"1", "true", "yes"}, {"0", "false", "no"}
    release_types = record.organization_tags.get("musicbrainz_albumtype", ())
    compilation_type = any("compilation" in normalize_match_text(value).split() for value in release_types)
    if flags - yes - no or flags & no and (flags & yes or compilation_type):
        raise ValueError("The compilation flag is unclear. Verify the release and album artist.")
    return bool(flags & yes or compilation_type)


def _number(value: str | None, field: str) -> tuple[int, int]:
    if not value:
        return 0, 0
    match = re.fullmatch(r"([0-9]{1,3})(?:\s*/\s*([0-9]{1,3}))?", value)
    if match:
        number = int(match[1])
        total = int(match[2] or 0) or number
        if 1 <= number <= total:
            return number, total
    raise ValueError(f"Verify the embedded {field} number before organizing this track.")


def _segment(raw: str, field: str) -> str:
    if _unsafe_text(raw):
        raise ValueError(f"The {field} contains unsupported characters.")
    if raw.startswith(("/", "\\")) or re.match(r"^[a-z]:[/\\]", raw, re.I) or any(
        part.strip() in {".", ".."} for part in re.split(r"[/\\]", raw)
    ):
        raise ValueError(f"The {field} looks like a filesystem path. Verify the tag.")
    name = sanitize_windows_filename(raw)
    if not name or RESERVED.fullmatch(name.split(".", 1)[0].rstrip(" ")):
        raise ValueError(f"The {field} cannot form a safe Windows path name.")
    if path_key((name,))[0] in INTERNAL_AUDIO_DIRS:
        raise ValueError(f"The {field} names an internal NyxCore folder.")
    return name


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _proposal(
    root: Path, entry: Entry, record: TrackRecord, album_guards: dict[str, set[str]], settings: HealthConfig,
) -> tuple[Proposal, tuple[str, str]]:
    blocked = Proposal(str(entry.path), "/".join(entry.parts), "needs_review", (), inspectable=True)
    if set(record.warnings) & {WarningCode.read_error, WarningCode.tag_parse_error}:
        raise ValueError("Embedded metadata could not be read. Inspect the file before organizing it.")
    fields = ("title", "artist", "album")
    title, artist, album = (_tag(record, field) or "" for field in fields)
    missing = [field for field, value in zip(fields, (title, artist, album), strict=True) if _unreliable(value, settings)]
    if missing:
        raise ValueError("Add reliable embedded tags for " + ", ".join(missing) + ".")
    album_artist = _tag(record, "albumartist")
    compilation = _compilation(record)
    if is_suspicious_swap(entry.path, record):
        raise ValueError("Title and artist appear swapped. Verify both embedded tags.")
    if album_artist:
        if _unreliable(album_artist, settings):
            raise ValueError("Verify the embedded album artist with a tag editor.")
        group = album_artist
        reasons = ["Grouped by embedded album artist and album."]
    else:
        group = artist
        credit_key = path_key((group,))[0]
        if (compilation or normalize_match_text(group) in VARIOUS or GUEST.search(group) or
                album_guards[path_key((album,))[0]] != {credit_key}):
            raise ValueError("Album credits are mixed or incomplete. Verify the release and set an explicit album artist.")
        reasons = ["Grouped by the one observed artist credit for this album title."]
    if CHANNEL.search(group) or CHANNEL.search(artist):
        raise ValueError("The artist credit looks like an uploader channel. Verify the artist tag.")

    version = _tag(record, "version")
    tagged_title = title
    version_text = normalize_match_text(version)
    title_has_version = version_text and f" {version_text} " in f" {normalize_match_text(title)} "
    if version and not title_has_version:
        tagged_title = f"{title} ({version})"
        reasons.append("The embedded version is retained.")
    artist_folder = _segment(group, "artist folder")
    album_folder = _segment(album, "album folder")
    stem = _segment(tagged_title, "track title")
    if (artist_folder, album_folder, stem) != (group, album, tagged_title):
        reasons.append("Windows-incompatible punctuation or trailing characters were trimmed for the preview.")

    track, track_total = _number(_tag(record, "tracknumber"), "track")
    disc, disc_total = _number(_tag(record, "discnumber"), "disc")
    if track:
        number = str(track).zfill(max(2, len(str(track_total))))
        if disc_total > 1:
            number = f"{str(disc).zfill(len(str(disc_total)))}-{number}"
        stem = f"{number} - {stem}"
        reasons.append("The filename prefix uses embedded track numbering.")
    destination = (artist_folder, album_folder, stem + entry.path.suffix)
    if (any(len(part.encode("utf-8")) > 255 or _utf16_length(part) > 255 for part in destination)
            or _utf16_length(destination[-1]) > MAX_FILENAME_LEN or _utf16_length(str(root.joinpath(*destination))) > 259):
        raise ValueError("The proposed path is too long. Verify shorter metadata or use a shorter library root.")
    status = "unchanged" if entry.parts == destination else "ready"
    return replace(blocked, status=status, reasons=tuple(reasons), destination=destination), (group, album)


def _album_guards(records: list[TrackRecord]) -> dict[str, set[str]]:
    """Fallback to artist only when the observed album cohort has a single consistent credit."""
    credits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        values = record.organization_tags.get("album", ()) or (record.tags.get("album") or "",)
        try:
            artist = _tag(record, "artist") or ""
            album_artist = _tag(record, "albumartist")
            evidence = {path_key((artist,))[0]}
            if album_artist:
                evidence.add(path_key((album_artist,))[0])
            if _compilation(record):
                evidence.add("")
        except ValueError:
            evidence = {""}
        for album in values:
            if album.strip():
                credits[path_key((album.strip(),))[0]].update(evidence)
    return credits


def _source_stat(parts: tuple[str, ...], root: Path, device: int) -> tuple[int, int, int, int] | None:
    """Recheck each source chain before reading tags or using cached scan metadata."""
    current = root
    try:
        for index, name in enumerate(parts, start=1):
            current = current / name
            node = current.lstat()
            if (is_linked_stat(node) or node.st_dev != device
                    or (index < len(parts) and not stat.S_ISDIR(node.st_mode))):
                return None
        if stat.S_ISREG(node.st_mode) and node.st_nlink == 1:
            return scan_stat_key(node)
    except OSError:
        pass
    return None


def _display(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def build_organization_preview(root: Path, settings: HealthConfig, cached: list[TrackRecord]) -> list[Proposal]:
    root = root.absolute()
    if (os.name == "nt" and str(root).startswith("\\\\")) or ".." in root.parts or _unsafe_text(str(root)):
        raise ValueError("Choose a canonical local library folder for organization.")
    info = root.lstat()
    if is_linked_stat(info) or not stat.S_ISDIR(info.st_mode) or root.resolve(strict=True) != root:
        raise ValueError("Choose the original library folder instead of a linked path.")
    occupied, entries = _inventory(root, info)
    cached_by_path = {record.path: record for record in cached}
    stamps = {str(entry.path): _source_stat(entry.parts, root, info.st_dev) for entry in entries if not entry.blocked}
    accepted = {
        path: record for path, stamp in stamps.items()
        if stamp is not None and (record := cached_by_path.get(path)) is not None and record.scanned_stat == stamp
    }
    to_scan = [path for path, stamp in stamps.items() if stamp is not None and path not in accepted]
    for start in range(0, len(to_scan), SCAN_BATCH_SIZE):
        batch = to_scan[start:start + SCAN_BATCH_SIZE]
        accepted.update({record.path: record for record in scan_audio_files(map(Path, batch))})
    records = {
        path: record for path, record in accepted.items()
        if record.scanned_stat is not None and record.scanned_stat == stamps.get(path)
        and _source_stat(Path(path).relative_to(root).parts, root, info.st_dev) == record.scanned_stat
    }
    album_guards = _album_guards(list(records.values()))

    rows = []
    all_destinations: dict[tuple[str, ...], int] = defaultdict(int)
    folder_spellings: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for entry in entries:
        record = records.get(str(entry.path))
        if record is None:
            reason = entry.blocked or "The source changed or could not be read. Build a new preview."
            rows.append(Proposal(_display(str(entry.path)), _display("/".join(entry.parts)), "needs_review", (reason,)))
            continue
        try:
            proposal, (group, album) = _proposal(root, entry, record, album_guards, settings)
        except ValueError as exc:
            rows.append(Proposal(str(entry.path), "/".join(entry.parts), "needs_review", (str(exc),), inspectable=True))
            continue
        assert proposal.destination is not None
        key = path_key(proposal.destination)
        all_destinations[key] += 1
        folder_spellings[key[:1]].add((proposal.destination[0], group))
        folder_spellings[key[:2]].add((*proposal.destination[:2], group, album))
        rows.append(proposal)

    # A native lookup also recognizes filesystem-specific aliases such as short names on NTFS.
    native: dict[tuple[str, ...], str | None] = {}
    return [_check_destination(row, root, occupied, all_destinations, folder_spellings, native) for row in rows]


def _native_occupant(root: Path, parts: tuple[str, ...], cache: dict[tuple[str, ...], str | None]) -> str | None:
    if parts not in cache:
        try:
            node = root.joinpath(*parts).lstat()
        except FileNotFoundError:
            cache[parts] = None
        except OSError:
            cache[parts] = "unsafe"
        else:
            cache[parts] = "unsafe" if is_linked_stat(node) else "alias"
    return cache[parts]


def _check_destination(
    row: Proposal,
    root: Path,
    occupied: dict[tuple[str, ...], list[Entry]],
    all_destinations: dict[tuple[str, ...], int],
    folder_spellings: dict[tuple[str, ...], set[tuple[str, ...]]],
    native: dict[tuple[str, ...], str | None],
) -> Proposal:
    if row.destination is None:
        return row
    key = path_key(row.destination)
    missing_parent = False
    for depth in (1, 2):
        siblings = occupied.get(key[:depth], [])
        alias = None if siblings or missing_parent else _native_occupant(root, row.destination[:depth], native)
        missing_parent = missing_parent or (not siblings and alias is None)
        if alias == "unsafe" or any(entry.blocked for entry in siblings):
            return replace(row, status="needs_review", reasons=("A proposed folder crosses a linked or unavailable path.",))
        if alias == "alias" or len(siblings) > 1 or any(entry.parts != row.destination[:depth] for entry in siblings):
            return replace(row, status="collision", reasons=("An existing folder has an equivalent spelling or filesystem alias.",))
        if any(entry.info is None or not stat.S_ISDIR(entry.info.st_mode) for entry in siblings):
            return replace(row, status="collision", reasons=("An existing item occupies a proposed folder path.",))
        if len(folder_spellings[key[:depth]]) > 1:
            return replace(row, status="collision", reasons=("Different metadata values produce the same or equivalent folder name.",))
    existing = occupied.get(key, [])
    if row.status == "unchanged" and len(existing) == 1:
        return replace(row, reasons=("The track is already at its proposed destination.", *row.reasons))
    reasons = []
    alias = None if existing or missing_parent else _native_occupant(root, row.destination, native)
    if alias == "unsafe" or any(entry.blocked for entry in existing):
        return replace(row, status="needs_review", reasons=("The proposed destination is linked or unavailable.",))
    if existing or alias == "alias":
        reasons.append("An existing item occupies the proposed destination, including equivalent spellings.")
    if all_destinations[key] > 1:
        reasons.append("Multiple tracks propose the same destination.")
    return replace(row, status="collision", reasons=tuple(reasons)) if reasons else row
