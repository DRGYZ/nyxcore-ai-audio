from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from nyxcore.core.audio_files import iter_audio_files
from nyxcore.core.scanner import scan_audio_files
from nyxcore.core.track import TrackRecord

INCREMENTAL_STATE_SCHEMA_VERSION = 2
INCREMENTAL_PATH_MODE_RELATIVE = "library_relative"


def _normalize_relative_key(value: str) -> str:
    return Path(value).as_posix()


def _key_for_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _resolve_key_path(key: str, root: Path | None) -> str:
    candidate = Path(key)
    if candidate.is_absolute() or root is None:
        return str(candidate)
    return str(root / Path(key))


def _load_state_key(raw_key: str, *, stored_root: Path | None, path_mode: str, fallback_path: str | None = None) -> str:
    candidate = Path(raw_key)
    if path_mode == INCREMENTAL_PATH_MODE_RELATIVE and not candidate.is_absolute():
        return _normalize_relative_key(raw_key)
    stored_path = fallback_path or raw_key
    stored_candidate = Path(stored_path)
    if stored_root is not None and stored_candidate.is_absolute():
        try:
            return stored_candidate.relative_to(stored_root).as_posix()
        except ValueError:
            return str(stored_candidate)
    if candidate.is_absolute():
        if stored_root is not None:
            try:
                return candidate.relative_to(stored_root).as_posix()
            except ValueError:
                return str(candidate)
        return str(candidate)
    return _normalize_relative_key(raw_key)


def _serialize_snapshot(snapshot: "FileSnapshot", *, key: str, path_mode: str) -> dict:
    data = snapshot.to_dict()
    if path_mode == INCREMENTAL_PATH_MODE_RELATIVE:
        data["path"] = key
    return data


def _serialize_record(record: TrackRecord, *, key: str, path_mode: str) -> dict:
    data = record.to_dict()
    if path_mode == INCREMENTAL_PATH_MODE_RELATIVE:
        data["path"] = key
    return data


def _display_changes(root: Path, changes: "ChangeSet") -> "ChangeSet":
    def _paths(values: list[str]) -> list[str]:
        return [_resolve_key_path(value, root) for value in values]

    return ChangeSet(
        added_files=_paths(changes.added_files),
        modified_files=_paths(changes.modified_files),
        removed_files=_paths(changes.removed_files),
        unchanged_files=_paths(changes.unchanged_files),
    )


@dataclass(slots=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path, *, stored_path: str | None = None) -> "FileSnapshot":
        stat = path.stat()
        return cls(path=stored_path or str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))

    @classmethod
    def from_dict(cls, data: dict) -> "FileSnapshot":
        return cls(path=str(data["path"]), size=int(data["size"]), mtime_ns=int(data["mtime_ns"]))


@dataclass(slots=True)
class IncrementalState:
    root: str
    schema_version: int = INCREMENTAL_STATE_SCHEMA_VERSION
    path_mode: str = INCREMENTAL_PATH_MODE_RELATIVE
    files: dict[str, FileSnapshot] = field(default_factory=dict)
    records: dict[str, TrackRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "path_mode": self.path_mode,
            "root": self.root,
            "files": {
                path: _serialize_snapshot(snapshot, key=path, path_mode=self.path_mode)
                for path, snapshot in sorted(self.files.items())
            },
            "records": {
                path: _serialize_record(record, key=path, path_mode=self.path_mode)
                for path, record in sorted(self.records.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict, *, current_root: Path | None = None) -> "IncrementalState":
        stored_root_text = str(data.get("root", ""))
        stored_root = Path(stored_root_text) if stored_root_text else None
        schema_version = int(data.get("schema_version", 1))
        path_mode = str(
            data.get(
                "path_mode",
                INCREMENTAL_PATH_MODE_RELATIVE if schema_version >= INCREMENTAL_STATE_SCHEMA_VERSION else "absolute",
            )
        )
        resolved_root = current_root or stored_root
        files: dict[str, FileSnapshot] = {}
        for raw_key, snapshot_data in data.get("files", {}).items():
            key = _load_state_key(
                str(raw_key),
                stored_root=stored_root,
                path_mode=path_mode,
                fallback_path=None if not isinstance(snapshot_data, dict) else snapshot_data.get("path"),
            )
            snapshot = FileSnapshot.from_dict(snapshot_data)
            snapshot.path = key
            files[key] = snapshot
        records: dict[str, TrackRecord] = {}
        for raw_key, record_data in data.get("records", {}).items():
            key = _load_state_key(
                str(raw_key),
                stored_root=stored_root,
                path_mode=path_mode,
                fallback_path=None if not isinstance(record_data, dict) else record_data.get("path"),
            )
            record = TrackRecord.from_dict(record_data)
            record.path = _resolve_key_path(key, resolved_root)
            records[key] = record
        return cls(
            root=str(resolved_root or stored_root_text),
            schema_version=INCREMENTAL_STATE_SCHEMA_VERSION,
            path_mode=INCREMENTAL_PATH_MODE_RELATIVE,
            files=dict(sorted(files.items())),
            records=dict(sorted(records.items())),
        )


@dataclass(slots=True)
class ChangeSet:
    added_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    unchanged_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RefreshSummary:
    mode: str
    changes: ChangeSet
    rescanned_files: int

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "changes": self.changes.to_dict(),
            "rescanned_files": self.rescanned_files,
        }


@dataclass(slots=True)
class IncrementalRefreshResult:
    state: IncrementalState
    records: list[TrackRecord]
    summary: RefreshSummary


def load_incremental_state(path: Path) -> IncrementalState | None:
    return load_incremental_state_for_root(path, current_root=None)


def load_incremental_state_for_root(path: Path, *, current_root: Path | None) -> IncrementalState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IncrementalState.from_dict(data, current_root=current_root)


def save_incremental_state(path: Path, state: IncrementalState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def build_file_snapshots(root: Path) -> dict[str, FileSnapshot]:
    snapshots: dict[str, FileSnapshot] = {}
    for path in iter_audio_files(root):
        try:
            key = _key_for_path(path, root)
            snapshot = FileSnapshot.from_path(path, stored_path=key)
        except OSError:
            continue
        snapshots[key] = snapshot
    return dict(sorted(snapshots.items()))


def diff_file_snapshots(previous: dict[str, FileSnapshot], current: dict[str, FileSnapshot]) -> ChangeSet:
    previous_paths = set(previous)
    current_paths = set(current)
    added = sorted(current_paths - previous_paths)
    removed = sorted(previous_paths - current_paths)
    unchanged: list[str] = []
    modified: list[str] = []
    for path in sorted(previous_paths & current_paths):
        before = previous[path]
        after = current[path]
        if before.size == after.size and before.mtime_ns == after.mtime_ns:
            unchanged.append(path)
        else:
            modified.append(path)
    return ChangeSet(
        added_files=added,
        modified_files=modified,
        removed_files=removed,
        unchanged_files=unchanged,
    )


def refresh_incremental_state(root: Path, state_path: Path) -> IncrementalRefreshResult:
    previous = load_incremental_state_for_root(state_path, current_root=root)
    current_snapshots = build_file_snapshots(root)

    if previous is None:
        records = scan_audio_files(root / Path(path) for path in current_snapshots)
        state = IncrementalState(
            root=str(root),
            schema_version=INCREMENTAL_STATE_SCHEMA_VERSION,
            path_mode=INCREMENTAL_PATH_MODE_RELATIVE,
            files=current_snapshots,
            records={_key_for_path(Path(record.path), root): record for record in records},
        )
        save_incremental_state(state_path, state)
        display_changes = ChangeSet(
            added_files=sorted(str(root / Path(path)) for path in current_snapshots),
            modified_files=[],
            removed_files=[],
            unchanged_files=[],
        )
        return IncrementalRefreshResult(
            state=state,
            records=sorted(records, key=lambda item: item.path),
            summary=RefreshSummary(
                mode="full",
                changes=display_changes,
                rescanned_files=len(records),
            ),
        )

    changes = diff_file_snapshots(previous.files, current_snapshots)
    rescanned_paths = [root / Path(path) for path in changes.added_files + changes.modified_files]
    rescanned_records = {_key_for_path(Path(record.path), root): record for record in scan_audio_files(rescanned_paths)}

    next_records: dict[str, TrackRecord] = {}
    for path in changes.unchanged_files:
        cached = previous.records.get(path)
        if cached is not None:
            next_records[path] = cached
    for path in changes.added_files + changes.modified_files:
        record = rescanned_records.get(path)
        if record is not None:
            next_records[path] = record

    filtered_snapshots: dict[str, FileSnapshot] = {}
    for path, snapshot in current_snapshots.items():
        if path in next_records:
            filtered_snapshots[path] = snapshot

    state = IncrementalState(
        root=str(root),
        schema_version=INCREMENTAL_STATE_SCHEMA_VERSION,
        path_mode=INCREMENTAL_PATH_MODE_RELATIVE,
        files=filtered_snapshots,
        records=next_records,
    )
    save_incremental_state(state_path, state)
    return IncrementalRefreshResult(
        state=state,
        records=[next_records[path] for path in sorted(next_records)],
        summary=RefreshSummary(
            mode="incremental",
            changes=_display_changes(root, changes),
            rescanned_files=len(rescanned_records),
        ),
    )


def watch_incremental_state(
    root: Path,
    state_path: Path,
    *,
    interval_seconds: float = 2.0,
    max_cycles: int | None = None,
    on_refresh: Callable[[IncrementalRefreshResult], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[IncrementalRefreshResult]:
    results: list[IncrementalRefreshResult] = []
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        result = refresh_incremental_state(root, state_path)
        results.append(result)
        if on_refresh is not None:
            on_refresh(result)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        sleep_fn(interval_seconds)
    return results
