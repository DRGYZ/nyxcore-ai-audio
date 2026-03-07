from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from nyxcore.core.audio_files import iter_audio_files
from nyxcore.core.scanner import scan_audio_files
from nyxcore.core.track import TrackRecord


@dataclass(slots=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> "FileSnapshot":
        stat = path.stat()
        return cls(path=str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))

    @classmethod
    def from_dict(cls, data: dict) -> "FileSnapshot":
        return cls(path=str(data["path"]), size=int(data["size"]), mtime_ns=int(data["mtime_ns"]))


@dataclass(slots=True)
class IncrementalState:
    root: str
    files: dict[str, FileSnapshot] = field(default_factory=dict)
    records: dict[str, TrackRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "files": {path: snapshot.to_dict() for path, snapshot in sorted(self.files.items())},
            "records": {path: record.to_dict() for path, record in sorted(self.records.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IncrementalState":
        return cls(
            root=str(data.get("root", "")),
            files={path: FileSnapshot.from_dict(snapshot) for path, snapshot in data.get("files", {}).items()},
            records={path: TrackRecord.from_dict(record) for path, record in data.get("records", {}).items()},
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
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IncrementalState.from_dict(data)


def save_incremental_state(path: Path, state: IncrementalState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def build_file_snapshots(root: Path) -> dict[str, FileSnapshot]:
    snapshots: dict[str, FileSnapshot] = {}
    for path in iter_audio_files(root):
        try:
            snapshot = FileSnapshot.from_path(path)
        except OSError:
            continue
        snapshots[str(path)] = snapshot
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
    previous = load_incremental_state(state_path)
    current_snapshots = build_file_snapshots(root)

    if previous is None:
        records = scan_audio_files(Path(path) for path in current_snapshots)
        state = IncrementalState(
            root=str(root),
            files=current_snapshots,
            records={record.path: record for record in records},
        )
        save_incremental_state(state_path, state)
        return IncrementalRefreshResult(
            state=state,
            records=sorted(records, key=lambda item: item.path),
            summary=RefreshSummary(
                mode="full",
                changes=ChangeSet(added_files=sorted(current_snapshots), modified_files=[], removed_files=[], unchanged_files=[]),
                rescanned_files=len(records),
            ),
        )

    changes = diff_file_snapshots(previous.files, current_snapshots)
    rescanned_paths = [Path(path) for path in changes.added_files + changes.modified_files]
    rescanned_records = {record.path: record for record in scan_audio_files(rescanned_paths)}

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

    state = IncrementalState(root=str(root), files=filtered_snapshots, records=next_records)
    save_incremental_state(state_path, state)
    return IncrementalRefreshResult(
        state=state,
        records=[next_records[path] for path in sorted(next_records)],
        summary=RefreshSummary(mode="incremental", changes=changes, rescanned_files=len(rescanned_records)),
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
