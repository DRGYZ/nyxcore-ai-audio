from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from nyxcore.config import NyxConfig
from nyxcore.core.track import TrackRecord
from nyxcore.incremental.service import build_file_snapshots
from nyxcore.report_pipeline import DuplicateHealthReports, build_duplicate_health_reports


@dataclass(frozen=True)
class LibrarySnapshot:
    generation: str
    started_at: datetime
    analyzed_at: datetime
    records: list[TrackRecord]
    reports: DuplicateHealthReports


class ReviewOnlyAnalysis:
    """Keep the last complete result and coordinate one build at a time."""

    def __init__(self, load_records: Callable[[Path], list[TrackRecord]]) -> None:
        self._load_records = load_records
        self._lock = Lock()
        self._current: tuple[tuple[Path, str], LibrarySnapshot] | None = None
        self._pending: tuple[tuple[Path, str], Future[LibrarySnapshot]] | None = None

    def get(self, music: Path, config: NyxConfig, *, refresh: bool = False) -> LibrarySnapshot:
        key = (music, config.model_dump_json())
        while True:
            with self._lock:
                if not refresh and self._current and self._current[0] == key:
                    return self._current[1]
                if self._pending is None:
                    future: Future[LibrarySnapshot] = Future()
                    self._pending = (key, future)
                    break
                pending_key, future = self._pending

            if pending_key == key:
                return future.result()
            # An unrelated profile cannot satisfy this request, even when its build succeeds.
            try:
                future.result()
            except Exception:
                pass

        try:
            result = self._build(music, config)
        except BaseException as exc:
            with self._lock:
                self._pending = None
                future.set_exception(exc)
            raise
        with self._lock:
            self._current = (key, result)
            self._pending = None
            future.set_result(result)
        return result

    def _build(self, music: Path, config: NyxConfig) -> LibrarySnapshot:
        started_at = datetime.now(UTC)
        next(music.iterdir(), None)  # Fail on an unavailable root before scanning.
        before = build_file_snapshots(music)
        records = self._load_records(music)
        reports = build_duplicate_health_reports(music, records, app_config=config)
        next(music.iterdir(), None)
        after = build_file_snapshots(music)
        if before != after or len(records) != len(after):
            raise OSError("Library files changed during analysis.")
        return LibrarySnapshot(
            generation=uuid4().hex,
            started_at=started_at,
            analyzed_at=datetime.now(UTC),
            records=records,
            reports=reports,
        )
