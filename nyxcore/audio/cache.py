from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from nyxcore.audio.models import AnalysisResult


class AnalysisCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_cache (
                path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                mtime_iso TEXT NOT NULL,
                backend TEXT NOT NULL,
                energy_0_10 REAL NOT NULL,
                bpm REAL,
                tags_json TEXT NOT NULL,
                genre_top TEXT,
                created_at_iso TEXT NOT NULL,
                confidence REAL,
                errors_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(path, file_size_bytes, mtime_iso)
            )
            """
        )
        self._ensure_columns()
        self.conn.commit()

    def _ensure_columns(self) -> None:
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(analysis_cache)").fetchall()}
        if "errors_json" not in cols:
            self.conn.execute("ALTER TABLE analysis_cache ADD COLUMN errors_json TEXT NOT NULL DEFAULT '[]'")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get(
        self,
        *,
        path: str,
        file_size_bytes: int,
        mtime_iso: str,
        backend: str | None = None,
    ) -> AnalysisResult | None:
        if backend is None:
            row = self.conn.execute(
                """
                SELECT energy_0_10, bpm, tags_json, genre_top, backend, created_at_iso, confidence, errors_json
                FROM analysis_cache
                WHERE path = ? AND file_size_bytes = ? AND mtime_iso = ?
                """,
                (path, file_size_bytes, mtime_iso),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT energy_0_10, bpm, tags_json, genre_top, backend, created_at_iso, confidence, errors_json
                FROM analysis_cache
                WHERE path = ? AND file_size_bytes = ? AND mtime_iso = ? AND backend = ?
                """,
                (path, file_size_bytes, mtime_iso, backend),
            ).fetchone()
        if row is None:
            return None
        return AnalysisResult(
            energy_0_10=float(row[0]),
            bpm=None if row[1] is None else float(row[1]),
            tags=list(json.loads(row[2])),
            genre_top=row[3],
            backend=row[4],
            created_at_iso=row[5],
            confidence=None if row[6] is None else float(row[6]),
            errors=list(json.loads(row[7])),
        )

    def set(self, *, path: str, file_size_bytes: int, mtime_iso: str, result: AnalysisResult) -> None:
        self.conn.execute(
            """
            INSERT INTO analysis_cache
              (path, file_size_bytes, mtime_iso, backend, energy_0_10, bpm, tags_json, genre_top, created_at_iso, confidence, errors_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path, file_size_bytes, mtime_iso) DO UPDATE SET
              backend=excluded.backend,
              energy_0_10=excluded.energy_0_10,
              bpm=excluded.bpm,
              tags_json=excluded.tags_json,
              genre_top=excluded.genre_top,
              created_at_iso=excluded.created_at_iso,
              confidence=excluded.confidence,
              errors_json=excluded.errors_json
            """,
            (
                path,
                file_size_bytes,
                mtime_iso,
                result.backend,
                result.energy_0_10,
                result.bpm,
                json.dumps(result.tags, ensure_ascii=False),
                result.genre_top,
                result.created_at_iso,
                result.confidence,
                json.dumps(result.errors, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def rows(self) -> list[dict]:
        fetched = self.conn.execute(
            """
            SELECT path, file_size_bytes, mtime_iso, backend, energy_0_10, bpm, tags_json, genre_top, created_at_iso, confidence, errors_json
            FROM analysis_cache
            """
        ).fetchall()
        rows: list[dict] = []
        for row in fetched:
            rows.append(
                {
                    "path": row[0],
                    "file_size_bytes": int(row[1]),
                    "mtime_iso": row[2],
                    "backend": row[3],
                    "energy_0_10": float(row[4]),
                    "bpm": None if row[5] is None else float(row[5]),
                    "tags": list(json.loads(row[6])),
                    "genre_top": row[7],
                    "created_at_iso": row[8],
                    "confidence": None if row[9] is None else float(row[9]),
                    "errors": list(json.loads(row[10])),
                }
            )
        return rows

