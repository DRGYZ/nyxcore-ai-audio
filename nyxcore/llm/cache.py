from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from nyxcore.llm.models import JudgeResult


class JudgeCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judge_cache (
                path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                mtime_iso TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                provider TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                genre_top TEXT,
                confidence REAL,
                reason TEXT NOT NULL,
                created_at_iso TEXT NOT NULL,
                errors_json TEXT NOT NULL,
                usage_prompt_tokens INTEGER,
                usage_completion_tokens INTEGER,
                usage_total_tokens INTEGER,
                PRIMARY KEY(path, file_size_bytes, mtime_iso, model, prompt_version)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def get(
        self,
        *,
        path: str,
        file_size_bytes: int,
        mtime_iso: str,
        model: str,
        prompt_version: str,
    ) -> JudgeResult | None:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT provider, tags_json, genre_top, confidence, reason, created_at_iso, errors_json,
                       usage_prompt_tokens, usage_completion_tokens, usage_total_tokens
                FROM judge_cache
                WHERE path = ? AND file_size_bytes = ? AND mtime_iso = ? AND model = ? AND prompt_version = ?
                """,
                (path, file_size_bytes, mtime_iso, model, prompt_version),
            ).fetchone()
        if row is None:
            return None
        return JudgeResult(
            tags=list(json.loads(row[1])),
            genre_top=row[2],
            confidence=None if row[3] is None else float(row[3]),
            reason=row[4],
            provider=row[0],
            model=model,
            prompt_version=prompt_version,
            created_at_iso=row[5],
            errors=list(json.loads(row[6])),
            usage_prompt_tokens=row[7],
            usage_completion_tokens=row[8],
            usage_total_tokens=row[9],
        )

    def set(
        self,
        *,
        path: str,
        file_size_bytes: int,
        mtime_iso: str,
        model: str,
        prompt_version: str,
        result: JudgeResult,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO judge_cache
                    (path, file_size_bytes, mtime_iso, model, prompt_version, provider,
                     tags_json, genre_top, confidence, reason, created_at_iso, errors_json,
                     usage_prompt_tokens, usage_completion_tokens, usage_total_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path, file_size_bytes, mtime_iso, model, prompt_version) DO UPDATE SET
                    provider=excluded.provider,
                    tags_json=excluded.tags_json,
                    genre_top=excluded.genre_top,
                    confidence=excluded.confidence,
                    reason=excluded.reason,
                    created_at_iso=excluded.created_at_iso,
                    errors_json=excluded.errors_json,
                    usage_prompt_tokens=excluded.usage_prompt_tokens,
                    usage_completion_tokens=excluded.usage_completion_tokens,
                    usage_total_tokens=excluded.usage_total_tokens
                """,
                (
                    path,
                    file_size_bytes,
                    mtime_iso,
                    model,
                    prompt_version,
                    result.provider,
                    json.dumps(result.tags, ensure_ascii=False),
                    result.genre_top,
                    result.confidence,
                    result.reason,
                    result.created_at_iso,
                    json.dumps(result.errors, ensure_ascii=False),
                    result.usage_prompt_tokens,
                    result.usage_completion_tokens,
                    result.usage_total_tokens,
                ),
            )
            self.conn.commit()
