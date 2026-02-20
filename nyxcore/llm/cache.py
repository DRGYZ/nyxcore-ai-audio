from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from nyxcore.llm.models import JudgeResult
from sqlmodel import Field, SQLModel, Session, create_engine, select


class JudgeCacheEntry(SQLModel, table=True):
    __tablename__ = "judge_cache"

    path: str = Field(primary_key=True)
    file_size_bytes: int = Field(primary_key=True)
    mtime_iso: str = Field(primary_key=True)
    model: str = Field(primary_key=True)
    prompt_version: str = Field(primary_key=True)
    provider: str
    tags_json: str
    genre_top: Optional[str] = None
    confidence: Optional[float] = None
    reason: str
    created_at_iso: str
    errors_json: str
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None


class CacheRepo:
    def __init__(self, engine) -> None:
        self.engine = engine

    def get(
        self,
        *,
        path: str,
        file_size_bytes: int,
        mtime_iso: str,
        model: str,
        prompt_version: str,
    ) -> JudgeCacheEntry | None:
        with Session(self.engine) as session:
            stmt = select(JudgeCacheEntry).where(
                JudgeCacheEntry.path == path,
                JudgeCacheEntry.file_size_bytes == file_size_bytes,
                JudgeCacheEntry.mtime_iso == mtime_iso,
                JudgeCacheEntry.model == model,
                JudgeCacheEntry.prompt_version == prompt_version,
            )
            return session.exec(stmt).first()

    def put(self, entry: JudgeCacheEntry) -> None:
        with Session(self.engine) as session:
            stmt = select(JudgeCacheEntry).where(
                JudgeCacheEntry.path == entry.path,
                JudgeCacheEntry.file_size_bytes == entry.file_size_bytes,
                JudgeCacheEntry.mtime_iso == entry.mtime_iso,
                JudgeCacheEntry.model == entry.model,
                JudgeCacheEntry.prompt_version == entry.prompt_version,
            )
            existing = session.exec(stmt).first()
            if existing is None:
                session.add(entry)
            else:
                existing.provider = entry.provider
                existing.tags_json = entry.tags_json
                existing.genre_top = entry.genre_top
                existing.confidence = entry.confidence
                existing.reason = entry.reason
                existing.created_at_iso = entry.created_at_iso
                existing.errors_json = entry.errors_json
                existing.usage_prompt_tokens = entry.usage_prompt_tokens
                existing.usage_completion_tokens = entry.usage_completion_tokens
                existing.usage_total_tokens = entry.usage_total_tokens
                session.add(existing)
            session.commit()


class JudgeCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(self.engine, tables=[JudgeCacheEntry.__table__])
        self.repo = CacheRepo(self.engine)

    def close(self) -> None:
        with self._lock:
            self.engine.dispose()

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
            row = self.repo.get(
                path=path,
                file_size_bytes=file_size_bytes,
                mtime_iso=mtime_iso,
                model=model,
                prompt_version=prompt_version,
            )
        if row is None:
            return None
        return JudgeResult(
            tags=list(json.loads(row.tags_json)),
            genre_top=row.genre_top,
            confidence=None if row.confidence is None else float(row.confidence),
            reason=row.reason,
            provider=row.provider,
            model=model,
            prompt_version=prompt_version,
            created_at_iso=row.created_at_iso,
            errors=list(json.loads(row.errors_json)),
            usage_prompt_tokens=row.usage_prompt_tokens,
            usage_completion_tokens=row.usage_completion_tokens,
            usage_total_tokens=row.usage_total_tokens,
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
        entry = JudgeCacheEntry(
            path=path,
            file_size_bytes=file_size_bytes,
            mtime_iso=mtime_iso,
            model=model,
            prompt_version=prompt_version,
            provider=result.provider,
            tags_json=json.dumps(result.tags, ensure_ascii=False),
            genre_top=result.genre_top,
            confidence=result.confidence,
            reason=result.reason,
            created_at_iso=result.created_at_iso,
            errors_json=json.dumps(result.errors, ensure_ascii=False),
            usage_prompt_tokens=result.usage_prompt_tokens,
            usage_completion_tokens=result.usage_completion_tokens,
            usage_total_tokens=result.usage_total_tokens,
        )
        with self._lock:
            self.repo.put(entry)
