from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


class MutationLockError(ValueError):
    """Raised when mutation lock acquisition or release is refused."""


@dataclass(slots=True)
class MutationLockRecord:
    token: str
    pid: int
    hostname: str
    library_root: str
    acquired_at: str


class LibraryMutationLock:
    def __init__(self, library_root: Path) -> None:
        self.library_root = library_root.resolve(strict=True)
        self.path = self.library_root / ".nyxcore_mutation.lock"
        self.record = MutationLockRecord(
            token=uuid.uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            library_root=str(self.library_root),
            acquired_at=datetime.now(tz=UTC).isoformat(),
        )
        self._owned = False

    def acquire(self, *, stale_owner_token: str | None = None) -> "LibraryMutationLock":
        if stale_owner_token is not None:
            raise MutationLockError(
                "Automatic stale-lock takeover is disabled. "
                "Retry without a takeover token once the library is unlocked."
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as exc:
            owner = self.read_owner(self.path)
            detail = "unreadable owner"
            if owner is not None:
                detail = f"pid={owner.pid} host={owner.hostname} since={owner.acquired_at} token={owner.token}"
            raise MutationLockError(f"library mutation lock is already held: {detail}") from exc
        try:
            payload = json.dumps(asdict(self.record), ensure_ascii=False).encode("utf-8")
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self._owned = True
        return self

    def release(self) -> None:
        if not self._owned:
            return
        owner = self.read_owner(self.path)
        if owner is None or owner.token != self.record.token:
            raise MutationLockError("mutation lock ownership changed; refusing to remove it")
        self.path.unlink()
        self._owned = False

    def __enter__(self) -> "LibraryMutationLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    @staticmethod
    def read_owner(path: Path) -> MutationLockRecord | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MutationLockRecord(
                token=str(data["token"]),
                pid=int(data["pid"]),
                hostname=str(data["hostname"]),
                library_root=str(data["library_root"]),
                acquired_at=str(data["acquired_at"]),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
