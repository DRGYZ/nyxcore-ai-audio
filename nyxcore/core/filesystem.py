from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


FileIdentity = tuple[int, int]


def is_linked_stat(info: os.stat_result) -> bool:
    """Recognize symlinks, Windows junctions, and other reparse points from lstat."""
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def scan_stat_key(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size


def bounded_path(path: Path, roots: tuple[Path, ...], *, label: str, must_exist: bool = False) -> Path:
    resolved = path.resolve(strict=must_exist)
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"{label} is outside the configured roots: {resolved}")
    return resolved


def path_present(path: Path) -> bool:
    """Include symlinks (even dangling ones), directories, and other collisions."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def regular_file_identity(path: Path) -> FileIdentity | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or not info.st_ino:
        return None
    return info.st_dev, info.st_ino


def file_snapshot(path: Path) -> tuple[FileIdentity, str] | None:
    identity = regular_file_identity(path)
    if identity is None:
        return None
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
            raise RuntimeError(f"file changed while opening: {path}")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if regular_file_identity(path) != identity:
        raise RuntimeError(f"file changed while fingerprinting: {path}")
    return identity, digest


def finish_linked_move(
    source: Path, destination: Path, identity: FileIdentity, *, expected_hash: str | None = None,
) -> None:
    if expected_hash is not None and file_snapshot(source) != (identity, expected_hash):
        raise RuntimeError("source fingerprint changed before move completion")
    if (source.resolve() == destination.resolve() or regular_file_identity(source) != identity
            or regular_file_identity(destination) != identity):
        raise RuntimeError("move paths do not identify the same regular file")
    source.unlink()


def move_file_no_replace(
    source: Path, destination: Path, *,
    expected_identity: FileIdentity | None = None, expected_hash: str | None = None,
) -> None:
    """Move one file without ever replacing an existing destination.

    The hard-link operation is the no-clobber commit point on both Windows and
    Unix. NyxCore review mutations stay within one configured library root, so
    crossing filesystems is intentionally rejected instead of falling back to
    a copy that could weaken the guarantee.
    """

    source = Path(source)
    destination = Path(destination)
    identity = regular_file_identity(source)
    if identity is None or (expected_identity is not None and identity != expected_identity):
        raise RuntimeError(f"source is missing or no longer the expected regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise RuntimeError(f"destination already exists: {destination}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"safe no-replace move failed for {source} -> {destination}: {exc}"
        ) from exc

    try:
        finish_linked_move(source, destination, identity, expected_hash=expected_hash)
    except Exception:
        # Preserve both entries if the source changed, and never clean up a
        # destination that has been replaced since the link was created.
        if regular_file_identity(source) == identity and regular_file_identity(destination) == identity:
            destination.unlink(missing_ok=True)
        raise
