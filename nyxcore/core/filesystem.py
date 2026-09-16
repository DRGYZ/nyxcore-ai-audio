from __future__ import annotations

import os
from pathlib import Path


def move_file_no_replace(source: Path, destination: Path) -> None:
    """Move one file without ever replacing an existing destination.

    The hard-link operation is the no-clobber commit point on both Windows and
    Unix. NyxCore review mutations stay within one configured library root, so
    crossing filesystems is intentionally rejected instead of falling back to
    a copy that could weaken the guarantee.
    """

    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise RuntimeError(f"source file does not exist: {source}")
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
        source.unlink()
    except Exception:
        destination.unlink(missing_ok=True)
        raise
