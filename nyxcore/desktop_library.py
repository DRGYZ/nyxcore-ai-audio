"""Desktop library choices and per-user workspace configuration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from nyxcore.core.atomic import atomic_write_text


@dataclass(frozen=True)
class LibraryChoice:
    music: Path | None  # None is an explicitly chosen synthetic demo.
    workspace: Path | None = None


@dataclass(frozen=True)
class LibrarySession:
    music: Path
    workspace: Path
    demo: bool


def overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


class DesktopLibraries:
    def __init__(self, state: Path):
        self.state = state
        self.settings = state / "settings.json"

    def remembered(self) -> LibraryChoice | None:
        try:
            data = json.loads(self.settings.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise ValueError("Saved library settings could not be read. Choose a folder to set them again.") from exc

        if isinstance(data, dict) and type(data.get("version")) is int and data["version"] == 1:
            if data.get("kind") == "demo":
                return LibraryChoice(None)
            path = data.get("path")
            if data.get("kind") == "music" and isinstance(path, str) and Path(path).is_absolute():
                return LibraryChoice(Path(path))
        raise ValueError("Saved library settings could not be read. Choose a folder to set them again.")

    def resolve(self, choice: LibraryChoice) -> LibrarySession:
        from nyxcore.launch import resolve_library_workspace

        try:
            if choice.workspace is not None:
                workspace = choice.workspace
            elif choice.music is None:
                workspace = self.state / "demo-workspace"
            else:
                canonical = os.path.normcase(str(choice.music.resolve()))
                identity = hashlib.sha256(os.fsencode(canonical)).hexdigest()
                workspace = self.state / "libraries" / identity
            music, workspace = resolve_library_workspace(choice.music, workspace)
        except PermissionError as exc:
            raise ValueError("NyxCore cannot read this music folder. Check its permissions or choose another folder.") from exc
        except RuntimeError as exc:
            raise ValueError("NyxCore could not resolve this folder. Choose a directly accessible music folder.") from exc
        if overlaps(music, self.state.resolve()):
            raise ValueError("Choose a music folder that is separate from NyxCore’s application data.")
        return LibrarySession(music, workspace, choice.music is None)

    def remember(self, session: LibrarySession) -> None:
        data = {"version": 1, "kind": "demo" if session.demo else "music"}
        if not session.demo:
            data["path"] = str(session.music)
        atomic_write_text(self.settings, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
