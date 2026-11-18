import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from nyxcore import launch
from nyxcore.desktop_controls import install_desktop_controls
from nyxcore.desktop_library import DesktopLibraries, LibraryChoice


def test_remembered_library_keeps_the_same_workspace(tmp_path, monkeypatch):
    state = tmp_path / "AppData"
    music = tmp_path / "Music Copy 音楽"
    other = tmp_path / "Other"
    demo = tmp_path / "demo/generated/sample-library"
    for folder in (music, other, demo):
        folder.mkdir(parents=True)
    (demo / "nyxcore_demo_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launch, "SOURCE_ROOT", tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    libraries = DesktopLibraries(state)

    assert libraries.remembered() is None
    first = libraries.resolve(LibraryChoice(music))
    assert first.workspace.parent == state / "libraries"
    first.workspace.mkdir(parents=True)
    review = first.workspace / "review.json"
    review.write_text("staged", encoding="utf-8")
    libraries.remember(first)
    assert json.loads(libraries.settings.read_text(encoding="utf-8")) == {
        "version": 1, "kind": "music", "path": str(music),
    }
    reopened = DesktopLibraries(state)
    assert reopened.resolve(reopened.remembered()) == first
    assert reopened.resolve(LibraryChoice(Path("~") / music.name)) == first
    assert review.read_text(encoding="utf-8") == "staged"
    assert reopened.resolve(LibraryChoice(other)).workspace != first.workspace
    assert reopened.resolve(LibraryChoice(music, tmp_path / "Manual output")).workspace == tmp_path / "Manual output"
    assert not list(music.iterdir())
    with pytest.raises(ValueError, match="separate"):
        reopened.resolve(LibraryChoice(music, music / "workspace"))
    with pytest.raises(ValueError, match="does not exist"):
        reopened.resolve(LibraryChoice(tmp_path / "Disconnected drive"))

    manual_demo = reopened.resolve(LibraryChoice(None))
    assert manual_demo.demo and manual_demo.workspace == state / "demo-workspace"
    reopened.remember(manual_demo)
    assert reopened.remembered() == LibraryChoice(None)


def test_corrupt_or_inaccessible_library_settings_are_recoverable(tmp_path, monkeypatch):
    state = tmp_path / "AppData"
    state.mkdir()
    libraries = DesktopLibraries(state)
    libraries.settings.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Choose a folder"):
        libraries.remembered()
    music = tmp_path / "Inaccessible"
    original = Path.iterdir

    def deny_music(path):
        if path == music:
            raise PermissionError("access denied")
        return original(path)

    music.mkdir()
    monkeypatch.setattr(Path, "iterdir", deny_music)
    with pytest.raises(ValueError, match="permissions"):
        libraries.resolve(LibraryChoice(music))


def test_portable_workspace_cannot_overlap_application_files(tmp_path):
    portable = tmp_path / "Portable NyxCore"
    music = tmp_path / "Music"
    portable.mkdir()
    music.mkdir()
    libraries = DesktopLibraries(tmp_path / "App Data", application=portable)
    with pytest.raises(ValueError, match="outside the extracted"):
        libraries.resolve(LibraryChoice(music, portable / "workspace"))
    with pytest.raises(ValueError, match="outside the extracted"):
        libraries.resolve(LibraryChoice(portable))


def test_desktop_controls_use_only_native_selections_and_the_current_report(tmp_path, monkeypatch):
    state = tmp_path / "AppData"
    portable = tmp_path / "Portable NyxCore"
    music = tmp_path / "Music"
    other = tmp_path / "Different Music"
    for path in (music, other, portable):
        path.mkdir()
    libraries = DesktopLibraries(state, application=portable)
    app = FastAPI()
    app.state.desktop_health_report = lambda: {"total": 3}
    keys = {"X-NyxCore-Window": "owner", "Origin": "http://127.0.0.1:41000"}
    selected = [None, other]
    destination = tmp_path / "health.json"
    destinations = [None, music / "blocked.json", portable / "blocked.json",
                    tmp_path / "missing" / "health.json", destination, destination]
    activated = []
    install_desktop_controls(
        app, libraries=libraries, session=libraries.resolve(LibraryChoice(music)),
        issue=None, retry_choice=None, window_key="owner", origin=lambda: keys["Origin"],
        choose_folder=lambda start: selected.pop(0),
        choose_export=lambda filename: destinations.pop(0),
        activate=lambda session, remember: activated.append((session, remember)) or "http://127.0.0.1:41001",
    )
    with TestClient(app) as client:
        control = "/api/desktop/"
        assert client.get(control + "session").status_code == 403
        assert client.post(control + "choose-folder", headers={**keys, "Origin": "https://example.org"}).status_code == 403
        assert client.post(control + "choose-folder", headers=keys).json() == {"path": None}
        assert client.post(control + "open-folder", headers=keys, json={"path": str(other)}).status_code == 409
        assert client.post(control + "choose-folder", headers=keys, json={"path": str(music)}).json() == {"path": str(other)}
        assert client.post(control + "export-health", headers=keys).json() == {"path": None}
        assert client.post(control + "export-health", headers=keys).status_code == 422
        assert client.post(control + "export-health", headers=keys).status_code == 422
        assert client.post(control + "export-health", headers=keys).status_code == 422
        assert not (tmp_path / "missing").exists()
        response = client.post(control + "export-health", headers=keys, json={"report": {"injected": True}})
        assert response.json() == {"path": str(destination)}
        assert json.loads(destination.read_text(encoding="utf-8")) == {"total": 3}

        def write_failed(*args):
            raise PermissionError("access denied")

        monkeypatch.setattr("nyxcore.desktop_controls.atomic_write_text", write_failed)
        assert client.post(control + "export-health", headers=keys).status_code == 422
        assert json.loads(destination.read_text(encoding="utf-8")) == {"total": 3}
        response = client.post(control + "open-folder", headers=keys, json={"path": str(music)})
        assert response.json() == {"origin": "http://127.0.0.1:41001"}
        assert activated == [(libraries.resolve(LibraryChoice(other)), True)]
        assert client.get("/api/health").status_code == 409
