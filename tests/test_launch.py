from __future__ import annotations

import asyncio
import json
import socket
import wave
from importlib import import_module
from threading import Event
from unittest.mock import Mock

import pytest
import uvicorn
from fastapi.testclient import TestClient

from nyxcore import launch


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    frontend = tmp_path / "web/dist"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (assets / "app.js").write_text("console.log('NyxCore')", encoding="utf-8")
    (assets / "app.css").write_text("body {}", encoding="utf-8")
    (frontend / "index.html").write_text(
        '<script type="module" src="/assets/app.js"></script><link rel="stylesheet" href="/assets/app.css">',
        encoding="utf-8",
    )
    monkeypatch.setattr(launch, "SOURCE_ROOT", tmp_path)
    monkeypatch.delenv("NYXCORE_WEB_CONFIG_PATH", raising=False)
    return tmp_path


def test_static_routes_and_api_remain_distinct(checkout):
    music = checkout / "music"
    out = checkout / "out"
    app = launch.create_local_app(frontend_dir=checkout / "web/dist", music_root=music, workspace_root=out)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for route in ("/", "/review?item=example", "/duplicates", "/history", "/search", "/health", "/playlists"):
            response = client.get(route)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "/assets/app.js" in response.text
        assert client.head("/review").status_code == 200
        assert client.get("/assets/app.js").text == "console.log('NyxCore')"
        assert client.get("/assets/app.css").headers["content-type"].startswith("text/css")
        assert client.get("/assets/missing.js").status_code == 404
        for route in ("/api", "/api/missing"):
            response = client.get(route)
            assert response.status_code == 404
            assert response.json()["detail"] == "Not Found"
        assert client.get("/api/review/plan").status_code == 405
        assert client.get("/api/status/", follow_redirects=False).status_code == 307
        assert client.get("/api/status", headers={"Host": "other.example"}).status_code == 400
        status = client.get("/api/status").json()
        assert status["review_only"] is True
        assert status["music_path"] == str(music)
        assert status["out_path"] == str(out)


def test_review_only_guards_direct_requests_without_blocking_triage_and_preview(checkout, monkeypatch):
    music = checkout / "music"
    music.mkdir()
    out = checkout / "out"
    source = music / "source.wav"
    duplicate = music / "copy.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 800)
    original = source.read_bytes()
    duplicate.write_bytes(original)
    app = launch.create_local_app(frontend_dir=checkout / "web/dist", music_root=music, workspace_root=out)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/search", params={"q": "source"}).status_code == 200
        findings = client.get("/api/review").json()["data"]["items"]
        item = next(item for item in findings if item["item_type"] == "exact_duplicate_group")
        triage = client.post("/api/review/state", json={"item_ids": [item["item_id"]], "action": "seen"})
        assert triage.status_code == 200
        assert (out / "review_state.json").is_file()
        preview = client.post("/api/review/plan", json={"item_ids": [item["item_id"]]})
        assert preview.status_code == 200
        report = preview.json()["data"]
        assert any(
            operation["operation_type"] == "quarantine_move"
            for plan in report["plans"] for operation in plan["proposed_operations"]
        )
        assert client.get("/api/history").json()["items"] == []

        blocked = Mock(side_effect=AssertionError("review-only requests must stop before work begins"))
        webapi = import_module("nyxcore.webapi.app")
        monkeypatch.setattr(webapi, "_load_records", blocked)
        monkeypatch.setattr(webapi, "load_operation_ledger", blocked)
        for path, payload in (
            ("/api/review/plan/apply", {"plan_report": report}),
            ("/api/history/example/restore", {}),
            ("/api/history/example/undo", {}),
        ):
            response = client.post(path, json=payload)
            assert response.status_code == 403
            assert "review-only mode" in response.json()["detail"]
        blocked.assert_not_called()

    assert {p.name: p.read_bytes() for p in music.iterdir()} == {source.name: original, duplicate.name: original}
    assert sorted(p.name for p in out.iterdir()) == ["review_state.json"]


@pytest.mark.parametrize("asset", ["index.html", "assets/app.js", "assets/app.css"])
def test_missing_build_fails_before_creating_workspace(checkout, capsys, asset):
    (checkout / "web/dist" / asset).unlink()
    music = checkout / "music"
    music.mkdir()
    out = checkout / "out"
    with pytest.raises(SystemExit) as result:
        launch.main(["--music", str(music), "--out", str(out)])
    assert result.value.code == 2
    assert "Frontend build is missing or incomplete" in capsys.readouterr().err
    assert not out.exists()


def test_preview_fails_clearly_when_a_bundled_asset_is_missing(checkout, capsys):
    assets = ["index.html", "assets/app.js", "assets/app.css", "assets/font.woff2"]
    (checkout / launch.PREVIEW_BUILD).write_text(json.dumps({"frontend_files": assets}), encoding="utf-8")
    font = checkout / "web/dist" / assets[-1]
    font.write_bytes(b"placeholder font")
    music = checkout / "music"
    music.mkdir()
    out = checkout / "out"
    launch.create_local_app(frontend_dir=checkout / "web/dist", music_root=music, workspace_root=out)
    font.unlink()

    with pytest.raises(SystemExit) as result:
        launch.main(["--music", str(music), "--out", str(out)])
    assert result.value.code == 2
    assert "Extract a fresh copy of the preview ZIP." in capsys.readouterr().err
    assert not out.exists()


def test_preview_explains_how_to_restore_missing_demo(checkout, capsys):
    (checkout / launch.PREVIEW_BUILD).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        launch.main(["--demo", "--out", str(checkout / "out")])
    assert result.value.code == 2
    message = capsys.readouterr().err
    assert "Synthetic demo is missing." in message
    assert "Extract a fresh copy of the preview ZIP." in message


def test_launcher_requires_explicit_library_and_valid_configuration(checkout, monkeypatch, capsys):
    music = checkout / "music"
    out = checkout / "out"
    config = checkout / "invalid.yaml"
    config.write_text("profile: unknown", encoding="utf-8")
    monkeypatch.setenv("NYXCORE_WEB_CONFIG_PATH", str(config))
    with pytest.raises(SystemExit):
        launch.main(["--out", str(out)])
    assert "--demo" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        launch.main(["--demo", "--out", str(out)])
    assert "create_demo_library.py" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        launch.main(["--music", str(music), "--out", str(out)])
    assert "Music directory does not exist" in capsys.readouterr().err
    music.mkdir()
    with pytest.raises(SystemExit):
        launch.main(["--music", str(music), "--out", str(music / "work")])
    assert "non-overlapping" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        launch.main(["--music", str(music), "--out", str(out), "--port", "0"])
    assert "--port must be between" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        launch.main(["--music", str(music), "--out", str(out)])
    assert "Invalid NyxCore configuration" in capsys.readouterr().err
    assert not out.exists()


def test_occupied_port_is_left_running(checkout, capsys):
    music = checkout / "music"
    music.mkdir()
    out = checkout / "out"
    with socket.socket() as existing:
        existing.bind((launch.HOST, 0))
        existing.listen()
        address = existing.getsockname()
        with pytest.raises(SystemExit) as result:
            launch.main(["--music", str(music), "--out", str(out), "--port", str(address[1])])
        assert result.value.code == 2
        assert "Try another --port" in capsys.readouterr().err
        with socket.create_connection(address, timeout=3), existing.accept()[0]:
            pass
    assert not out.exists()


@pytest.mark.parametrize(("started", "open_browser"), [(False, True), (True, False), (True, True)])
def test_browser_open_requires_successful_startup(checkout, monkeypatch, started, open_browser):
    app = launch.create_local_app(frontend_dir=checkout / "web/dist", music_root=checkout, workspace_root=checkout)
    server = launch.LocalServer(app, port=8001, music=checkout, out=checkout, open_browser=open_browser)
    browser_called = Event()
    opened_urls = []

    async def startup(instance, sockets=None):
        assert not browser_called.is_set()
        instance.started = started

    def open_url(url):
        opened_urls.append(url)
        browser_called.set()

    monkeypatch.setattr(uvicorn.Server, "startup", startup)
    monkeypatch.setattr(launch, "_open_browser", open_url)
    start_thread = Mock(wraps=launch.Thread)
    monkeypatch.setattr(launch, "Thread", start_thread)
    asyncio.run(server.startup())
    if started and open_browser:
        assert browser_called.wait(timeout=3)
        assert opened_urls == ["http://127.0.0.1:8001/"]
    else:
        start_thread.assert_not_called()
