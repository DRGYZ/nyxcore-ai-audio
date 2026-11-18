import json
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from nyxcore import desktop, launch


def test_desktop_backend_serves_only_the_review_app_and_releases_the_port(tmp_path, monkeypatch):
    frontend = tmp_path / "web/dist"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text('<script src="/assets/app.js"></script>', encoding="utf-8")
    (frontend / "assets/app.js").write_text("", encoding="utf-8")
    music, workspace = tmp_path / "music", tmp_path / "workspace"
    music.mkdir()
    monkeypatch.setattr(launch, "SOURCE_ROOT", tmp_path)
    monkeypatch.setenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT", str(music))
    backend = desktop.DesktopBackend(lambda: desktop.create_desktop_app(music, workspace))
    backend.start()
    try:
        deadline = monotonic() + 5
        while not backend.ready() and not backend.finished.is_set() and monotonic() < deadline:
            sleep(0.02)
        assert backend.ready(), backend.error
        assert backend.origin.startswith("http://127.0.0.1:")
        with urlopen(backend.origin + "/api/status", timeout=2) as response:
            assert json.load(response)["review_only"] is True
        with urlopen(backend.origin + "/organization", timeout=2) as response:
            assert response.status == 200
        for path, body in (("/api/review/plan/apply", {"plan_report": {}}), ("/api/history/test/undo", {})):
            request = Request(
                backend.origin + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
            )
            with pytest.raises(HTTPError) as rejected:
                urlopen(request, timeout=2)
            assert rejected.value.code == 403
    finally:
        backend.close()
    assert backend.finished.is_set() and not backend.thread.is_alive()
    with pytest.raises(URLError):
        urlopen(backend.origin + "/api/status", timeout=2)


def test_desktop_backend_reports_startup_failure_without_a_listener(tmp_path):
    backend = desktop.DesktopBackend(lambda: desktop.create_desktop_app(tmp_path / "missing", tmp_path / "out"))
    backend.start()
    try:
        assert backend.finished.wait(5)
        assert not backend.ready() and backend.origin is None
        assert "Music directory does not exist" in str(backend.error)
    finally:
        backend.close()
    assert not (tmp_path / "out").exists()


def test_desktop_navigation_keeps_the_local_origin_and_external_references_separate():
    origin = "http://127.0.0.1:54000"
    assert desktop.local_navigation("about:blank", None)
    assert desktop.local_navigation(origin + "/organization?desktop=1", origin)
    for url in ("file:///C:/Music", "https://musicbrainz.org", "http://localhost:54000", "http://127.0.0.1:54001"):
        assert not desktop.local_navigation(url, origin)
    assert desktop.external_reference("https://musicbrainz.org/recording/example")
    for url in ("http://musicbrainz.org", "https://musicbrainz.org.example.com", "https://other@musicbrainz.org"):
        assert not desktop.external_reference(url)
