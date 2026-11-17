import wave
from concurrent.futures import Future, ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
import pytest

from nyxcore import report_pipeline
from nyxcore.webapi import analysis as analysis_module

webapi = import_module("nyxcore.webapi.app")


def write_track(path: Path, value: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(value.to_bytes(2, "little") * 800)


def generation(payload: dict) -> str:
    return payload["meta"]["analysis_generation"]


def request_json(client, method, url, **kwargs):
    response = client.request(method, url, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def library(tmp_path, monkeypatch):
    music = tmp_path / "music"
    music.mkdir()
    write_track(music / "original.wav")
    write_track(music / "copy.wav")
    out = tmp_path / "workspace"
    monkeypatch.delenv("NYXCORE_WEB_CONFIG_PATH", raising=False)
    return music, out


def test_cold_readers_share_one_analysis(library, monkeypatch):
    scans = []
    duplicate_passes = []
    entered, waiting, release = Event(), Event(), Event()
    scan = webapi.scan_music_folder
    analyze = report_pipeline.analyze_duplicates

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiting.set()
            return super().result(timeout)

    def held_scan(root):
        scans.append(root)
        entered.set()
        assert release.wait(10)
        return scan(root)

    def counted_analysis(records, **kwargs):
        duplicate_passes.append(len(records))
        return analyze(records, **kwargs)

    monkeypatch.setattr(analysis_module, "Future", ObservedFuture)
    monkeypatch.setattr(webapi, "scan_music_folder", held_scan)
    monkeypatch.setattr(report_pipeline, "analyze_duplicates", counted_analysis)
    music, out = library
    app = webapi.create_app(music_root=music, workspace_root=out, review_only=True)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(request_json, client, "GET", "/api/library")
        try:
            assert entered.wait(10)
            second = executor.submit(request_json, client, "GET", "/api/duplicates")
            assert waiting.wait(10)
        finally:
            release.set()
        shared = first.result(timeout=10)
        duplicate = second.result(timeout=10)
        assert generation(shared) == generation(duplicate)
        assert shared["duplicates"] == duplicate["data"]
        for route in ("/api/health", "/api/review", "/api/search?q=original", "/api/library"):
            assert generation(request_json(client, "GET", route)) == generation(shared)
        assert scans == [music]
        assert duplicate_passes == [2]


def test_refresh_publishes_add_remove_and_changed_content_together(library):
    music, out = library
    app = webapi.create_app(music_root=music, workspace_root=out, review_only=True)
    with TestClient(app) as client:
        old = request_json(client, "GET", "/api/library")
        assert old["duplicates"]["summary"]["exact_group_count"] == 1
        assert old["health"]["overview"]["total_audio_files"] == 2
        write_track(music / "new.wav", 2)
        refreshed = request_json(client, "POST", "/api/library/refresh")
        added = request_json(client, "GET", "/api/library")
        assert generation(added) == refreshed["analysis_generation"] != generation(old)
        assert added["health"]["overview"]["total_audio_files"] == 3

        write_track(music / "copy.wav", 3)
        request_json(client, "POST", "/api/library/refresh")
        changed = request_json(client, "GET", "/api/library")
        assert changed["health"]["overview"]["total_audio_files"] == 3
        assert changed["duplicates"]["summary"]["exact_group_count"] == 0
        assert generation(changed) != generation(added)

        (music / "original.wav").unlink()
        assert generation(request_json(client, "GET", "/api/library")) == generation(changed)
        assert request_json(client, "GET", "/api/search?q=original")["total_matches"] == 1
        request_json(client, "POST", "/api/library/refresh")
        current = request_json(client, "GET", "/api/library")
        assert current["health"]["overview"]["total_audio_files"] == 2
        assert current["duplicates"]["summary"]["exact_group_count"] == 0
        assert request_json(client, "GET", "/api/search?q=original")["total_matches"] == 0
        assert request_json(client, "GET", "/api/search?q=new")["total_matches"] == 1
        assert all(item["item_type"] != "exact_duplicate_group" for item in current["review"]["items"])
        assert generation(current) != generation(changed)


def test_failed_or_inconsistent_refresh_keeps_last_complete_analysis(library, monkeypatch):
    music, out = library
    analyze = analysis_module.build_duplicate_health_reports
    write_track_once = Event()

    def destabilize(root, records, **kwargs):
        result = analyze(root, records, **kwargs)
        if not write_track_once.is_set():
            write_track(root / "arrived-mid-scan.wav", 4)
            write_track_once.set()
        return result

    app = webapi.create_app(music_root=music, workspace_root=out, review_only=True)
    with TestClient(app) as client:
        baseline = request_json(client, "GET", "/api/library")
        with monkeypatch.context() as patch:
            patch.setattr(analysis_module, "build_duplicate_health_reports", destabilize)
            failed = client.post("/api/library/refresh")
            assert failed.status_code == 503
            assert "Refresh Library" in failed.json()["detail"]
            assert client.get("/api/status").status_code == 200
            preserved = request_json(client, "GET", "/api/library")
            assert generation(preserved) == generation(baseline)
            assert preserved["analyzed_at"] == baseline["analyzed_at"]
            assert preserved["health"] == baseline["health"]
        request_json(client, "POST", "/api/library/refresh")
        current = request_json(client, "GET", "/api/library")
        assert current["health"]["overview"]["total_audio_files"] == 3
        assert generation(current) != generation(baseline)


def test_refresh_readers_see_old_result_until_new_generation_is_ready(library, monkeypatch):
    music, out = library
    entered, waiting, release = Event(), Event(), Event()
    build = analysis_module.ReviewOnlyAnalysis._build
    calls = []

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiting.set()
            return super().result(timeout)

    def held_build(self, root, config):
        calls.append(root)
        entered.set()
        assert release.wait(10)
        return build(self, root, config)

    app = webapi.create_app(music_root=music, workspace_root=out, review_only=True)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        baseline = request_json(client, "GET", "/api/library")
        write_track(music / "third.wav", 3)
        monkeypatch.setattr(analysis_module.ReviewOnlyAnalysis, "_build", held_build)
        monkeypatch.setattr(analysis_module, "Future", ObservedFuture)
        first = executor.submit(request_json, client, "POST", "/api/library/refresh")
        try:
            assert entered.wait(10)
            second = executor.submit(request_json, client, "POST", "/api/library/refresh")
            assert waiting.wait(10)
            reading = request_json(client, "GET", "/api/library")
            assert generation(reading) == generation(baseline)
            assert reading["health"]["overview"]["total_audio_files"] == 2
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
        latest = request_json(client, "GET", "/api/library")
        assert latest["health"]["overview"]["total_audio_files"] == 3
        assert calls == [music]


def test_review_state_is_saved_outside_the_analysis_and_reapplied_after_refresh(library):
    music, out = library
    app = webapi.create_app(music_root=music, workspace_root=out, review_only=True)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        baseline = request_json(client, "GET", "/api/library")
        first, second = (item["item_id"] for item in baseline["review"]["items"][:2])
        futures = [
            executor.submit(request_json, client, "POST", "/api/review/state", json={"item_ids": [item], "action": action})
            for item, action in ((first, "resolved"), (second, "ignored"))
        ]
        for future in futures:
            future.result(timeout=10)

        def statuses():
            report = request_json(client, "GET", "/api/review?include_resolved=true&include_ignored=true")
            return {item["item_id"]: item["review_status"] for item in report["data"]["items"]}

        assert statuses()[first] == "resolved"
        assert statuses()[second] == "ignored"
        request_json(client, "POST", "/api/library/refresh")
        assert statuses()[first] == "seen"
        assert statuses()[second] == "ignored"


def test_profile_changes_and_restart_create_new_analyses(library):
    music, out = library

    def make_app():
        return webapi.create_app(music_root=music, workspace_root=out, review_only=True)

    with TestClient(make_app()) as client:
        default = request_json(client, "GET", "/api/library")
        collector = request_json(client, "GET", "/api/library?profile=collector")
        assert collector["meta"]["active_profile"] == "collector"
        assert generation(collector) != generation(default)
        assert generation(request_json(client, "GET", "/api/search?q=copy&profile=collector")) == generation(collector)
    with TestClient(make_app()) as client:
        restarted = request_json(client, "GET", "/api/library")
        assert generation(restarted) not in {generation(default), generation(collector)}
        assert restarted["meta"]["active_profile"] == "default"
    with TestClient(webapi.create_app(music_root=music, workspace_root=out)) as client:
        assert client.post("/api/library/refresh").status_code == 404
        assert request_json(client, "GET", "/api/duplicates")["meta"]["mode"] == "live"
