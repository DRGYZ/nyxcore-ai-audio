import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from nyxcore.webapi.app import create_app


def _client(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a.mp3", "b.mp3"):
        (music / name).write_bytes(b"synthetic duplicate fixture" * 100)
    return music, TestClient(create_app())


def _plan(client):
    report = client.get("/api/review").json()["data"]
    item = next(item for item in report["items"] if item["item_type"] == "exact_duplicate_group")
    response = client.post("/api/review/plan", json={"item_ids": [item["item_id"]]})
    assert response.status_code == 200
    return response.json()["data"]


def test_cleanup_rescan_restore_cycle(tmp_path):
    music, client = _client(tmp_path)
    with patch.dict(os.environ, {"NYXCORE_WEB_MUSIC_DIR": str(music), "NYXCORE_WEB_OUT_DIR": str(tmp_path / "out")}):
        plan = _plan(client)
        applied = client.post("/api/review/plan/apply", json={"plan_report": plan})
        assert applied.status_code == 200
        assert applied.json()["results"][0]["status"] == "ok"
        assert client.get("/api/duplicates").json()["data"]["exact_duplicates"] == []
        batch = client.get("/api/history").json()["items"][0]
        assert batch["reversible"] is True
        restored = client.post(f'/api/history/{batch["batch_id"]}/undo', json={})
        assert restored.status_code == 200
        assert all(op["undo_status"] == "ok" for op in restored.json()["changed_operations"])
        assert len(client.get("/api/duplicates").json()["data"]["exact_duplicates"]) == 1
        assert client.get("/api/history").json()["items"][0]["reversible"] is False


def test_changed_duplicate_rejected_before_any_move(tmp_path):
    music, client = _client(tmp_path)
    with patch.dict(os.environ, {"NYXCORE_WEB_MUSIC_DIR": str(music), "NYXCORE_WEB_OUT_DIR": str(tmp_path / "out")}):
        plan = _plan(client)
        (music / "a.mp3").write_bytes(b"new unique content")
        response = client.post("/api/review/plan/apply", json={"plan_report": plan})
        assert response.status_code == 409
        assert "current review output" in response.json()["detail"]
        assert (music / "a.mp3").read_bytes() == b"new unique content"
        assert (music / "b.mp3").exists()
        assert not (music / ".nyxcore_quarantine").exists()


def test_missing_preferred_rejected(tmp_path):
    from pathlib import Path

    music, client = _client(tmp_path)
    with patch.dict(os.environ, {"NYXCORE_WEB_MUSIC_DIR": str(music), "NYXCORE_WEB_OUT_DIR": str(tmp_path / "out")}):
        plan = _plan(client)
        preferred = next(op for op in plan["plans"][0]["proposed_operations"] if op["operation_type"] == "keep_preferred")
        Path(preferred["path"]).unlink()
        response = client.post("/api/review/plan/apply", json={"plan_report": plan})
        assert response.status_code == 409
        assert "current review output" in response.json()["detail"]
        assert len(list(music.glob("*.mp3"))) == 1
        assert not (music / ".nyxcore_quarantine").exists()


def test_both_copies_changed_since_plan_are_rejected(tmp_path):
    music, client = _client(tmp_path)
    with patch.dict(os.environ, {"NYXCORE_WEB_MUSIC_DIR": str(music), "NYXCORE_WEB_OUT_DIR": str(tmp_path / "out")}):
        plan = _plan(client)
        for path in music.glob("*.mp3"):
            path.write_bytes(b"matching but different from reviewed content")
        response = client.post("/api/review/plan/apply", json={"plan_report": plan})
        assert response.status_code == 409
        assert "source changed since preview" in response.json()["detail"]
        assert len(list(music.glob("*.mp3"))) == 2
        assert not (music / ".nyxcore_quarantine").exists()
