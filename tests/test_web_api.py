from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from nyxcore.saved_playlists.service import (
    create_saved_playlist_definition,
    load_saved_playlist_store,
    save_saved_playlist_definition,
)
from nyxcore.webapi.app import create_app

TEST_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TEST_ROOT / "_runtime_web_api"


def _reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _reset_dir(RUNTIME_ROOT / self._testMethodName)
        self.music = self.root / "music"
        self.out = self.root / "out"
        self.music.mkdir()
        self.out.mkdir()
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_file(self, name: str, content: bytes) -> Path:
        path = self.music / name
        path.write_bytes(content)
        return path

    def test_status_endpoint(self) -> None:
        response = self.client.get(
            "/api/status",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["music_path"], str(self.music.resolve()))
        self.assertEqual(payload["out_path"], str(self.out.resolve()))

    def test_playlists_endpoint_returns_saved_playlist_summaries(self) -> None:
        store_root = self.out / "saved_playlists"
        store = load_saved_playlist_store(store_root)
        definition = create_saved_playlist_definition(
            name="Focus Set",
            query="focus music",
            profile="default",
            max_tracks=10,
            min_score=0.0,
        )
        store.playlists[definition.playlist_id] = definition
        save_saved_playlist_definition(store_root, store)

        response = self.client.get(
            "/api/playlists",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["playlist_id"], definition.playlist_id)
        self.assertEqual(payload["items"][0]["name"], "Focus Set")

    def test_review_state_mutation_endpoint(self) -> None:
        self._write_file("missing.mp3", b"not-audio-but-local")
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        self.assertEqual(review_response.status_code, 200)
        item_id = review_response.json()["data"]["items"][0]["item_id"]

        response = self.client.post(
            "/api/review/state",
            json={
                "item_ids": [item_id],
                "action": "seen",
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "seen")
        self.assertEqual(payload["updated_item_ids"], [item_id])

    def test_review_plan_generation_endpoint(self) -> None:
        left = self._write_file("dup-a.mp3", b"dup" * 400)
        self._write_file("dup-b.flac", left.read_bytes())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        self.assertEqual(review_response.status_code, 200)
        exact_item = next(
            item for item in review_response.json()["data"]["items"] if item["item_type"] == "exact_duplicate_group"
        )

        response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [exact_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["summary"]["generated_plan_count"], 1)
        self.assertEqual(payload["data"]["plans"][0]["action_type"], "exact_duplicate_keep_plan")

    def test_health_endpoint_uses_live_bitrate_bucket_keys(self) -> None:
        self._write_file("sample.mp3", b"not-audio-but-local")

        response = self.client.get(
            "/api/health",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["data"]["quality"]["bitrate_buckets"],
            {
                "unknown": 1,
                "<128k": 0,
                "128k-191k": 0,
                "192k-255k": 0,
                ">=256k": 0,
            },
        )

    def test_history_restore_endpoint(self) -> None:
        left = self._write_file("dup-a.mp3", b"dup" * 400)
        self._write_file("dup-b.flac", left.read_bytes())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        exact_item = next(
            item for item in review_response.json()["data"]["items"] if item["item_type"] == "exact_duplicate_group"
        )
        plan_response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [exact_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )
        self.assertEqual(plan_response.status_code, 200)
        apply_response = self.client.post(
            "/api/review/plan/apply",
            json={
                "plan_report": plan_response.json()["data"],
                "out_path": str(self.out),
            },
        )
        self.assertEqual(apply_response.status_code, 200)
        apply_payload = apply_response.json()
        self.assertEqual(apply_payload["result_count"], 1)
        self.assertEqual(apply_payload["results"][0]["status"], "ok")
        operation_statuses = {item["status"] for item in apply_payload["results"][0]["operation_results"]}
        self.assertTrue(operation_statuses.issubset({"ok", "error", "skipped"}))
        self.assertIn("ok", operation_statuses)
        batch_id = apply_payload["batch_id"]

        restore_response = self.client.post(
            f"/api/history/{batch_id}/restore",
            json={"out_path": str(self.out)},
        )

        self.assertEqual(restore_response.status_code, 200)
        payload = restore_response.json()
        self.assertEqual(payload["batch_id"], batch_id)
        self.assertTrue(any(item["undo_status"] == "ok" for item in payload["changed_operations"]))
        self.assertIn(payload["changed_operations"][0]["status"], {"ok", "error", "skipped"})
        self.assertIn(payload["changed_operations"][0]["undo_status"], {"pending", "ok", "error", "not_supported"})
        self.assertTrue((self.music / "dup-a.mp3").exists())


if __name__ == "__main__":
    unittest.main()
