from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from nyxcore.action_plan.service import execute_reviewed_action_plan
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
        self.environment = patch.dict(
            os.environ,
            {
                "NYXCORE_WEB_MUSIC_DIR": str(self.music),
                "NYXCORE_WEB_OUT_DIR": str(self.out),
            },
        )
        self.environment.start()
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.environment.stop()
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

    def test_status_rejects_unconfigured_music_and_output_roots(self) -> None:
        other_music = self.root / "other-music"
        other_out = self.root / "other-out"

        music_response = self.client.get(
            "/api/status",
            params={"music_path": str(other_music), "out_path": str(self.out)},
        )
        out_response = self.client.get(
            "/api/status",
            params={"music_path": str(self.music), "out_path": str(other_out)},
        )

        self.assertEqual(music_response.status_code, 403)
        self.assertIn("server-configured root", music_response.json()["detail"])
        self.assertEqual(out_response.status_code, 403)
        self.assertIn("server-configured root", out_response.json()["detail"])

    def test_status_rejects_unconfigured_config_file(self) -> None:
        custom_config = self.root / "custom.yaml"
        custom_config.write_text("profile: default\n", encoding="utf-8")

        response = self.client.get(
            "/api/status",
            params={
                "music_path": str(self.music),
                "out_path": str(self.out),
                "config_path": str(custom_config),
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("config_path overrides are disabled", response.json()["detail"])

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

    def test_create_and_refresh_playlist_endpoints(self) -> None:
        self._write_file("focus-one.mp3", b"one")

        create_response = self.client.post(
            "/api/playlists",
            json={
                "name": "Focus Set",
                "query": "focus",
                "max_tracks": 10,
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        playlist_id = created["item"]["playlist_id"]
        self.assertEqual(created["item"]["track_count"], 1)
        self.assertEqual(len(created["item"]["latest_tracks"]), 1)
        self.assertTrue(Path(created["m3u_path"]).exists())

        second = self._write_file("focus-two.mp3", b"two")
        refresh_response = self.client.post(
            f"/api/playlists/{playlist_id}/refresh",
            json={
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(refresh_response.status_code, 200)
        refreshed = refresh_response.json()
        self.assertEqual(refreshed["item"]["track_count"], 2)
        self.assertIn(str(second), refreshed["item"]["latest_refresh_diff"]["tracks_added"])

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
        with patch(
            "nyxcore.webapi.app.execute_reviewed_action_plan",
            wraps=execute_reviewed_action_plan,
        ) as mutation_service:
            apply_response = self.client.post(
                "/api/review/plan/apply",
                json={
                    "plan_report": plan_response.json()["data"],
                    "music_path": str(self.music),
                    "out_path": str(self.out),
                },
            )
        self.assertEqual(apply_response.status_code, 200)
        mutation_service.assert_called_once()
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

        history_response = self.client.get(
            "/api/history",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        self.assertEqual(history_response.status_code, 200)
        history_batch = next(item for item in history_response.json()["items"] if item["batch_id"] == batch_id)
        self.assertTrue(any(item["undo_status"] == "ok" for item in history_batch["operations"]))

    def test_history_restore_rejects_paths_outside_configured_roots(self) -> None:
        left = self._write_file("dup-a.mp3", b"dup" * 400)
        self._write_file("dup-b.flac", left.read_bytes())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        exact_item = next(
            item for item in review_response.json()["data"]["items"]
            if item["item_type"] == "exact_duplicate_group"
        )
        plan_response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [exact_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )
        apply_response = self.client.post(
            "/api/review/plan/apply",
            json={
                "plan_report": plan_response.json()["data"],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )
        batch_id = apply_response.json()["batch_id"]
        outside = self.root.parent / "outside-nyxcore-root"

        alternate_response = self.client.post(
            f"/api/history/{batch_id}/restore",
            json={"out_path": str(self.out), "alternate_restore_dir": str(outside)},
        )
        target_response = self.client.post(
            f"/api/history/{batch_id}/restore",
            json={"out_path": str(self.out), "target_path": str(outside / "track.mp3")},
        )

        self.assertEqual(alternate_response.status_code, 403)
        self.assertIn("alternate_restore_dir is outside", alternate_response.json()["detail"])
        self.assertEqual(target_response.status_code, 403)
        self.assertIn("target_path is outside", target_response.json()["detail"])

        ledger_path = self.out / "review_history.json"
        ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger_payload["batches"][0]["operations"][0]["current_path"] = str(outside / "forged.mp3")
        ledger_path.write_text(json.dumps(ledger_payload), encoding="utf-8")
        ledger_response = self.client.post(
            f"/api/history/{batch_id}/restore",
            json={"out_path": str(self.out)},
        )

        self.assertEqual(ledger_response.status_code, 403)
        self.assertIn("history current path is outside", ledger_response.json()["detail"])
        self.assertFalse((outside / "forged.mp3").exists())

    def test_search_endpoint_returns_ranked_unicode_filename_matches(self) -> None:
        self._write_file("снег.mp3", b"one")
        self._write_file("live-снег-version.mp3", b"two")
        self._write_file("sunlight.mp3", b"three")

        response = self.client.get("/api/search", params={"q": "СНЕГ", "limit": 1})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "СНЕГ")
        self.assertEqual(payload["total_matches"], 2)
        self.assertEqual(payload["returned_count"], 1)
        self.assertEqual(payload["items"][0]["filename"], "снег.mp3")
        self.assertIn("filename", payload["items"][0]["match_fields"])

    def test_search_endpoint_validates_query_and_limit(self) -> None:
        short_query = self.client.get("/api/search", params={"q": "x"})
        excessive_limit = self.client.get("/api/search", params={"q": "valid", "limit": 51})

        self.assertEqual(short_query.status_code, 422)
        self.assertEqual(excessive_limit.status_code, 422)

    def test_apply_rejects_client_attempt_to_enable_review_only_plan(self) -> None:
        for index in range(51):
            self._write_file(f"Artist - Song {index:03d}.mp3", f"track-{index}".encode())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        missing_item = next(
            item for item in review_response.json()["data"]["items"] if item["item_type"] == "missing_metadata"
        )
        plan_response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [missing_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )
        plan_payload = plan_response.json()["data"]
        blocked_plan = next(
            plan for plan in plan_payload["plans"] if plan["action_type"] == "metadata_fix_plan"
        )
        self.assertFalse(blocked_plan["apply_supported"])
        blocked_plan["apply_supported"] = True
        blocked_plan["proposed_operations"] = [blocked_plan["proposed_operations"][0]]
        blocked_plan["proposed_operations"][0]["apply_supported"] = True
        plan_payload["plans"] = [blocked_plan]

        apply_response = self.client.post(
            "/api/review/plan/apply",
            json={
                "plan_report": plan_payload,
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(apply_response.status_code, 409)
        self.assertIn("review-only", apply_response.json()["detail"])
        self.assertEqual(len(list(self.music.glob("*.mp3"))), 51)

    def test_apply_rejects_client_attempt_to_change_operation_details(self) -> None:
        left = self._write_file("dup-a.mp3", b"dup" * 400)
        self._write_file("dup-b.flac", left.read_bytes())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        exact_item = next(
            item for item in review_response.json()["data"]["items"]
            if item["item_type"] == "exact_duplicate_group"
        )
        plan_response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [exact_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )
        plan_payload = plan_response.json()["data"]
        quarantine_operation = next(
            operation
            for operation in plan_payload["plans"][0]["proposed_operations"]
            if operation["operation_type"] == "quarantine_move"
        )
        quarantine_operation["destination_path"] = str(self.root / "forged-destination.mp3")

        apply_response = self.client.post(
            "/api/review/plan/apply",
            json={
                "plan_report": plan_payload,
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        self.assertEqual(apply_response.status_code, 409)
        self.assertIn("operation details changed", apply_response.json()["detail"])
        self.assertTrue((self.music / "dup-a.mp3").exists())
        self.assertTrue((self.music / "dup-b.flac").exists())
        self.assertFalse((self.root / "forged-destination.mp3").exists())

    def test_apply_rejects_backup_directory_outside_configured_output(self) -> None:
        left = self._write_file("dup-a.mp3", b"dup" * 400)
        self._write_file("dup-b.flac", left.read_bytes())
        review_response = self.client.get(
            "/api/review",
            params={"music_path": str(self.music), "out_path": str(self.out)},
        )
        exact_item = next(
            item for item in review_response.json()["data"]["items"]
            if item["item_type"] == "exact_duplicate_group"
        )
        plan_response = self.client.post(
            "/api/review/plan",
            json={
                "item_ids": [exact_item["item_id"]],
                "music_path": str(self.music),
                "out_path": str(self.out),
            },
        )

        apply_response = self.client.post(
            "/api/review/plan/apply",
            json={
                "plan_report": plan_response.json()["data"],
                "music_path": str(self.music),
                "out_path": str(self.out),
                "backup_dir": str(self.root / "outside-output"),
            },
        )

        self.assertEqual(apply_response.status_code, 403)
        self.assertIn("backup_dir is outside", apply_response.json()["detail"])
        self.assertTrue((self.music / "dup-a.mp3").exists())
        self.assertTrue((self.music / "dup-b.flac").exists())


if __name__ == "__main__":
    unittest.main()
