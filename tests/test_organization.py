from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

from nyxcore.config import HealthConfig
from nyxcore.action_plan.ledger import find_batch, load_operation_ledger
from nyxcore.action_plan.lifecycle import inspect_recorded_batch, recover_recorded_batch
from nyxcore.core import filesystem
from nyxcore.core.filesystem import is_linked_stat
from nyxcore.core.scanner import scan_audio_files
from nyxcore.rename import organization
from nyxcore.webapi.app import create_app


@pytest.fixture
def music(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required to create disposable tracks.")
    library = tmp_path / "Music"
    library.mkdir()
    seed = tmp_path / "seed.flac"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.08", str(seed)],
                   check=True, capture_output=True, timeout=15)

    def track(name, **tags):
        path = library / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed, path)
        audio = FLAC(path)
        for key, value in tags.items():
            audio[key] = value if isinstance(value, list) else [value]
        audio.save()
        return path

    return library, track


def _state(root):
    state = {}
    pending = [root]
    while pending:
        path = pending.pop()
        info = path.lstat()
        digest = None
        if not is_linked_stat(info):
            if path.is_dir():
                pending.extend(path.iterdir())
            elif path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state[path.relative_to(root).as_posix()] = (info.st_mode, info.st_mtime_ns, info.st_size, digest)
    return state


def _rows(library, records=()):
    return {row.source: row for row in organization.build_organization_preview(library, HealthConfig(), list(records))}


def test_organization_policy_and_blocked_metadata(music):
    root, track = music
    track("studio.FLAC", title="Glass / Waves", artist="Élodie", album="Echoes...", tracknumber="03/10", discnumber="2/2")
    track("live.flac", title="Glass / Waves (Live at Lyon)", artist="Élodie", album="Echoes...")
    track("remix.flac", title="Glass / Waves", version="Aurora Remix", artist="Élodie", album="Echoes...")
    track("demo.flac", title="Glass / Waves (Demo)", artist="Élodie", album="Echoes...")
    track("symbol.flac", title="🌒", artist="!!!", album="Moon")
    track("distinct-version.flac", title="Aliveness", version="Live", artist="Seaside", album="Hours")
    track("long-order.flac", title="Finale", artist="Seaside", album="Hours", tracknumber="9/120", discnumber="2/10")
    track("Seaside/Hours/01 - Dusk.flac", title="Dusk", artist="Seaside", album="Hours", tracknumber="1")
    track("compilation.flac", title="One", artist="Singer", album="Sampler", albumartist="Various Artists", compilation="1")
    track("guest.flac", title="Two", artist="Guest", album="Sampler", albumartist="Various Artists")
    track("untagged.flac")
    track("missing-album.flac", title="No Album Yet", artist="Élodie")
    track("conflicting.flac", title=["Name One", "Name Two"], artist="Élodie", album="Variants")
    track("compilation-missing.flac", title="Three", artist="Visitor", album="Different Sampler", compilation="1")
    track("mixed-a.flac", title="A", artist="A Band", album="Mix Tape")
    track("mixed-b.flac", title="B", artist="B Band", album="Mix Tape")
    track("reserved.flac", title="Dusk", artist="COM¹", album="Records")
    track("path-tag.flac", title="Dusk", artist="Artist", album="../../somewhere")
    track("long.flac", title="X" * 182, artist="Band", album="Songs")
    track("broken-number.flac", title="Dusk", artist="Band", album="Songs", tracknumber="9/3")
    (root / "not-audio.flac").write_bytes(b"unreadable audio")
    before = _state(root)
    actual = _rows(root)
    assert actual["studio.FLAC"].destination == ("Élodie", "Echoes", "2-03 - Glass Waves.FLAC")
    assert actual["live.flac"].destination[-1] == "Glass Waves (Live at Lyon).flac"
    assert actual["remix.flac"].destination[-1] == "Glass Waves (Aurora Remix).flac"
    assert actual["demo.flac"].destination[-1] == "Glass Waves (Demo).flac"
    assert actual["symbol.flac"].destination == ("!!!", "Moon", "🌒.flac")
    assert actual["distinct-version.flac"].destination[-1] == "Aliveness (Live).flac"
    assert actual["long-order.flac"].destination[-1] == "02-009 - Finale.flac"
    assert actual["Seaside/Hours/01 - Dusk.flac"].status == "unchanged"
    for name in ("compilation.flac", "guest.flac"):
        assert actual[name].destination[:2] == ("Various Artists", "Sampler")
    expectations = {
        "untagged.flac": "title, artist, album", "missing-album.flac": "album.", "conflicting.flac": "Conflicting",
        "compilation-missing.flac": "album artist", "mixed-a.flac": "album artist", "mixed-b.flac": "album artist",
        "reserved.flac": "Windows", "path-tag.flac": "filesystem path", "long.flac": "too long",
        "broken-number.flac": "track number", "not-audio.flac": "could not be read",
    }
    for name, reason in expectations.items():
        assert actual[name].status == "needs_review" and actual[name].destination is None
        assert reason in actual[name].reasons[0]
    assert _state(root) == before


def test_collision_keys_existing_destinations_and_folder_aliases(music):
    root, track = music
    tags = dict(artist="Artist", album="Album")
    track("one.flac", title="Café", **tags)
    track("two.flac", title="CAFE\u0301", **tags)
    track("from-inbox.flac", title="Occupied", **tags)
    track("Artist/Album/Occupied.flac", title="Occupied", **tags)
    # An arbitrary directory can also occupy a destination filename.
    (root / "Artist" / "Album" / "Folder.flac").mkdir()
    track("folder.flac", title="Folder", **tags)
    track("slash-a.flac", title="Song One", artist="A/B", album="Record", albumartist="A/B")
    track("slash-b.flac", title="Song Two", artist="A:B", album="Record", albumartist="A:B")
    (root / "UPPER").mkdir()
    track("case-folder.flac", title="Dusk", artist="Upper", album="Case")
    before = _state(root)
    rows = _rows(root)
    for name in ("one.flac", "two.flac", "from-inbox.flac", "folder.flac", "slash-a.flac", "slash-b.flac", "case-folder.flac"):
        assert rows[name].status == "collision", (name, rows[name])
    assert rows["Artist/Album/Occupied.flac"].status == "unchanged"
    assert _state(root) == before


def test_local_links_are_reported_without_reading_targets(music, tmp_path, monkeypatch):
    root, track = music
    source = track("Source.flac", title="Track", artist="Linked", album="Album")
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copyfile(source, outside / "Outside.flac")
    linked = root / "Linked"
    if os.name == "nt":
        subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(linked), str(outside)],
                       check=True, capture_output=True, timeout=10)
    else:
        linked.symlink_to(outside, target_is_directory=True)
    os.link(source, root / "HardLink.flac")
    track("Second.flac", title="Second", artist="Linked", album="Album")
    scans = []
    def observe(paths):
        paths = list(paths)
        scans.extend(paths)
        return scan_audio_files(paths)
    monkeypatch.setattr(organization, "scan_audio_files", observe)
    before = _state(tmp_path)
    rows = _rows(root)
    assert set(scans) == {root / "Second.flac"}
    assert rows["Linked"].status == "needs_review" and not rows["Linked"].inspectable
    assert "linked" in rows["Second.flac"].reasons[0]
    assert not rows["HardLink.flac"].inspectable
    assert _state(tmp_path) == before


def test_native_mp4_compilation_is_recognized(music):
    root, _ = music
    path = root / "compilation.m4a"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.08", str(path)],
                   check=True, capture_output=True, timeout=15)
    item = MP4(path)
    item["\xa9nam"], item["\xa9ART"], item["\xa9alb"] = ["Finale"], ["Artist"], ["Sampler"]
    item["cpil"] = True
    item.save()
    assert _rows(root)["compilation.m4a"].status == "needs_review"
    item["aART"] = ["Various Artists"]
    item.save()
    assert _rows(root)["compilation.m4a"].destination == ("Various Artists", "Sampler", "Finale.m4a")


def test_pages_are_bounded_and_share_one_conflict_checked_preview(music, tmp_path, monkeypatch):
    root, track = music
    first = track("001.flac", title="One", artist="Artist", album="Record")
    for index in range(2, 55):
        track(f"{index:03d}.flac", title=f"Song {index}", artist="Artist", album="Record")
    track("Last.flac", title="One", artist="Artist", album="Record")
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(music_root=root, workspace_root=workspace, review_only=True))
    monkeypatch.setattr("nyxcore.core.filesystem.file_snapshot", lambda *args: pytest.fail("Unexpected audio hashing"))
    before = _state(tmp_path)
    preview = client.post("/api/organization/preview", json={"music_path": str(tmp_path)}).json()
    assert preview["counts"] == {"ready": 53, "needs_review": 0, "collision": 2, "unchanged": 0}
    endpoint = f"/api/organization/preview/{preview['id']}"
    head = client.get(endpoint).json()
    tail = client.get(endpoint, params={"offset": 50}).json()
    assert head["rows"][0]["status"] == "collision"
    assert len(head["rows"]) == 50 and len(tail["rows"]) == 5 and tail["total"] == 55
    assert client.get(endpoint, params={"view": "folders"}).json()["total"] == 53
    assert not workspace.exists()
    assert client.post(endpoint + "/apply").status_code == 403
    assert _state(tmp_path) == before

    cached = scan_audio_files(sorted(root.glob("*.flac")))
    previous = scan_audio_files
    scanned = []
    def updated(paths):
        batch = list(paths)
        scanned.extend(batch)
        return previous(batch)
    monkeypatch.setattr(organization, "scan_audio_files", updated)
    assert _rows(root, cached)["001.flac"].status == "collision"
    assert not scanned
    new = FLAC(first)
    new["title"] = ["Revised Song"]
    new.save()
    assert _rows(root, cached)["001.flac"].status == "ready"
    assert scanned == [first]
    assert client.post("/api/organization/preview").status_code == 200
    assert client.get(endpoint).status_code == 409


@pytest.fixture
def reviewed(music, tmp_path, monkeypatch):
    root, track = music
    workspace = tmp_path / "Organization Workspace"
    monkeypatch.setenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT", str(root))
    source = track("Inbox/Original.flac", title="First", artist="Artist", album="Record")
    second = track("Inbox/Second.flac", title="Second", artist="Artist", album="Record")
    untouched = track("Inbox/Unselected.flac", title="Third", artist="Artist", album="Record")
    client = TestClient(create_app(music_root=root, workspace_root=workspace))
    return root, workspace, client, source, second, untouched


def _approval(client, *sources):
    preview = client.post("/api/organization/preview").json()
    assert preview["apply_enabled"]
    endpoint = f"/api/organization/preview/{preview['id']}"
    selected = {"paths": list(map(str, sources))}
    review = client.post(endpoint + "/review", json=selected)
    assert review.status_code == 200, review.text
    return endpoint, {**selected, "approval": review.json()["approval"], "confirm": True}, review.json()["moves"]


@pytest.mark.parametrize("count", [1, 2])
def test_reviewed_organization_move_and_exact_history_undo(reviewed, count):
    root, workspace, client, source, second, untouched = reviewed
    original = {path: (filesystem.regular_file_identity(path), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (source, second, untouched)}
    endpoint, approval, moves = _approval(client, *(source, second)[:count])
    assert [item["destination"] for item in moves] == [
        str(root / "Artist" / "Record" / name) for name in ("First.flac", "Second.flac")[:count]
    ]
    assert not (root / "Artist").exists()
    assert client.post(endpoint + "/apply", json={**approval, "paths": [str(untouched)]}).status_code == 409
    assert client.post(endpoint + "/apply", json={**approval, "destination": str(root / "arbitrary")}).status_code == 422
    assert client.post(endpoint + "/apply", json={**approval, "confirm": 1}).status_code == 422
    response = client.post(endpoint + "/apply", json=approval)
    assert response.status_code == 200, response.text
    result = response.json()
    assert [item["status"] for item in result["operations"]] == ["ok"] * count
    for move in moves:
        assert not Path(move["source"]).exists()
        assert filesystem.file_snapshot(Path(move["destination"])) == original[Path(move["source"])]
    assert filesystem.file_snapshot(untouched) == original[untouched]
    if count == 1:
        assert filesystem.file_snapshot(second) == original[second]
    assert client.post(endpoint + "/apply", json=approval).status_code == 409
    ledger = load_operation_ledger(workspace / "review_history.json")
    batch = find_batch(ledger, result["batch_id"])
    assert all(item.move_identity == original[Path(item.original_path)][0] for item in batch.operations)
    reverse = f"/api/history/{batch.batch_id}/undo"
    source.write_bytes(b"new occupant")
    rejected = client.post(reverse, json={"target_path": str(source)})
    assert rejected.status_code == 200 and rejected.json()["changed_operations"][0]["undo_status"] == "error"
    assert source.read_bytes() == b"new occupant"
    source.unlink()
    assert client.post(reverse, json={"alternate_restore_dir": str(workspace)}).status_code == 403
    undone = client.post(reverse, json={})
    assert undone.status_code == 200, undone.text
    assert all(item["undo_status"] == "ok" for item in undone.json()["changed_operations"])
    assert all(filesystem.file_snapshot(path) == identity_hash for path, identity_hash in original.items())
    assert all(not Path(move["destination"]).exists() for move in moves)
    assert (root / "Artist" / "Record").is_dir()


@pytest.mark.parametrize("change", ["source", "occupied", "case", "unicode", "hardlink", "reparse", "new-preview"])
def test_approval_refuses_stale_or_unsafe_selection(reviewed, tmp_path, change):
    root, workspace, client, source, _, _ = reviewed
    if change == "unicode":
        item = FLAC(source)
        item["title"] = ["Café"]
        item.save()
    endpoint, approval, moves = _approval(client, source)
    destination = Path(moves[0]["destination"])
    original = source.read_bytes()
    if change == "source":
        with source.open("ab") as stream:
            stream.write(b"edited after approval")
    elif change in {"occupied", "case", "unicode"}:
        destination.parent.mkdir(parents=True)
        occupied = destination if change == "occupied" else destination.with_name(
            "CAFE\u0301.flac" if change == "unicode" else "FIRST.FLAC"
        )
        occupied.write_bytes(b"new occupant")
    elif change == "hardlink":
        os.link(source, root / "Additional link.flac")
    elif change == "reparse":
        outside = tmp_path / "Outside"
        outside.mkdir()
        if os.name == "nt":
            subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(root / "Artist"), str(outside)],
                           check=True, timeout=10, capture_output=True)
        else:
            (root / "Artist").symlink_to(outside, target_is_directory=True)
    else:
        assert client.post("/api/organization/preview").status_code == 200
    assert client.post(endpoint + "/apply", json=approval).status_code == 409
    assert not (workspace / "review_history.json").exists()
    if change != "source":
        assert source.read_bytes() == original
    if change == "reparse":
        assert not list(outside.iterdir())


@pytest.mark.parametrize("removed_source", [False, True])
def test_mid_move_interruption_is_recorded_and_recovered(reviewed, monkeypatch, removed_source):
    root, workspace, client, source, _, _ = reviewed
    before = filesystem.file_snapshot(source)
    endpoint, approval, moves = _approval(client, source)
    destination = Path(moves[0]["destination"])
    original = filesystem.finish_linked_move
    def interrupt(*args, **kwargs):
        if removed_source:
            original(*args, **kwargs)
        raise OSError("controlled interruption during move completion")
    monkeypatch.setattr(filesystem, "finish_linked_move", interrupt)
    response = client.post(endpoint + "/apply", json=approval)
    assert response.status_code == 200, response.text
    batch_id = response.json()["batch_id"]
    assert response.json()["operations"][0]["status"] == "error"
    assert filesystem.file_snapshot(source) == (None if removed_source else before)
    assert filesystem.file_snapshot(destination) == before
    ledger_path = workspace / "review_history.json"
    batch, assessments = inspect_recorded_batch(ledger_path, batch_id, library_root=root, workspace_root=workspace)
    assert batch.operations[0].journal_state == "applying"
    assert assessments[0].classification == ("completed_unfinalized" if removed_source else "destination_linked_source_present")
    monkeypatch.setattr(filesystem, "finish_linked_move", original)
    monkeypatch.setenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT", str(root.parent))
    with pytest.raises(ValueError, match="disposable development library"):
        recover_recorded_batch(ledger_path=ledger_path, batch_id=batch_id, library_root=root, workspace_root=workspace,
                               action="finalize")
    assert filesystem.file_snapshot(source) == (None if removed_source else before)
    assert filesystem.file_snapshot(destination) == before
    monkeypatch.setenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT", str(root))
    recover_recorded_batch(ledger_path=ledger_path, batch_id=batch_id, library_root=root, workspace_root=workspace,
                           action="finalize")
    assert not source.exists() and filesystem.file_snapshot(destination) == before
    assert client.post(f"/api/history/{batch_id}/undo", json={}).json()["changed_operations"][0]["undo_status"] == "ok"
    assert filesystem.file_snapshot(source) == before and not destination.exists()


def test_partial_batch_and_review_only_boundary(reviewed, monkeypatch):
    root, workspace, client, source, second, third = reviewed
    endpoint, approval, moves = _approval(client, source, second, third)
    review_only = TestClient(create_app(music_root=root, workspace_root=workspace, review_only=True))
    preview = review_only.post("/api/organization/preview").json()
    assert not preview["apply_enabled"]
    review_endpoint = f"/api/organization/preview/{preview['id']}"
    assert review_only.post(review_endpoint + "/apply", json=approval).status_code == 403
    assert review_only.post(review_endpoint + "/review", json={"paths": approval["paths"]}).status_code == 403
    assert review_only.post("/api/history/example/restore", json={}).status_code == 403
    assert review_only.post("/api/history/example/undo", json={}).status_code == 403
    from nyxcore.core.move_scope import MovePaths
    original = MovePaths.link
    def fail_second(paths):
        if paths.source == second:
            raise OSError("controlled link failure")
        original(paths)
    monkeypatch.setattr(MovePaths, "link", fail_second)
    response = client.post(endpoint + "/apply", json=approval)
    assert response.status_code == 200, response.text
    result = response.json()
    assert [op["status"] for op in result["operations"]] == ["ok", "error", "skipped"]
    assert not source.exists() and second.exists() and third.exists()
    assert Path(moves[0]["destination"]).exists()
    assert not any(Path(move["destination"]).exists() for move in moves[1:])
    monkeypatch.delenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT")
    assert client.post(f"/api/history/{result['batch_id']}/undo", json={}).status_code == 403
    monkeypatch.setenv("NYXCORE_WEB_ORGANIZATION_DISPOSABLE_ROOT", str(root))
    assert client.post(f"/api/history/{result['batch_id']}/undo", json={}).status_code == 200
    assert all(path.exists() for path in (source, second, third))


def test_destination_parent_substitution_at_link_boundary(reviewed, tmp_path, monkeypatch):
    from nyxcore.core.move_scope import MovePaths

    root, _, client, source, _, _ = reviewed
    before = filesystem.file_snapshot(source)
    endpoint, approval, moves = _approval(client, source)
    destination = Path(moves[0]["destination"])
    displaced = tmp_path / "displaced"
    external = tmp_path / "external"
    external.mkdir()
    original = MovePaths.link

    def replace_parent(paths):
        if os.name == "nt":
            with pytest.raises(OSError):
                destination.parent.rename(displaced)
        else:
            destination.parent.rename(displaced)
            destination.parent.symlink_to(external, target_is_directory=True)
        original(paths)

    monkeypatch.setattr(MovePaths, "link", replace_parent)
    response = client.post(endpoint + "/apply", json=approval)
    assert response.status_code == 200, response.text
    result = response.json()["operations"][0]
    assert not list(external.iterdir())
    if os.name == "nt":
        assert result["status"] == "ok" and filesystem.file_snapshot(destination) == before, result["message"]
    else:
        assert result["status"] == "error" and filesystem.file_snapshot(source) == before
        assert not list(displaced.iterdir())
