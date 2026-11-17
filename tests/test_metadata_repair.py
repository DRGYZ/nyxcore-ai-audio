from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mutagen import File as MutagenFile
from mutagen.id3 import COMM, ID3, TCON, TPE1, TXXX

from nyxcore.action_plan import service
from nyxcore.action_plan.lock import LibraryMutationLock
from nyxcore.webapi.app import create_app


@pytest.fixture
def library(tmp_path):
    music = tmp_path / "Música 音楽"
    out = tmp_path / "workspace"
    music.mkdir()
    return music, out


def synth(music: Path, name: str, **tags: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for disposable format fixtures")
    path = music / name
    command = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.16"]
    if path.suffix == ".mp3":
        command.extend(["-id3v2_version", "3"])
    for field, val in tags.items():
        command.extend(["-metadata", f"{field}={val}"])
    subprocess.run([*command, str(path)], check=True, capture_output=True, timeout=20)
    return path


def audio_hash(path):
    return subprocess.check_output([
        shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
    ], timeout=15)


def client_for(library, **options):
    return TestClient(create_app(music_root=library[0], workspace_root=library[1], **options))


def inspection(client, path):
    response = client.get("/api/metadata/track", params={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def entry(track, **values):
    return {**{key: track[key] for key in ("path", "inspection", "generation")}, "values": values}


def preview(client, entries):
    response = client.post("/api/metadata/plan", json={"entries": entries})
    assert response.status_code == 200, response.text
    return {"entries": entries, "preview": response.json()["preview"], "confirm": True}, response.json()


def test_disposable_batch_applies_verifies_and_undoes_all_three_formats(library):
    music, out = library
    paths = [
        synth(music, "Årtist - Aurora.mp3", genre="Ambient", title="Track 01"),
        synth(music, "Årtist - Aurora.flac", genre="Ambient", title="Aurora"),
        synth(music, "Årtist - Aurora.m4a", genre="Ambient", album="Original"),
    ]
    original_mp3 = ID3(paths[0])
    original_mp3.add(COMM(encoding=1, lang="eng", desc="", text=["Retain this text"]))
    original_mp3.save(paths[0], v2_version=3)
    before = {path: path.read_bytes() for path in paths}
    packets = {path: audio_hash(path) for path in paths}
    client = client_for(library)
    tracks = [inspection(client, path) for path in paths]
    assert all(track["supported"] for track in tracks)
    assert "placeholder_metadata" in tracks[0]["issues"]
    assert tracks[0]["suggestions"]["title"]["source"] == "filename"
    assert tracks[0]["suggestions"]["album"]["value"] is None
    entries = [
        entry(tracks[0], title="Fréquence", artist="Échos", album="静寂"),
        entry(tracks[1], artist="Échos", album="静寂"),
        entry(tracks[2], title="Dérive", artist="Échos"),
    ]
    request, plan = preview(client, entries)
    assert [file["changes"] for file in plan["files"]] == [item["values"] for item in entries]
    assert plan["files"][0]["before"]["title"] == ["Track 01"]
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 200, response.text
    batch = response.json()["batch_id"]
    operations = response.json()["results"][0]["operation_results"]
    assert [item["status"] for item in operations] == ["ok", "ok", "ok"]
    for path, operation, proposal in zip(paths, operations, entries):
        backup = Path(operation["backup_path"])
        assert backup.is_relative_to(out) and backup.read_bytes() == before[path]
        assert audio_hash(path) == packets[path]
        tags = MutagenFile(path, easy=True).tags
        assert tags["genre"] == ["Ambient"]
        for field, value in proposal["values"].items():
            assert tags[field] == [value]
        assert not any(issue in inspection(client, path)["issues"] for issue in ("missing_title", "missing_artist", "missing_album"))
    assert MutagenFile(paths[1], easy=True).tags["title"] == ["Aurora"]
    assert MutagenFile(paths[2], easy=True).tags["album"] == ["Original"]
    raw_mp3 = ID3(paths[0])
    assert raw_mp3.version[:2] == (2, 3)
    assert any(frame.text == ["Retain this text"] for frame in raw_mp3.getall("COMM"))
    assert client.get("/api/metadata/issues", params={"issue": "missing_artist"}).json()["total"] == 0
    ledger = client.get("/api/history").json()["items"][0]
    assert ledger["batch_id"] == batch and ledger["reversible"]
    undo = client.post(f"/api/history/{batch}/undo", json={})
    assert undo.status_code == 200, undo.text
    assert all(path.read_bytes() == before[path] for path in paths)
    assert client.get("/api/metadata/issues", params={"issue": "missing_artist"}).json()["total"] == 3
    assert client.post("/api/metadata/plan/apply", json=request).status_code == 409


def test_id3v23_repair_preserves_unrelated_multiple_values(library):
    path = synth(library[0], "Artist - Dawn.mp3")
    tags = ID3(path, translate=False)
    frames = [
        TPE1(encoding=1, text=["Écho", "第二"]),
        TCON(encoding=1, text=["Ambient", "Electronic"]),
        TXXX(encoding=1, desc="catalog", text=["First", "二"]),
    ]
    for frame in frames:
        tags.add(frame)
    tags.save(path, v2_version=3, v23_sep=None)
    before = path.read_bytes()
    client = client_for(library)
    request, _ = preview(client, [entry(inspection(client, path), album="New album")])
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 200
    assert response.json()["results"][0]["operation_results"][0]["status"] == "ok"
    current = ID3(path, translate=False)
    assert current.version[:2] == (2, 3)
    assert all(current[frame.HashKey].text == frame.text for frame in frames)
    assert current["TALB"].text == ["New album"]
    batch_id = response.json()["batch_id"]
    assert client.post(f"/api/history/{batch_id}/undo", json={}).status_code == 200
    assert path.read_bytes() == before


def test_metadata_reports_unreadable_tags_and_rechecks_write_permissions(library):
    path = synth(library[0], "Artist - Dawn.flac")
    malformed = library[0] / "Broken.flac"
    malformed.write_bytes(b"fLaC" + bytes(49))
    client = client_for(library)
    broken = inspection(client, malformed)
    assert not broken["readable"] and not broken["supported"]
    assert not broken["issues"] and not broken["suggestions"]
    assert "could not be read" in broken["write_blocker"]
    before = path.read_bytes()
    request, _ = preview(client, [entry(inspection(client, path), title="New title")])
    original_mode = path.stat().st_mode
    try:
        path.chmod(stat.S_IREAD)
        if os.access(path, os.W_OK):
            pytest.skip("The effective user can write to a read-only file")
        track = inspection(client, path)
        assert track["readable"] and not track["supported"]
        assert "read-only" in track["write_blocker"]
        response = client.post("/api/metadata/plan/apply", json=request)
        assert response.status_code == 409 and "read-only" in response.json()["detail"]
        assert not (library[1] / "review_history.json").exists()
        assert path.read_bytes() == before
    finally:
        path.chmod(original_mode)


def test_inspection_suggestions_and_unsupported_files(library):
    music, _ = library
    swapped = synth(music, "Artiste - Mélodie.flac", title="Artiste", artist="Mélodie")
    ambiguous = synth(music, "A - B - C.flac")
    unsupported = synth(music, "Read Only.wav")
    formatted = synth(music, "Formatting.flac", title="Safe   Spaces")
    client = client_for(library)
    track = inspection(client, swapped)
    assert track["swap"]
    assert track["suggestions"]["title"]["value"] == "Mélodie"
    assert track["suggestions"]["artist"]["value"] == "Artiste"
    assert track["suggestions"]["title"]["source"] == "swap"
    assert inspection(client, ambiguous)["suggestions"]["title"]["confidence"] == "ambiguous"
    assert inspection(client, formatted)["suggestions"]["title"]["value"] == "Safe Spaces"
    assert inspection(client, formatted)["suggestions"]["title"]["source"] == "formatting"
    only = inspection(client, unsupported)
    assert not only["supported"]
    response = client.post("/api/metadata/plan", json={"entries": [entry(track, title="Mélodie"), entry(only, title="Blocked")]})
    assert response.status_code == 409
    assert "unavailable" in response.json()["detail"]


def test_draft_persistence_refresh_and_review_only_denial(library):
    music, out = library
    path = synth(music, "Composer - Score.flac")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    client = client_for(library, review_only=True)
    track = inspection(client, path)
    old = client.get("/api/metadata/drafts").json()
    selection = [entry(track, artist="Composer", title="Score")]
    draft_response = client.post("/api/metadata/drafts", json={"revision": old["revision"], "entries": selection})
    assert draft_response.status_code == 200, draft_response.text
    assert (out / "metadata_drafts.json").is_file()
    assert client.get("/api/metadata/drafts").json()["entries"] == selection
    assert client.post("/api/metadata/drafts", json={"revision": old["revision"], "entries": []}).status_code == 409
    request, plan = preview(client, selection)
    assert plan["review_only"]
    assert client.post("/api/metadata/plan/apply", json=request).status_code == 403
    assert client.post("/api/metadata/plan/apply", json={}).status_code == 403
    for route, body in [
        ("/api/review/plan/apply", {"plan_report": {}}),
        ("/api/history/any/undo", {}),
        ("/api/history/any/restore", {}),
    ]:
        assert client.post(route, json=body).status_code == 403
    assert client.post("/api/library/refresh").status_code == 200
    assert client.get("/api/metadata/drafts").json()["generation"] != track["generation"]
    assert client.post("/api/metadata/plan", json={"entries": selection}).status_code == 409
    restarted = client_for(library, review_only=True)
    assert restarted.get("/api/metadata/drafts").json()["entries"] == selection
    assert restarted.post("/api/metadata/plan", json={"entries": selection}).status_code == 409
    renewed = [entry(inspection(restarted, path), artist="Composer", title="Score")]
    assert preview(restarted, renewed)[1]["review_only"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_rejects_changed_bytes_replaced_identity_duplicate_and_tampered_values(library):
    music, out = library
    path = synth(music, "Artist - Song.flac", genre="Original")
    client = client_for(library)
    item = entry(inspection(client, path), artist="Replacement")
    assert client.post("/api/metadata/plan", json={"entries": [item, item]}).status_code == 409
    for values in ({"genre": "Invalid field"}, {"artist": ""}, {"artist": "UNKNOWN"}, {"artist": "a" * 513}):
        assert client.post("/api/metadata/plan", json={"entries": [{**item, "values": values}]}).status_code == 422
    request, _ = preview(client, [item])
    with LibraryMutationLock(music):
        held = client.post("/api/metadata/plan/apply", json=request)
        assert held.status_code == 409 and "already held" in held.json()["detail"]
    forged = {**request, "entries": [{**item, "values": {"artist": "Tampered"}}]}
    assert client.post("/api/metadata/plan/apply", json=forged).status_code == 409
    assert client.post("/api/metadata/plan/apply", json={**request, "plan_report": {"plans": []}}).status_code == 422
    replacement = path.with_suffix(".tmp")
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    assert client.post("/api/metadata/plan/apply", json=request).status_code == 409
    request, _ = preview(client, [entry(inspection(client, path), artist="Replacement")])
    external = MutagenFile(path, easy=True)
    external["genre"] = ["External edit"]
    external.save()
    external_bytes = path.read_bytes()
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 409 and "changed" in response.json()["detail"]
    assert path.read_bytes() == external_bytes
    assert not (out / "review_history.json").exists()


def test_development_health_refresh_invalidates_an_approved_preview(library):
    path = synth(library[0], "Artist - Dawn.flac")
    client = client_for(library)
    initial = path.read_bytes()
    request, _ = preview(client, [entry(inspection(client, path), artist="Artist")])
    assert client.get("/api/health", params={"refresh": True}).status_code == 200
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 409 and "analysis changed" in response.json()["detail"]
    assert path.read_bytes() == initial


def test_metadata_scope_excludes_external_traversal_and_links(library):
    music, _ = library
    external = synth(music.parent, "Other.flac")
    client = client_for(library)
    for path in (external, music / ".." / external.name):
        assert client.get("/api/metadata/track", params={"path": str(path)}).status_code == 403
    internal = synth(music, "In.flac")
    alias = music / "Alias.flac"
    os.link(internal, alias)
    assert not inspection(client, internal)["supported"]
    if os.name == "nt":
        junction = music / "Library junction"
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "New-Item -ItemType Junction -Path $env:NYX_TEST_LINK -Target $env:NYX_TEST_TARGET | Out-Null"],
            env={**os.environ, "NYX_TEST_LINK": str(junction), "NYX_TEST_TARGET": str(music.parent)},
            capture_output=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert client.get("/api/metadata/track", params={"path": str(junction / external.name)}).status_code == 403
        return
    try:
        link = music / "Outside.flac"
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Creating test symlinks is unavailable")
    assert client.get("/api/metadata/track", params={"path": str(link)}).status_code == 403


@pytest.mark.parametrize("uncertain", [False, True])
def test_partial_batch_records_skipped_files_and_reports_uncertain_writes(library, monkeypatch, uncertain):
    root, _ = library
    paths = [synth(root, f"Artist - Title {i}.flac") for i in range(3)]
    original = {path: path.read_bytes() for path in paths}
    client = client_for(library)
    request, _ = preview(client, [entry(inspection(client, path), artist="Replacement") for path in paths])
    real_write = service.write_tags

    def fail_second(path, **kwargs):
        if path == paths[1]:
            if uncertain:
                real_write(path, **kwargs)
            raise RuntimeError("Disposable writer failure")
        real_write(path, **kwargs)

    monkeypatch.setattr(service, "write_tags", fail_second)
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 200
    batch_id = response.json()["batch_id"]
    result = response.json()["results"][0]
    assert result["status"] == "partial_failure"
    assert [(operation["path"], operation["status"]) for operation in result["operation_results"]] == [
        (str(paths[0]), "ok"), (str(paths[1]), "error"), (str(paths[2]), "skipped"),
    ]
    assert paths[2].read_bytes() == original[paths[2]]
    history = client.get("/api/history").json()["items"][0]
    assert history["recovery_required"] is uncertain
    undo = client.post(f"/api/history/{batch_id}/undo", json={})
    if uncertain:
        assert "File may have changed" in result["operation_results"][1]["message"]
        assert undo.status_code == 409
        assert paths[1].read_bytes() != original[paths[1]]
    else:
        assert undo.status_code == 200
        assert all(path.read_bytes() == original[path] for path in paths)
