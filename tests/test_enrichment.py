from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from mutagen import File as MutagenFile
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, COMM, ID3, TCON, TPE1, TXXX
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
from PIL import Image

from nyxcore import enrichment
from nyxcore.action_plan.lock import LibraryMutationLock
from nyxcore.enrichment import CAA, MB, MAX_ART_BYTES, ProviderError, PublicResolver, Transport, approved_url
from nyxcore.tagging.writer import front_covers
from nyxcore.webapi.app import create_app

FIXTURE = json.loads((Path(__file__).parent / "fixtures/enrichment-replay.json").read_text(encoding="utf-8"))
FIRST = FIXTURE["recordings"]["recordings"][1]
RELEASE = FIRST["releases"][0]
OTHER = FIRST["releases"][1]


def png(color="royalblue", size=(300, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class Replay:
    """Exercise provider normalization with stable API-shaped response snapshots."""

    def __init__(self):
        self.calls = []
        self.failure = None
        self.original = png()

    async def get(self, url, *, artwork, maximum):
        self.calls.append(url)
        if self.failure:
            raise self.failure
        parsed = urlsplit(url)
        if parsed.hostname == MB:
            query = parse_qs(parsed.query)["query"][0]
            payload = {"recordings": []} if "No match" in query else FIXTURE["recordings"]
            return json.dumps(payload, ensure_ascii=False).encode()
        assert parsed.hostname == CAA
        if parsed.path.endswith("/"):
            return None if str(OTHER["id"]) in parsed.path else json.dumps(FIXTURE["covers"]).encode()
        if parsed.path.endswith("-250.jpg") and "/916201342" in parsed.path:
            return None
        return self.original


@pytest.fixture
def library(tmp_path, monkeypatch):
    root, workspace = tmp_path / "Library 音楽", tmp_path / "Workspace"
    root.mkdir()
    replay = Replay()
    monkeypatch.setattr(enrichment, "TRANSPORT", replay)
    return root, workspace, replay


def app_for(library, *, review_only=False):
    return TestClient(create_app(music_root=library[0], workspace_root=library[1], review_only=review_only))


def search(client, **changes):
    return client.post("/api/metadata/lookup", json={
        "title": "Meridian", "artist": "Écho Ensemble", "album": "Lumière — 夜", "duration_ms": 192000,
        "filename_title": "Méridian", **changes,
    })


def select_cover(client, source=None):
    if source is None:
        candidate = search(client).json()["candidates"][0]
        source = next(item["source"] for item in candidate["releases"] if item["id"] == RELEASE["id"])
    covers = client.post("/api/metadata/covers", json=source)
    assert covers.status_code == 200, covers.text
    choices = covers.json()["covers"]
    assert len(choices) == 2
    assert choices[0]["thumbnail"].startswith("data:image/jpeg;base64,")
    assert choices[1]["thumbnail"] is None
    choice = {key: choices[0][key] for key in ("id", "release_id", "extension", "token")}
    response = client.post("/api/metadata/cover-select", json={"source": source, "choice": choice})
    assert response.status_code == 200, response.text
    return source, response.json()["selection"]


def synth(root, name, *, cover=False):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for disposable audio fixtures")
    path = root / name
    args = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=410:duration=0.18"]
    if path.suffix == ".mp3":
        args += ["-id3v2_version", "3"]
    elif path.suffix == ".m4a":
        chapters = root.parent / "chapters.ffmeta"
        chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=170\ntitle=Chapter A\n", encoding="ascii")
        args += ["-f", "ffmetadata", "-i", str(chapters), "-map", "0:a", "-map_chapters", "1"]
    subprocess.run([*args, str(path)], check=True, capture_output=True, timeout=20)
    if path.suffix == ".mp3":
        tags = ID3(path, translate=False)
        tags.add(TPE1(encoding=1, text=["Original", "第二"]))
        tags.add(TCON(encoding=1, text=["Ambient", "Électro"]))
        tags.add(TXXX(encoding=1, desc="catalog", text=["Preserve", "二"]))
        tags.add(COMM(encoding=1, lang="eng", desc="test", text=["留める"]))
        tags.add(APIC(encoding=1, mime="image/png", type=4, desc="Front cover", data=png("yellow", (2, 2))))
        tags.save(path, v2_version=3, v23_sep=None)
    elif path.suffix == ".flac":
        audio = FLAC(path)
        audio["comment"] = ["Keep me", "第二"]
        image = Picture()
        image.type = 4
        image.mime = "image/png"
        image.data = png("yellow", (2, 2))
        audio.add_picture(image)
        audio.save()
    elif path.suffix == ".m4a":
        audio = MP4(path)
        audio["©cmt"] = ["Keep me"]
        audio["©lyr"] = ["Paroles 二"]
        audio["----:com.apple.iTunes:keep"] = [MP4FreeForm(b"native")]
        if cover:
            audio["covr"] = [MP4Cover(png("orange", (25, 25)), imageformat=MP4Cover.FORMAT_PNG)]
        audio.save()
    return path


def audio_digest(path):
    return subprocess.check_output([
        shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
        "-f", "hash", "-hash", "sha256", "-",
    ], timeout=10)


def proposal(client, path, values, *, source=None, art=None):
    inspection = client.get("/api/metadata/track", params={"path": str(path)}).json()
    return {"path": inspection["path"], "generation": inspection["generation"], "inspection": inspection["inspection"],
            "values": values, "source": source, "artwork": art}


def confirm(client, entries):
    response = client.post("/api/metadata/plan", json={"entries": entries})
    assert response.status_code == 200, response.text
    return {"entries": entries, "preview": response.json()["preview"], "confirm": True}, response.json()


def test_recording_ranking_multiple_releases_and_privacy(library):
    client = app_for(library)
    data = search(client).json()
    assert data["status"] == "ambiguous"
    best = data["candidates"][0]
    assert best["id"] == FIRST["id"] and best["title"] == "Méridian"
    assert best["evidence"] == {"title": "similar", "artist": "agrees", "album": "agrees", "duration": "agrees", "filename": "agrees"}
    assert best["artists"][0]["name"] == "Écho Ensemble"
    assert sum(item["kind"] == "release" for item in best["releases"]) == 2
    assert sum(item["kind"] == "release-group" for item in best["releases"]) == 2
    url = library[2].calls[0]
    assert set(parse_qs(urlsplit(url).query)) == {"query", "fmt", "limit"}
    assert "192000" not in url and "Lumi" not in url
    for title in ("C:\\Users\\private\\audio.mp3", "/home/user/audio.flac", "a" * 64):
        assert search(client, title=title).status_code == 422
    assert len(library[2].calls) == 1
    assert search(client).status_code == 200 and len(library[2].calls) == 1
    assert search(client, title="No match").json() == {"candidates": [], "status": "no_match"}


def test_missing_cover_and_release_choice_receipt(library):
    client = app_for(library)
    result = search(client).json()["candidates"][0]
    source = next(item["source"] for item in result["releases"] if item["id"] == OTHER["id"])
    assert client.post("/api/metadata/covers", json=source).json() == {"covers": []}
    group = next(item["source"] for item in result["releases"] if item["id"] == RELEASE["release-group"]["id"])
    assert select_cover(client, group)[1]["release_id"] == RELEASE["id"]
    assert client.post("/api/metadata/covers", json={**group, "title": "Tampered"}).status_code == 409


def test_mixed_artwork_metadata_batch_and_byte_exact_undo(library):
    root, workspace, _ = library
    paths = [synth(root, "Aria — MP3.mp3"), synth(root, "静寂.flac"), synth(root, "Café.m4a", cover=True)]
    before = {path: path.read_bytes() for path in paths}
    audio = {path: audio_digest(path) for path in paths}
    chapters = [(part.start, part.title) for part in MP4(paths[2]).chapters]
    assert chapters
    client = app_for(library)
    source, new = select_cover(client)
    entries = [
        proposal(client, paths[0], {"album": "Lumière — 夜"}, source=source, art=new),
        proposal(client, paths[1], {}, source=source, art=new),
        proposal(client, paths[2], {"title": "Version choisie"}, source=source, art=new),
    ]
    entries[0]["source_fields"] = ["album"]
    denied = client.post("/api/metadata/plan", json={"entries": entries})
    assert denied.status_code == 409 and "explicitly approve" in denied.json()["detail"].lower()
    entries[-1]["artwork"] = {**new, "replace_existing": True}
    drafts = client.get("/api/metadata/drafts").json()
    saved = client.post("/api/metadata/drafts", json={"revision": drafts["revision"], "entries": entries})
    assert saved.status_code == 200, saved.text
    assert len(client.get("/api/metadata/drafts").json()["entries"]) == 3
    stale_tab = client.post("/api/metadata/drafts", json={"revision": drafts["revision"], "entries": []})
    assert stale_tab.status_code == 409
    request, report = confirm(client, entries)
    assert report["files"][0]["source_fields"] == ["album"]
    assert report["files"][2]["source_fields"] == []
    falsely_attributed = [{**entries[2], "source_fields": ["title"]}]
    assert client.post("/api/metadata/plan", json={"entries": falsely_attributed}).status_code == 422
    assert all(item["artwork"]["digest"] == new["digest"] for item in report["files"])
    assert not report["files"][0]["artwork"]["before_present"] and report["files"][2]["artwork"]["before_present"]
    assert report["files"][2]["artwork"]["before"].startswith("data:image/jpeg;base64,")
    with LibraryMutationLock(root):
        assert client.post("/api/metadata/plan/apply", json=request).status_code == 409
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    operations = result["results"][0]["operation_results"]
    assert [item["status"] for item in operations] == ["ok", "ok", "ok"], result
    for path, operation in zip(paths, operations):
        [embedded] = front_covers(path)
        assert hashlib.sha256(embedded).hexdigest() == new["digest"]
        assert audio_digest(path) == audio[path]
        assert Path(operation["backup_path"]).is_relative_to(workspace)
        assert Path(operation["backup_path"]).read_bytes() == before[path]
    mp3 = ID3(paths[0], translate=False)
    assert mp3.version[:2] == (2, 3) and mp3["TPE1"].text == ["Original", "第二"]
    assert mp3["TCON"].text == ["Ambient", "Électro"] and mp3["TXXX:catalog"].text == ["Preserve", "二"]
    assert any(item.type == 4 and item.data == png("yellow", (2, 2)) for item in mp3.getall("APIC"))
    flac = FLAC(paths[1])
    assert flac["comment"] == ["Keep me", "第二"] and any(item.type == 4 for item in flac.pictures)
    assert MP4(paths[2])["©lyr"] == ["Paroles 二"] and MP4(paths[2])["----:com.apple.iTunes:keep"] == [b"native"]
    assert [(part.start, part.title) for part in MP4(paths[2]).chapters] == chapters
    assert MutagenFile(paths[0], easy=True).tags["album"] == ["Lumière — 夜"]
    assert MutagenFile(paths[2], easy=True).tags["title"] == ["Version choisie"]
    batch = result["batch_id"]
    assert client.post(f"/api/history/{batch}/undo", json={}).status_code == 200
    assert all(path.read_bytes() == before[path] for path in paths)


@pytest.mark.parametrize("damage", ["missing", "modified", "source"])
def test_apply_rejects_modified_or_missing_cache_and_audio(library, damage):
    root, workspace, _ = library
    path = synth(root, "Song.flac")
    client = app_for(library)
    source, cover = select_cover(client)
    entry = proposal(client, path, {}, source=source, art=cover)
    request, _ = confirm(client, [entry])
    forged = {**request, "entries": [{**entry, "source": {**source, "recording_id": "00000000-0000-0000-0000-000000000000"}}]}
    assert client.post("/api/metadata/plan/apply", json=forged).status_code == 409
    cached, = (workspace / "music-info-cache").glob("approved-*.cache")
    if damage == "missing":
        cached.unlink()
    elif damage == "modified":
        cached.write_bytes(b"tampered")
    else:
        audio = FLAC(path)
        audio["title"] = ["Edited outside NyxCore"]
        audio.save()
    original = path.read_bytes()
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 409 and path.read_bytes() == original
    assert not (workspace / "review_history.json").exists()


def test_provider_failure_keeps_drafts_and_review_only_enforcement(library):
    root, workspace, replay = library
    path = synth(root, "Local.flac")
    client = app_for(library, review_only=True)
    source, cover = select_cover(client)
    entry = proposal(client, path, {"title": "Local edit"}, source=source, art=cover)
    past = client.get("/api/metadata/drafts").json()
    assert client.post("/api/metadata/drafts", json={"revision": past["revision"], "entries": [entry]}).status_code == 200
    snapshot = (workspace / "metadata_drafts.json").read_bytes()
    replay.failure = ProviderError("The provider timed out.", 503)
    assert search(client, title="Fresh query").status_code == 503
    assert client.get("/api/health").status_code == 200
    assert (workspace / "metadata_drafts.json").read_bytes() == snapshot
    request, report = confirm(client, [entry])
    assert report["review_only"]
    for route, payload in [
        ("/api/metadata/plan/apply", request), ("/api/review/plan/apply", {"plan_report": {}}),
        ("/api/history/example/restore", {}), ("/api/history/example/undo", {}),
    ]:
        assert client.post(route, json=payload).status_code == 403
    assert front_covers(path) == []


def test_partial_artwork_verification_failure_enters_existing_recovery(library, monkeypatch):
    root, _, _ = library
    first, failed, untouched = [synth(root, f"Part {index}.flac") for index in range(3)]
    original = untouched.read_bytes()
    client = app_for(library)
    source, cover = select_cover(client)
    request, _ = confirm(client, [proposal(client, path, {}, source=source, art=cover)
                                  for path in (first, failed, untouched)])
    from nyxcore.action_plan import service

    verify = service.verify_written_artwork

    def fault(path, artwork):
        if path == failed:
            raise RuntimeError("Injected cover verification failure")
        verify(path, artwork)

    monkeypatch.setattr(service, "verify_written_artwork", fault)
    response = client.post("/api/metadata/plan/apply", json=request)
    assert response.status_code == 200, response.text
    assert [op["status"] for op in response.json()["results"][0]["operation_results"]] == ["ok", "error", "skipped"]
    batch = client.get("/api/history").json()["items"][0]
    assert batch["recovery_required"] and not batch["reversible"]
    assert untouched.read_bytes() == original


@pytest.mark.parametrize("payload", [b"<svg onload='alert(1)' />", b"x" * (MAX_ART_BYTES + 1), png(size=(6500, 3))])
def test_bad_original_image_never_becomes_a_selection(library, payload):
    client = app_for(library)
    library[2].original = payload
    source = next(item["source"] for item in search(client).json()["candidates"][0]["releases"] if item["id"] == RELEASE["id"])
    cover = client.post("/api/metadata/covers", json=source).json()["covers"][0]
    response = client.post("/api/metadata/cover-select", json={
        "source": source, "choice": {key: cover[key] for key in ("id", "release_id", "extension", "token")},
    })
    assert response.status_code == 502
    assert not list((library[1] / "music-info-cache").glob("approved-*.cache"))


@pytest.mark.parametrize("url", [
    "https://127.0.0.1/image", "https://coverartarchive.org.example.com/image",
    "https://coverartarchive.org@127.0.0.1/image", "https://coverartarchive.org:4431/image",
    "file:///etc/passwd", "https://10.0.0.4/image", "https://[::1]/image",
])
def test_host_allowlist_rejects_unapproved_destinations(url):
    with pytest.raises(ProviderError):
        approved_url(url, artwork=True)


def test_known_archive_mirrors_are_upgraded_to_https():
    assert approved_url("http://dn721805.ca.archive.org/0/image.jpg", artwork=True) == "https://dn721805.ca.archive.org/0/image.jpg"
    assert approved_url("https://ia800123.us.archive.org/image.jpg", artwork=True).startswith("https://")


def test_resolver_refuses_private_address_before_connecting(monkeypatch):
    async def run():
        async def dns(*args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", dns)
        with pytest.raises(OSError):
            await PublicResolver().resolve(CAA, 443)
    asyncio.run(run())


class FakeResponse:
    def __init__(self, status, headers):
        self.status, self.headers, self.content_length = status, headers, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *ignored):
        pass


def test_redirect_is_validated_before_following_and_service_backoff_is_preserved():
    class Session:
        def __init__(self, status, headers):
            self.calls = []
            self.response = FakeResponse(status, headers)

        def get(self, url, *, allow_redirects):
            assert not allow_redirects
            self.calls.append(url)
            return self.response

    redirect = Session(307, {"Location": "https://127.0.0.1/private"})
    with pytest.raises(ProviderError):
        asyncio.run(Transport()._request(redirect, "https://coverartarchive.org/release/item", True, 100))
    assert len(redirect.calls) == 1
    limited = Session(503, {"Retry-After": "40"})
    assert asyncio.run(Transport()._request(limited, "https://musicbrainz.org/ws/2/recording", False, 100)) == 40


def test_transport_spacing_bounded_retry_and_timeout(monkeypatch):
    async def run():
        transport = Transport()
        times = []

        async def temporary_failure(*args):
            times.append(time.monotonic())
            return 1 if len(times) == 1 else b"{}"

        monkeypatch.setattr(transport, "_request", temporary_failure)
        assert await transport.get("https://musicbrainz.org/ws/2/recording", artwork=False, maximum=100) == b"{}"
        assert len(times) == 2 and times[1] - times[0] >= 1

        async def blocked(*args):
            times.append(time.monotonic())
            return 40

        transport = Transport()
        monkeypatch.setattr(transport, "_request", blocked)
        for _ in range(2):
            with pytest.raises(ProviderError) as result:
                await transport.get("https://musicbrainz.org/ws/2/recording", artwork=False, maximum=100)
            assert result.value.status == 503
        assert len(times) == 3

        async def timed_out(*args):
            raise TimeoutError

        transport = Transport()
        monkeypatch.setattr(transport, "_request", timed_out)
        with pytest.raises(ProviderError, match="Manual editing is still available"):
            await transport.get("https://musicbrainz.org/ws/2/recording", artwork=False, maximum=100)

    asyncio.run(run())


def test_only_workspace_cache_entries_are_evicted(library, monkeypatch):
    monkeypatch.setattr(enrichment, "CACHE_BYTES", 24)
    monkeypatch.setattr(enrichment, "CACHE_FILES", 2)
    workspace = library[1]
    cache = enrichment.WorkspaceCache(lambda: workspace)
    cache.write("response", "one", b"x" * 12)
    extra = workspace / "music-info-cache" / "unrelated.cache"
    extra.write_bytes(b"outside NyxCore cache")
    cache.write("response", "two", b"y" * 12)
    cache.write("response", "three", b"z" * 17)
    assert cache.read("response", "one", age=60, maximum=30) is None
    assert cache.read("response", "two", age=60, maximum=30) is None
    assert cache.read("response", "three", age=60, maximum=30) == b"z" * 17
    assert extra.read_bytes() == b"outside NyxCore cache"
