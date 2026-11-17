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
    assert client.post(endpoint + "/apply").status_code == 404
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
