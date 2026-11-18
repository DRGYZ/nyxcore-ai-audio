import hashlib
import json
import subprocess
from importlib import import_module
from pathlib import Path
from zipfile import ZipFile

import pytest


LICENSE_BYTES = b"Synthetic packaging test bytes. This is not a project license.\n"
LICENSE_SHA256 = hashlib.sha256(LICENSE_BYTES).hexdigest()
COMMIT = "a" * 40


@pytest.fixture
def packaging(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("build_windows_candidate"), import_module("smoke_windows_candidate")


@pytest.mark.parametrize("committed", [False, True])
def test_staging_includes_only_the_optional_license_in_the_selected_snapshot(
    tmp_path, monkeypatch, packaging, committed,
):
    builder, _ = packaging
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(builder, "SOURCE_PATHS", ("README.md",))
    monkeypatch.setattr(builder, "ROOT", checkout)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=checkout)

    git("init", "--quiet")
    (checkout / "README.md").write_bytes(b"Fixture")
    (checkout / "LICENSE").write_bytes(LICENSE_BYTES)
    git("add", "README.md", *(("LICENSE",) if committed else ()))
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Fixture")
    commit = git("rev-parse", "HEAD").decode().strip()

    staged = builder.stage_project("git", tmp_path / "commit", commit, working_tree=False)
    assert ("LICENSE" in staged) == committed
    if committed:
        assert staged["LICENSE"].read_bytes() == LICENSE_BYTES
    working = builder.stage_project("git", tmp_path / "working", commit, working_tree=True)
    assert working["LICENSE"].read_bytes() == LICENSE_BYTES


@pytest.fixture
def archive_pair(tmp_path, packaging):
    builder, verifier = packaging

    def check(*, licensed=True, approved=None, omit=None, replace=None, source_commit=COMMIT,
              source_binary=None, companion_hash=None, expected_commit=COMMIT,
              omit_gpl=False, replace_mutagen=False, omit_native=False):
        license_file = tmp_path / "fixture-license"
        license_file.write_bytes(LICENSE_BYTES)
        incorrect_license = tmp_path / "changed-license"
        incorrect_license.write_bytes(b"Different fixture bytes\n")
        upstream = tmp_path / "upstream-source"
        upstream.write_bytes(b"Source fixture")
        upstream_sha256 = builder.digest(upstream)
        runtime_source = "upstream/native/runtime-fixture.tar.gz"
        python_sources = {
            name: {"file": f"{name}-fixture.tar.gz", "sha256": upstream_sha256, "notices": ["COPYING"]}
            for name in verifier.REQUIRED_SOURCE_DISTRIBUTIONS
        }
        inventory = builder.save_json(tmp_path / "inventory.json", {
            "python": python_sources,
            "pydantic_core_locked_rust_sources": [],
            "runtime": {"source_archives": {"runtime-fixture.tar.gz": {"sha256": upstream_sha256}}},
        })
        gpl_notice = tmp_path / "gpl-fixture"
        gpl_notice.write_bytes(b"GPL notice fixture")
        common = {
            "WINDOWS_DEPENDENCIES.json": inventory,
            **{name: gpl_notice for name in verifier.GPL_NOTICES},
            **{f"licenses/python/{name}/COPYING": gpl_notice for name in python_sources},
        }
        copies = {}
        if licensed:
            copies = {
                section: incorrect_license if section == replace else license_file
                for section in ("binary", "source", "snapshot") if section != omit
            }
        recorded = {path: builder.digest(gpl_notice) for path in verifier.GPL_NOTICES.values()}
        if licensed:
            recorded["LICENSE"] = LICENSE_SHA256
        provenance = {
            "source_state": "commit", "source_commit": COMMIT, "base_commit": None,
            "first_party_license_sha256": LICENSE_SHA256 if licensed else None,
            "source_files_sha256": hashlib.sha256(json.dumps(recorded, sort_keys=True).encode()).hexdigest(),
        }
        package, source = tmp_path / builder.ZIP_NAME, tmp_path / builder.SOURCE_NAME
        source_files = {
            **common,
            "SOURCE_FILES.json": builder.save_json(tmp_path / "SOURCE_FILES.json", recorded),
            **{f"project/{path}": gpl_notice for path in verifier.GPL_NOTICES.values()},
            **{f"upstream/python/{item['file']}": upstream for item in python_sources.values()},
        }
        if not omit_native:
            source_files[runtime_source] = upstream
        if replace_mutagen:
            source_files["upstream/python/mutagen-fixture.tar.gz"] = incorrect_license
        if "source" in copies:
            source_files["LICENSE"] = copies["source"]
        if "snapshot" in copies:
            source_files["project/LICENSE"] = copies["snapshot"]
        builder.write_archive(source, source_files, {
            **provenance, "source_commit": source_commit, "binary_archive": source_binary or package.name,
        })
        provenance["source_companion"] = {
            "file": source.name, "sha256": companion_hash or builder.digest(source),
        }
        binary_files = dict(common)
        if omit_gpl:
            del binary_files["licenses/GPL-3.0.txt"]
        if "binary" in copies:
            binary_files["LICENSE"] = copies["binary"]
        builder.write_archive(package, binary_files, provenance)
        for path in (package, source):
            path.with_name(path.name + ".sha256").write_text(
                f"{builder.digest(path)}  {path.name}\n", encoding="ascii",
            )

        verifier.verify_zip(package)
        with ZipFile(package) as binary:
            manifest = json.loads(binary.read(f"{package.stem}/PREVIEW_BUILD.json"))
            verifier.verify_pair(package, binary, manifest, expected_commit=expected_commit,
                                 approved_license_sha256=approved)

    return check


def test_archive_pair_accepts_exact_approved_license(archive_pair):
    archive_pair(approved=LICENSE_SHA256)


def test_unlicensed_archives_can_only_pass_internal_verification(archive_pair):
    archive_pair(licensed=False)
    with pytest.raises(ValueError, match="approved first-party LICENSE"):
        archive_pair(licensed=False, approved=LICENSE_SHA256)


def test_license_filename_does_not_grant_approval(archive_pair):
    with pytest.raises(ValueError, match="approved first-party LICENSE"):
        archive_pair(approved="0" * 64)


@pytest.mark.parametrize("section", ["binary", "source", "snapshot"])
def test_missing_or_different_license_is_rejected(archive_pair, section):
    with pytest.raises(ValueError, match="Expected exactly one first-party license"):
        archive_pair(approved=LICENSE_SHA256, omit=section)
    with pytest.raises(ValueError, match="First-party license checksum mismatch"):
        archive_pair(approved=LICENSE_SHA256, replace=section)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"companion_hash": "0" * 64}, "belongs to another binary build"),
        ({"source_binary": "different.zip"}, "different binary ZIP"),
        ({"source_commit": "b" * 40}, "provenance differ at source_commit"),
        ({"expected_commit": "b" * 40}, "not built from the expected commit"),
    ],
)
def test_archive_pair_rejects_mismatched_provenance(archive_pair, change, error):
    with pytest.raises(ValueError, match=error):
        archive_pair(approved=LICENSE_SHA256, **change)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"omit_gpl": True}, "Missing GPL notice"),
        ({"replace_mutagen": True}, "Upstream source checksum mismatch: .*mutagen"),
        ({"omit_native": True}, "Missing upstream source: upstream/native"),
    ],
)
def test_archive_pair_requires_verified_source_materials(archive_pair, change, error):
    with pytest.raises(ValueError, match=error):
        archive_pair(approved=LICENSE_SHA256, **change)
