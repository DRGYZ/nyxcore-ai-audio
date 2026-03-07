from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nyxcore.config import NyxConfig
from nyxcore.incremental.service import RefreshSummary
from nyxcore.playlist_query.service import PlaylistReport, build_playlist_report
from nyxcore.core.track import TrackRecord


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "playlist"


def _playlist_id(name: str, query: str, *, profile: str, max_tracks: int | None, min_score: float | None) -> str:
    payload = json.dumps(
        {
            "name": name,
            "query": query,
            "profile": profile,
            "max_tracks": max_tracks,
            "min_score": min_score,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(name)}-{digest}"


@dataclass(slots=True)
class SavedPlaylistDefinition:
    playlist_id: str
    name: str
    query: str
    max_tracks: int | None
    min_score: float | None
    profile: str
    created_at: str
    updated_at: str | None = None
    last_refreshed_at: str | None = None
    last_refresh_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SavedPlaylistDefinition":
        return cls(
            playlist_id=str(data["playlist_id"]),
            name=str(data["name"]),
            query=str(data["query"]),
            max_tracks=None if data.get("max_tracks") is None else int(data.get("max_tracks")),
            min_score=None if data.get("min_score") is None else float(data.get("min_score")),
            profile=str(data.get("profile", "default")),
            created_at=str(data["created_at"]),
            updated_at=None if data.get("updated_at") is None else str(data.get("updated_at")),
            last_refreshed_at=None if data.get("last_refreshed_at") is None else str(data.get("last_refreshed_at")),
            last_refresh_summary=dict(data.get("last_refresh_summary", {})),
        )


@dataclass(slots=True)
class SavedPlaylistLatestResult:
    playlist_id: str
    refreshed_at: str
    refresh_mode: str
    active_profile: str
    report: dict
    summary: dict[str, object]
    refresh_diff: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SavedPlaylistLatestResult":
        return cls(
            playlist_id=str(data["playlist_id"]),
            refreshed_at=str(data["refreshed_at"]),
            refresh_mode=str(data.get("refresh_mode", "full")),
            active_profile=str(data.get("active_profile", "default")),
            report=dict(data.get("report", {})),
            summary=dict(data.get("summary", {})),
            refresh_diff=dict(data.get("refresh_diff", {})),
        )


@dataclass(slots=True)
class SavedPlaylistStore:
    schema_version: int = 1
    playlists: dict[str, SavedPlaylistDefinition] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "playlists": {playlist_id: item.to_dict() for playlist_id, item in sorted(self.playlists.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SavedPlaylistStore":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            playlists={
                playlist_id: SavedPlaylistDefinition.from_dict(item)
                for playlist_id, item in data.get("playlists", {}).items()
            },
        )


def _definitions_path(store_root: Path) -> Path:
    return store_root / "saved_playlists.json"


def _playlist_dir(store_root: Path, playlist_id: str) -> Path:
    return store_root / "playlists" / playlist_id


def _latest_result_path(store_root: Path, playlist_id: str) -> Path:
    return _playlist_dir(store_root, playlist_id) / "latest_result.json"


def _latest_tracklist_path(store_root: Path, playlist_id: str) -> Path:
    return _playlist_dir(store_root, playlist_id) / "latest_tracks.json"


def _latest_m3u_path(store_root: Path, playlist_id: str) -> Path:
    return _playlist_dir(store_root, playlist_id) / "latest.m3u"


def load_saved_playlist_store(store_root: Path) -> SavedPlaylistStore:
    path = _definitions_path(store_root)
    if not path.exists():
        return SavedPlaylistStore()
    return SavedPlaylistStore.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_saved_playlist_definition(store_root: Path, store: SavedPlaylistStore) -> None:
    path = _definitions_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def create_saved_playlist_definition(
    *,
    name: str,
    query: str,
    profile: str,
    max_tracks: int | None,
    min_score: float | None,
    now: datetime | None = None,
) -> SavedPlaylistDefinition:
    now = now or datetime.now(tz=UTC)
    return SavedPlaylistDefinition(
        playlist_id=_playlist_id(name, query, profile=profile, max_tracks=max_tracks, min_score=min_score),
        name=name,
        query=query,
        max_tracks=max_tracks,
        min_score=min_score,
        profile=profile,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )


def rename_saved_playlist_definition(
    definition: SavedPlaylistDefinition,
    *,
    name: str,
    now: datetime | None = None,
) -> SavedPlaylistDefinition:
    now = now or datetime.now(tz=UTC)
    definition.name = name
    definition.updated_at = now.isoformat()
    return definition


def edit_saved_playlist_definition(
    definition: SavedPlaylistDefinition,
    *,
    query: str | None = None,
    max_tracks: int | None = None,
    min_score: float | None = None,
    profile: str | None = None,
    clear_max_tracks: bool = False,
    clear_min_score: bool = False,
    now: datetime | None = None,
) -> SavedPlaylistDefinition:
    now = now or datetime.now(tz=UTC)
    if query is not None:
        definition.query = query
    if clear_max_tracks:
        definition.max_tracks = None
    elif max_tracks is not None:
        definition.max_tracks = max_tracks
    if clear_min_score:
        definition.min_score = None
    elif min_score is not None:
        definition.min_score = min_score
    if profile is not None:
        definition.profile = profile
    definition.updated_at = now.isoformat()
    return definition


def delete_saved_playlist_definition(store_root: Path, store: SavedPlaylistStore, playlist_id: str) -> SavedPlaylistDefinition | None:
    definition = store.playlists.pop(playlist_id, None)
    if definition is None:
        return None
    playlist_dir = _playlist_dir(store_root, playlist_id)
    if playlist_dir.exists():
        shutil.rmtree(playlist_dir)
    return definition


def read_saved_playlist_latest_result(store_root: Path, playlist_id: str) -> SavedPlaylistLatestResult | None:
    path = _latest_result_path(store_root, playlist_id)
    if not path.exists():
        return None
    return SavedPlaylistLatestResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_saved_playlist_latest_result(store_root: Path, result: SavedPlaylistLatestResult) -> None:
    path = _latest_result_path(store_root, result.playlist_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _ranked_track_map(payload: dict) -> dict[str, dict]:
    tracks = payload.get("ranked_tracks", [])
    return {str(track.get("path")): dict(track) for track in tracks if track.get("path")}


def _build_refresh_diff(previous: SavedPlaylistLatestResult | None, current_report: PlaylistReport) -> dict[str, object]:
    if previous is None:
        return {
            "has_previous_result": False,
            "tracks_added": [],
            "tracks_removed": [],
            "track_count_delta": current_report.summary.track_count,
            "estimated_duration_delta_seconds": current_report.summary.estimated_total_duration_seconds,
            "rank_changes": [],
        }

    previous_tracks = _ranked_track_map(previous.report)
    current_tracks = {track.path: track for track in current_report.ranked_tracks}
    added = sorted(path for path in current_tracks if path not in previous_tracks)
    removed = sorted(path for path in previous_tracks if path not in current_tracks)
    previous_ranks = {path: index + 1 for index, path in enumerate(previous_tracks)}
    current_ranks = {track.path: index + 1 for index, track in enumerate(current_report.ranked_tracks)}
    rank_changes = []
    for path in sorted(set(previous_ranks).intersection(current_ranks)):
        old_rank = previous_ranks[path]
        new_rank = current_ranks[path]
        if old_rank != new_rank:
            rank_changes.append(
                {
                    "path": path,
                    "old_rank": old_rank,
                    "new_rank": new_rank,
                    "rank_delta": old_rank - new_rank,
                }
            )
    rank_changes.sort(key=lambda item: (-abs(int(item["rank_delta"])), str(item["path"])))
    return {
        "has_previous_result": True,
        "tracks_added": added,
        "tracks_removed": removed,
        "track_count_delta": current_report.summary.track_count - int(previous.summary.get("track_count", 0)),
        "estimated_duration_delta_seconds": (
            current_report.summary.estimated_total_duration_seconds
            - float(previous.summary.get("estimated_total_duration_seconds", 0.0))
        ),
        "rank_changes": rank_changes[:10],
    }


def refresh_saved_playlist(
    store_root: Path,
    definition: SavedPlaylistDefinition,
    *,
    records: list[TrackRecord],
    refresh_summary: RefreshSummary,
    app_config: NyxConfig,
    analysis_cache_path: Path | None,
    profile_override: str | None = None,
    max_tracks_override: int | None = None,
    min_score_override: float | None = None,
    now: datetime | None = None,
) -> SavedPlaylistLatestResult:
    now = now or datetime.now(tz=UTC)
    active_profile = profile_override or definition.profile
    previous = read_saved_playlist_latest_result(store_root, definition.playlist_id)
    report = build_playlist_report(
        records,
        query=definition.query,
        settings=app_config.playlist,
        max_tracks=max_tracks_override if max_tracks_override is not None else definition.max_tracks,
        min_score=min_score_override if min_score_override is not None else definition.min_score,
        analysis_cache_path=analysis_cache_path,
    )
    refresh_diff = _build_refresh_diff(previous, report)
    latest = SavedPlaylistLatestResult(
        playlist_id=definition.playlist_id,
        refreshed_at=now.isoformat(),
        refresh_mode=refresh_summary.mode,
        active_profile=active_profile,
        report=report.to_dict(),
        summary={
            "track_count": report.summary.track_count,
            "estimated_total_duration_seconds": report.summary.estimated_total_duration_seconds,
            "average_bpm": report.summary.average_bpm,
            "average_energy_0_10": report.summary.average_energy_0_10,
            "added_files": len(refresh_summary.changes.added_files),
            "modified_files": len(refresh_summary.changes.modified_files),
            "removed_files": len(refresh_summary.changes.removed_files),
            "unchanged_files": len(refresh_summary.changes.unchanged_files),
        },
        refresh_diff=refresh_diff,
    )
    definition.last_refreshed_at = latest.refreshed_at
    definition.last_refresh_summary = dict(latest.summary)
    definition.updated_at = now.isoformat()
    write_saved_playlist_latest_result(store_root, latest)
    return latest


def export_saved_playlist_m3u(store_root: Path, playlist_id: str, latest: SavedPlaylistLatestResult) -> Path:
    path = _latest_m3u_path(store_root, playlist_id)
    lines = ["#EXTM3U"] + [str(track["path"]) for track in latest.report.get("ranked_tracks", [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_saved_playlist_json(store_root: Path, playlist_id: str, latest: SavedPlaylistLatestResult) -> Path:
    path = _latest_tracklist_path(store_root, playlist_id)
    payload = {
        "playlist_id": playlist_id,
        "refreshed_at": latest.refreshed_at,
        "tracks": latest.report.get("ranked_tracks", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
