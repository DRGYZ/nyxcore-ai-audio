from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

REVIEW_STATUSES = {"new", "seen", "snoozed", "ignored", "resolved"}


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(slots=True)
class ReviewStateEntry:
    item_id: str
    status: str
    updated_at: str
    snooze_until: str | None = None
    item_type: str | None = None
    summary: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewStateEntry":
        return cls(
            item_id=str(data["item_id"]),
            status=str(data["status"]),
            updated_at=str(data["updated_at"]),
            snooze_until=None if data.get("snooze_until") in {None, ""} else str(data.get("snooze_until")),
            item_type=None if data.get("item_type") in {None, ""} else str(data.get("item_type")),
            summary=None if data.get("summary") in {None, ""} else str(data.get("summary")),
        )


@dataclass(slots=True)
class ReviewStateStore:
    schema_version: int = 1
    items: dict[str, ReviewStateEntry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "items": {item_id: entry.to_dict() for item_id, entry in sorted(self.items.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewStateStore":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            items={item_id: ReviewStateEntry.from_dict(entry) for item_id, entry in data.get("items", {}).items()},
        )


def load_review_state(path: Path) -> ReviewStateStore:
    if not path.exists():
        return ReviewStateStore()
    return ReviewStateStore.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_review_state(path: Path, state: ReviewStateStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def normalize_review_state(
    state: ReviewStateStore,
    *,
    active_item_ids: set[str] | None = None,
    now: datetime | None = None,
) -> bool:
    now = now or utc_now()
    changed = False
    now_iso = now.isoformat()
    for entry in state.items.values():
        if entry.status == "snoozed" and entry.snooze_until is not None:
            snooze_until = _parse_iso(entry.snooze_until)
            if snooze_until is not None and snooze_until <= now:
                entry.status = "seen"
                entry.snooze_until = None
                entry.updated_at = now_iso
                changed = True
        if active_item_ids is not None and entry.status == "resolved" and entry.item_id in active_item_ids:
            entry.status = "seen"
            entry.updated_at = now_iso
            changed = True
    return changed


def apply_review_action(
    state: ReviewStateStore,
    *,
    item_ids: list[str],
    status: str,
    days: int | None = None,
    item_type_by_id: dict[str, str] | None = None,
    summary_by_id: dict[str, str] | None = None,
    now: datetime | None = None,
) -> None:
    if status not in REVIEW_STATUSES - {"new"}:
        raise ValueError(f"Unsupported review action status: {status}")
    now = now or utc_now()
    now_iso = now.isoformat()
    snooze_until = None
    if status == "snoozed":
        if days is None or days < 1:
            raise ValueError("Snoozed review items require days >= 1")
        snooze_until = (now + timedelta(days=days)).isoformat()

    for item_id in item_ids:
        entry = state.items.get(item_id)
        if entry is None:
            entry = ReviewStateEntry(item_id=item_id, status="new", updated_at=now_iso)
            state.items[item_id] = entry
        entry.status = status
        entry.updated_at = now_iso
        entry.snooze_until = snooze_until
        if item_type_by_id is not None and item_id in item_type_by_id:
            entry.item_type = item_type_by_id[item_id]
        if summary_by_id is not None and item_id in summary_by_id:
            entry.summary = summary_by_id[item_id]

