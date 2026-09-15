from __future__ import annotations

import unittest

from nyxcore.core.track import TrackRecord
from nyxcore.search.service import search_tracks


def _track(path: str, **tags: str | None) -> TrackRecord:
    values: dict[str, str | None] = {
        "title": None,
        "artist": None,
        "album": None,
        "albumartist": None,
        "genre": None,
    }
    values.update(tags)
    return TrackRecord(
        path=path,
        file_size_bytes=123,
        mtime_iso="2026-01-01T00:00:00+00:00",
        tags=values,
        has_cover_art=False,
        duration_seconds=180.0,
    )


class SearchTests(unittest.TestCase):
    def test_matches_unicode_in_filenames_and_tags(self) -> None:
        records = [
            _track("C:/Music/снег_CaF1w01yFNY_140.mp3"),
            _track("C:/Music/arabic.mp3", title="أغاني الليل", artist="ليلى"),
            _track("C:/Music/other.mp3", title="Sunlight"),
        ]

        cyrillic_total, cyrillic = search_tracks(records, "СНЕГ")
        arabic_total, arabic = search_tracks(records, "أغاني")

        self.assertEqual(cyrillic_total, 1)
        self.assertEqual(cyrillic[0].filename, "снег_CaF1w01yFNY_140.mp3")
        self.assertIn("filename", cyrillic[0].match_fields)
        self.assertEqual(arabic_total, 1)
        self.assertEqual(arabic[0].record.tags["artist"], "ليلى")
        self.assertIn("title", arabic[0].match_fields)

    def test_requires_every_query_token_across_searchable_fields(self) -> None:
        records = [
            _track("C:/Music/blue.mp3", title="Blue Monday", artist="New Order"),
            _track("C:/Music/monday.mp3", title="Monday", artist="Someone Else"),
        ]

        total, results = search_tracks(records, "blue order")

        self.assertEqual(total, 1)
        self.assertEqual(results[0].record.tags["title"], "Blue Monday")
        self.assertEqual(results[0].match_fields[:2], ("title", "artist"))

    def test_ranking_is_deterministic_and_limit_preserves_total(self) -> None:
        records = [
            _track("C:/Music/zeta-night.mp3", title="Night Drive"),
            _track("C:/Music/night.mp3", title="Night"),
            _track("C:/Music/alpha-night.mp3", title="A Night Away"),
        ]

        total, results = search_tracks(records, "night", limit=2)

        self.assertEqual(total, 3)
        self.assertEqual([item.record.tags["title"] for item in results], ["Night", "Night Drive"])

    def test_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            search_tracks([], "night", limit=0)


if __name__ == "__main__":
    unittest.main()
