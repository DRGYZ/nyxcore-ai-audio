from unittest.mock import patch

import pytest

from nyxcore.core.track import TrackRecord
from nyxcore.duplicates.service import _normalize_text, analyze_duplicates


@pytest.mark.parametrize('text', ['Девочку несёт', 'أغنية جميلة', '東京の夜', 'प्यार', 'BÖ'])
def test_preserves_international_titles(text):
    assert _normalize_text(text)
    assert _normalize_text(text) != _normalize_text('')
    assert any(ord(char) > 127 for char in _normalize_text(text))


def test_equivalent_unicode_and_case_match():
    assert _normalize_text('CAFÉ') == _normalize_text('cafe\u0301')
    assert _normalize_text('Привет_МИР!') == 'привет мир'
    assert _normalize_text('Artist - Song') == 'artist song'


@pytest.mark.parametrize('titles,expected', [
    (['GRIVINA - Девочку несёт Official Video', 'GRIVINA - Я хочу Official Video'], 0),
    (['IC3PEAK - Плак-Плак', 'IC3PEAK - Смерти Больше Нет'], 0),
    (['Девочку несёт', 'Девочку несёт'], 1),
])
def test_duplicate_detection_with_cyrillic_titles(tmp_path, titles, expected):
    records = []
    for index, title in enumerate(titles):
        path = tmp_path / f'{title}.{index}.mp3'
        path.write_bytes(bytes([index]) * 1024)
        records.append(TrackRecord(
            path=str(path), file_size_bytes=1024, mtime_iso='',
            tags={'title': title, 'artist': 'Artist', 'album': None},
            has_cover_art=False, duration_seconds=200.0, warnings=[],
        ))
    with patch('nyxcore.duplicates.service._load_bitrate_bps', return_value=192000):
        report = analyze_duplicates(records)
    assert report.summary.likely_group_count == expected
