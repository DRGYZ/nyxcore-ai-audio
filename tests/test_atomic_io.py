from __future__ import annotations

from unittest.mock import patch

import pytest

from nyxcore.core.atomic import atomic_write_text


def test_atomic_write_replaces_complete_file_and_cleans_temporary_file(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text("old", encoding="utf-8")

    atomic_write_text(destination, "new content")

    assert destination.read_text(encoding="utf-8") == "new content"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_preserves_previous_file_when_replace_fails(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text("old", encoding="utf-8")

    with patch("nyxcore.core.atomic.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_text(destination, "incomplete replacement")

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
