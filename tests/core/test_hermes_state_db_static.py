"""RED test for HermesStateDbImporter class-level API.

22.06.2026 bugfix: upstream `build_dataset_from_external` calls
`importer_cls.extract_messages()` as a static method (no `self`). Our
class is defined with `self` for state (state_db_path, etc.) which
breaks the call site:

  File ".../external_importers.py", line 648, in build_dataset_from_external
    msgs = importer_cls.extract_messages()
TypeError: extract_messages() missing 1 required positional argument: 'self'

The fix: extract_messages should be a staticmethod (or classmethod) so
it can be called as `importer_cls.extract_messages(...)` without instantiation.
Path configuration is done via env vars (HERMES_STATE_DB_PATH) at module
import, not via instance state.
"""

import os
from pathlib import Path

import pytest


def test_extract_messages_is_static_method():
    """extract_messages must be callable as HermesStateDbImporter.extract_messages(...)
    without instantiating the class — that's how upstream's build_dataset_from_external
    calls it.
    """
    from evolution.core.hermes_state_db import HermesStateDbImporter
    import inspect

    # The function should not have 'self' as first parameter when called via class.
    # If it's a regular method, calling it via class without instance raises TypeError.
    # We can verify by checking that the function is callable as a static/classmethod.
    method = getattr(HermesStateDbImporter, "extract_messages")
    # Static method or classmethod: can be called without instance
    # Regular method (unbound): would need self

    # This is the actual call that fails in production:
    import unittest.mock as mock
    # Mock the state.db read so we don't need a real file.
    with mock.patch.object(HermesStateDbImporter, "_read_usage_json", return_value=None), \
         mock.patch.object(Path, "exists", return_value=False):
        # If extract_messages is a regular method, this raises TypeError.
        # If staticmethod/classmethod, this works and returns [].
        result = HermesStateDbImporter.extract_messages()
    assert result == []


def test_extract_messages_accepts_skill_name_as_positional():
    """extract_messages(skill_name) should be callable positionally too,
    matching upstream API which passes skill_name as first arg.
    """
    from evolution.core.hermes_state_db import HermesStateDbImporter
    import unittest.mock as mock

    with mock.patch.object(HermesStateDbImporter, "_read_usage_json", return_value=None), \
         mock.patch.object(Path, "exists", return_value=False):
        # Both positional and keyword must work.
        result_pos = HermesStateDbImporter.extract_messages("daniil-protocol")
        result_kw = HermesStateDbImporter.extract_messages(skill_name="daniil-protocol")
    assert result_pos == result_kw == []


def test_build_dataset_from_external_can_call_hermes_state_db_importer():
    """Integration: call build_dataset_from_external with hermes-state-db source.
    This is the actual code path that broke."""
    from evolution.core.external_importers import build_dataset_from_external
    from evolution.core.dataset_builder import EvalDataset
    from evolution.core.hermes_state_db import HermesStateDbImporter
    import tempfile
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmpdir:
        # Fake the db path so importer doesn't try to read real state.db
        fake_db = Path(tmpdir) / "state.db"
        # HermesStateDbImporter reads HERMES_STATE_DB_PATH env var
        os.environ["HERMES_STATE_DB_PATH"] = str(fake_db)
        os.environ["HERMES_USAGE_JSON_PATH"] = str(Path(tmpdir) / "usage.json")

        # Make a minimal valid SQLite db
        import sqlite3
        conn = sqlite3.connect(str(fake_db))
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER DEFAULT 1)")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', 'test message long enough', 1.0)")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'assistant', 'test response', 2.0)")
        conn.commit()
        conn.close()

        # Mock the relevance filter so we don't need LLM
        with mock.patch("evolution.core.external_importers.RelevanceFilter") as mock_rf:
            mock_rf.return_value.filter_and_score.return_value = []

            # This call should NOT raise TypeError on importer_cls.extract_messages()
            try:
                dataset = build_dataset_from_external(
                    skill_name="test-skill",
                    skill_text="# test skill body",
                    sources=["hermes-state-db"],
                    output_path=Path(tmpdir),
                    model="openai/test-model",
                )
            except TypeError as e:
                if "self" in str(e):
                    pytest.fail(
                        f"build_dataset_from_external broke on hermes-state-db: {e}. "
                        f"extract_messages must be static/classmethod."
                    )
                raise
            except Exception as e:
                # Other exceptions are OK (e.g. LLM call failed) — we only care
                # about the static-method TypeError.
                pass
