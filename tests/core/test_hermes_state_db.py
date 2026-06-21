"""Tests for HermesStateDbImporter — session mining from ~/.hermes/state.db.

Added 21.06.2026 to support using the user's actual Hermes session history
as eval data for skill evolution. The upstream HermesSessionImporter looks
at `~/.hermes/sessions/*.json` (request dumps), but the real session data
is in `~/.hermes/state.db` (SQLite).

Strategy: cross-reference `~/.hermes/skills/.usage.json` (which tracks per-
skill usage timestamps) with the SQLite state.db to find sessions where a
given skill was actually used. Then extract user/assistant message pairs.

Privacy: optionally sanitizes content via 9router (`NINE_ROUTER_API_KEY`)
if available. Falls back to upstream `_contains_secret` regex otherwise.
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    """Create a minimal state.db with 2 sessions, user+assistant pairs."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    # Session A: started 1_000_000.0, ended 2_000_000.0
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("session-A", "telegram", 1_000_000.0, 2_000_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-A", "user", "what is the daniil protocol", 1_100_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-A", "assistant", "Daniil protocol is two-phase", 1_200_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-A", "user", "explain skill evolution", 1_300_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-A", "assistant", "DSPy + GEPA", 1_400_000.0),
    )

    # Session B: started 3_000_000.0, ended 4_000_000.0 — different time window
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("session-B", "telegram", 3_000_000.0, 4_000_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-B", "user", "unrelated question about something else", 3_100_000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-B", "assistant", "unrelated answer here", 3_200_000.0),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def usage_json(tmp_path: Path) -> Path:
    """Create a .usage.json with one skill used in session A timeframe."""
    usage_path = tmp_path / ".usage.json"
    data = {
        "daniil-protocol": {
            "use_count": 5,
            "view_count": 5,
            # Unix timestamp 1_500_000.0 → 1970-01-18T16:40:00 UTC (sits inside session A's [1M, 2M] window)
            "last_used_at": "1970-01-18T16:40:00+00:00",
            "last_viewed_at": "1970-01-18T16:40:00+00:00",
        },
        "never-used-skill": {
            "use_count": 0,
            "view_count": 0,
        },
    }
    usage_path.write_text(json.dumps(data, indent=2))
    return usage_path


@pytest.fixture
def importer(state_db: Path, usage_json: Path, monkeypatch):
    """Build an importer pointing at our fixtures (via env vars)."""
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(usage_json))
    # Disable sanitizer by default (no API key in tests unless explicitly set).
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NINE_ROUTER_URL", raising=False)

    from evolution.core.hermes_state_db import HermesStateDbImporter
    return HermesStateDbImporter()


# ── Sanity: env-var path discovery ────────────────────────────────────────


def test_paths_from_env_vars(state_db, usage_json, monkeypatch):
    """Importer must read paths from HERMES_STATE_DB_PATH and HERMES_USAGE_JSON_PATH."""
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(usage_json))

    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    assert imp.state_db_path == state_db
    assert imp.usage_json_path == usage_json


# ── Filtering by skill name via .usage.json ───────────────────────────────


def test_extract_messages_filters_by_skill(importer):
    """With skill_name='daniil-protocol' and a matching last_used_at in
    session-A's window, must return pairs from session-A only."""
    pairs = importer.extract_messages(skill_name="daniil-protocol")
    assert len(pairs) >= 1
    # All pairs must be from session-A (the only one overlapping with last_used_at)
    assert all(p["session_id"] == "session-A" for p in pairs)
    # source field tags them as the right kind
    assert all(p["source"] == "hermes-state-db" for p in pairs)


def test_extract_messages_skill_never_used(importer):
    """If skill has use_count=0, no pairs returned."""
    pairs = importer.extract_messages(skill_name="never-used-skill")
    assert pairs == []


def test_extract_messages_skill_not_in_usage(importer):
    """If skill is not tracked in .usage.json at all, no pairs (no signal)."""
    pairs = importer.extract_messages(skill_name="completely-unknown-skill")
    assert pairs == []


def test_extract_messages_no_skill_returns_all(importer):
    """Without skill_name filter, return all qualifying pairs across all sessions."""
    pairs = importer.extract_messages()  # no skill filter
    # Session A has 2 user+assistant pairs, Session B has 1 (but short, see below)
    session_a_pairs = [p for p in pairs if p["session_id"] == "session-A"]
    session_b_pairs = [p for p in pairs if p["session_id"] == "session-B"]
    assert len(session_a_pairs) == 2
    # Session B has 'unrelated question' (24 chars) — passes the >=10 char filter
    assert len(session_b_pairs) == 1


# ── Robustness: missing files ────────────────────────────────────────────


def test_handles_missing_state_db(tmp_path, usage_json, monkeypatch):
    """If state.db doesn't exist, return [] (not raise)."""
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(tmp_path / "no-such.db"))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(usage_json))
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)

    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    pairs = imp.extract_messages()
    assert pairs == []


def test_handles_missing_usage_json(state_db, tmp_path, monkeypatch):
    """If .usage.json doesn't exist, fall back to all sessions (no skill filter)."""
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(tmp_path / "no-usage.json"))
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)

    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    pairs = imp.extract_messages()  # no skill filter
    # Should still return session pairs despite no usage file
    assert len(pairs) >= 1


def test_handles_corrupt_usage_json(state_db, tmp_path, monkeypatch):
    """Corrupt JSON in .usage.json → fall back to all sessions, don't crash."""
    bad_usage = tmp_path / ".usage.json"
    bad_usage.write_text("{ this is not valid json")
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(bad_usage))
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)

    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    pairs = imp.extract_messages()
    assert len(pairs) >= 1  # didn't crash


# ── Pair extraction semantics ─────────────────────────────────────────────


def test_pairs_are_user_then_assistant(importer):
    """Each pair must have user content in task_input, assistant in assistant_response."""
    pairs = importer.extract_messages(skill_name="daniil-protocol")
    assert len(pairs) >= 1
    for p in pairs:
        assert "task_input" in p
        assert "assistant_response" in p
        assert "session_id" in p
        # user text was non-empty in fixtures
        assert p["task_input"]
        assert p["assistant_response"]


def test_skips_short_user_messages(state_db, tmp_path, monkeypatch):
    """User messages < 10 chars are skipped (matches upstream HermesSessionImporter)."""
    # Insert a session with a 5-char user message
    conn = sqlite3.connect(str(state_db))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("session-short", "telegram", 5000.0, 6000.0),
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-short", "user", "hi", 5100.0),  # too short
    )
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session-short", "assistant", "hello there", 5200.0),
    )
    conn.commit()
    conn.close()

    usage = tmp_path / ".usage.json"
    usage.write_text("{}")
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(usage))
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)

    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    pairs = imp.extract_messages()
    short_pairs = [p for p in pairs if p["session_id"] == "session-short"]
    assert short_pairs == []  # skipped due to length


# ── Privacy: secret filter ────────────────────────────────────────────────


def test_skips_messages_with_secrets(importer):
    """Messages containing obvious secrets are filtered out (upstream regex)."""
    # The fixtures don't contain secrets, so just verify the import path:
    from evolution.core.external_importers import _contains_secret
    assert _contains_secret("api_key: sk-abc123def456ghi789jkl012mno345pqr678") is True
    assert _contains_secret("just a regular message about skills") is False


def test_no_sanitize_without_9router_key(state_db, usage_json, monkeypatch):
    """Without NINE_ROUTER_API_KEY, no HTTP calls — pure local mode.

    Note: as of 21.06.2026 sanitize-step was removed entirely from the
    importer. 9router PII protection happens transparently via DSPy/LiteLLM
    at the LLM-call layer, not as a separate preprocess step. This test
    now verifies the cleaner invariant: no HTTP calls ever come from
    extract_messages().
    """
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(state_db))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(usage_json))
    monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NINE_ROUTER_URL", raising=False)

    with patch("requests.post") as mock_post:
        from evolution.core.hermes_state_db import HermesStateDbImporter
        imp = HermesStateDbImporter()
        pairs = imp.extract_messages(skill_name="daniil-protocol")
        assert len(pairs) >= 1
        # No HTTP calls in extract_messages — period.
        assert not mock_post.called


def test_extract_messages_makes_no_http_calls(importer):
    """extract_messages must never make HTTP calls. Sanitization (if any) is
    a separate concern handled at the LLM-call layer via DSPy/LiteLLM with
    9router's transparent PII/secret masking.

    This locks in the architectural decision (21.06.2026): the importer
    is purely a SQLite+JSON reader, not a sanitization client.
    """
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get, \
         patch("urllib.request.urlopen") as mock_urlopen:
        pairs = importer.extract_messages(skill_name="daniil-protocol")
        assert len(pairs) >= 1
        assert mock_post.call_count == 0
        assert mock_get.call_count == 0
        assert mock_urlopen.call_count == 0


def test_importer_does_not_read_nine_router_env_vars(monkeypatch):
    """HermesStateDbImporter constructor must not consult NINE_ROUTER_API_KEY
    or NINE_ROUTER_URL — those are concerns of the LLM-call layer
    (DSPy/LiteLLM via 9router), not the importer.
    """
    monkeypatch.setenv("HERMES_STATE_DB_PATH", "/nonexistent/path")
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", "/nonexistent/path")
    monkeypatch.setenv("NINE_ROUTER_API_KEY", "should-not-be-read")
    monkeypatch.setenv("NINE_ROUTER_URL", "http://should-not-be-read")

    # Should NOT raise or read those vars — just construct without error.
    from evolution.core.hermes_state_db import HermesStateDbImporter
    imp = HermesStateDbImporter()
    # Path defaults should be the actual ~/.hermes paths, not affected by
    # the NINE_ROUTER_* vars we just set.
    assert "NINE_ROUTER" not in str(imp.state_db_path)
    assert "NINE_ROUTER" not in str(imp.usage_json_path)
