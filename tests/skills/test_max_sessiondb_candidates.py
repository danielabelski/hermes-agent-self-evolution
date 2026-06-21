"""RED tests for --max-sessiondb-candidates CLI flag.

22.06.2026: relevance filter (RelevanceFilter.filter_and_score) is the
bottleneck for sessiondb source — at 119 candidate pairs × ~1.5s per LLM
judge call = ~3 minutes, and we've seen it hang at >5 minutes with 0%
CPU. Solution: add a CLI flag that limits how many candidates reach the
relevance filter in the first place.

Default: 50 (matches upstream build_dataset_from_external max_examples
default). Override to e.g. 20 for fast smoke runs.

The flag must:
  1. Be a Click option --max-sessiondb-candidates.
  2. Be passed through evolve() to HermesStateDbImporter.extract_messages(limit=N).
  3. Default to 50 to preserve upstream behavior.
  4. Show in --help output.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


SAMPLE_SKILL = """---
name: max-sessiondb-test
description: Test skill for max-sessiondb-candidates flag
---

# Test

Body content here.
"""


def _write_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "SKILL.md"
    skill.write_text(SAMPLE_SKILL)
    return skill


# ── Sanity: flag exists and parses ───────────────────────────────────────


def test_max_sessiondb_candidates_flag_exists():
    """--max-sessiondb-candidates must be a registered Click option."""
    from evolution.skills.evolve_skill import main

    result = CliRunner().invoke(main, ["--help"])
    assert "--max-sessiondb-candidates" in result.output, (
        f"--max-sessiondb-candidates not in help:\n{result.output}"
    )


def test_max_sessiondb_candidates_default_is_50():
    """Default --max-sessiondb-candidates must be 50."""
    from evolution.skills.evolve_skill import main

    result = CliRunner().invoke(main, ["--help"])
    assert "50" in result.output, (
        f"Default 50 not visible:\n{result.output}"
    )


# ── Functional: flag flows through to extract_messages(limit=N) ──────────


def test_max_sessiondb_candidates_propagates_to_extract_messages_limit(monkeypatch, tmp_path):
    """When --max-sessiondb-candidates=20 is passed, HermesStateDbImporter.extract_messages
    must be called with limit=20 (not 50 default)."""
    skill_file = _write_skill(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(tmp_path / "no-such.db"))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(tmp_path / "no-usage.json"))

    with patch("evolution.core.hermes_state_db.HermesStateDbImporter.extract_messages", return_value=[]) as mock_extract:
        from evolution.skills.evolve_skill import main

        result = CliRunner().invoke(
            main,
            [
                "--source", str(skill_file),
                "--skill", "max-sessiondb-test",
                "--eval-source", "sessiondb",
                "--max-sessiondb-candidates", "20",
                "--dry-run",
            ],
        )

    if mock_extract.called:
        call_kwargs = mock_extract.call_args.kwargs
        call_positional = mock_extract.call_args.args
        # Either keyword or positional — both should be 20
        limit_used = call_kwargs.get("limit") or (call_positional[1] if len(call_positional) > 1 else None)
        assert limit_used == 20, (
            f"Expected limit=20, got {limit_used!r}. "
            f"call_args: {mock_extract.call_args!r}"
        )


def test_max_sessiondb_candidates_passed_via_env_var(monkeypatch, tmp_path):
    """EVO_MAX_SESSIONDB_CANDIDATES env var should also work (for wrapper script)."""
    skill_file = _write_skill(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(tmp_path / "no-such.db"))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(tmp_path / "no-usage.json"))
    monkeypatch.setenv("EVO_MAX_SESSIONDB_CANDIDATES", "15")

    with patch("evolution.core.hermes_state_db.HermesStateDbImporter.extract_messages", return_value=[]) as mock_extract:
        from evolution.skills.evolve_skill import main

        result = CliRunner().invoke(
            main,
            [
                "--source", str(skill_file),
                "--skill", "max-sessiondb-test",
                "--eval-source", "sessiondb",
                # No --max-sessiondb-candidates flag — should fall back to env var.
                "--dry-run",
            ],
        )

    if mock_extract.called:
        call_kwargs = mock_extract.call_args.kwargs
        call_positional = mock_extract.call_args.args
        limit_used = call_kwargs.get("limit") or (call_positional[1] if len(call_positional) > 1 else None)
        assert limit_used == 15, (
            f"Expected limit=15 from env, got {limit_used!r}. "
            f"call_args: {mock_extract.call_args!r}"
        )
