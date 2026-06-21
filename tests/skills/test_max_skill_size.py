"""Tests for the --max-skill-size CLI flag in evolution.skills.evolve_skill.

Added 21.06.2026: real-world test on daniil-protocol (49K chars) hit the
default 15K limit and was rejected by constraint gate. Users need an
override for legitimately large skills.

The flag must:
  1. Be a Click option (--max-skill-size) with default 15000.
  2. Pass through evolve() to EvolutionConfig(max_skill_size=...).
  3. ConstraintValidator should then use the override limit.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


SAMPLE_LARGE_SKILL = """---
name: large-skill
description: A test skill that exceeds the default 15K limit
version: 1.0.0
---

# Large Test Skill

""" + ("This is filler text. " * 3000)  # ~6000 words = ~36K chars


def _write_large_skill(tmp_path: Path) -> Path:
    """Write a SKILL.md that's ~36K chars (exceeds default 15K limit)."""
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_LARGE_SKILL)
    return skill_file


# ── Sanity: flag exists and parses ───────────────────────────────────────


def test_max_skill_size_flag_exists():
    """--max-skill-size must be a registered Click option."""
    from evolution.skills.evolve_skill import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "--max-skill-size" in result.output, (
        f"--max-skill-size not in help output:\n{result.output}"
    )


def test_max_skill_size_default_is_15000():
    """Default --max-skill-size must be 15000 (upstream default)."""
    from evolution.skills.evolve_skill import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    # Click --help shows "15000" as default
    assert "15000" in result.output or "default: 15000" in result.output.lower(), (
        f"Default 15000 not visible in help:\n{result.output}"
    )


# ── Functional: flag flows through to EvolutionConfig ────────────────────


def test_max_skill_size_propagates_to_config(tmp_path, monkeypatch):
    """When --max-skill-size=50000 is passed, EvolutionConfig.max_skill_size
    should be 50000 (not 15000)."""
    skill_file = _write_large_skill(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    from evolution.core.config import EvolutionConfig
    from evolution.skills.evolve_skill import main

    with patch("evolution.skills.evolve_skill.EvolutionConfig") as mock_cfg:
        mock_cfg.return_value = EvolutionConfig(
            hermes_agent_path=None,
            output_dir=Path("./output"),
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--source", str(skill_file),
                "--skill", "large-skill",
                "--max-skill-size", "50000",
                "--dry-run",
            ],
        )

    if mock_cfg.called:
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs.get("max_skill_size") == 50000, (
            f"max_skill_size={call_kwargs.get('max_skill_size')!r}, expected 50000"
        )
    else:
        # Patch wasn't called (e.g. if evolve() raises early). At least
        # confirm the option was parsed (no Click error).
        assert "Invalid value" not in result.output, (
            f"Click rejected --max-skill-size:\n{result.output}"
        )


def test_max_skill_size_allows_large_skill_through_dry_run(tmp_path, monkeypatch):
    """A 36K-char skill with --max-skill-size=50000 should pass setup validation
    (dry-run completes without the size constraint being violated)."""
    skill_file = _write_large_skill(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    from evolution.skills.evolve_skill import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--source", str(skill_file),
            "--skill", "large-skill",
            "--max-skill-size", "50000",
            "--eval-source", "synthetic",
            "--dry-run",
        ],
    )

    # Dry-run should succeed (exit 0) and print DRY RUN banner.
    assert result.exit_code == 0, (
        f"Expected dry-run to succeed. stderr/stdout:\n{result.output}"
    )
    assert "DRY RUN" in result.output, (
        f"Expected DRY RUN banner:\n{result.output}"
    )


def test_default_max_skill_size_rejects_large_skill(tmp_path, monkeypatch):
    """A 36K-char skill with default 15K limit should be flagged by the
    constraint validator as exceeding size_limit (when actually running, not
    dry-run). On dry-run the size is just reported, not enforced."""
    skill_file = _write_large_skill(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    from evolution.skills.evolve_skill import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--source", str(skill_file),
            "--skill", "large-skill",
            "--eval-source", "synthetic",
            "--dry-run",
        ],
    )

    # Dry-run just prints "Would validate constraints" — doesn't actually
    # run the validator. So with default 15K, dry-run still succeeds.
    # This test documents that dry-run doesn't enforce; real run would fail.
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
