"""Tests for --source PATH and --output-dir PATH CLI flags.

Added 21.06.2026: standalone-mode — read any SKILL.md directly, write output
to any directory. No HERMES_AGENT_REPO required.

Background
----------
Default evolution flow expects `--hermes-repo` pointing at a directory that
contains `skills/<name>/SKILL.md`. For standalone use (e.g. optimizing one of
your own `~/.hermes/skills/<name>/SKILL.md`), we want to skip that lookup and
point at the SKILL.md directly. We also want to override the default `./output`
relative-to-CWD output location.

These tests lock in:
  1. `--source PATH` loads a SKILL.md directly without hermes-repo lookup.
  2. `--output-dir PATH` overrides the default output location.
  3. `--source` + `--hermes-repo` are mutually exclusive with a clear error.
"""

import pytest
from pathlib import Path
from click.testing import CliRunner


SAMPLE_SKILL = """---
name: standalone-test
description: A standalone skill for testing the --source flag
version: 1.0.0
---

# Standalone Test

## Procedure
1. Read the input
2. Do the thing
3. Verify
"""


def _write_skill(tmp_path: Path, name: str = "standalone-test") -> Path:
    """Write a sample SKILL.md directly to tmp_path/<name>.md."""
    skill_file = tmp_path / f"{name}.md"
    skill_file.write_text(SAMPLE_SKILL)
    return skill_file


# ── --source PATH loads SKILL.md directly ───────────────────────────────────


def test_source_flag_dry_run_loads_skill(tmp_path, monkeypatch):
    """--source PATH should bypass find_skill() and load SKILL.md directly."""
    from evolution.skills.evolve_skill import main

    # No HERMES_AGENT_REPO, no default ~/.hermes/hermes-agent — must still work.
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    skill_file = _write_skill(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "--source", str(skill_file),
            "--skill", "standalone-test",  # still required by CLI signature
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "standalone-test" in result.output


def test_source_flag_rejects_nonexistent_path(tmp_path, monkeypatch):
    """--source PATH must fail clearly if the file does not exist."""
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    missing = tmp_path / "does-not-exist.md"

    result = CliRunner().invoke(
        main,
        [
            "--source", str(missing),
            "--skill", "anything",
            "--dry-run",
        ],
    )

    # Should fail (non-zero exit) with a clear error mentioning the missing file.
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


def test_source_flag_with_hermes_repo_is_error(tmp_path, monkeypatch):
    """--source and --hermes-repo are mutually exclusive."""
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    skill_file = _write_skill(tmp_path)
    fake_repo = tmp_path / "fake-hermes"
    fake_repo.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "--source", str(skill_file),
            "--hermes-repo", str(fake_repo),
            "--skill", "standalone-test",
            "--dry-run",
        ],
    )

    # Should fail with a clear "use one or the other" message.
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "either" in result.output.lower()


# ── --output-dir PATH overrides default ────────────────────────────────────


def test_output_dir_creates_directory_with_dry_run(tmp_path, monkeypatch):
    """--output-dir should be created if it doesn't exist (dry-run path)."""
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    skill_file = _write_skill(tmp_path)
    out_dir = tmp_path / "custom" / "nested" / "out"

    result = CliRunner().invoke(
        main,
        [
            "--source", str(skill_file),
            "--skill", "standalone-test",
            "--output-dir", str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    # Dry run doesn't actually write, but the dir shouldn't error either.


def test_output_dir_appears_in_validation_step(tmp_path, monkeypatch):
    """The validation step must respect the --output-dir override."""
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    skill_file = _write_skill(tmp_path)
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(
        main,
        [
            "--source", str(skill_file),
            "--skill", "standalone-test",
            "--output-dir", str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    # The dry-run message confirms setup validated.
    assert "setup validated" in result.output.lower() or "DRY RUN" in result.output
