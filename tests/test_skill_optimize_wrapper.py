"""Tests for scripts/skill_optimize.sh — the wrapper around hermes-agent-self-evolution.

Locks in two architectural invariants (21.06.2026):

1. NO global `export VAR=value` calls in the script. The script must only
   use `set -a; . file; set +a` which exports into the python subprocess
   only, not into the caller's shell. This protects the user from having
   their interactive bash environment polluted.

2. NINE_ROUTER_API_KEY (which lives in ~/.hermes/.env as the global Hermes
   secret-store key name) is auto-mapped to OPENAI_API_KEY (which DSPy /
   LiteLLM clients read by default). Without this mapping, evolution would
   fail with "OPENAI_API_KEY not set" even though the user has the right key
   in their global Hermes secrets.
"""

import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_DIR / "scripts" / "skill_optimize.sh"


# ── Sanity ────────────────────────────────────────────────────────────────


def test_skill_optimize_sh_exists():
    """Wrapper script must exist at the expected location."""
    assert SCRIPT_PATH.exists()
    assert SCRIPT_PATH.is_file()


def test_skill_optimize_sh_is_executable():
    """Wrapper script must be executable (chmod +x)."""
    import stat
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT_PATH} is not user-executable"


# ── No global export ──────────────────────────────────────────────────────


def test_skill_optimize_sh_does_not_export_globally():
    """The script must not call `export VAR=value` for any env var.

    `export` modifies the calling shell's environment permanently. This
    is hostile to the user (pollutes their interactive bash). The correct
    pattern is `set -a; . file; set +a` which exports into the python
    subprocess only.

    Allow `export` ONLY when it appears in a comment line (the right way to
    document the intended env var names without actually exporting them).
    """
    content = SCRIPT_PATH.read_text()
    for i, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        # Skip comments and blank lines.
        if not line or line.startswith("#"):
            continue
        # `export` as a bare command (with or without leading whitespace).
        assert not line.startswith("export "), (
            f"Line {i}: bare 'export VAR=...' is forbidden — use "
            f"`set -a; . file; set +a` instead. Got: {raw_line!r}"
        )
        # `export VAR=value` mid-line (e.g. inside a function body).
        # Use a regex that's strict enough to avoid false positives on
        # words like "export_path".
        import re
        if re.search(r"\bexport\s+[A-Z_][A-Z0-9_]*=", line):
            assert False, (
                f"Line {i}: 'export VAR=' pattern is forbidden. "
                f"Use `set -a; . file; set +a`. Got: {raw_line!r}"
            )


def test_skill_optimize_sh_uses_set_a_pattern():
    """The script must use the `set -a; . file; set +a` pattern to load .env
    files. This auto-exports assigned vars to subprocess env without
    modifying the caller shell."""
    content = SCRIPT_PATH.read_text()
    assert "set -a" in content, (
        "Script must use `set -a` to auto-export vars loaded from .env files"
    )
    assert "set +a" in content, (
        "Script must use `set +a` to stop auto-export after loading .env"
    )


# ── 9router key mapping ───────────────────────────────────────────────────


def test_skill_optimize_sh_loads_hermes_dot_env():
    """Script must source ~/.hermes/.env (where NINE_ROUTER_API_KEY lives)."""
    content = SCRIPT_PATH.read_text()
    assert ".hermes/.env" in content, (
        "Script must load ~/.hermes/.env (the global Hermes secret store)"
    )


def test_skill_optimize_sh_loads_project_dot_env():
    """Script must source the project-local .env (where OPENAI_API_BASE and
    EVAL_MODEL defaults live)."""
    content = SCRIPT_PATH.read_text()
    # Either an explicit path or a derived path via PROJECT_DIR.
    assert "$PROJECT_DIR" in content or "PROJECT_DIR" in content, (
        "Script must derive and load the project-local .env"
    )


def test_skill_optimize_sh_maps_nine_router_to_openai_key():
    """If NINE_ROUTER_API_KEY is set but OPENAI_API_KEY is empty, the script
    must assign OPENAI_API_KEY=$NINE_..._KEY so DSPy/LiteLLM clients
    (which read OPENAI_API_KEY by default) can authenticate to 9router."""
    content = SCRIPT_PATH.read_text()
    # Look for any conditional assignment that maps NINE_ROUTER → OPENAI.
    # Acceptable forms (in comments or code):
    #   OPENAI_API_KEY="$NIN..."
    #   if [ -n "${NINE_ROUTER_API_KEY:-}" ]; then OPENAI_API_KEY="$NIN...Y"; fi
    assert "NINE_ROUTER_API_KEY" in content, (
        "Script must reference NINE_ROUTER_API_KEY (the Hermes global key name)"
    )
    assert "OPENAI_API_KEY" in content, (
        "Script must reference OPENAI_API_KEY (DSPy/LiteLLM default)"
    )
    # Ensure the mapping is via assignment referencing NINE_ROUTER_API_KEY,
    # not just comment.
    # Use regex: OPENAI_API_KEY=...$NINE_ROUTER_API_KEY...
    import re
    pattern = re.compile(r"OPENAI_API_KEY=[\"']?\$\{?NINE_ROUTER_API_KEY", re.MULTILINE)
    assert pattern.search(content), (
        "Script must explicitly map NINE_ROUTER_API_KEY → OPENAI_API_KEY "
        "via an assignment (e.g. OPENAI_API_KEY=\"$NIN..._KEY\")"
    )


# ── Functional smoke ──────────────────────────────────────────────────────


def test_skill_optimize_sh_help_works():
    """The script must be runnable: `bash skill_optimize.sh --help` should
    display evolution's help text."""
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Expected --help to succeed. stderr: {result.stderr}"
    )
    assert "Evolve a Hermes Agent skill" in result.stdout, (
        "Expected evolution's help banner in output"
    )
    assert "--source" in result.stdout, "Expected --source flag documented"
    assert "--output-dir" in result.stdout, "Expected --output-dir flag documented"
    assert "--sessiondb-source" in result.stdout, "Expected --sessiondb-source flag documented"
