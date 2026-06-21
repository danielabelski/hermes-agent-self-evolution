"""RED tests for dspy.LM env-only configuration.

22.06.2026: When wrappers set OPENAI_API_BASE / OPENAI_API_KEY in env,
evolution should call `dspy.LM(model)` WITHOUT explicit api_base/api_key.
LiteLLM auto-detects these from env. Adding explicit api_base/api_key
in the dspy.LM call broke 9router routing in proc_97d990fe99fd
(APIConnectionError: model_not_found for provider: openai).

Lock in: dspy.LM called from evolve_skill.py must NOT pass api_base/api_key
explicitly when wrapper has set them via env. LiteLLM env heuristic works
(verified in proc_0d9b68a2deae with 49 examples, baseline 0.581).
"""

from unittest.mock import patch, MagicMock
import os


def test_evolve_skill_does_not_pass_explicit_api_base_to_dspy_lm(monkeypatch):
    """evolve_skill.py must call dspy.LM(model) WITHOUT api_base/api_key.
    The wrapper script already sets OPENAI_API_BASE/OPENAI_API_KEY in env.
    LiteLLM auto-detection handles routing. Adding explicit kwargs breaks
    9router (verified in proc_97d990fe99fd with APIConnectionError: model_not_found).

    We can't test the dry-run path (it exits before dspy.LM is called).
    Instead we patch dspy.LM and call main() with --iterations=0 to make
    the runner exit early via existing iteration=0 guard. Actually,
    iterations=0 might still call dspy.LM. The cleanest test is to just
    verify the SOURCE of evolve_skill.py has no "dspy.LM(...api_base=...)" pattern.
    """
    import re
    from pathlib import Path
    src = Path(
        "/home/daniel/HermesProjects/default/hermes-agent-self-evolution/evolution/skills/evolve_skill.py"
    ).read_text()

    # The forbidden pattern: dspy.LM with explicit api_base or api_key.
    forbidden_patterns = [
        re.compile(r"dspy\.LM\([^)]*api_base="),
        re.compile(r"dspy\.LM\([^)]*api_key="),
    ]
    for pattern in forbidden_patterns:
        matches = pattern.findall(src)
        assert not matches, (
            f"evolve_skill.py contains forbidden dspy.LM(..., api_base=...) or "
            f"api_key=... pattern. This broke 9router routing in "
            f"proc_97d990fe99fd (APIConnectionError: model_not_found for provider: openai). "
            f"Use dspy.LM(model) — LiteLLM reads OPENAI_API_BASE/OPENAI_API_KEY from env. "
            f"Found: {matches[:3]}"
        )
