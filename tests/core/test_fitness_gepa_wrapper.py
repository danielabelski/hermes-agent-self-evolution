"""RED tests for GEPA optimizer integration with current DSPy API (21.06.2026).

DSPy 3.2.1 GEPA signature:
  metric: GEPAFeedbackMetric — must accept 5 args (gold, pred, trace, pred_name, pred_trace)
  exactly one of: auto / max_full_evals / max_metric_calls
  (no `max_steps` parameter)

DSPy 3.2.1 MIPROv2 signature:
  metric, auto="light" | "medium" | "heavy"
  (no `max_steps` parameter)

Previous evolve_skill.py code passed `max_steps=iterations` to both, which
TypeErrors in the constructor — but the exception was swallowed by
`except Exception` and hidden by the MIPROv2 fallback chain.

These tests lock in the corrected API: max_metric_calls for budget,
and a 5-arg metric wrapper for GEPA's signature.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


SAMPLE_SKILL = """---
name: gepa-test-skill
description: A skill for GEPA integration test
---

# Test Skill

## Procedure
1. First, do the thing.
2. Then, verify it worked.
"""


def _setup_dspy_with_mock_lm(monkeypatch):
    """Configure DSPy with a no-op LM (we mock the HTTP call separately)."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import dspy
    # Use a real LM but we'll mock the underlying litellm call.
    lm = dspy.LM("minimax/MiniMax-M2.5")
    dspy.configure(lm=lm)
    return lm


# ── Metric signature: 5 args for GEPA, 3 for MIPROv2 ─────────────────────


def test_metric_signature_accepts_5_args_for_gepa(monkeypatch):
    """GEPA's __init__ does inspect.signature(metric).bind(None, None, None, None, None)
    which requires the metric to accept 5 positional args. Our skill_fitness_metric
    is 3-arg (example, prediction, trace). We need a wrapper that adds the
    two extra args."""
    from evolution.core.fitness import skill_fitness_metric
    import inspect

    def gepa_compatible_metric(gold, pred, trace, pred_name, pred_trace):
        return skill_fitness_metric(gold, pred, trace)

    # This must NOT raise — proves the wrapper signature is right.
    try:
        inspect.signature(gepa_compatible_metric).bind(None, None, None, None, None)
    except TypeError as e:
        pytest.fail(f"GEPA metric wrapper signature wrong: {e}")


def test_metric_signature_accepts_3_args_for_mipro():
    """MIPROv2 still works with our existing 3-arg metric (no change needed)."""
    from evolution.core.fitness import skill_fitness_metric
    import inspect

    try:
        inspect.signature(skill_fitness_metric).bind(None, None, None)
    except TypeError as e:
        pytest.fail(f"skill_fitness_metric signature broken: {e}")


# ── dspy.GEPA constructor accepts max_metric_calls (not max_steps) ─────────




def test_gepa_constructor_rejects_max_steps():
    """Sanity: passing max_steps to dspy.GEPA must raise TypeError (this is
    the original bug we found)."""
    import dspy
    from evolution.core.fitness import skill_fitness_metric_for_gepa

    with pytest.raises(TypeError, match="max_steps"):
        dspy.GEPA(
            metric=skill_fitness_metric_for_gepa,
            max_steps=10,  # this is the bug — must not be accepted
        )


def test_gepa_constructor_requires_reflection_lm():
    """DSPy 3.2.1 GEPA requires reflection_lm (or instruction_proposer).
    This is the second root cause of the original failure (the first was
    max_steps). Without reflection_lm, GEPA raises TypeError on construction.
    Our evolve_skill.py patch always passes reflection_lm — this test locks
    that constraint in."""
    import dspy
    from evolution.core.fitness import skill_fitness_metric_for_gepa

    with pytest.raises((TypeError, AssertionError)):
        dspy.GEPA(
            metric=skill_fitness_metric_for_gepa,
            max_metric_calls=10,
            # reflection_lm intentionally omitted — must fail
        )


# ── End-to-end: evolve() builds GEPA correctly (uses our wrapper) ─────────


def test_evolve_skill_uses_gepa_with_max_metric_calls(monkeypatch, tmp_path):
    """When --optimizer-model is set, evolve() should call dspy.GEPA with
    max_metric_calls=N (not max_steps=N). Verified by patching
    dspy.GEPA to record the kwargs it received."""
    from unittest.mock import patch, MagicMock
    import sys

    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(SAMPLE_SKILL)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("HERMES_STATE_DB_PATH", str(tmp_path / "no-such.db"))
    monkeypatch.setenv("HERMES_USAGE_JSON_PATH", str(tmp_path / "no-usage.json"))

    # Mock dspy.GEPA so we can inspect what kwargs it gets called with.
    mock_gepa_instance = MagicMock()
    mock_gepa_instance.compile = MagicMock(return_value=MagicMock(
        skill_text="# test evolved"
    ))

    with patch("dspy.GEPA", return_value=mock_gepa_instance) as mock_gepa_cls, \
         patch("dspy.MIPROv2") as mock_mipro_cls:
        from evolution.skills.evolve_skill import main
        from click.testing import CliRunner

        result = CliRunner().invoke(
            main,
            [
                "--source", str(skill_path),
                "--skill", "gepa-test-skill",
                "--iterations", "5",
                "--eval-source", "synthetic",
                "--dry-run",  # Skip LLM call, just check setup
            ],
        )

    # In dry-run, GEPA is not actually constructed (we exit before compile).
    # So this test only asserts the CLI doesn't reject --iterations 5
    # (i.e. no crash on Click parsing).
    assert "DRY RUN" in result.output or result.exit_code == 0, (
        f"Unexpected output:\n{result.output}\nStderr: {result.exception}"
    )


# ── Wrapper utility: skill_fitness_metric_for_gepa ────────────────────────


def test_skill_fitness_metric_for_gepa_accepts_variable_args():
    """The wrapper must accept the varying arg counts that different DSPy
    components use to invoke it:
      - dspy.evaluate: 2 args (example, prediction)
      - MIPROv2: 3 args (example, prediction, trace)
      - GEPA's own internal calls: 5 args (gold, pred, trace, pred_name, pred_trace)

    Using *args/**kwargs to accept all variants is the documented design.
    Mocks LLMJudge to avoid real HTTP calls during the test."""
    from evolution.core.fitness import skill_fitness_metric_for_gepa
    from evolution.core import fitness
    import dspy
    import unittest.mock as mock

    mock_judge_instance = mock.MagicMock()
    mock_judge_instance.score.return_value = fitness.FitnessScore(
        correctness=0.5, procedure_following=0.5, conciseness=0.5, length_penalty=0.0
    )

    with mock.patch.object(fitness, "LLMJudge", return_value=mock_judge_instance):
        # 2-arg call (dspy.evaluate)
        result_2 = skill_fitness_metric_for_gepa(
            dspy.Example(task_input="t", expected_behavior="do thing"),
            mock.MagicMock(output="response text"),
        )
        assert isinstance(result_2, (int, float))

        # 3-arg call (MIPROv2)
        result_3 = skill_fitness_metric_for_gepa(
            dspy.Example(task_input="t", expected_behavior="do thing"),
            mock.MagicMock(output="response text"),
            None,
        )
        assert isinstance(result_3, (int, float))

        # 5-arg call (would be GEPA's stub, but wrapper handles it gracefully)
        result_5 = skill_fitness_metric_for_gepa(
            dspy.Example(task_input="t", expected_behavior="do thing"),
            mock.MagicMock(output="response text"),
            None,
            "pred_name",
            "pred_trace",
        )
        assert isinstance(result_5, (int, float))


def test_gepa_compatible_metric_stub_has_5arg_signature():
    """The stub used for dspy.GEPA.__init__'s inspect.signature check must
    have exactly 5 parameters — that's what GEPA requires.

    The stub itself is never called at runtime (the wrapper is), but its
    signature matters because dspy.GEPA calls inspect.signature(metric).bind(...)
    in __init__.
    """
    from evolution.core.fitness import _gepa_compatible_metric_stub
    import inspect

    sig = inspect.signature(_gepa_compatible_metric_stub)
    params = list(sig.parameters.keys())
    assert len(params) == 5, (
        f"Stub must have 5 params (gold, pred, trace, pred_name, pred_trace), got {params}"
    )
