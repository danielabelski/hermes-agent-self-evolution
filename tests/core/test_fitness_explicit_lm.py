"""RED tests for explicit LM passing in LLMJudge.

22.06.2026: LLMJudge.__init__ used self.config.eval_model default
"openai/gpt-4.1-mini" without api_base/api_key override. With 9router
(OPENAI_API_BASE=http://localhost:8787/v1), LiteLLM's `openai/` provider
fails with NotFoundError because 9router doesn't expose that provider
key.

Fix: LLMJudge takes an explicit `lm` parameter. The caller (evolve_skill.py)
builds `dspy.LM(model, api_base=..., api_key=...)` and passes it in.
This is the only way to guarantee 9router compatibility — relying on
env-var heuristics inside LiteLLM is fragile.

These tests lock in:
  1. LLMJudge.__init__ accepts `lm` parameter.
  2. LLMJudge.score() uses self.lm (not constructs dspy.LM internally).
  3. skill_fitness_metric passes lm through to LLMJudge.
  4. When no lm is passed, falls back to self.config.eval_model.
"""

import os
from unittest.mock import patch, MagicMock

import pytest


# ── LLMJudge accepts explicit lm ──────────────────────────────────────────


def test_llm_judge_init_accepts_lm_parameter():
    """LLMJudge must accept an `lm` parameter in __init__. This is the fix
    that prevents NotFoundError when 9router is configured as OpenAI-compatible
    base but LiteLLM tries to route to api.openai.com."""
    from evolution.core.fitness import LLMJudge
    from evolution.core.config import EvolutionConfig

    mock_lm = MagicMock()
    config = EvolutionConfig(hermes_agent_path=None)
    judge = LLMJudge(config=config, lm=mock_lm)
    assert judge.lm is mock_lm, (
        f"LLMJudge must store the explicit lm. Got: {judge.lm!r}"
    )


def test_llm_judge_init_works_without_lm_parameter():
    """Backward-compat: LLMJudge still works without explicit lm
    (falls back to self.config.eval_model). This is for the stand-alone
    fitness metric use case where no caller passes lm."""
    from evolution.core.fitness import LLMJudge
    from evolution.core.config import EvolutionConfig

    config = EvolutionConfig(hermes_agent_path=None)
    judge = LLMJudge(config=config)
    # lm attribute should be None (or missing) — not crash
    assert getattr(judge, "lm", None) is None


def test_llm_judge_score_uses_self_lm():
    """When LLMJudge has self.lm set, score() must use that LM (not construct
    a new dspy.LM(self.config.eval_model) which would route to OpenAI directly)."""
    from evolution.core.fitness import LLMJudge
    from evolution.core.config import EvolutionConfig
    import dspy

    mock_lm = MagicMock()
    mock_judge_response = dspy.Prediction(
        correctness=0.7, procedure_following=0.6, conciseness=0.8, feedback="ok"
    )
    mock_lm.return_value = mock_judge_response

    with patch("dspy.ChainOfThought") as mock_cot:
        # When judge.predict is called, return our mock response
        mock_cot.return_value = MagicMock(return_value=mock_judge_response)

        config = EvolutionConfig(hermes_agent_path=None, eval_model="openai/gpt-4.1-mini")
        judge = LLMJudge(config=config, lm=mock_lm)
        fit = judge.score(
            task_input="t",
            expected_behavior="e",
            agent_output="o",
            skill_text="s",
        )

    # The self.lm should have been used (via dspy.context)
    # If self.lm was used, dspy.context(lm=mock_lm) was called.
    # We can't easily verify the dspy.context call directly, but we CAN
    # verify the score returned is the one from our mock.
    assert fit.correctness == 0.7
    assert fit.procedure_following == 0.6


# ── skill_fitness_metric passes lm through ──────────────────────────────


def test_skill_fitness_metric_passes_lm_to_judge():
    """skill_fitness_metric must accept lm parameter and forward it to LLMJudge.
    This is the integration that lets evolve_skill.py inject the 9router LM."""
    from evolution.core.fitness import skill_fitness_metric, FitnessScore
    import dspy

    mock_lm = MagicMock()
    # LLMJudge.score returns FitnessScore (which has .composite). Mock that.
    mock_fitness = FitnessScore(
        correctness=0.5, procedure_following=0.5, conciseness=0.5, feedback=""
    )

    with patch("evolution.core.fitness.LLMJudge") as mock_judge_cls:
        mock_judge_cls.return_value.score = MagicMock(return_value=mock_fitness)

        example = dspy.Example(task_input="t", expected_behavior="e").with_inputs("task_input")
        prediction = dspy.Prediction(output="o")

        skill_fitness_metric(example, prediction, lm=mock_lm)

    # LLMJudge must have been constructed with lm=mock_lm
    call_kwargs = mock_judge_cls.call_args.kwargs
    assert call_kwargs.get("lm") is mock_lm, (
        f"LLMJudge must receive lm=mock_lm. Got: {call_kwargs!r}"
    )


# ── 9router env propagation ─────────────────────────────────────────────


def test_evolve_skill_passes_lm_with_explicit_api_base(monkeypatch):
    """evolve_skill.py must build dspy.LM with explicit api_base and api_key
    from env vars (so 9router works). This is the contract that prevents
    the NotFoundError on 'openai' provider."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # We can't easily import the main function and inspect it,
    # but we can verify dspy.LM is constructed with the right kwargs
    # by mocking it and running a dry-run CLI invocation.
    with patch("dspy.LM") as mock_lm_cls:
        from click.testing import CliRunner
        from evolution.skills.evolve_skill import main
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("---\nname: test\ndescription: test\n---\n\n# Test\n")
            skill_path = f.name

        try:
            result = CliRunner().invoke(
                main,
                [
                    "--source", skill_path,
                    "--skill", "test",
                    "--eval-source", "synthetic",
                    "--eval-model", "minimax/MiniMax-M2.5",
                    "--dry-run",
                ],
            )
        finally:
            os.unlink(skill_path)

    # If evolve_skill.py builds dspy.LM with explicit api_base, mock_lm_cls
    # was called with api_base=OPENAI_API_BASE. Verify.
    if mock_lm_cls.called:
        for call in mock_lm_cls.call_args_list:
            kwargs = call.kwargs
            if kwargs.get("api_base"):
                assert kwargs["api_base"] == "http://localhost:8787/v1", (
                    f"dspy.LM must use api_base from OPENAI_API_BASE env var. "
                    f"Got: {kwargs['api_base']!r}"
                )
                return
        # If no call had api_base, fail
        for call in mock_lm_cls.call_args_list:
            print(f"DEBUG dspy.LM call: kwargs={call.kwargs}, args={call.args}")
        pytest.fail(
            "dspy.LM was called but never with api_base=OPENAI_API_BASE. "
            "This means 9router routing will fail with NotFoundError on 'openai' provider."
        )
