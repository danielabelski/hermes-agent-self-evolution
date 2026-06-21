"""RED tests for LLM-as-judge skill_fitness_metric.

22.06.2026: the current `skill_fitness_metric` (evolution/core/fitness.py:107-136)
uses keyword overlap as fitness function. This is a poor proxy for actual
skill quality — a skill that paraphrases expected_behavior or improves
structure without keyword-matching gets a low score. All 4 real runs on
daniil-protocol returned improvement=0.0 because GEPA can't find
improvements the proxy can detect.

`LLMJudge` (line 34-104) is already implemented but not wired up. This
file locks in: skill_fitness_metric should use LLMJudge (not keyword
overlap) for fitness scoring, returning a float 0-1.

Trade-offs:
  + Real quality signal (correctness, procedure_following, conciseness)
  + Sees structural improvements, persona, format
  - Costs +1 LLM call per skill evaluation
  - Slower per GEPA iteration (~1-2s extra)
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest


# ── Sanity: LLM-as-judge returns FitnessScore ───────────────────────────


def test_skill_fitness_metric_uses_llm_judge_when_available(monkeypatch):
    """skill_fitness_metric should call LLMJudge.score() and return
    FitnessScore.composite, not compute keyword overlap locally.
    """
    # Make sure dspy.LM doesn't try real HTTP
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from evolution.core import fitness
    from evolution.core.config import EvolutionConfig

    # Mock LLMJudge to return a known FitnessScore
    mock_judge_instance = MagicMock()
    mock_judge_instance.score.return_value = fitness.FitnessScore(
        correctness=0.9,
        procedure_following=0.8,
        conciseness=0.7,
        length_penalty=0.0,
        feedback="Looks good",
    )

    with patch.object(fitness, "LLMJudge", return_value=mock_judge_instance) as MockJudge:
        config = EvolutionConfig(hermes_agent_path=None)
        import dspy
        example = dspy.Example(
            task_input="do thing",
            expected_behavior="rubric",
        ).with_inputs("task_input")
        prediction = dspy.Prediction(output="agent response")

        score = fitness.skill_fitness_metric(example, prediction, trace=None)

    # Expected: 0.5*0.9 + 0.3*0.8 + 0.2*0.7 = 0.45 + 0.24 + 0.14 = 0.83
    assert abs(score - 0.83) < 0.01, (
        f"Expected score ≈ 0.83 (composite of mocked FitnessScore), got {score}. "
        f"Either skill_fitness_metric still uses keyword overlap, or composite "
        f"formula changed."
    )
    # The LLMJudge must have been called with right args
    assert mock_judge_instance.score.called
    call_kwargs = mock_judge_instance.score.call_args.kwargs
    assert call_kwargs["task_input"] == "do thing"
    assert call_kwargs["expected_behavior"] == "rubric"
    assert call_kwargs["agent_output"] == "agent response"


def test_skill_fitness_metric_returns_zero_for_empty_output():
    """Empty agent output should return 0.0 (no LLM call needed)."""
    from evolution.core import fitness
    import dspy

    example = dspy.Example(task_input="t", expected_behavior="r").with_inputs("task_input")
    prediction = dspy.Prediction(output="")  # empty

    with patch.object(fitness, "LLMJudge") as MockJudge:
        score = fitness.skill_fitness_metric(example, prediction, trace=None)

    assert score == 0.0
    # LLMJudge must NOT have been called (no point scoring empty output)
    assert not MockJudge.called


def test_skill_fitness_metric_applies_length_penalty(monkeypatch):
    """When artifact_size > 90% of max_size, length_penalty > 0 and
    composite score is reduced. Test that the LLM-judge score is passed
    artifact_size and max_size correctly.

    Since LLMJudge.score() applies the length_penalty internally based on
    artifact_size/max_size kwargs, we verify the call passes these args and
    that the composite reflects the penalty.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from evolution.core import fitness
    from evolution.core.config import EvolutionConfig
    import dspy

    # Real LLMJudge but with the score() method mocked to return a
    # FitnessScore with length_penalty set to what a 95% ratio would give.
    # This isolates whether artifact_size/max_size reach judge.score().
    mock_judge_instance = MagicMock()
    mock_judge_instance.score.return_value = fitness.FitnessScore(
        correctness=1.0,
        procedure_following=1.0,
        conciseness=1.0,
        length_penalty=0.15,  # What 95% ratio would compute to internally
        feedback="",
    )

    with patch.object(fitness, "LLMJudge", return_value=mock_judge_instance):
        example = dspy.Example(task_input="t", expected_behavior="r").with_inputs("task_input")
        prediction = dspy.Prediction(output="x" * 1000)

        # Call with artifact_size and max_size such that ratio > 0.9
        score = fitness.skill_fitness_metric(
            example, prediction, trace=None,
            artifact_size=950,
            max_size=1000,
        )

    # composite = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0; minus length_penalty 0.15 = 0.85
    assert abs(score - 0.85) < 0.01, (
        f"Expected score ≈ 0.85 (composite 1.0 - length_penalty 0.15), got {score}"
    )
    # Verify the call passed artifact_size and max_size
    call_kwargs = mock_judge_instance.score.call_args.kwargs
    assert call_kwargs.get("artifact_size") == 950, (
        f"Expected artifact_size=950 in judge.score() call, got {call_kwargs.get('artifact_size')!r}"
    )
    assert call_kwargs.get("max_size") == 1000, (
        f"Expected max_size=1000 in judge.score() call, got {call_kwargs.get('max_size')!r}"
    )


# ── GEPA wrapper integration ────────────────────────────────────────────


def test_skill_fitness_metric_for_gepa_uses_llm_judge(monkeypatch):
    """The GEPA wrapper must call the LLM-judge path through skill_fitness_metric,
    not the old keyword overlap."""
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from evolution.core import fitness
    import dspy

    mock_judge_instance = MagicMock()
    mock_judge_instance.score.return_value = fitness.FitnessScore(
        correctness=0.8, procedure_following=0.7, conciseness=0.6,
        length_penalty=0.0, feedback=""
    )

    with patch.object(fitness, "LLMJudge", return_value=mock_judge_instance):
        example = dspy.Example(task_input="t", expected_behavior="r").with_inputs("task_input")
        prediction = dspy.Prediction(output="agent output")

        # 5-arg call (as GEPA would do internally)
        score = fitness.skill_fitness_metric_for_gepa(
            gold=example, pred=prediction, trace=None,
            pred_name="predict", pred_trace=None,
        )

    # Expected composite: 0.5*0.8 + 0.3*0.7 + 0.2*0.6 = 0.4 + 0.21 + 0.12 = 0.73
    assert abs(score - 0.73) < 0.01, (
        f"Expected score ≈ 0.73 (LLM-judge composite), got {score}. "
        f"Wrapper still using keyword overlap?"
    )
    # The judge.score() must have been called via skill_fitness_metric
    # (proving the wrapper actually invoked the LLM-judge path, not the
    # old keyword overlap).
    assert mock_judge_instance.score.called, (
        "LLMJudge.score() was not called from skill_fitness_metric_for_gepa — "
        "wrapper is still using keyword overlap, not the LLM-judge path."
    )
