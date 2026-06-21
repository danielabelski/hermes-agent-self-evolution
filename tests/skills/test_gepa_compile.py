"""RED tests for GEPA optimizer integration (21.06.2026).

Real run on daniil-protocol hit:
  File "dspy/teleprompt/mipro_optimizer_v2.py", line ...
  ...
  ImportError: MIPROv2 requires optional dependency 'optuna'.

Which was the FALLBACK error (after GEPA failed). The original GEPA
exception was hidden by the `except Exception` in evolve_skill.py:166-176.

These tests reproduce the GEPA path directly so we can see the real
exception and fix root cause, not just the fallback.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock


def test_gepa_compile_does_not_raise():
    """Direct dspy.GEPA(...).compile() call with our model + metric + reflection_lm.

    Must not raise. If this fails, real evolution also fails at the same point.
    Locks in the full fix: metric wrapper + max_metric_calls + reflection_lm.
    """
    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8787/v1")
    os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-test")

    import dspy
    from evolution.core.fitness import skill_fitness_metric_for_gepa
    from evolution.skills.skill_module import SkillModule

    # Mirror real run: 1 trial, 1 train example, 1 val example, simple body.
    lm = dspy.LM("minimax/MiniMax-M2.5")
    dspy.configure(lm=lm)

    baseline = SkillModule("# Test skill body. Do the thing step by step.")
    trainset = [
        dspy.Example(
            task_input="task one",
            expected_behavior="do thing correctly",
        ).with_inputs("task_input")
    ]
    valset = [
        dspy.Example(
            task_input="task two",
            expected_behavior="do other thing",
        ).with_inputs("task_input")
    ]

    optimizer = dspy.GEPA(
        metric=skill_fitness_metric_for_gepa,
        max_metric_calls=1,
        reflection_lm=lm,  # required in DSPy 3.x
    )

    # The line that failed in real run. We catch and re-raise with
    # full traceback for visibility.
    try:
        result = optimizer.compile(
            baseline,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        import traceback
        pytest.fail(
            f"dspy.GEPA.compile() raised: {type(e).__name__}: {e}\n"
            f"Full traceback:\n{traceback.format_exc()}"
        )
