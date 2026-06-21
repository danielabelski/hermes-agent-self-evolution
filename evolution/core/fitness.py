"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import os

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig, lm: Optional[dspy.LM] = None):
        self.config = config
        # Explicit LM passed by caller (preferred — guarantees 9router routing).
        # When None, score() falls back to dspy.LM(self.config.eval_model).
        self.lm = lm
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        # Use explicit self.lm if provided (caller pre-built it with correct
        # api_base/api_key for 9router). Fall back to constructing dspy.LM
        # from self.config.eval_model (legacy path, may fail with 9router
        # because LiteLLM routes "openai/gpt-4.1-mini" to api.openai.com
        # which we have no credentials for).
        lm_to_use = self.lm if self.lm is not None else dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm_to_use):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    artifact_size: Optional[int] = None,
    max_size: Optional[int] = None,
    eval_model: Optional[str] = None,
    lm: Optional[dspy.LM] = None,
) -> float:
    """DSPy-compatible metric function for skill optimization.

    Uses LLM-as-judge (LLMJudge class) to score agent output against the
    expected behavior rubric. The composite score is computed from three
    dimensions: correctness (50%), procedure_following (30%), conciseness (20%),
    with an optional length_penalty applied for skills near their size limit.

    Changed 22.06.2026: was using keyword overlap (a poor proxy for skill
    quality — paraphrased improvements got low scores). LLMJudge was already
    implemented at line 34 but never wired up. This change unlocks the
    pre-existing infrastructure.

    Args:
        example: dspy.Example with task_input and expected_behavior.
        prediction: dspy.Prediction with 'output' field.
        trace: optional DSPy trace (unused, present for API compat).
        artifact_size: optional size of evolved skill body in chars (for length penalty).
        max_size: optional max size for length penalty (typically from EvolutionConfig).
        eval_model: model string for LLM-judge (e.g. "minimax/MiniMax-M2.5"). If None,
            falls back to EvolutionConfig default. Pass explicitly when using
            9router — otherwise LiteLLM tries to route to "openai" provider
            which 9router doesn't have credentials for.

    Returns:
        Float 0-1, FitnessScore.composite.
    """
    agent_output = getattr(prediction, "output", "") or ""

    if not agent_output.strip():
        return 0.0

    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    skill_text = getattr(example, "skill_text", "") or ""

    # Use LLM-as-judge for fitness scoring. The default config has eval_model
    # fallback for stand-alone use; in evolve_skill.py the dspy.configure(lm=...)
    # upstream is what makes the actual LLM call work regardless of this fallback.
    from evolution.core.config import EvolutionConfig
    config = EvolutionConfig(hermes_agent_path=None)
    if eval_model:
        config.eval_model = eval_model
    # Pass explicit lm (preferred for 9router — caller already built it
    # with correct api_base/api_key). Fall back to dspy.LM(self.config.eval_model)
    # if no lm was passed (legacy path, see LLMJudge.score docstring).
    judge = LLMJudge(config=config, lm=lm)
    fit = judge.score(
        task_input=task,
        expected_behavior=expected,
        agent_output=agent_output,
        skill_text=skill_text,
        artifact_size=artifact_size,
        max_size=max_size,
    )
    return fit.composite


# Sentinel default config for the standalone skill_fitness_metric (used by
# relevance filter and unit tests). When called from evolve_skill.py, the
# EvolutionConfig has the right eval_model — but for stand-alone use this
# lets the function still work without explicit config.


def skill_fitness_metric_for_gepa(*args, **kwargs) -> float:
    """GEPA-compatible metric wrapper.

    DSPy 3.2.1 has a mix of metric signatures:
      * `dspy.evaluate.Evaluate` calls metric(example, prediction) — 2 args
      * `dspy.GEPA.__init__` does inspect.signature(metric).bind(None, None, None, None, None)
        requiring 5 args: (gold, pred, trace, pred_name, pred_trace)
      * `dspy.MIPROv2` calls metric(example, prediction, trace=None) — 3 args

    Our base `skill_fitness_metric` is 3-arg (example, prediction, trace).
    This wrapper accepts *args, **kwargs and forwards the first 3 positional
    to skill_fitness_metric. It satisfies:
      - the 2-arg call from dspy.evaluate (just (example, prediction))
      - the 3-arg call from MIPROv2 ((example, prediction, trace))
      - the 5-arg signature inspection in dspy.GEPA.__init__ — by accepting
        5 params via signature inspection below (separate stub function)

    Added 21.06.2026 to fix the GEPA integration bug where the
    `inspect.signature(metric).bind(None, None, None, None, None)` check
    in dspy.GEPA.__init__ failed because our metric only accepted 3 args.
    """
    # Forward positional args to skill_fitness_metric. GEPA may pass either:
    #   - 5 positional args (gold, pred, trace, pred_name, pred_trace)
    #   - 5 keyword args (gold=..., pred=..., trace=..., pred_name=..., pred_trace=...)
    # but skill_fitness_metric only needs 3 (example, prediction, trace).
    # Extra args (pred_name, pred_trace) are GEPA-specific metadata and not used.
    if len(args) >= 3:
        return skill_fitness_metric(
            args[0], args[1], args[2],
            eval_model=os.getenv("EVAL_MODEL", "openai/gpt-4.1-mini"),
        )
    elif len(args) == 2:
        return skill_fitness_metric(
            args[0], args[1], None,
            eval_model=os.getenv("EVAL_MODEL", "openai/gpt-4.1-mini"),
        )
    elif len(args) == 1:
        # Shouldn't happen in practice but be safe.
        return 0.0
    elif "gold" in kwargs or "pred" in kwargs:
        # GEPA call with keyword args. GEPA names its args gold/pred/trace;
        # skill_fitness_metric uses example/prediction/trace. Map them.
        return skill_fitness_metric(
            example=kwargs["gold"],
            prediction=kwargs["pred"],
            trace=kwargs.get("trace"),
            eval_model=os.getenv("EVAL_MODEL", "openai/gpt-4.1-mini"),
        )
    else:
        # Called with no args — also shouldn't happen.
        return 0.0


# This is a separate function used purely for the inspect.signature() check
# in dspy.GEPA.__init__. It exists so that the signature looks like 5 args
# (which dspy.GEPA requires), but the runtime path goes through
# skill_fitness_metric_for_gepa above. Keep them in sync if either changes.
#
# All params have default=None so that:
#   - inspect.signature(metric).bind(None, None, None, None, None) in
#     dspy.GEPA.__init__ succeeds (5 required params satisfied by None)
#   - Runtime calls with fewer args (e.g. dspy.evaluate passes only 2) also
#     work because missing args default to None
def _gepa_compatible_metric_stub(
    gold=None, pred=None, trace=None, pred_name=None, pred_trace=None
):
    """Stub with 5-arg signature for dspy.GEPA.__init__ signature check.

    NOT actually called at runtime — dspy.GEPA.compile() invokes the metric
    via the wrapper above (skill_fitness_metric_for_gepa). This stub exists
    only so that inspect.signature(...).bind(None, None, None, None, None)
    in GEPA's __init__ succeeds.

    Both functions are passed to dspy.GEPA via `metric=skill_fitness_metric_for_gepa`
    (the wrapper handles runtime, this stub handles signature). The wrapper
    must accept the 5-arg call too, but it ignores the extra args.
    """
    return skill_fitness_metric_for_gepa(gold, pred, trace, pred_name, pred_trace)


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
