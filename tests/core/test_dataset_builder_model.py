"""RED test: DatasetBuilder must use eval_model, not judge_model default.

22.06.2026: DatasetBuilder line 126 used self.config.judge_model which
defaults to "openai/gpt-4.1". When wrapper passes --eval-model
minimax/MiniMax-M2.5 to LLMJudge/RelevanceFilter (which DO use eval_model),
DatasetBuilder still uses judge_model default and fails on 9router with
NotFoundError: model_not_found for provider: openai.

Fix: DatasetBuilder must use config.eval_model (the CLI-flag-passed model),
not config.judge_model. Or: both should default to the same model that
works with 9router.
"""
import os
from pathlib import Path


def test_synthetic_dataset_builder_uses_eval_model_not_judge_model():
    """DatasetBuilder must call dspy.LM(self.config.eval_model), not
    dspy.LM(self.config.judge_model). The judge_model default
    'openai/gpt-4.1' is not in 9router's catalog."""
    from evolution.core.dataset_builder import SyntheticDatasetBuilder

    src = Path(
        "/home/daniel/HermesProjects/default/hermes-agent-self-evolution/evolution/core/dataset_builder.py"
    ).read_text()

    # Look for any dspy.LM(self.config.<field>) call in the file.
    # The forbidden pattern is judge_model (not in 9router).
    assert "dspy.LM(self.config.judge_model)" not in src, (
        "SyntheticDatasetBuilder still uses self.config.judge_model "
        "which defaults to 'openai/gpt-4.1' — this model is NOT in 9router's "
        "catalog and causes NotFoundError. Use self.config.eval_model instead "
        "(which is passed via --eval-model CLI flag and propagated to the "
        "real LLM call)."
    )


def test_evolution_config_eval_model_is_used_by_synthetic_builder():
    """Confirm the synthetic builder uses eval_model. This is the positive
    assertion (paired with the negative above)."""
    from evolution.core.dataset_builder import SyntheticDatasetBuilder

    src = Path(
        "/home/daniel/HermesProjects/default/hermes-agent-self-evolution/evolution/core/dataset_builder.py"
    ).read_text()

    # Synthetic builder must use self.config.eval_model.
    assert "dspy.LM(self.config.eval_model)" in src, (
        "SyntheticDatasetBuilder should call dspy.LM(self.config.eval_model) — "
        "same model LLMJudge and RelevanceFilter use."
    )
