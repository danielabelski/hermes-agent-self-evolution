"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    source: Optional[str] = None,
    output_dir_arg: Optional[str] = None,
    sessiondb_sources: Optional[list[str]] = None,
    max_skill_size: int = 15000,
    max_sessiondb_candidates: Optional[int] = None,
):
    """Main evolution function — orchestrates the full optimization loop.

    Two modes:
      * Default: --hermes-repo + --skill (finds skill via find_skill in repo/skills/)
      * Standalone: --source PATH (loads SKILL.md directly from disk)

    --source and --hermes-repo are mutually exclusive. --source bypasses
    HERMES_AGENT_REPO entirely — useful for optimizing ~/.hermes/skills/<name>/SKILL.md
    without copying it into a hermes-agent-style tree.
    """

    # ── 0a. Validate mode flags ──────────────────────────────────────
    if source and hermes_repo:
        console.print(
            "[red]✗ --source and --hermes-repo are mutually exclusive. "
            "Use one or the other.[/red]"
        )
        sys.exit(1)

    # ── 0b. Resolve output directory (CLI override or default ./output) ──
    resolved_output_dir = Path(output_dir_arg) if output_dir_arg else Path("./output")

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo) if not source else None,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        output_dir=resolved_output_dir,
        max_skill_size=max_skill_size,
    )

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    if source:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            console.print(f"[red]✗ --source file not found: {source_path}[/red]")
            sys.exit(1)
        if not source_path.is_file():
            console.print(f"[red]✗ --source path is not a file: {source_path}[/red]")
            sys.exit(1)
        skill_path = source_path
        console.print(f"  Mode: standalone (--source)")
    else:
        if not config.hermes_agent_path:
            console.print(
                "[red]✗ No --hermes-repo and no --source. Provide one.[/red]"
            )
            sys.exit(1)
        skill_path = find_skill(skill_name, config.hermes_agent_path)
        if not skill_path:
            console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
            sys.exit(1)
        console.print(f"  Mode: repo (--hermes-repo)")

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        # Default to hermes-state-db (our local SQLite + .usage.json cross-reference).
        # Override via --sessiondb-source for other upstream sources (e.g. legacy
        # hermes importer that reads ~/.hermes/sessions/*.json).
        source_list = sessiondb_sources if sessiondb_sources else ["hermes-state-db"]
        # Cap candidates before relevance filter to avoid hanging on huge
        # session histories. Default 50 matches upstream build_dataset_from_external
        # max_examples default. Override via --max-sessiondb-candidates or env
        # EVO_MAX_SESSIONDB_CANDIDATES. The cap is applied per-importer via
        # extract_messages(limit=N).
        candidate_limit = max_sessiondb_candidates or 50
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=source_list,
            output_path=save_path,
            model=eval_model,
            max_examples=candidate_limit,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    # BUGFIX 21.06.2026: was passing skill["body"] but the skill_structure check
    # looks for frontmatter (---, name:, description:) in the first 500 chars.
    # body never has those — frontmatter does. Must pass skill["raw"] (full file).
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy. The wrapper script (scripts/skill_optimize.sh) already
    # sources ~/.hermes/.env which sets OPENAI_API_BASE and OPENAI_API_KEY
    # (the latter is NINE_ROUTER_API_KEY auto-mapped). LiteLLM auto-detects
    # these from env — we must NOT pass them explicitly as kwargs to dspy.LM.
    # Verified: proc_0d9b68a2deae (env-only) gave 49 examples + 0.581 baseline;
    # proc_97d990fe99fd (explicit api_base) failed with APIConnectionError
    # because explicit api_base broke LiteLLM's internal model resolution.
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        # BUGFIX 21.06.2026: DSPy 3.2.1 GEPA signature changed. Old code passed
        # `max_steps=iterations` which is no longer a valid kwarg (TypeError in
        # constructor). Use `max_metric_calls` instead. Also need to pass
        # `reflection_lm` (required since DSPy 3.x — GEPA uses it to reflect on
        # proposed instructions) and the GEPA-specific 5-arg metric signature.
        # Fall back to MIPROv2 if GEPA fails for any reason.
        from evolution.core.fitness import skill_fitness_metric_for_gepa, _gepa_compatible_metric_stub

        reflection_lm = dspy.LM(optimizer_model)
        # Use the 5-arg stub so dspy.GEPA.__init__'s signature inspection passes.
        # At runtime, dspy.evaluate calls the metric with 2 args (example, prediction);
        # dspy.MIPROv2 with 3 args (example, prediction, trace). The wrapper above
        # handles all these cases via *args/**kwargs.
        optimizer = dspy.GEPA(
            metric=_gepa_compatible_metric_stub,
            max_metric_calls=max(1, iterations),
            reflection_lm=reflection_lm,
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version.
        # BUGFIX 21.06.2026: MIPROv2 in DSPy 3.x uses `auto` parameter (light/medium/heavy),
        # not `max_steps`. Pass `auto="light"` for fastest variant.
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        from evolution.core.fitness import skill_fitness_metric_for_gepa
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric_for_gepa,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    # BUGFIX 21.06.2026: pass skill["raw"] (full file) not skill["body"] —
    # skill_structure check needs frontmatter in text[:500]. The body alone
    # never has frontmatter; raw has both.
    evolved_full_for_check = reassemble_skill(skill["frontmatter"], evolved_body)
    evolved_constraints = validator.validate_all(
        evolved_full_for_check, "skill", baseline_text=skill["raw"]
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred, lm=lm)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score = skill_fitness_metric(ex, evolved_pred, lm=lm)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = config.output_dir / skill_name / timestamp
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir_path / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir_path / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "source": str(skill_path),
        "mode": "standalone" if source else "repo",
    }
    (output_dir_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir_path}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir_path}/baseline_skill.md {output_dir_path}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--source", default=None,
              help="Direct path to SKILL.md (standalone mode, bypasses --hermes-repo)")
@click.option("--output-dir", default=None,
              help="Output directory (default: ./output relative to CWD)")
@click.option("--sessiondb-source", default=None,
              help=("Override the sessiondb sources list (comma-separated). "
                    "Default when --eval-source sessiondb: 'hermes-state-db'. "
                    "Other available: 'claude-code', 'copilot', 'hermes' (legacy)."))
@click.option("--max-skill-size", default=15000, type=int,
              help=("Override the maximum size (in chars) for evolved skill bodies. "
                    "Default 15000. Use 50000+ for large skills like daniil-protocol."))
@click.option("--max-sessiondb-candidates", default=None, type=int,
              help=("Cap sessiondb candidates before relevance filter (avoids hanging "
                    "on huge histories). Default 50. Override via env EVO_MAX_SESSIONDB_CANDIDATES."))
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, source, output_dir, sessiondb_source, max_skill_size, max_sessiondb_candidates):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization.

    Two modes:
      * --hermes-repo PATH  Find skill in <PATH>/skills/<name>/SKILL.md
      * --source PATH        Load SKILL.md directly from PATH (standalone)

    --output-dir PATH overrides the default ./output location.

    --sessiondb-source accepts comma-separated list, e.g.
    --sessiondb-source 'hermes-state-db,claude-code'.
    """
    sessiondb_sources = None
    if sessiondb_source:
        sessiondb_sources = [s.strip() for s in sessiondb_source.split(",") if s.strip()]
    # CLI flag > env var > default (50)
    if max_sessiondb_candidates is None:
        max_sessiondb_candidates = int(os.getenv("EVO_MAX_SESSIONDB_CANDIDATES", "50"))

    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        source=source,
        output_dir_arg=output_dir,
        sessiondb_sources=sessiondb_sources,
        max_skill_size=max_skill_size,
        max_sessiondb_candidates=max_sessiondb_candidates,
    )


if __name__ == "__main__":
    main()
