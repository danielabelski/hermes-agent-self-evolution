# hermes-agent-self-evolution (fork)

> **Fork** of [NousResearch/hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution)
> customized for danielabelski/Hermes Agent setup (9router + state.db mining).
>
> Upstream README still applies. See [PLAN.md](PLAN.md) for architecture
> and [docs/configuration.md](docs/configuration.md) for our custom setup.

## What's different from upstream

This fork adds:

- **Standalone mode**: `--source PATH` and `--output-dir PATH` to evolve
  a SKILL.md without needing a full `hermes-agent` repo checkout.
- **Custom HermesStateDbImporter** (`evolution/core/hermes_state_db.py`):
  mines user/assistant pairs from `~/.hermes/state.db` filtered by
  `~/.hermes/skills/.usage.json` (skips sessions not in the skill's
  usage window).
- **9router integration** via `scripts/skill_optimize.sh` wrapper:
  - Sources `~/.hermes/.env` (NINE_ROUTER_API_KEY) and project `.env`
    (OPENAI_API_BASE, EVAL_MODEL).
  - Auto-maps NINE_ROUTER_API_KEY → OPENAI_API_KEY for DSPy/LiteLLM.
  - `set -a; . file; set +a` pattern: vars auto-export to python
    subprocess only, never to caller shell.
- **`./configure.sh`**: interactive setup for changing LLM proxy.
  Re-run anytime to switch from 9router to LiteLLM proxy, OpenAI direct,
  or any OpenAI-compatible API.
- **Custom CLI flags**:
  - `--max-skill-size N` — default 15000, override per-run.
  - `--max-sessiondb-candidates N` — cap candidates before relevance
    filter (avoids DSPy parallelizer race on large sessions).
- **LLM-as-judge fitness** (`evolution/core/fitness.py`): correctness +
  procedure_following + conciseness composite.
- **`SkillModule` architecture fix** (`evolution/skills/skill_module.py`):
  skill text is embedded in `dspy.Signature` docstring so GEPA can
  actually mutate it (otherwise skill files stay byte-identical).

## Quick start

```bash
# 1. Clone this fork
git clone https://github.com/danielabelski/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution

# 2. Configure LLM proxy (one-time, or whenever you switch)
./configure.sh

# 3. Install
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 4. Run evolution (synthetic eval data, no LLM cost for setup)
bash scripts/skill_optimize.sh \
  --source /home/daniel/.hermes/skills/daniil-protocol/SKILL.md \
  --skill daniil-protocol \
  --iterations 5 \
  --eval-source synthetic

# 5. Or use real session history from Hermes state.db
bash scripts/skill_optimize.sh \
  --source /home/daniel/.hermes/skills/daniil-protocol/SKILL.md \
  --skill daniil-protocol \
  --iterations 30 \
  --max-sessiondb-candidates 10 \
  --max-skill-size 60000 \
  --eval-source sessiondb
```

## Switching to a different LLM proxy

```bash
./configure.sh            # answer 3 prompts
# (or)
./configure.sh --show     # see current config
```

See [docs/configuration.md](docs/configuration.md) for details.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

204 tests passing (as of 2026-06-21).

## Sync with upstream

This fork diverges from upstream — rebase carefully. Suggested workflow:

```bash
git remote add upstream https://github.com/NousResearch/hermes-agent-self-evolution.git
git fetch upstream
git rebase upstream/main
# Resolve conflicts (likely in evolution/skills/evolve_skill.py and
# evolution/core/fitness.py where we made the most changes).
```

We do **not** auto-merge upstream changes — the local customizations
(standalone mode, hermes-state-db, 9router integration) are
upstream-incompatible.

## Files of interest

```
configure.sh                       # interactive setup
scripts/skill_optimize.sh          # wrapper (sources .env, activates venv, invokes evolution)
evolution/core/hermes_state_db.py # custom sessiondb importer
evolution/core/fitness.py          # LLMJudge (3-dim composite)
evolution/skills/skill_module.py   # SkillModule with skill_text in docstring
docs/configuration.md              # config docs
.env.example                       # template (if we add one)
```

## Status (as of 2026-06-21)

- Phase 1 (skill files): end-to-end pipeline working.
- Real run on `daniil-protocol` skill: 30 iterations, M2.7 model,
  constraint gate 4/4 PASS, holdout eval 0.525 (52.5%) baseline.
  Skill already well-written — no improvement found by GEPA.
- 9 прогонов, 4 bugfixes, 12 new tests за сессию 21-22.06.2026.
