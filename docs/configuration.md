# Configuration

This project reads LLM proxy configuration from **two files**:

1. **`~/.hermes/.env`** — global Hermes secret store (API keys, owned by
   Hermes; project reads but does not write the keys themselves).
2. **`.env`** in the project root — project-local config (base URL, model).

## Quick start

```bash
./configure.sh        # interactive: 3 questions (base URL, model, key)
```

The script writes both files. Re-run any time to point the project at a
different LLM proxy.

## Show current config

```bash
./configure.sh --show
```

## Files

### `~/.hermes/.env` (global secret store)

Holds the **API key** for the LLM proxy. Hermes manages this file —
don't hand-edit unless you know what you're doing.

```bash
# Examples of keys in this file:
NINE_ROUTER_API_KEY=***      # 9router (localhost:8787)
OPENROUTER_API_KEY=***       # openrouter.ai
GOOGLE_API_KEY=***           # Google AI
ANTHROPIC_API_KEY=***        # Anthropic
```

The wrapper (`scripts/skill_optimize.sh`) auto-maps these to `OPENAI_API_KEY`
if needed (DSPy/LiteLLM require this specific name).

### `.env` (project-local config)

Holds the **base URL** and **default model**.

```bash
OPENAI_API_BASE=http://localhost:8787/v1
EVAL_MODEL=minimax/MiniMax-M2.7
```

## Switch to a different LLM proxy

1. `./configure.sh` — answer the 3 prompts with the new proxy's URL, model, and key.
2. Done. The next invocation of `scripts/skill_optimize.sh` uses the new proxy.

No code changes needed.

## How the wrapper reads config

```bash
# Inside scripts/skill_optimize.sh:
load_env_file "$HOME/.hermes/.env"   # secrets (API keys)
load_env_file "$PROJECT_DIR/.env"    # project config (base URL, model)

# Auto-map: NINE_ROUTER_API_KEY → OPENAI_API_KEY
# (only if OPENAI_API_KEY is unset; OPENAI_API_KEY wins if both are set)
```

The wrapper uses `set -a; . file; set +a` so vars are auto-exported to the
python subprocess **without** polluting the caller shell. See
`tests/test_skill_optimize_wrapper.py` for the contract.

## Models

Tested on 9router (per `evolution/skills/skill_optimize.sh` and `configure.sh`
prompts):

| Model | Notes |
|---|---|
| `minimax/MiniMax-M2.7` | **Default.** JSON-clean. Recommended for judge/eval. |
| `minimax/MiniMax-M2.5` | Older, sometimes returns prose instead of JSON → `AdapterParseError`. |
| `openai/gpt-4.1-mini` | Works on 9router (LiteLLM routes `openai/` provider). |
| `haiku`, `sonnet`, `opus` | Bare aliases — 9router serves them, but LiteLLM doesn't recognize the implicit provider → `NotFoundError: provider: openai`. |

## Troubleshooting

### "No active credentials for provider: openai"

The wrapper didn't get an API key. Check:

```bash
grep "NINE_ROUTER_API_KEY" ~/.hermes/.env       # must be uncommented + non-empty
echo $NINE_ROUTER_API_KEY | head -c 7           # must start with "sk-" or similar
./configure.sh --show | head -10              # verify what wrapper sees
```

### "AdapterParseError: LM response cannot be serialized to JSON"

Model returned prose instead of JSON. Switch to `minimax/MiniMax-M2.7`
(the only one we tested that consistently follows JSON format on 9router).

### "RuntimeError: cannot schedule new futures after shutdown"

DSPy parallelizer race condition in relevance filter. Reduce candidates:

```bash
bash scripts/skill_optimize.sh \
    --eval-source sessiondb \
    --max-sessiondb-candidates 10 \
    ...
```

## Files written by configure.sh

| File | Owner | Contains |
|---|---|---|
| `~/.hermes/.env` | Hermes | API keys (e.g. `NINE_ROUTER_API_KEY`) |
| `./.env` | This project | `OPENAI_API_BASE`, `EVAL_MODEL` |

Both are in `.gitignore` so secrets don't leak.
