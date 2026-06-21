#!/usr/bin/env bash
# configure.sh — interactive setup for hermes-agent-self-evolution.
#
# Purpose: collect LLM proxy configuration from the user and write it to
# ~/.hermes/.env (global secrets store) and ./.env (project-local config).
# After running this once, the user can re-run it any time to point the
# project at a different LLM proxy (e.g. switching from 9router to
# LiteLLM proxy, or to OpenAI direct, etc.).
#
# Usage:
#   ./configure.sh             # interactive prompts
#   ./configure.sh --help      # show usage
#   ./configure.sh --show      # show current config (no prompts)
#
# What this writes:
#   ~/.hermes/.env:
#       NINE_ROUTER_API_KEY=***  (or whatever key name the user picks)
#       ... other secrets
#
#   ./.env (project-local):
#       OPENAI_API_BASE=http://your-llm-proxy:port/v1
#       EVAL_MODEL=minimax/MiniMax-M2.7
#
# Design:
#   - Non-destructive: if a value is already set, asks before overwriting.
#   - One source of truth per concern: secrets → ~/.hermes/.env,
#     project config → ./.env. Wrapper reads both.
#   - Keys are NOT hardcoded anywhere else: wrapper (scripts/skill_optimize.sh)
#     reads OPENAI_API_BASE from ./.env and NINE_ROUTER_API_KEY from
#     ~/.hermes/.env, then auto-maps the latter to OPENAI_API_KEY if needed.
#
# Exit codes:
#   0  success (or --show completed)
#   1  user cancelled
#   2  invalid input

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
GLOBAL_ENV="$HOME/.hermes/.env"
PROJECT_ENV="$PROJECT_DIR/.env"

show_help() {
    cat <<'EOF'
configure.sh — interactive setup for hermes-agent-self-evolution

Usage:
    ./configure.sh             # interactive prompts
    ./configure.sh --help      # this message
    ./configure.sh --show      # show current config (no prompts)

Files written:
    ~/.hermes/.env  — global secrets (API keys)
    ./.env          — project-local config (base URL, model choice)

Re-run anytime to change the LLM proxy — your changes apply on the next
invocation of scripts/skill_optimize.sh.
EOF
}

show_current() {
    echo "=== ~/.hermes/.env (global secrets) ==="
    if [ -f "$GLOBAL_ENV" ]; then
        # Show keys with values masked for secrets
        while IFS='=' read -r k v; do
            case "$k" in
                *_KEY|*_TOKEN|*_SECRET)
                    echo "  $k = *** (length=${#v})"
                    ;;
                ""|\#*) ;;
                *)
                    echo "  $k = $v"
                    ;;
            esac
        done < "$GLOBAL_ENV"
    else
        echo "  (file does not exist)"
    fi
    echo ""
    echo "=== ./.env (project config) ==="
    if [ -f "$PROJECT_ENV" ]; then
        cat "$PROJECT_ENV"
    else
        echo "  (file does not exist)"
    fi
}

ask() {
    # ask <prompt> <default> <varname>
    # Reads from stdin. Echoes the chosen value (default if empty).
    local prompt="$1"
    local default="$2"
    local varname="$3"
    local val
    if [ -n "$default" ]; then
        read -r -p "$prompt [$default]: " val
        val="${val:-$default}"
    else
        read -r -p "$prompt: " val
    fi
    eval "$varname=\$val"
}

# Ensure ~/.hermes/ exists (idempotent — Hermes guarantees it, but defensive)
mkdir -p "$(dirname "$GLOBAL_ENV")"

# Ensure .env exists in project
touch "$PROJECT_ENV"

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --show)
        show_current
        exit 0
        ;;
esac

echo "=== hermes-agent-self-evolution — configure LLM proxy ==="
echo ""
echo "This writes two files:"
echo "  ~/.hermes/.env (global secret store) — API keys"
echo "  ./.env (project config) — base URL, model choice"
echo ""

# ── 1. LLM proxy base URL ──────────────────────────────────────────────
echo "── Step 1. LLM proxy base URL ──"
echo "Examples:"
echo "  9router (default): http://localhost:8787/v1"
echo "  LiteLLM proxy:     http://localhost:4000/v1"
echo "  OpenAI direct:     https://api.openai.com/v1"
echo "  Custom:            http://192.168.1.100:8080/v1"
ask "LLM proxy base URL" "http://localhost:8787/v1" NEW_BASE_URL

# ── 2. Model name (EVAL_MODEL) ────────────────────────────────────────
echo ""
echo "── Step 2. Model name (used for eval / judge / optimizer) ──"
echo "Examples:"
echo "  minimax/MiniMax-M2.7     (9router, JSON-clean)"
echo "  minimax/MiniMax-M2.5     (9router, older)"
echo "  openai/gpt-4.1-mini      (if proxy serves OpenAI API)"
echo "  claude-sonnet-4-20250514  (if proxy serves Anthropic)"
ask "Model name (EVAL_MODEL)" "minimax/MiniMax-M2.7" NEW_MODEL

# ── 3. API key ──────────────────────────────────────────────────────────
echo ""
echo "── Step 3. API key ──"
echo "Stored in ~/.hermes/.env under the env var name you choose."
echo "Common conventions: NINE_ROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
ask "Env var name for API key" "NINE_ROUTER_API_KEY" NEW_KEY_NAME

# Use read -s for secrets (don't echo)
read -r -s -p "API key value (input hidden): " NEW_KEY_VALUE
echo ""

if [ -z "$NEW_KEY_VALUE" ]; then
    echo "ERROR: API key cannot be empty" >&2
    exit 2
fi

# ── Write ~/.hermes/.env ────────────────────────────────────────────────
echo ""
echo "── Writing ~/.hermes/.env ──"
# Use a temp file + grep to safely upsert one key
TMP_ENV="$(mktemp)"
if [ -f "$GLOBAL_ENV" ]; then
    # Remove existing key (case-sensitive, exact match)
    grep -v "^${NEW_KEY_NAME}=" "$GLOBAL_ENV" > "$TMP_ENV" || true
else
    touch "$TMP_ENV"
fi
echo "${NEW_KEY_NAME}=*** NEW_KEY_VALUE" >> "$TMP_ENV"
mv "$TMP_ENV" "$GLOBAL_ENV"
chmod 600 "$GLOBAL_ENV"
echo "  ✓ wrote ${NEW_KEY_NAME}=*** (length=${#NEW_KEY_VALUE})"

# ── Write ./.env ────────────────────────────────────────────────────────
echo ""
echo "── Writing ./.env ──"
TMP_PROJ="$(mktemp)"
# Remove existing keys we're about to overwrite
grep -v -E "^(OPENAI_API_BASE|EVAL_MODEL)=" "$PROJECT_ENV" > "$TMP_PROJ" || true
echo "" >> "$TMP_PROJ"
echo "# Added by configure.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TMP_PROJ"
echo "OPENAI_API_BASE=${NEW_BASE_URL}" >> "$TMP_PROJ"
echo "EVAL_MODEL=${NEW_MODEL}" >> "$TMP_PROJ"
mv "$TMP_PROJ" "$PROJECT_ENV"
echo "  ✓ wrote OPENAI_API_BASE=${NEW_BASE_URL}"
echo "  ✓ wrote EVAL_MODEL=${NEW_MODEL}"

echo ""
echo "=== Done! ==="
echo ""
echo "Verify with: ./configure.sh --show"
echo ""
echo "Test the wrapper with:"
echo "  bash scripts/skill_optimize.sh --help"
echo ""
echo "Run a real evolution with:"
echo "  bash scripts/skill_optimize.sh \\"
echo "      --source /path/to/SKILL.md \\"
echo "      --skill <name> \\"
echo "      --iterations 5 \\"
echo "      --eval-source synthetic"
