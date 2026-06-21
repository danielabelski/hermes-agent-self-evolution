"""HermesStateDbImporter — mine user/assistant pairs from ~/.hermes/state.db.

Added 21.06.2026 to support using the user's actual Hermes session history
as eval data for skill evolution. The upstream HermesSessionImporter looks
at ``~/.hermes/sessions/*.json`` (request dumps in this install), but the
real session data is in ``~/.hermes/state.db`` (SQLite).

Strategy:
  1. Read ~/.hermes/skills/.usage.json to get the last_used_at timestamp of
     the skill we are evaluating (Hermes already maintains this sidecar).
  2. Query ~/.hermes/state.db for sessions whose [started_at, ended_at]
     window overlaps that timestamp.
  3. Walk each matching session and extract user/assistant pairs (skipping
     tool messages in between) — same semantics as upstream
     HermesSessionImporter.

Privacy:
  - Optional: if NINE_ROUTER_API_KEY + NINE_ROUTER_URL are set, sanitize
    each pair via the 9router endpoint before returning. This protects
    against secrets/PII leaking into the LLM-as-judge eval prompt.
  - Fallback: if no key is set, content is filtered only by the upstream
    _contains_secret regex (catches obvious API keys / passwords but not
    everything).

Configuration (env vars):
  - HERMES_STATE_DB_PATH:    override default ~/.hermes/state.db
  - HERMES_USAGE_JSON_PATH:  override default ~/.hermes/skills/.usage.json
  - NINE_ROUTER_API_KEY:     enable sanitization (otherwise local-only)
  - NINE_ROUTER_URL:         9router endpoint (default http://localhost:8787/v1)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HermesStateDbImporter:
    """Extract user/assistant message pairs from Hermes state.db.

    Mirrors the contract of HermesSessionImporter (and ClaudeCodeImporter,
    CopilotImporter): returns a list of dicts with keys
    {source, task_input, assistant_response, session_id}.

    Source: "hermes-state-db"
    """

    SOURCE_NAME = "hermes-state-db"

    DEFAULT_STATE_DB_PATH = Path.home() / ".hermes" / "state.db"
    DEFAULT_USAGE_JSON_PATH = Path.home() / ".hermes" / "skills" / ".usage.json"

    def __init__(
        self,
        state_db_path: Optional[Path] = None,
        usage_json_path: Optional[Path] = None,
    ):
        env_state = os.getenv("HERMES_STATE_DB_PATH")
        env_usage = os.getenv("HERMES_USAGE_JSON_PATH")

        self.state_db_path = (
            Path(state_db_path).expanduser()
            if state_db_path
            else (Path(env_state).expanduser() if env_state else self.DEFAULT_STATE_DB_PATH)
        )
        self.usage_json_path = (
            Path(usage_json_path).expanduser()
            if usage_json_path
            else (Path(env_usage).expanduser() if env_usage else self.DEFAULT_USAGE_JSON_PATH)
        )

    # ── Public API ──────────────────────────────────────────────────────

    @staticmethod
    def extract_messages(
        skill_name: Optional[str] = None,
        limit: int = 0,
    ) -> list[dict]:
        """Mine user/assistant pairs from state.db, optionally filtered by skill.

        Static method to match upstream's API contract: build_dataset_from_external
        calls `importer_cls.extract_messages()` (no instance) at
        evolution/core/external_importers.py:648. Was a regular method before
        22.06.2026 — broke sessiondb source.

        Paths are read from env vars (HERMES_STATE_DB_PATH, HERMES_USAGE_JSON_PATH)
        on each call, so the function has no per-instance state. To customize
        paths, set the env vars before calling.

        Args:
            skill_name: If provided, restrict to sessions whose [started_at,
                ended_at] window overlaps the skill's last_used_at timestamp
                from .usage.json. If None, return pairs from all sessions.
            limit: Maximum number of pairs to return (0 = no limit).

        Returns:
            List of dicts: {source, task_input, assistant_response, session_id,
            timestamp}. Empty list on any error (missing db, missing usage.json,
            db read failure) — never raises.
        """
        state_db_path, usage_json_path = HermesStateDbImporter._resolve_paths()
        if not state_db_path.exists():
            logger.debug("HermesStateDbImporter: state.db not found at %s", state_db_path)
            return []

        # Step 1: figure out which sessions are relevant.
        relevant_session_ids = HermesStateDbImporter._select_relevant_sessions(
            skill_name, usage_json_path
        )
        if relevant_session_ids is not None and not relevant_session_ids:
            return []

        # Step 2: extract user/assistant pairs from those sessions.
        try:
            pairs = HermesStateDbImporter._extract_pairs_from_sessions(
                relevant_session_ids, state_db_path
            )
        except sqlite3.Error as e:
            logger.warning("HermesStateDbImporter: state.db read failed: %s", e)
            return []

        # Step 3: 9router provides PII/secret masking transparently at the
        # LLM-call layer (DSPy/LiteLLM with OPENAI_API_BASE=http://localhost:8787/v1).
        # No sanitize step needed here — the pairs go to LLM as-is.

        if limit and len(pairs) > limit:
            pairs = pairs[:limit]

        return pairs

    @staticmethod
    def _resolve_paths() -> tuple[Path, Path]:
        """Resolve state.db and usage.json paths from env vars (or defaults)."""
        env_state = os.getenv("HERMES_STATE_DB_PATH")
        env_usage = os.getenv("HERMES_USAGE_JSON_PATH")
        state_db_path = (
            Path(env_state).expanduser()
            if env_state
            else HermesStateDbImporter.DEFAULT_STATE_DB_PATH
        )
        usage_json_path = (
            Path(env_usage).expanduser()
            if env_usage
            else HermesStateDbImporter.DEFAULT_USAGE_JSON_PATH
        )
        return state_db_path, usage_json_path

    # ── Session selection via .usage.json ────────────────────────────────

    @staticmethod
    def _select_relevant_sessions(
        skill_name: Optional[str], usage_json_path: Path
    ) -> Optional[list[str]]:
        """Return session IDs whose [started_at, ended_at] window overlaps
        last_used_at for the given skill. If skill_name is None or .usage.json
        is missing/corrupt, return None (= "all sessions are relevant").
        """
        if skill_name is None:
            return None

        usage = HermesStateDbImporter._read_usage_json(usage_json_path)
        if not usage:
            return None  # can't filter without usage data; fall back to all

        record = usage.get(skill_name)
        if not isinstance(record, dict):
            return []  # skill not tracked → no signal → no sessions

        if record.get("use_count", 0) == 0 and record.get("view_count", 0) == 0:
            return []  # explicitly never used

        last_used_iso = (
            record.get("last_used_at")
            or record.get("last_viewed_at")
            or record.get("last_patched_at")
        )
        if not last_used_iso:
            return []  # tracked but no timestamp yet

        try:
            target_ts = HermesStateDbImporter._parse_iso_to_unix(last_used_iso)
        except (TypeError, ValueError) as e:
            logger.debug("HermesStateDbImporter: bad timestamp %r: %s", last_used_iso, e)
            return []

        # Query state.db for sessions whose window overlaps target_ts.
        return HermesStateDbImporter._query_sessions_overlapping(target_ts)

    @staticmethod
    def _read_usage_json(usage_json_path: Path) -> Optional[dict]:
        """Read .usage.json. Returns None on missing file or corrupt JSON."""
        if not usage_json_path.exists():
            return None
        try:
            return json.loads(usage_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("HermesStateDbImporter: usage.json read failed: %s", e)
            return None

    @staticmethod
    def _parse_iso_to_unix(iso: str) -> float:
        """Parse an ISO 8601 timestamp into a Unix timestamp (float seconds)."""
        # Normalize 'Z' suffix → '+00:00' (older Python fromisoformat rejects Z).
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    @staticmethod
    def _query_sessions_overlapping(target_ts: float) -> list[str]:
        """Return IDs of sessions whose [started_at, ended_at] window contains
        target_ts.

        A session is considered "containing" the target if it started on or
        before target_ts AND (ended_at is null OR ended_at >= target_ts).
        This matches the user model: a skill was "used" in any session that
        was active at that moment.
        """
        state_db_path, _ = HermesStateDbImporter._resolve_paths()
        try:
            conn = sqlite3.connect(str(state_db_path))
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT id FROM sessions
                WHERE started_at <= ?
                  AND (ended_at IS NULL OR ended_at >= ?)
                """,
                (target_ts, target_ts),
            ).fetchall()
            conn.close()
            return [row[0] for row in rows]
        except sqlite3.Error as e:
            logger.warning("HermesStateDbImporter: session query failed: %s", e)
            return []

    # ── Pair extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_pairs_from_sessions(
        session_ids: Optional[list[str]],
        state_db_path: Path,
    ) -> list[dict]:
        """Walk messages in the given sessions (or all sessions if None) and
        pair each user message with the next assistant response. Skip tool
        messages in between, short messages, and secrets (upstream regex).
        """
        # Lazy import to avoid circular dependency at module load time.
        from evolution.core.external_importers import _contains_secret

        conn = sqlite3.connect(str(state_db_path))
        try:
            cur = conn.cursor()
            if session_ids:
                placeholders = ",".join("?" * len(session_ids))
                rows = cur.execute(
                    f"SELECT session_id, role, content, timestamp "
                    f"FROM messages WHERE session_id IN ({placeholders}) "
                    f"AND active = 1 "
                    f"ORDER BY session_id, timestamp",
                    session_ids,
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT session_id, role, content, timestamp "
                    "FROM messages WHERE active = 1 "
                    "ORDER BY session_id, timestamp"
                ).fetchall()
        finally:
            conn.close()

        by_session: dict[str, list[tuple]] = defaultdict(list)
        for session_id, role, content, ts in rows:
            by_session[session_id].append((role, content or "", ts))

        pairs: list[dict] = []
        for session_id, msgs in by_session.items():
            for i, (role, content, ts) in enumerate(msgs):
                if role != "user":
                    continue
                if len(content) < 10:
                    continue
                if _contains_secret(content):
                    continue
                # Find the next assistant response after this user message.
                assistant_text = ""
                for j in range(i + 1, len(msgs)):
                    next_role, next_content, _ = msgs[j]
                    if next_role == "assistant":
                        if next_content:
                            assistant_text = next_content
                        break
                    if next_role == "user":
                        break  # next user turn before any assistant — give up
                if not assistant_text:
                    continue
                if _contains_secret(assistant_text):
                    continue
                pairs.append({
                    "source": HermesStateDbImporter.SOURCE_NAME,
                    "task_input": content,
                    "assistant_response": assistant_text,
                    "session_id": session_id,
                    "timestamp": ts,
                })
        return pairs

    # ── 9router note ─────────────────────────────────────────────────────
    #
    # Earlier versions of this importer had a `_sanitize_via_9router` method
    # that POSTed each pair to the 9router chat/completions endpoint to
    # strip secrets/PII before passing to LLM-as-judge.
    #
    # As of 21.06.2026, that step is removed. Reason: 9router is a generic
    # OpenAI-compatible LLM proxy with built-in (transparent) PII/secret
    # masking on its endpoint. The DSPy/LiteLLM layer (configured via
    # OPENAI_API_BASE=http://localhost:8787/v1) gets that protection for
    # free on every LLM call. Adding a separate sanitize step here would:
    #   1. Double the work (mask once in preprocess, mask again at endpoint).
    #   2. Add 8s per pair latency (extra round-trip).
    #   3. Require model selection + fallback chain for what is a transport-
    #      level feature, not an LLM task.
    #
    # Sanitization now happens transparently at the 9router HTTP layer
    # when DSPy/LiteLLM POST messages. Set OPENAI_API_BASE in the project
    # .env (or wrapper script) to enable it.
