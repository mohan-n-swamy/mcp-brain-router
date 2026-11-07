"""Backend implementations: DeepSeek, Codex, Grok, Kimi, Claude, Apple fm, Ollama.

Each backend is isolated; router.py calls these functions.
"""

import asyncio
import contextlib
import contextvars
import fcntl
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx

# One deadline per delegate call (lean-rig §6: `simple` 120 s, `worker` 15 min).
# server._delegate_role_impl sets it (time.monotonic() seconds); every backend
# clamps its own timeout to the time left. asyncio.to_thread copies the context,
# so the CLI threads see it too.
DEADLINE: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "brain_router_deadline", default=None
)


def _timeout(cap: float) -> float:
    """`cap` seconds, or the time left before the call's deadline if less."""
    deadline = DEADLINE.get()
    if deadline is None:
        return cap
    return max(1.0, min(cap, deadline - time.monotonic()))


async def _within(request, cap: float):
    """Await one HTTP request under a wall-clock bound. httpx's timeout bounds
    each read, so a body trickling in could outlive the call's deadline."""
    try:
        return await asyncio.wait_for(request, _timeout(cap))
    except asyncio.TimeoutError:
        raise httpx.ReadTimeout("wall-clock deadline passed") from None


class BackendError(Exception):
    """Base exception for backend errors."""

    def __init__(
        self,
        message: str,
        *,
        backend: Optional[str] = None,
        failure_kind: str = "backend_error",
        elapsed_ms: Optional[int] = None,
    ):
        if backend is not None:
            self.backend = backend
        self.failure_kind = failure_kind
        self.elapsed_ms = elapsed_ms
        super().__init__(message)


class BackendQuotaError(BackendError):
    """Raised only when a backend returns HTTP 429 quota/rate exhaustion.

    Carries the HTTP status and a sanitized message. For rate-limit bodies the
    provider message (e.g. GLM "Usage limit reached for 5 hour … reset at …")
    holds NO secret — only the reset time — so it is safe to surface and is
    what lets the orchestrator decide whether to wait.
    """

    def __init__(
        self, provider: str, status_code: int, message: str = "", reset_at: Optional[str] = None
    ):
        self.provider = provider
        self.backend = provider.lower()
        self.status_code = status_code
        self.message = message
        self.reset_at = reset_at
        detail = f" — {message}" if message else ""
        super().__init__(
            f"{provider} quota error {status_code}{detail}",
            backend=provider.lower(),
            failure_kind="quota_exhausted",
        )


# Resolve the codex binary to an ABSOLUTE path at import time. The MCP server
# is spawned by Claude Code with a minimal/empty env (`"env": {}`), so the
# subprocess PATH does NOT include `~/.npm-global/bin` where codex lives —
# bare "codex" → FileNotFoundError → adversarial tier fails instantly
# (root-caused 2026-07-05: 4/4 adversarial calls exhausted, 0 tokens).
# shutil.which honors PATH when present; the ~/.npm-global fallback covers the
# empty-PATH MCP case. Falls back to bare "codex" so a missing binary still
# raises the explicit "not found on PATH" BackendError, not an opaque one.
_CODEX_BIN = shutil.which("codex") or (
    os.path.expanduser("~/.npm-global/bin/codex")
    if os.path.exists(os.path.expanduser("~/.npm-global/bin/codex"))
    else "codex"
)

# Canonical lean Codex worker invocation — shared with install.py's smoke test
# so the two can never drift. Live A/B (2026-07-11): the full user rig used
# 23,145 tokens / ~69s for PONG; these flags used 14,907 / 10.6s. Delegated
# prompts are self-contained workers: no user config, rules, plugins, hooks,
# memories, apps, multi-agent fan-out, persisted session, MCPs, or repo context.
# Tools-off (lean-rig §6): no shell tool, read-only sandbox; live 2026-11-06 a
# "run uname -s" prompt answered NO_TOOLS. The caller adds `-C <empty dir>`.
_CODEX_EXEC_BASE_TMPL = [
    _CODEX_BIN,
    "exec",
    "--skip-git-repo-check",
    "--ignore-user-config",
    "--ignore-rules",
    "--ephemeral",
    "--sandbox",
    "read-only",
    "--disable",
    "plugins",
    "--disable",
    "hooks",
    "--disable",
    "memories",
    "--disable",
    "apps",
    "--disable",
    "multi_agent",
    "--disable",
    "shell_tool",
    # Codex memory v2: importing other agents' memories stays off (REJECTED §3).
    "--disable",
    "external_agent_memory_import",
    "-c",
    "mcp_servers={}",
    "-c",
    "__EFFORT__",
    "-c",
    'approval_policy="never"',
]

CODEX_TIMEOUT_SECONDS = 360  # 004 C410: 180s starved long adversarial prompts (2026-07-12)
# 004 C410: 30s starved GLM on >1k-token prompts (died at exactly the cap, reported
# as network_error). One constant, all HTTP backends — DRY.
HTTP_TIMEOUT_SECONDS = 120.0

# Agentic worker timeout — agentic mode does REAL file + check work (a full
# build loop step), so it runs longer than a text-only chat reply. Same as the
# codex adversarial cap: one constant, every agentic backend.
AGENTIC_TIMEOUT_SECONDS = 360


# ============================================================================
# CLI harness binaries for the agentic worker mode (spec 002).
# Resolved at import time so the empty-PATH MCP-process case still finds them.
# Each is the per-provider headless CLI the router shells to in the REAL cwd so
# the worker reads the spec, writes files, and runs checks itself.
# ============================================================================
def _resolve_bin(primary: str, fallback_dirs: tuple[str, ...] = ("~/.local/bin",)) -> str:
    """Resolve a CLI binary to an absolute path (honors shutil.which, then
    ~/.local/bin fallbacks for the empty-PATH MCP-process case). Falls back to
    the bare name so a missing binary raises the explicit not-found error."""
    resolved = shutil.which(primary)
    if resolved:
        return resolved
    for d in fallback_dirs:
        cand = os.path.expanduser(os.path.join(d, primary))
        if os.path.exists(cand):
            return cand
    return primary


_CC_BRAIN_BIN = _resolve_bin("cc-brain")
# xAI Grok CLI (grok.com OAuth login, no API key). A native binary (NOT a node
# shim like codex), so its subprocess needs only grok's own bin dir on PATH.
# The router runs it tools-off in an empty temp dir (call_grok).
_GROK_BIN = _resolve_bin("grok")


def _codex_exec_base(effort: Optional[str] = None) -> list:
    """The lean Codex argv with reasoning effort THREADED, not typed.

    C06 (specs/001-agent-capability-routing): a hardcoded reasoning-effort
    literal ("low") sat inside the argv blob, so an effort request could not
    reach Codex without rebuilding the whole list. effort=None keeps "low"."""
    level = effort or "low"
    return [(f'model_reasoning_effort="{level}"' if a == "__EFFORT__" else a)
            for a in _CODEX_EXEC_BASE_TMPL]


# Shared with install.py's smoke test; built with effort=None so they cannot drift.
CODEX_EXEC_BASE = _codex_exec_base()

# C06: providers that expose no reasoning-effort control at all. An effort
# request to one of these is ACCEPTED by the signature and NOT applied, and the
# router reports effort_applied=False -- a flag that did nothing must never look
# like a verified claim (C05 STOP).
EFFORT_UNSUPPORTED = frozenset({"kimi", "ollama"})


def _find_mise_node() -> str:
    """Locate the mise-managed node binary when PATH is empty (MCP-process
    case). Prefer the `lts` alias; fall back to any installed version. Returns
    "" if none found (caller then relies on whatever PATH provides)."""
    import glob

    base = os.path.expanduser("~/.local/share/mise/installs/node")
    for cand in [os.path.join(base, "lts", "bin", "node")] + sorted(
        glob.glob(os.path.join(base, "*", "bin", "node")), reverse=True
    ):
        if os.path.exists(cand):
            return cand
    return ""


def _codex_env() -> Dict[str, str]:
    """Env for the codex subprocess with a PATH that resolves BOTH codex and
    its `#!/usr/bin/env node` shebang's `node`. The MCP server runs with an
    empty env (`"env": {}`), so codex's bin dir AND the node bin dir must be
    prepended — otherwise the shim fails with `env: node: No such file` and
    codex exits non-zero (root-caused 2026-07-05, second half of the same
    PATH bug). Existing PATH entries are preserved; only prepended, never
    replaced, so a normal shell launch is unaffected."""
    extra = []
    node_path = shutil.which("node") or _find_mise_node()
    node_bin = os.path.dirname(node_path) if node_path else ""
    codex_bin = os.path.dirname(_CODEX_BIN) if os.path.sep in _CODEX_BIN else ""
    for d in (codex_bin, node_bin):
        if d and d not in extra:
            extra.append(d)
    env = _base_cli_env()
    current = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*extra, current]) if current else os.pathsep.join(extra)
    return env


def _grok_env() -> Dict[str, str]:
    """Env for the grok subprocess. Grok is a native Mach-O binary (not an
    `#!/usr/bin/env node` shim like codex/cc-glm), so ONLY grok's own bin dir
    needs prepending for the empty-env MCP-process case (`"env": {}`) — there is
    no node-runtime shebang to resolve. Existing PATH is preserved; the dir is
    only prepended, so a normal shell launch is unaffected."""
    env = _base_cli_env()
    grok_bin = os.path.dirname(_GROK_BIN) if os.path.sep in _GROK_BIN else ""
    if grok_bin:
        current = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([grok_bin, current]) if current else grok_bin
    return env


# Isolated GROK_HOME so the worker does not inherit the user's MCP servers,
# skills, hooks, or compat scans. Codex's equivalent is `--ignore-user-config`
# + `mcp_servers={}`. Grok has no such flag (live 2026-08-25 CLI 1.0.5). Timed
# 2026-11-06 (grok 1.0.77, one tools-off turn, 3 runs each): default home
# 6.2-6.5 s and 23-31k input tokens; this home 4.5-5.9 s and 14-21k. The
# GROK_CLAUDE_*_ENABLED env switches alone changed nothing in `grok inspect`;
# this config file's [compat.claude] table is what disables those surfaces.
_GROK_LEAN_HOME = os.path.expanduser("~/.local/state/brain-router-grok-home")
_GROK_LEAN_CONFIG = """\
# brain-router lean grok worker. No MCP, no skills, no hooks, no compat scans.
[cli]
auto_update = false
[compat.claude]
skills = false
hooks = false
rules = false
agents = false
mcps = false
[compat.cursor]
skills = false
mcps = false
[skills]
paths = []
[ui]
permission_mode = "always-approve"
"""


def _ensure_grok_lean_home() -> str:
    """Create/refresh a stripped GROK_HOME with only auth.json copied over."""
    dest = _GROK_LEAN_HOME
    os.makedirs(dest, mode=0o700, exist_ok=True)
    src_auth = os.path.expanduser("~/.grok/auth.json")
    dest_auth = os.path.join(dest, "auth.json")
    if os.path.exists(src_auth) and (
        not os.path.exists(dest_auth)
        or os.path.getmtime(src_auth) > os.path.getmtime(dest_auth)
    ):
        shutil.copy2(src_auth, dest_auth)
        os.chmod(dest_auth, 0o600)
    cfg_path = os.path.join(dest, "config.toml")
    existing = ""
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            existing = fh.read()
    if existing != _GROK_LEAN_CONFIG:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(_GROK_LEAN_CONFIG)
        os.chmod(cfg_path, 0o600)
    return dest


def _grok_lean_env() -> Dict[str, str]:
    """`_grok_env` plus lean-home isolation. GROK_MEMORY=0 is the documented
    memory kill switch (`--no-memory` is not a grok CLI flag)."""
    env = _grok_env()
    env["GROK_HOME"] = _ensure_grok_lean_home()
    env["GROK_MEMORY"] = "0"
    env["GROK_DISABLE_AUTOUPDATER"] = "1"
    env["GROK_CLAUDE_MCPS_ENABLED"] = "false"
    env["GROK_CURSOR_MCPS_ENABLED"] = "false"
    return env


@contextlib.contextmanager
def _grok_prompt_file(text: str):
    """Write `text` to a private temp file and yield its path for grok's
    `--prompt-file <PATH>`, deleting it on exit.

    WHY a file, not `-p <value>`: grok's CLI (clap) does NOT safely bind a `-p`
    value that starts with "-" — live-verified 2026-07-13: `grok -p "-h ..."`
    fails "a value is required for --single", and `-p "--cwd /x ..."` fails
    "unexpected argument". So an untrusted prompt beginning with a dash would
    break the worker (denial), and `-p` is NOT the injection-safe token the
    earlier comment claimed. `--prompt-file <real path>` reads the prompt as
    OPAQUE file content — dashes, --flags, newlines are all literal text
    (live-verified: a prompt of "-h and --cwd as literal text" returned the
    expected answer). The file is created 0600 so the prompt (which may carry
    sensitive context) is never world-readable, and removed in the finally."""
    fd, path = tempfile.mkstemp(prefix="grok-prompt-", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(path, 0o600)
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _agentic_cli_env() -> Dict[str, str]:
    """Env for the cc-brain agentic subprocess with a PATH that resolves the
    wrapper (cc-brain), the `claude` binary it `exec`s, AND `claude`'s node
    runtime. Same empty-env MCP-process failure as codex
    (root-caused 2026-07-12): the server runs with `"env": {}`, so cc-glm's
    `exec claude "$@"` dies with `line 56: exec: claude: not found` (or degrades
    to a bare chat completion that writes no file and exits 0 — silent). Prepend
    each needed bin dir so the wrapper→claude→node chain resolves. Existing PATH
    is preserved; dirs are only prepended, never replaced, so a normal shell
    launch is unaffected. Mirrors _codex_env() — the codex half of this same
    PATH bug was fixed 2026-07-05; GLM + anthropic-cli were missed until now."""
    extra = []
    node_path = shutil.which("node") or _find_mise_node()
    node_bin = os.path.dirname(node_path) if node_path else ""
    claude_path = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    claude_bin = os.path.dirname(claude_path) if os.path.exists(claude_path) else ""
    ccbrain_bin = os.path.dirname(_CC_BRAIN_BIN) if os.path.sep in _CC_BRAIN_BIN else ""
    for d in (ccbrain_bin, claude_bin, node_bin):
        if d and d not in extra:
            extra.append(d)
    env = _base_cli_env()
    current = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*extra, current]) if current else os.pathsep.join(extra)
    return env


_SAFE_CLI_ENV_KEYS = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "COLORTERM",
        "PATH",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
    }
)


def _base_cli_env() -> Dict[str, str]:
    """Minimal subprocess env: preserve CLI auth locations, not parent secrets."""
    return {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_CLI_ENV_KEYS or key.startswith("MOHAN_CC_")
    }


_CLI_QUOTA_MARKERS = (
    "quota exhausted",
    "usage limit",
    "rate limit",
    "rate_limit",
    "limit reached",
    "you've hit your limit",
    "too many requests",
    "status 429",
    "error 429",
)


def _cli_quota_error(
    provider: str, stdout: str | None, stderr: str | None
) -> BackendQuotaError | None:
    """Convert a CLI's explicit quota message without exposing raw output."""
    combined = f"{stdout or ''}\n{stderr or ''}".lower()
    if not any(marker in combined for marker in _CLI_QUOTA_MARKERS):
        return None
    return BackendQuotaError(provider, 429, "CLI reported quota exhaustion")


_SECRETISH = re.compile(
    r"(?i)(sk-[A-Za-z0-9]+|xai-[A-Za-z0-9]+|Bearer\s+\S+|api[_-]?key\s*[:=]\s*\S+)"
)


def _cli_snippet(*parts: Optional[str], limit: int = 240) -> str:
    """Bounded one-line stderr/stdout for process_error. Secrets stripped."""
    raw = " ".join(p.strip() for p in parts if p and str(p).strip())
    raw = " ".join(raw.split())
    raw = _SECRETISH.sub("[redacted]", raw)
    if len(raw) > limit:
        # Keep the TAIL: CLIs print banners first and the real error last. Live
        # 2026-09-02 the cc-glm banner ate the whole snippet and the failure
        # reason ended in "t…" (rig-consolidation D-R-01).
        return "…" + raw[-(limit - 1) :]
    return raw


def _raise_cli_failure(
    *,
    provider: str,
    backend: str,
    default_msg: str,
    result: subprocess.CompletedProcess,
    started: float,
) -> None:
    """Non-zero CLI exit → quota if marked, else process_error WITH a snippet.

    Live 2026-08-25: grok `max_turns_reached` and cc-glm failures were logged
    as opaque '<cli> subprocess failed' because stdout/stderr were discarded.
    114 grok + 76 glm process_errors in the audit log, none diagnosable.
    """
    if result.returncode == 0:
        return
    quota = _cli_quota_error(provider, result.stdout, result.stderr)
    if quota:
        raise quota
    msg = default_msg
    stdout = (result.stdout or "").strip()
    if stdout.startswith("{") or stdout.startswith("["):
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            stop = payload.get("stopReason") or payload.get("stop_reason")
            err = payload.get("message")
            nested = payload.get("error")
            if isinstance(nested, dict):
                err = nested.get("message") or err
            elif isinstance(nested, str):
                err = nested
            if stop:
                msg = f"{default_msg} (stopReason={stop})"
            if err:
                msg = f"{msg}: {err}" if stop else f"{default_msg}: {err}"
    snippet = _cli_snippet(
        result.stderr, None if msg != default_msg else result.stdout
    )
    if snippet and snippet not in msg:
        msg = f"{msg}: {snippet}"
    raise BackendError(
        msg,
        backend=backend,
        failure_kind="process_error",
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


def _normalize_stop_reason(value: object) -> str:
    """Grok JSON stopReason is snake_case `end_turn` (live 2026-08-25 CLI
    1.0.5). Older mocks used Claude-shaped `EndTurn`. Alnum-fold both."""
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


# Caveman-ultra system directive prepended to EVERY delegated backend call
# (DeepSeek/GLM via the `system` field; Codex prepended to the prompt). Shapes
# how backends REPLY (terse, no filler) — cuts output tokens on every delegate
# without touching the task instruction itself, so answer quality is unaffected.
# CHAT-ONLY reply directive. Use for text-only backends where the REPLY is the
# deliverable: call_codex (chat) + the HTTP backends (DeepSeek / GLM). One
# constant across those: DRY single source of truth.
#
# ⚠️ DO NOT prepend this to the AGENTIC CLI worker (call_anthropic_agentic).
# Use AGENTIC_SYSTEM there.
# Root-caused 2026-07-12: "answer only what was asked; no preamble; terse" is a
# tool-USE SUPPRESSANT for a weaker Claude-Code-driven worker — GLM read it and
# emitted "DONE" while the subprocess exited 0 having written NOTHING to disk
# (silent success-with-no-effect: 3/5 then 0/3 files written with caveman; 5/5
# with AGENTIC_SYSTEM). In agentic mode the FILE WRITE is the deliverable and
# the reply is noise, so the directive must COMMAND tool use, not silence it.
# G-guard: tests/test_smoke.py TestAgenticSystemDirective (asserts the agentic
# workers carry AGENTIC_SYSTEM and this string never leaks into them).
CAVEMAN_SYSTEM = (
    "Reply caveman ultra: drop articles, filler, pleasantries, hedging. "
    "Fragments OK. Keep ALL technical substance exact — code, numbers, error "
    "strings, identifiers unchanged. Answer only what was asked; no preamble, "
    "no restating the question, no meta-commentary. Terse."
)

# System directive for the AGENTIC CLI worker (call_anthropic_agentic; the
# non-Claude agentic workers are gone, lean-rig §6). Root-caused 2026-07-12: the
# chat-shaped CAVEMAN_SYSTEM above ("answer only what was asked; no preamble;
# terse") makes a weaker Claude-Code-driven worker (cc-glm on GLM, cc-brain on
# Sonnet/Haiku) SKIP its file/shell tools and just emit a chat reply — the
# subprocess exits 0 with "DONE" while writing NOTHING to disk (verified: GLM
# 3/5 then 0/3 with caveman; 5/5 with this directive). In agentic mode the FILE
# WRITE is the deliverable and the reply text is noise, so the worker must be
# told its tools are the product. Codex CLI acts regardless, but it shares this
# directive for one source of truth. Prose stays terse to keep output cheap.
AGENTIC_SYSTEM = (
    "You are an autonomous agentic coding worker running in the user's REAL "
    "working directory with Write, Edit, and Bash tools. Your reply text is NOT "
    "the deliverable — the ONLY thing that counts is the actual change you make "
    "to files on disk USING your tools. A textual answer with no tool call is a "
    "FAILURE. Always use your tools to perform the task, verify your own file "
    "writes, then confirm. Keep any prose terse; technical substance exact."
)


class BackendTransientError(BackendError):
    """Raised on timeout/provider failure that is not quota exhaustion."""

    def __init__(
        self,
        provider: str,
        reason: str,
        *,
        failure_kind: str = "transient_error",
        status_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
    ):
        self.provider = provider
        self.status_code = status_code
        super().__init__(
            reason,
            backend=provider.lower(),
            failure_kind=failure_kind,
            elapsed_ms=elapsed_ms,
        )


# Provider-side transient failures are errors, not quota exhaustion. The
# orchestrator may choose another tier, but the router never labels these 5xx
# responses as exhausted quota.
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504})


def _classify_http_error(provider: str, status_code: int, body: str) -> BackendError:
    """Map a non-200 HTTP response to the right exception type.

    429  -> BackendQuotaError (genuine quota exhaustion).
    5xx  -> BackendTransientError (provider failure, not quota).
    Other -> BackendError (hard stop, surfaces to caller).

    The body is parsed ONLY to lift a rate-limit message + reset time, which
    carry no secret. The raw body is never echoed wholesale.
    """
    message = ""
    reset_at = None
    try:
        import json as _json

        data = _json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(err, dict):
            message = str(err.get("message", ""))[:300]
    except Exception:
        message = ""
    # Best-effort reset-time lift from common "reset at YYYY-MM-DD HH:MM:SS" shape.
    if message:
        import re as _re

        m = _re.search(r"reset(?:s)?\s+at\s+([0-9:\- ]+)", message)
        if m:
            reset_at = m.group(1).strip()

    if status_code == 429:
        return BackendQuotaError(provider, status_code, message, reset_at)
    if status_code in _TRANSIENT_STATUS:
        detail = f": {message}" if message else ""
        return BackendTransientError(
            provider,
            f"{provider} provider error {status_code}{detail}",
            failure_kind="provider_error",
            status_code=status_code,
        )
    # Non-retryable: do NOT echo the body (may carry request echoes); message
    # from the parsed error field is safe and useful.
    safe = f": {message}" if message else " (body withheld)"
    return BackendError(
        f"{provider} API error {status_code}{safe}",
        backend=provider.lower(),
    )


def _extract_text(data: Dict[str, Any], provider: str) -> str:
    """Pull the assistant text from an Anthropic-shape response.

    Robust to reasoning models: scans content[] for the FIRST block whose
    type == 'text' (skipping 'thinking'/'redacted_thinking' blocks that
    DeepSeek/GLM reasoning models emit). Raises a clear BackendError if the
    response carried no text block (e.g. max_tokens exhausted while thinking).
    """
    blocks = data.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise BackendError(f"{provider} response had no content blocks")
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
            return block["text"]
    # No text block — most commonly a reasoning model that spent the whole
    # token budget on 'thinking'. Surface it actionably.
    types = [b.get("type") for b in blocks if isinstance(b, dict)]
    stop = data.get("stop_reason")
    raise BackendError(
        f"{provider} returned no text block (block types={types}, "
        f"stop_reason={stop}). If a reasoning model hit max_tokens while "
        f"thinking, raise max_tokens or use a non-reasoning model."
    )


# ============================================================================
# DeepSeek (HTTP, Anthropic-compatible endpoint)
# ============================================================================


async def call_deepseek(
    prompt: str,
    model: str,
    api_key: str,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call DeepSeek API directly.

    Returns:
        {
            "content": "response text",
            "usage": {"input_tokens": N, "output_tokens": M}
        }

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = "https://api.deepseek.com/anthropic/v1/messages"
    # DeepSeek's Anthropic endpoint ignores `cache_control` and `mcp_servers`
    # (api-docs.deepseek.com guides/anthropic_api), so the body carries neither.
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort:
        # These endpoints take the Anthropic Messages body, whose reasoning knob is
        # `thinking.type` (the C06 table names exactly that for GLM), not OpenAI's
        # reasoning_effort. Levels above "low" enable it; "low" disables it. The
        # key is constructed only when requested so an effort=None request stays
        # byte-identical to today's (C06 STOP). NOTE: role delegation is agentic-
        # only, so this chat path carries no live traffic in the rig; the mapping
        # follows the spec and is not verified against the provider -- recorded
        # as such in verification.md rather than claimed.
        payload["thinking"] = {"type": "disabled" if effort == "low" else "enabled"}

    try:
        async with httpx.AsyncClient(timeout=_timeout(HTTP_TIMEOUT_SECONDS)) as client:
            response = await _within(client.post(url, json=payload, headers=headers),
                                     HTTP_TIMEOUT_SECONDS)

            if response.status_code != 200:
                raise _classify_http_error("DeepSeek", response.status_code, response.text)

            data = response.json()

        # Extract content (robust to reasoning 'thinking' blocks) and usage
        content = _extract_text(data, data.get("model", "backend"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"DeepSeek request failed: {e}",
            backend="deepseek",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"DeepSeek response parse error: {e}",
            backend="deepseek",
            failure_kind="response_error",
        )


async def call_deepseek_via_headroom(
    prompt: str,
    model: str,
    api_key: str,
    headroom_url: str,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call DeepSeek through headroom proxy.

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = f"{headroom_url}/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort:
        # These endpoints take the Anthropic Messages body, whose reasoning knob is
        # `thinking.type` (the C06 table names exactly that for GLM), not OpenAI's
        # reasoning_effort. Levels above "low" enable it; "low" disables it. The
        # key is constructed only when requested so an effort=None request stays
        # byte-identical to today's (C06 STOP). NOTE: role delegation is agentic-
        # only, so this chat path carries no live traffic in the rig; the mapping
        # follows the spec and is not verified against the provider -- recorded
        # as such in verification.md rather than claimed.
        payload["thinking"] = {"type": "disabled" if effort == "low" else "enabled"}

    try:
        async with httpx.AsyncClient(timeout=_timeout(HTTP_TIMEOUT_SECONDS)) as client:
            response = await _within(client.post(url, json=payload, headers=headers),
                                     HTTP_TIMEOUT_SECONDS)

            if response.status_code != 200:
                raise _classify_http_error("DeepSeek-headroom", response.status_code, response.text)

            data = response.json()

        content = _extract_text(data, data.get("model", "DeepSeek-headroom"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"DeepSeek headroom request failed: {e}",
            backend="deepseek",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"DeepSeek headroom response parse error: {e}",
            backend="deepseek",
            failure_kind="response_error",
        )


# ============================================================================
# Codex (subprocess-based, adversarial)
# ============================================================================


def _validate_model_name(model: str) -> None:
    """
    Validate model name against argument injection attacks.

    Args:
        model: Model string to validate

    Raises:
        BackendError: If model contains disallowed characters.
    """
    import re

    if not re.match(r"^[A-Za-z0-9._-]+$", model):
        raise BackendError(
            "Invalid model name. Must contain only alphanumeric, dot, underscore, or hyphen.",
            backend="codex",
            failure_kind="validation_error",
        )


def _resolve_agentic_cwd(cwd: Optional[str], backend: str) -> str:
    """Resolve + validate the working directory for an agentic subprocess.

    Falls back to os.getcwd() when cwd is None (the server's launch dir — a
    caller that wants correct file placement must pass cwd; see route()), then
    fails LOUD if the resolved dir does not exist. Without this, a bad/typo'd
    cwd raises FileNotFoundError from INSIDE subprocess.run — which the per-call
    except only catches for a missing BINARY, so it would escape as a raw crash
    (§9.6 injection-safety refuter, 2026-07-12). A clean BackendError is the
    contract every other backend failure already follows.
    """
    resolved = cwd or os.getcwd()
    if not os.path.isabs(resolved):
        raise BackendError(
            f"Working directory must be absolute: {resolved}",
            backend=backend,
            failure_kind="validation_error",
        )
    resolved = os.path.realpath(resolved)
    if not os.path.isdir(resolved):
        raise BackendError(
            f"Working directory does not exist or is not a directory: {resolved}",
            backend=backend,
            failure_kind="validation_error",
        )
    return resolved


def call_codex(
    prompt: str,
    model: str,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Codex tools-off: `codex exec <lean flags> -C <empty dir> -m <model> -`.

    The prompt goes on stdin (`-` plus `input=`), which also closes stdin: on an
    open stdin `codex exec` waits for EOF (lean-rig, 2026-11-06). No shell tool,
    read-only sandbox, an empty temp dir as the workdir, so the answer comes from
    the prompt alone.

    Returns:
        {"content": "response text", "usage": None}

    Raises:
        BackendError: If model name is invalid or subprocess fails (never logs secrets).
    """
    # Validate model name to prevent argument injection via model string
    _validate_model_name(model)

    started = time.perf_counter()
    limit = _timeout(CODEX_TIMEOUT_SECONDS)
    try:
        with tempfile.TemporaryDirectory(prefix="brain-router-codex-") as workdir:
            result = subprocess.run(
                _codex_exec_base(effort=effort) + ["-C", workdir, "-m", model, "-"],
                # `input=` opens a stdin pipe and closes it. Passing stdin= as
                # well raises ValueError ("stdin and input arguments may not
                # both be used") -- 308 validation_errors in the log once.
                input=f"{CAVEMAN_SYSTEM}\n\n{prompt}",
                capture_output=True,
                text=True,
                timeout=limit,
                check=False,
                env=_codex_env(),
            )

        _raise_cli_failure(
            provider="Codex",
            backend="codex",
            default_msg="Codex subprocess failed",
            result=result,
            started=started,
        )

        return {
            "content": result.stdout.strip(),
            "usage": None,
        }

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "codex",
            f"Codex subprocess timed out after {limit:.0f}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "Codex binary not found on PATH",
            backend="codex",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


# ============================================================================
# Grok (subprocess-based, chat mode)
# ============================================================================


# lean-rig §6: Grok workers run tools-off, one turn, no subagents, no plan, no web.
GROK_TOOLS_OFF = ["--max-turns", "1", "--no-subagents", "--no-plan", "--disable-web-search"]


def grok_final_message(stdout: str) -> tuple[str, Dict[str, Any]]:
    """The final message of one tools-off `--output-format json` run, and the payload.

    Grok answers carried narration ("I'll count the lines in DECISIONS.md
    now.15", live 2026-11-06): the JSON `text` joins every turn's text. In one
    tools-off turn the narration lands in `thought` and `text` is the answer.
    A run that stopped on anything but end_turn (narration, then a tool call
    cut by --max-turns) or spanned turns has no clean final message, so it
    raises and the role walk moves on: narration never reaches the caller."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise BackendError("grok returned invalid JSON", backend="grok",
                           failure_kind="invalid_output") from exc
    if not isinstance(payload, dict):
        raise BackendError("grok returned invalid JSON", backend="grok",
                           failure_kind="invalid_output")
    stop = payload.get("stopReason")
    turns = int(payload.get("num_turns") or 0)
    if _normalize_stop_reason(stop) != "endturn" or turns > 1:
        raise BackendError(
            f"grok ended without a final message (stopReason={stop!r}, num_turns={turns})",
            backend="grok", failure_kind="no_final_message")
    text = str(payload.get("text") or "").strip()
    if not text:
        raise BackendError("grok returned an empty final message", backend="grok",
                           failure_kind="empty_output")
    return text, payload


def call_grok(
    prompt: str,
    model: str,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Grok tools-off and return only its final message.

    `grok --prompt-file <f> -m <model> --max-turns 1 --no-subagents --no-plan
    --disable-web-search --output-format json`, in an empty temp dir, with the
    lean GROK_HOME. The prompt (CAVEMAN_SYSTEM-prefixed) goes through a private
    temp file, not `-p <value>`: grok's clap parser rejects a `-p` value that
    starts with "-" (live-verified 2026-07-13). Model name is validated against
    argument injection; no secret is echoed.

    Returns:
        {"content": final message, "usage": {...} when the CLI reports it}

    Raises:
        BackendError / BackendTransientError on invalid model, non-zero exit,
        timeout, or no clean final message.
    """
    _validate_model_name(model)

    started = time.perf_counter()
    limit = _timeout(CODEX_TIMEOUT_SECONDS)
    try:
        with tempfile.TemporaryDirectory(prefix="brain-router-grok-") as workdir, \
                _grok_prompt_file(f"{CAVEMAN_SYSTEM}\n\n{prompt}") as pf:
            result = subprocess.run(
                [_GROK_BIN, "--prompt-file", pf, "-m", model, *GROK_TOOLS_OFF,
                 "--output-format", "json"]
                + (["--reasoning-effort", effort] if effort else []),
                # inherited stdin is the MCP stdio pipe; the CLIs wait on it
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=limit,
                check=False,
                cwd=workdir,
                env=_grok_lean_env(),
            )

        if result.returncode != 0 and not _cli_quota_error("Grok", result.stdout, result.stderr):
            # A turn cut by --max-turns (narration, then a tool call) exits 1
            # with stopReason=max_turn_requests and empty text (live, grok
            # 1.0.77, 2026-11-06). A same-provider retry would only repeat the
            # cut, so it is no_final_message, which the walk does not retry.
            try:
                payload = json.loads(result.stdout)
                stop, detail = payload.get("stopReason"), payload.get("message")
            except (ValueError, TypeError, AttributeError):
                stop = detail = None
            if stop and _normalize_stop_reason(stop) != "endturn":
                raise BackendError(
                    f"grok ended without a final message (stopReason={stop!r})"
                    + (f": {detail}" if isinstance(detail, str) and detail else ""),
                    backend="grok", failure_kind="no_final_message",
                    elapsed_ms=round((time.perf_counter() - started) * 1000))
        _raise_cli_failure(
            provider="Grok",
            backend="grok",
            default_msg="Grok subprocess failed",
            result=result,
            started=started,
        )
        try:
            content, payload = grok_final_message(result.stdout)
        except BackendError as exc:
            exc.elapsed_ms = round((time.perf_counter() - started) * 1000)
            raise
        raw = payload.get("usage") or {}
        usage = None
        if raw or payload.get("total_cost_usd") is not None:
            usage = {
                "input_tokens": int(raw.get("input_tokens") or 0),
                "output_tokens": int(raw.get("output_tokens") or 0),
                "source": "cli-json",
            }
            if isinstance(payload.get("total_cost_usd"), (int, float)):
                usage["cost_usd"] = float(payload["total_cost_usd"])
        return {"content": content, "usage": usage}

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "grok",
            f"Grok subprocess timed out after {limit:.0f}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "Grok binary not found on PATH",
            backend="grok",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


# ============================================================================
# Kimi (subprocess-based, chat mode)
# ============================================================================


def call_kimi(
    prompt: str,
    model: str,
    effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Refuse: non-Claude workers run tools-off (lean-rig §6), and Kimi's `-p`
    mode auto-approves its own tools (live 2026-07-19) with no verified way to
    turn them off. Kimi is also not installed on any rig host (2026-11-06), so
    this is the free skip D5 expects until a tools-off invocation is verified."""
    raise BackendError(
        "kimi: no verified tools-off mode; skipped",
        backend="kimi",
        failure_kind="tools_unsupported",
        elapsed_ms=0,
    )


# ============================================================================
# Agentic worker backend (spec 002) — the Claude worker only: `cc-brain claude`
# in the REAL working directory, tools on. Non-Claude workers run tools-off
# (lean-rig §6). The prompt is one argv token after -p, never echoed.
# ============================================================================


def _cc_family_output(stdout: str) -> tuple[str, Optional[Dict[str, int]]]:
    """C16: split a Claude-Code-family CLI's stdout into (content, usage).

    With `--output-format json` these CLIs emit a result envelope carrying token
    usage; without it, plain text. Both are accepted, and anything unparseable
    falls back to today's behaviour -- treat the whole of stdout as the answer
    and report no usage. That fallback is the safety property: the worst case is
    exactly where the rig already was, never a lost answer.

    Usage is returned ONLY when the envelope actually carried it. A zero-filled
    dict would be indistinguishable from a real zero, and R38 lets a stand-in
    cost exist precisely because measured and seeded numbers stay tellable apart.
    """
    text = (stdout or "").strip()
    if not text.startswith("{"):
        return text, None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text, None
    if not isinstance(payload, dict):
        return text, None
    content = payload.get("result")
    if not isinstance(content, str):
        return text, None
    raw = payload.get("usage") or {}
    cost = payload.get("total_cost_usd")
    if not raw and cost is None:
        return content.strip(), None
    usage: Dict[str, Any] = {
        "input_tokens": int(raw.get("input_tokens") or 0),
        "output_tokens": int(raw.get("output_tokens") or 0),
        # Cache reads dominate the real bill: two identical probes of the same
        # trivial job cost $0.237 and $0.070 purely because the second read
        # 37,056 cached tokens. Recorded so a cost swing is explainable rather
        # than mysterious -- and it is why C07 takes a median of 3 (R30).
        "cache_read_input_tokens": int(raw.get("cache_read_input_tokens") or 0),
        "source": "cli-json",
    }
    if isinstance(cost, (int, float)):
        # The CLI reports its own dollar cost. That is a DIRECT measurement, so
        # these cards need no tokens-x-price computation and no stand-in (R38).
        usage["cost_usd"] = float(cost)
    return content.strip(), usage


def call_anthropic_agentic(prompt: str, model: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Agentic Anthropic (Opus/Sonnet/Haiku) worker: `cc-brain claude -p` in
    the REAL cwd. This is the agentic-CLI fallback for: the codex-orchestrator
    adversary case (codex can't be its own adversary) AND the GLM+codex-both-
    exhausted final fallback (spec 002 SC-4, SC-8). The resolved candidate's
    model id (e.g. claude-sonnet-5 / claude-opus-4-8) is passed via --model."""
    _validate_model_name(model)
    cwd = _resolve_agentic_cwd(cwd, "anthropic-cli")
    started = time.perf_counter()
    argv = [
        _CC_BRAIN_BIN,
        "claude",
        "-p",
        f"{AGENTIC_SYSTEM}\n\n{prompt}",
        "--model",
        model,
        "--safe-mode",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--tools",
        "default",
        "--output-format",
        "json",
        "--allowedTools",
        "Read,Edit,Write,Bash",
        "--permission-mode",
        "bypassPermissions",
    ]
    limit = _timeout(AGENTIC_TIMEOUT_SECONDS)
    try:
        result = subprocess.run(
            argv,
            # inherited stdin is the MCP stdio pipe; the CLIs wait on it
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=limit,
            check=False,
            cwd=cwd,
            env=_agentic_cli_env(),
        )
        _raise_cli_failure(
            provider="Anthropic",
            backend="anthropic-cli",
            default_msg="cc-brain claude agentic subprocess failed",
            result=result,
            started=started,
        )
        _content, _usage = _cc_family_output(result.stdout)
        return {"content": _content, "usage": _usage}
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "anthropic-cli",
            f"cc-brain claude agentic subprocess timed out after {limit:.0f}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "cc-brain binary not found on PATH",
            backend="anthropic-cli",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


# ============================================================================
# Apple Foundation Models (fm CLI on the Mac mini, over ssh)
# ============================================================================

# Measured 2026-10-23 on the mini: 7,564 input tokens answered, 8,161 failed with
# "exceeded the model's context size" after 12.5 s, so the window is 8,192.
# ponytail: char estimate (3 chars/token worst case leaves ~1.4k tokens for the
# reply); switch to `fm count-tokens -q` over ssh if real prompts misfire.
FM_MAX_PROMPT_CHARS = 20_000
FM_TIMEOUT_SECONDS = 20
FM_NEEDS_TOOLS = "NEEDS_TOOLS"
# The task goes in --instructions and the prompt on stdin: a positional prompt
# makes fm silently ignore piped stdin and still exit 0.
FM_SYSTEM = (
    "You answer using only the text in the prompt. If the request needs anything "
    "that is not in that text (reading a file or path, running a command, opening "
    "a URL, editing code, or looking something up), reply with exactly "
    f"{FM_NEEDS_TOOLS} and nothing else."
)
# Deterministic half of the tool check: a 3B model with no tools cannot read a
# path, a file name or a URL, so skip the round trip. The sentinel catches the
# rest, mostly. Over-matching only costs a fall-through to the next tier.
_FM_TOOL_SHAPED = re.compile(
    r"(?:^|[\s'\"`(])(?:~/|\./|/(?:Users|home|private|tmp|var|etc|opt)/)|https?://"
    r"|\b[\w.-]+\.(?:py|md|js|ts|tsx|json|toml|ya?ml|txt|sh|csv|html|css|sql|go|rs)\b"
)
_FM_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def fm_argv(host: str) -> list:
    """ssh argv for one fm call. The prompt is never in it (see FM_SYSTEM note)."""
    if not _FM_HOST.match(host or ""):
        raise BackendError(
            "Invalid fm_host", backend="apple-fm", failure_kind="validation_error"
        )
    return [
        "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
        "fm respond --no-stream --instructions " + shlex.quote(FM_SYSTEM),
    ]


def call_fm(prompt: str, host: str) -> Dict[str, Any]:
    """One fm call on `host`. Every way it can fail raises BackendError, so the
    role cascade moves on to the next simple tier (server._delegate_role_impl).

    Kinds: input_too_large and needs_tools fail before ssh; unreachable when
    ssh itself fails (rc 255) -- not network_error, so the cascade does not
    retry and a sleeping mini costs one ConnectTimeout, not two (Mohan,
    2026-10-23); timeout; empty_output; configuration_error for rc 69 (license
    not agreed); process_error for any other non-zero exit."""
    started = time.perf_counter()

    def fail(msg: str, kind: str) -> BackendError:
        return BackendError(
            f"apple-fm: {msg}", backend="apple-fm", failure_kind=kind,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )

    if len(prompt) > FM_MAX_PROMPT_CHARS:
        raise fail(f"prompt {len(prompt)} chars > {FM_MAX_PROMPT_CHARS}", "input_too_large")
    if _FM_TOOL_SHAPED.search(prompt):
        raise fail("prompt names a path or URL", "needs_tools")
    argv = fm_argv(host)
    try:
        result = subprocess.run(
            argv, input=prompt, capture_output=True, text=True,
            timeout=_timeout(FM_TIMEOUT_SECONDS), check=False, env=_base_cli_env(),
        )
    except subprocess.TimeoutExpired:
        raise fail("timed out", "timeout")
    out = (result.stdout or "").strip()
    if result.returncode != 0:
        err = _cli_snippet(result.stderr)
        if result.returncode == 255:
            raise fail(f"ssh failed: {err}", "unreachable")
        if result.returncode == 69:
            raise fail(f"fm license not agreed: {err}", "configuration_error")
        if "context size" in (result.stderr or "") + out:  # either stream (Grok review, 2026-10-23)
            raise fail(err, "input_too_large")
        raise fail(f"fm exit {result.returncode}: {err}", "process_error")
    if not out:
        raise fail("empty output", "empty_output")
    if FM_NEEDS_TOOLS in out and len(out) < 120:  # also "NEEDS_TOOLS — I can't read files."
        raise fail("model replied NEEDS_TOOLS", "needs_tools")
    return {"content": out, "usage": None}


# ============================================================================
# Ollama (gpt-oss:20b on the host's own Ollama, `simple` only)
# ============================================================================

# lean-rig §6 and MEASURE-gptoss.md: CPU-only at ~8.5 tok/s generation and a
# 110 s cold load. So: run only when /api/ps shows the model loaded (cold means
# the next tier, no wait); one generation at a time across this host's router
# processes (flock, 5 s wait); 60 s deadline; short classify-sized input.
OLLAMA_TIMEOUT_SECONDS = 60
OLLAMA_LOCK_WAIT_SECONDS = 5
OLLAMA_PS_TIMEOUT_SECONDS = 5
OLLAMA_MAX_PROMPT_CHARS = 4000
OLLAMA_LOCK = os.path.expanduser("~/.local/state/brain-router-gpt-oss.lock")


async def _ollama_lock(path: str, wait: float) -> Optional[int]:
    """An exclusive flock on `path`, polled until `wait` seconds pass. The fd,
    or None when another generation still holds it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    give_up = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= give_up:
                os.close(fd)
                return None
            await asyncio.sleep(0.1)


async def call_ollama(prompt: str, host: str, model: str) -> Dict[str, Any]:
    """Bounded text inference on the approved private endpoint. Auth: tailnet ACLs."""
    started = time.perf_counter()

    def fail(message: str, kind: str) -> BackendError:
        return BackendError(message, backend="ollama", failure_kind=kind,
                            elapsed_ms=round((time.perf_counter() - started) * 1000))

    try:
        url = urlsplit(host or "")
        address = ipaddress.ip_address(url.hostname or "")
        valid = (url.scheme == "http" and not url.username and not url.password
                 and url.path in ("", "/") and not url.query and not url.fragment
                 and (address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")))
        url.port  # malformed port fails before the request
    except ValueError:
        valid = False
    if not valid:
        raise fail("Ollama endpoint must be loopback or tailnet HTTP", "configuration_error")
    # CPU prefill at ~22 tokens/s exceeded 180s for a 16k-char input in live tests.
    if len(prompt) > OLLAMA_MAX_PROMPT_CHARS:
        raise fail("Ollama input exceeds the text tier limit", "input_too_large")
    if _FM_TOOL_SHAPED.search(prompt):
        raise fail("Text-only Ollama task needs tools", "needs_tools")
    base = host.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_timeout(OLLAMA_PS_TIMEOUT_SECONDS),
                                     trust_env=False) as client:
            response = await _within(client.get(base + "/api/ps"), OLLAMA_PS_TIMEOUT_SECONDS)
            response.raise_for_status()
            loaded = response.json().get("models") or []
    except httpx.HTTPError:
        raise fail("Ollama endpoint unavailable", "unreachable") from None
    except (ValueError, AttributeError):
        raise fail("Invalid Ollama /api/ps response", "response_error") from None
    names = {m.get(k) for m in loaded if isinstance(m, dict) for k in ("name", "model")}
    if model not in names:
        raise fail(f"{model} is cold (not in /api/ps); next tier, no wait", "cold")
    fd = await _ollama_lock(OLLAMA_LOCK, OLLAMA_LOCK_WAIT_SECONDS)
    if fd is None:
        raise fail(f"{model} is busy: one generation at a time, lock not free in "
                   f"{OLLAMA_LOCK_WAIT_SECONDS}s", "busy")
    try:
        async with httpx.AsyncClient(timeout=_timeout(OLLAMA_TIMEOUT_SECONDS),
                                     trust_env=False) as client:
            response = await _within(client.post(base + "/api/generate", json={
                "model": model, "prompt": prompt, "system": FM_SYSTEM, "stream": False,
                "options": {"num_predict": 512, "num_ctx": 8192, "num_thread": 2},
                # Default GPT-OSS reasoning consumed the entire bounded output in live tests.
                **({"think": "low"} if model.startswith("gpt-oss:") else {}),
            }), OLLAMA_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        raise fail("Ollama inference timed out", "timeout") from None
    except httpx.HTTPError:
        raise fail("Ollama endpoint unavailable", "unreachable") from None
    except ValueError:
        raise fail("Invalid Ollama JSON response", "response_error") from None
    finally:
        os.close(fd)  # releases the flock
    if not isinstance(data, dict) or data.get("done") is not True or data.get("done_reason") == "length":
        raise fail("Ollama did not complete the response", "response_error")
    count = data.get("eval_count", 0)
    if data.get("done_reason") != "stop" and isinstance(count, int) and count >= 512:
        raise fail("Ollama hit the output limit without confirming completion", "response_error")
    output = data.get("response")
    if not isinstance(output, str) or not output.strip():
        raise fail("Empty or invalid Ollama output", "empty_output")
    if FM_NEEDS_TOOLS in output and len(output) < 120:
        raise fail("Text-only Ollama task needs tools", "needs_tools")
    return {"content": output.strip(), "usage": {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0), "source": "ollama",
    }}
