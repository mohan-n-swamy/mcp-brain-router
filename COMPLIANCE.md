# Compliance Analysis — mcp-brain-router

**Invalidated by role-shard migration: 2026-07-14.** The June analysis below
describes the former non-Anthropic-only architecture. It is retained as
historical context, not a current compliance conclusion.

---

## TL;DR (BLUF)

**PRODUCTION CAUTION:** Treat this tool as personal/testing use only. Role
shards may now launch `cc-brain claude` as an automated subscription-backed CLI
worker. That behavior invalidates the former claim that this MCP never accesses
Anthropic Services by bot or script. No current policy approval has been
verified for this architecture.

Current safe statement: provider credentials remain inside their native CLI/API
auth paths and are not copied into router configuration. Compliance of automated
Claude subscription use is unresolved. Anthropic has not reviewed or approved
this tool.

> Historical analysis starts below. Any sentence asserting "no automated
> Anthropic access" is superseded by this notice.

---

## What this tool does / does not do

| Behavior | Status | Note |
|----------|--------|------|
| Orchestrator (Claude Code) remains native, unmodified | ✓ Yes | First-party Anthropic client, unchanged |
| Shares Anthropic credentials with third-party software | ✗ No | Never passes Claude API key or login tokens |
| Delegates to other LLM providers using THEIR keys | ✓ Yes | DeepSeek key, GLM key, Codex CLI — independent |
| Proxies or wraps Anthropic subscription traffic | ✗ No | Anthropic traffic is direct, client→Anthropic only |
| Bypasses Anthropic's access controls | ✗ No | Uses MCP, an Anthropic-created standard for extensions |
| Introduces automated/bot access to Anthropic Services | ⚠ Yes | Role fallback can launch `cc-brain claude -p`; policy status unresolved |
| Accepts third-party responses and forwards to Claude | ✓ Yes | Labeled as untrusted external data (prompt-injection guard) |

---

## The relevant terms (VERBATIM QUOTES)

### Source A: Anthropic Consumer Terms of Service

**URL:** https://www.anthropic.com/legal/consumer-terms _(accessed 2026-06-29)_

#### Prohibited automated access (Section "Permitted Use")

> Except when you are accessing our Services via an Anthropic API Key or where we otherwise explicitly permit it, to access the Services through automated or non-human means, whether through a bot, script, or otherwise.

#### Account sharing and credential protection

> You may not share your Account login information, Anthropic API key, or Account credentials with anyone else. You also may not make your Account available to anyone else.

#### Abuse and interference (Section 3)

> You also must not abuse, harm, interfere with, or disrupt our Services, including, for example, introducing viruses or malware, spamming or DDoSing Services, or bypassing any of our systems or protective measures.

**Note:** The Consumer Terms explicitly state that the Commercial Terms (API Console / API keys) are governed by a **separate document** and do NOT govern Claude.ai / Claude Pro individual use.

---

### Source C: Get started with custom connectors using remote MCP

**URL:** https://support.claude.com/en/articles/11175166 _(accessed 2026-06-29)_

> The Model Context Protocol (MCP) is an open standard, created by Anthropic, for AI applications to connect to tools and data.

---

### Source D: Use connectors to extend Claude's capabilities

**URL:** https://support.claude.com/en/articles/11176164 _(accessed 2026-06-29)_

> Connectors work across Claude, Claude Desktop, Claude Code, and API (via MCP Connector).

---

## Why this design is consistent with those terms

### On automated access (Consumer Terms "Permitted Use")

The tool **does not introduce automated or bot access to Anthropic Services**. The orchestrator is Claude Code, a native first-party Anthropic client run by the human user. The user initiates each request; there is no unattended script, no background daemon making decisions, no bot. The MCP standard itself (Source C) is Anthropic's own creation for extending Claude, and connectors are explicitly supported in Claude Code (Source D). This is intentional, designed use, not a workaround.

### On credential sharing (Consumer Terms "Account")

The tool **does not share Anthropic credentials**. No Anthropic API key, login token, or session credential is passed to the MCP, written to config files, or shared with third-party software. The only secrets stored are API keys for OTHER providers (DeepSeek, GLM)—which the user controls and rotates independently. The Anthropic credential (the Claude Code subscription, whether Free/Pro/Max) remains in Claude Code's sole control.

### On bypassing protective measures (Consumer Terms Section 3)

The tool **does not bypass Anthropic's protective measures**. It uses MCP, an Anthropic-created standard for safe extension. It does not modify Claude Code, inject code, or interfere with Anthropic's systems. Delegation to other providers happens at the prompt layer—the MCP receives a request, queries external LLMs, and returns untrusted data labeled as such. Claude Code's normal safety / moderation stack still applies to everything.

---

## What we could NOT verify (the "Fin" support-agent assertion)

In June 2026, an Anthropic support agent ("Fin") asserted in a chat that Anthropic's Terms contain a clause addressing *"third-party tools that misrepresent identity or route third-party traffic against subscription limits."*

**We could not independently verify this claim.** The exact language does not appear in the public Consumer Terms (Source A) or the Acceptable Use Policy (https://www.anthropic.com/legal/aup, accessed 2026-06-29). The support-agent statement is included here for transparency, but we treat it as an **unverified assertion from a support agent**, not as the governing legal text or definitive policy.

**Critical distinction:** When an individual support agent makes a statement, it is not the same as Anthropic's official policy. Official policy lives in:
1. The published legal terms (Consumer Terms, Commercial Terms, AUP)
2. Direct communication from Anthropic's legal or policy team
3. Official documentation (support articles, help center)

An individual support agent's ad-hoc assertion ≠ official policy. If Anthropic's legal team has a concern, it must appear in one of the sources above to be binding.

**What "Fin" may have intended:** Anthropic may have concerns about a *different* pattern—one where a proxy or agent software **wraps the Anthropic subscription token** and routes it through third-party infrastructure, making it appear to Anthropic that the traffic is legitimate when it is actually being intercepted and modified. That pattern *would* be closer to "misrepresenting identity" and "bypassing protective measures."

**Why this tool is NOT that pattern:** This tool does not proxy the Anthropic subscription. Claude Code is the unmodified, direct client. The delegation to other providers happens entirely in the application layer (the MCP), using independent credentials, and the traffic never touches Anthropic's infrastructure. There is nothing to misrepresent.

---

## The contrast that matters

| Pattern | Status | Risk |
|---------|--------|------|
| Proxy wraps subscription token through third-party software (contested) | ✗ Not this tool | Misrepresents identity; may violate "bypassing protective measures" |
| Tool delegates to other providers, orchestrator stays native (this tool) | ✓ This approach | Traffic is direct; no credential sharing; Anthropic stack unchanged |

---

## Verify this yourself

Before accepting this memo, confirm each quote and claim independently:

1. **Open** https://www.anthropic.com/legal/consumer-terms
2. **Find** the "Permitted Use" section and locate the clause beginning with "Except when you are accessing our Services via an Anthropic API Key..."
3. **Find** the "Account" section and confirm the clause on credential sharing.
4. **Find** Section 3 and confirm the clause on "bypassing any of our systems or protective measures."
5. **Open** https://support.claude.com/en/articles/11175166 and confirm that MCP is described as "an open standard, created by Anthropic."
6. **Open** https://support.claude.com/en/articles/11176164 and confirm that Claude Code is listed as a supported client for connectors.
7. **Search** the public Anthropic Consumer Terms and AUP (https://www.anthropic.com/legal/aup) for exact language matching "misrepresent identity" or "third-party traffic against subscription limits." Report the results.

If any source contradicts this memo, file an issue immediately.

---

## If Anthropic clarifies otherwise

**If Anthropic's legal team, support, or a definitive Terms update clarifies that this tool's design violates their policy,** the response is straightforward:

1. Disable the connector: `claude mcp remove brain-router`
2. Stop delegating to other providers immediately.
3. The tool is fail-safe because it adds nothing to Anthropic traffic — removing it has zero impact on Claude Code's functionality.

No existing conversations, no cached data, no configuration pollution. The tool was never integrated into Anthropic's infrastructure.

---

## Sources

| Source | URL | Purpose | Accessed |
|--------|-----|---------|----------|
| Consumer Terms of Service | https://www.anthropic.com/legal/consumer-terms | Governs prohibited uses, credential sharing, abuse | 2026-06-29 |
| Acceptable Use Policy | https://www.anthropic.com/legal/aup | Governs harmful use cases (CSAM, weapons, fraud, etc.) | 2026-06-29 |
| Get started with custom connectors using remote MCP | https://support.claude.com/en/articles/11175166 | Confirms MCP is Anthropic's standard for extensions | 2026-06-29 |
| Use connectors to extend Claude's capabilities | https://support.claude.com/en/articles/11176164 | Confirms Claude Code is a supported MCP client | 2026-06-29 |

---

## Disclaimer

This memo reflects analysis of publicly available Anthropic terms as of 2026-06-29. It is not legal advice. Anthropic has not reviewed this tool and makes no endorsement. If you have concerns about compliance, contact Anthropic's support team directly before using this tool in a production environment.
