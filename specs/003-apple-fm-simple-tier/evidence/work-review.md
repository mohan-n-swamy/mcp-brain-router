## work-review

- F1 — manuf-product-design was not invoked. Exemption claimed: internal routing tier, no product or UI surface; authority = Mohan's pasted brief plus .rigor.md. Mohan may reject it.
- F2 — n/a: no UI.
- F3 — Gate misfires, recorded with evidence and not worked around by editing correct artifacts: the integration gate fired on the word "codex" in an edit (switched to the Edit tool); the claims hook cannot match paths containing spaces (re-ran a probe whose object path has none); doctrine-check H6 (F9 above).
- F4 — Cross-project claims: "kimi/glm/ds caches read by provider-quota.sh, statusline-hosts.sh, tmux" = grep hits only, files not read. No reuse claims made.
- F5 — Debugging conclusions and their data: checklist lint failure ← Source lines include ~/.agents/AGENTS.md (grep) + lint passes after re-stamp; Codex failure ← 401 lines in its log; brew failure ← curl 35 in brew output and a working 302 to release-assets.githubusercontent.com.

