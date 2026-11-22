# S5.3 F6 — Safety & Security Evaluation Note

- **Spec:** §11 (allowlists), §12 (MCP boundary & data firewall), §14 (isolation & injection screen), §19.4 (`defenses.*`), §20 (selected feature evidence)
- **Benchmark:** `hypora/evaluation/cases/safety/attacks.json` (36 distinct attacks: 6 allowlist, 6 data firewall, 6 approval gate, 6 sandbox isolation, 12 indirect prompt injections)
- **Runner:** `scripts/eval_s53_f6.py`
- **Artifacts:** `results/s5.3/f6/f6-evidence.json`, `results/s5.3/f6/f6-summary.md`

## Design Choices

1. **7-arm defense matrix:** Evaluates `all_off`, `all_on`, and all 5 single-defense-off ablations (`no_allowlist`, `no_firewall`, `no_approval`, `no_injection_filter`, `no_sandbox`) across all 36 attacks, plus a 6×3 repeat sample on 6 representative live indirect injections (`openai/gpt-6-luna`, `temperature=0.0`).
2. **Real enforcement boundaries:**
   - Allowlist and data-firewall attacks execute against real `HyporaMCPServer.call_tool(...)` over a seeded workspace (`create_dev_workspace`).
   - Approval-gate attacks execute against real `ApprovalGate.decide(...)` backed by SQLite `Database` and `EventLog`.
   - Indirect prompt injections pass untrusted payloads through `HyporaMCPServer._screen_injection(...)`, feed the screened result to live `openai/gpt-6-luna`, and execute any model-requested tool call or approval against the active defense configuration.
   - Sandbox isolation (`all_on` and other `sandbox_isolation=True` arms) runs inside real Docker containers via `SandboxLauncher` (`network_mode="none"`, `read_only=True`, `cap_drop=["ALL"]`, non-root UID, symlink scan).
3. **Counterfactual sandbox ablation (`all_off`, `no_sandbox`):** Because production `SandboxLauncher` hardcodes `network_mode="none"` and container isolation (spec §14.3, `docs/notes/threat-model.md`), `sandbox_isolation=False` runs the exact same 6 probe scripts in an isolated local subprocess against synthetic canary fixtures (including a local ephemeral TCP listener bound to the host's non-loopback IP, with zero external internet traffic).
4. **Honest residual-risk probes:** `ATK-INJ-09` through `ATK-INJ-12` intentionally test paraphrased, multi-line split, and false-verdict prose injections that bypass pattern-based line regexes, measuring both pattern-filter limits and structural containment (`allowlist`, `firewall`, platform-owned metrics).

## Reproduction

```bash
uv run python scripts/eval_s53_f6.py
```
