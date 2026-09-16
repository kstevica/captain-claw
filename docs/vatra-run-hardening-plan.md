# Vatra — Run Hardening Plan (long-form deliverables on weak models)

> Status: **PLAN — not started** (2026-09-15). Synthesised from the Captain Spark
> incident report (`docs/vatra-run-hardening-incident.md`, "story run 3",
> DeepSeek v4.1 Flash on every slot), eight subsystem code maps, three independently
> drafted plans scored by three judges, and a claim-by-claim verification pass
> (36 load-bearing claims: 30 confirmed, 6 confirmed with the nuances folded in below).
> Every file:line below was opened on `main` at bec2b8d0; re-check line numbers
> before editing, symbols are the stable reference.
> Backend restart required on deploy; no DB migrations; FD bundle rebuild only for
> the optional Increment 9 UI surface.

Date: 2026-09-15. Related: `docs/vatra-quality-tightening-plan.md` (envelope conventions),
`docs/vatra-execution-groups-plan.md` (phases/wait/clarify), `docs/weak-model-tending-plan.md`
(Phase 0 only shipped; Phases 1–4 unstarted).
Goal: a Vatra run driven over the HTTP API with a fast/weak model on every slot still
produces ONE correct, complete, verified long-form deliverable — because every place the
runtime currently trusts the model (write tool, tool loop, board wait, dispatch order,
reporter reply, done flip) gets a deterministic check with a bounded retry.

---

## Context — what happened, and the confirmed root causes

Run 3 spawned five specialists (A: Long-Horizon Planner, Deep Researcher, Fact Checker;
B: two `editor-writer` slots writing Part One / Part Two) and a Reporter. The deliverable
that reached Spark was a ~3 KB hand-off note pointing at a `fair-measure.md` that never
existed; one part file was an 88-byte stub; chapters were duplicated and missing across
the seam; the QA specialist audited a "reconstructed" bible; every board wait expired
while its producer was still running; the Lead denied the one gap-closing request.

### The key insight: the placeholder write is the compaction pointer round-trip

After every successful `write` whose content is ≥ 200 chars, the agent runtime rewrites
the model's OWN history: `_compact_write_tool_call` (`captain_claw/agent_session_mixin.py:966-1035`)
replaces the assistant tool-call's `content` argument with

```
compact_ref = (f"[written to disk: {path}, {lines} lines, {size_kb:.1f}KB"
               f" — use read tool to view]")          # agent_session_mixin.py:997-1000
args_dict["content"] = compact_ref                    # agent_session_mixin.py:1019
```

and rewrites the persisted assistant message's `function.arguments` in place as a JSON
string (`agent_session_mixin.py:1020-1022`). The context builder passes that stored message
through unchanged (`agent_context_mixin.py:3761`, ~3797) and the wire serialiser forwards a
string argument verbatim (`llm/__init__.py:922-923` for OpenAI-style providers, 1262-1280
for the Responses API), so on every provider the marker is what the model re-reads as its
own prior write content. The call site fires on the tool's self-reported success alone
(`agent_tool_loop_mixin.py:1267-1273`). So a weak model that later "re-issues its last
write" or "saves work" from its own compacted history emits the marker as the file body.
Nothing stops it:

- `WriteTool.execute` writes any string verbatim (`captain_claw/tools/write.py:393-395`);
  the only content guards (`_STATUS_CONFIRMATION_PATTERNS` at 17-33, shrink guard at
  345-350) live inside `if not append and file_path.exists()` (292) and require an EXISTING
  file ≥ 1024 bytes — a first 88-byte write to a new path passes, is not compacted (< 200
  chars, `agent_session_mixin.py:993`), and returns `Written 88 chars …` with the hint
  "Do NOT read this file back — you already know its contents" (`write.py:428-440`).
- `validate_arguments` checks key presence only (`tools/registry.py:196-202`); empty
  `content` passes.
- No readback/size/hash verify exists on the tool-loop path; `_write_file_with_verification`
  (`agent_file_ops_mixin.py:763`) checks `is_file()` only and is called solely by the
  auto-write paths (343/494/894/930).
- The BotPort prompt explains compaction ("This is NORMAL — the file IS saved",
  `botport_client.py:68-70`); the Vatra/Basna worker directive (`agent_context_mixin.py:3086-3095`)
  and `_vfs_directive` (`vatra_routes.py:200-218`) carry no such note — the Deep
  Researcher's 18:53 "the write captured a placeholder" self-correction is that confusion.

### The "truncated path" is the action log, not the write path

Nothing on the write path shortens a filename (`vfs.py:418-447` joins segments verbatim;
`write.py:388-390` only uses the suffix for HTML unescape), and a length-cut JSON argument
becomes `{"raw": …}` → `Missing required argument: path`, never a short path. The strings
Spark saw come from `_summarize_tool_args` (`basna_routes.py:2343`):
`", ".join(f"{k}={str(v)[:40]}" …)` — `vfs:vatra-<sid8>/eppo-authenticity-pac` and
`vfs:vatra-<sid8>/s5-story-bridge-ch4-7` are each exactly 40 characters, and that string is
what `_on_action` renders into the progress feed (`vatra_routes.py:1463-1466`). The stub
that shipped kept its `.md`. Fix the log; keep the extension rule only as a cheap guard.

### The other confirmed causes

| Incident item | Cause (verified) |
|---|---|
| Reporter returned a pointer note as `truth` | `reporter.md:28` licenses "keep your reply to a short pointer"; `_run_reporter` returns any non-empty reply (`vatra_routes.py:3158-3165`); `_capture_generated` scans only `DATA_DIR/<slug>/data/workspace` (793-822), never the VFS folder the reporter is told to write to (3141); no existence check; `status=done` unconditional (2342-2345). |
| Waits expired on live producers | Flat clamp `wait = min(_MAX_WAIT_S, remaining, max(requested, _WAIT_MIN_S))` (3577) with `_MAX_WAIT_S=120 / _WAIT_TOTAL_BUDGET_S=300 / _WAIT_MAX_ATTEMPTS=6` (143-146); tool caps a call at 90 s (`tools/vatra.py:265`) and always renders "Proceed with your best alternative" (286-289); producer liveness exists only as a local in `_send_chat_and_collect` (`basna_routes.py:2600`). |
| Query wait unsatisfiable | `search_vatra_board` matches the WHOLE phrase as one `LIKE` (`db.py:1617-1619`). |
| Same-archetype writers invisible to each other | `CLAW_VATRA_OWNER` = archetype id; board read/search/wait exclude `from_owner == caller` (`db.py:1621-1623`); ledger keyed by owner (3557). Both writers and the default reporter are `editor-writer`. |
| Two writers raced | Same-group owners start in one `asyncio.gather` (1848; flat 1942); `resolve_groups` pulls a violated dependency DOWN into the dependent's group (`vatra_groups.py:174-178`; user-locked `group_lock` dependencies are exempt and stay later), so producer and consumer run concurrently and only the board can bridge them; `results_by_id`/`done` are stamped after the group's gather (1936-1938) except for owners called forward by `_run_pulled` (1781); no declared artifact per subtask. |
| Lead denied the gap fix | `_lead_clarify` = one `_creds("reason")` call, `default_max=400`, deny on any exception (1730-1745); `parse_clarify` "Defaults to DENY on any parse trouble" (`vatra_groups.py:335-336`); rule 3 "DENY … anything the requester could get itself from the shared board" (324-325); roster keyed by archetype id (1880-1881); denial `continue`s (1907-1915). |
| No canon QA, no gate | Only research-shaped checks exist (`research_consistency.py:40-41` kinds figure/date/identifier); all passes read `truth` (the note); done is unconditional. |
| Fact Checker audited a reconstruction | Prompt-compliant: "never stop to say you're missing teammate input — produce your best version" (`vatra_routes.py:2743-2745`). |

## Decisions locked (2026-09-15)

1. **Absent config == byte-for-byte today**, with exactly two default-on exceptions,
   both bug-fixes no correct run can trip: (a) the write tool refuses empty bodies and
   bodies matching the runtime's own acknowledgement/compaction markers and verifies the
   write on disk (`CLAW_WRITE_GUARD=0` kill-switch); (b) the action log stops cutting paths.
2. Everything else rides the `quality_profile` envelope or new request fields. A new
   `long_form` preset turns on the FREE/structural levers; paid levers (`canon_pass`,
   `gate_blocks_done`, `reject_pointer_truth` retry) stay explicit, per the existing rule.
3. **No new terminal status.** A failed deliverable/canon gate persists `status="error"`
   with `truth`, `files`, `analysis.quality_verdict` kept — the branch the "no usable
   output" path already uses (`vatra_routes.py:2085`). Verified: `/continue` requires only
   a non-empty parent `truth` (`vatra_routes.py:4253-4255`) and `/resume` REFUSES only
   `done`+truth sessions (4128), so an `error` run is both continuable and resumable.
4. **Names come from code, not the Lead.** Per-subtask artifact filenames are assigned
   deterministically (caller manifest → derived slug); the Lead/planner may enrich, never
   define.
5. **Scheduler waits, not the model.** Declared inputs gate dispatch before a slot is
   acquired; `vatra wait` stays the fallback for ad-hoc inputs and never blocks longer than
   `_MAX_WAIT_S` per HTTP call (the waiter's own 180 s activity window must keep ticking).
6. Pure logic in new shared modules (`captain_claw/write_guard.py`,
   `flight_deck/deliverable_manifest.py`, `flight_deck/research_canon.py`); route files get
   thin call sites. Worker context via `CLAW_*` env; `FD_*` stays server/deployment.
7. The compaction marker TEXT is not changed (BotPort prompt + any UI matcher would need
   coordinated edits); the write tool rejects it and the worker directive explains it.
8. Pointer→file substitution in the reporter is flag-gated (`resolve_pointer_truth`) and
   requires a pointer-SHAPED reply, never a mere `vfs:` mention — a research reporter
   citing `vfs:proj/sources.md` in prose must not have its truth replaced.

## What exists already (reuse, do not rebuild)

- Producer-side heartbeat extension: `_EXTEND_ACTIVITY_S=180`, `_EXTEND_SLICE_S=300`,
  `_EXTEND_CAP_X=3`, `_extend_deadline` (`basna_routes.py:110-124`), `last_activity` bumped
  on every ws message (2600), `timed_out` flagged. Only the WAITER side is missing.
- `_dispatch_one(file_paths=…)` (`basna_routes.py:2670-2674`) already attaches files to a
  dispatch; used for user uploads at `vatra_routes.py:1559-1563`.
- Max-parallel gate is acquired INSIDE `_dispatch_one` (2676-2679), so anything awaited
  before it holds no slot.
- `quality.push_deps` / `_dep_output_block` (`vatra_routes.py:319-322`, injected 1540-1541,
  cap `_DEP_PUSH_CAP=12_000`) inlines a FINISHED earlier-phase producer's output.
- `quality.acted_gate` corrective re-dispatch (1565-1577); `ACTED_CORRECTIVE` pattern.
- Bounded revise-until-clean gate: `quality_findings.run_gate` (132), `LOOP_KINDS` (34),
  `block_on_critical` + `block_max_rounds` + `TokenBudget`; wired at `vatra_routes.py:2287-2321`.
- Extract→code-verify→bounded-revise shape with audit OUTSIDE the deliverable:
  `research_consistency.run_check` (362) / `write_audit` (502) → `deliverable.consistency.md`.
- Stall retry lever: `MAX_STALL_RETRIES=2` (`agent_orchestration_mixin.py:272`) and
  `setattr(self.provider, "_tool_choice_override", "required")` (2656), honoured by
  `LiteLLMProvider` (`llm/__init__.py:2907-2910`), Ollama (2536-2538), Responses (1457-1460).
- `LLMResponse.finish_reason` (`llm/__init__.py:65`) exists; inside the agent loop it is
  only ledgered (`agent_guard_mixin.py:558, 587`) — no continuation-on-truncation logic.
  Flight Deck route code already branches on it elsewhere (Lead planner
  `vatra_routes.py:528-539` fails with 502; team forge `server.py:6504` retries once with a
  bumped max_tokens) — reuse that shape for the worker-side handling in Increment 2.
- `_creds("reason")` for Lead decompose (`vatra_routes.py:1006` Group-0 route, 1145
  execute route) and clarify (1733); reporter tier = archetype tier (3089; `editor-writer`
  is `"tier": "balanced"`, `instructions/archetypes.json:118`, tools include `edit`).
- `quality` is honoured on `/plan/approve` (3924 → 3976) and inherited by continuation
  rounds; `/start` accepts it (3744) but never persists it (4198-4204).
- `_vfs_resolve_under` (= `captain_claw.vfs.resolve_under`, imported at `vatra_routes.py:107`).
- Later-group schedule guard + pull-forward for query waits (3519-3521, `match_later_owner`).
- Blind-rewrite guard (`agent_tool_loop_mixin.py:967-1005`) and duplicate-call guard (874-880,
  `duplicate_call_max=1` at `config.py:543-550`).
- `run_tests` conventions: `asyncio_mode = "auto"` (`pyproject.toml:117`); route-handler
  harness with `_FakeDB` + monkeypatched `plan_vatra_group0`/`asyncio.create_task`
  (`tests/test_flight_deck/test_vatra_start_tiers.py:46-59`).

## Config knobs — incident §3 name → actual field

| Incident knob | Actual name | Where | Default | `long_form` preset |
|---|---|---|---|---|
| `vfs.write.reject_placeholder` | env `CLAW_WRITE_GUARD` (unset/1 = on) + `tools.write.reject_placeholder` | `write_guard.py`, `config.py ToolsConfig.write` | **on** | n/a (global) |
| `vfs.write.verify_readback` | env `CLAW_WRITE_GUARD` + `tools.write.verify_readback`, `verify_retries=2` | same | **on** | n/a |
| (extension / empty / size floor) | `quality.write_guard` → worker env `CLAW_WRITE_STRICT=1`, `CLAW_WRITE_MIN_BYTES`, `CLAW_DECLARED_FILES`, `CLAW_MY_ARTIFACT` | `_vatra_env` | off | **on** |
| (retry malformed tool calls) | `tools.malformed_call_retries` (config) / env `CLAW_MALFORMED_CALL_RETRIES` | `config.py`, `_spawn_worker` | 0 (workers: 2) | injected for every Vatra worker |
| (declared deliverable) | `deliverable: dict` request field | `VatraStartRequest`, `VatraPlanApproveRequest`, `ExecuteRequest` | None | — |
| — | `quality.derive_manifest` | `quality_profile.py` | False | **on** |
| `synthesis.emit` | `quality.synthesis_emit` `""` \| `"rewrite"` \| `"concat_then_smooth"` | `quality_profile.py` | `""` | `concat_then_smooth` |
| (description-as-deliverable) | `quality.resolve_pointer_truth` (resolve + one corrective retry, budget-gated) | `quality_profile.py` | False | **on** |
| `run.complete.require_deliverable_file` | `quality.require_deliverable_file`, `quality.deliverable_min_chars`, `quality.deliverable_min_sections` | `quality_profile.py` | False / 0 / 0 | **on** |
| `synthesis.require_inputs_present` | `quality.require_inputs` (scheduler gate + BLOCKED semantics) | `quality_profile.py` | False | **on** |
| `board.wait.mode = heartbeat` | `quality.wait_heartbeat` | `quality_profile.py` | False | **on** |
| `board.wait.default_ms` / `long_form` profile | `quality.wait_max_total_s` (un-charged alive-wait ceiling; 0 → the run's `dispatch_timeout`) — per-call cap stays `_MAX_WAIT_S` by design | `quality_profile.py` | 0 | 0 |
| (sequential seam) | `quality.strict_deps` | `quality_profile.py` | False | **on** |
| (gap-closing clarify) | `quality.clarify_dep_grant` | `quality_profile.py` | False | **on** |
| `qa.canon_pass` | `quality.canon_pass` | `quality_profile.py` | False | off (paid) |
| `qa.gate_on_contradictions` | `quality.gate_blocks_done` (pairs with `block_on_critical`) | `quality_profile.py` | False | off (paid) |
| (QA out of the deliverable) | `quality.deliverable_kind` `""` \| `"document"` \| `"fiction"` | `quality_profile.py` | `""` | `""` (Spark sets `fiction`) |
| (stronger Lead/Reporter) | `role_tiers: {lead, planner, clarify, reporter, qa}` request field (tier NAMES) | Vatra request models + `ExecuteRequest` | None | — |
| (dispatch wall) | `dispatch_timeout` exposed on `VatraStartRequest` / `VatraPlanApproveRequest` | `vatra_routes.py` | 600 | — |
| (weak model signal) | `tiers[<name>].weak: true` → env `CLAW_MODEL_WEAK=1` | `_spawn_worker` | absent | — |

`long_form` = `balanced` ∪ {`write_guard`, `derive_manifest`, `resolve_pointer_truth`,
`require_deliverable_file`, `require_inputs`, `wait_heartbeat`, `strict_deps`,
`clarify_dep_grant`, `push_deps`} + `synthesis_emit: concat_then_smooth`. `judgment_ledger`
is deliberately NOT in it. New bool flags join `bool_flags` in `from_dict` (`quality_profile.py:206-215`)
and stay OUT of `_BOOL_FLAGS` (249-256) like `push_deps`/`micro_workers`/`blast_radius_gate`/
`constraint_learning` (all four are in `bool_flags` at 214 but not in `_BOOL_FLAGS`, so they
do not count toward `any_enabled`, 258-261).

---

## Increment 1 (P0) — Write boundary: placeholder refusal, readback receipt, honest action log

**Goal.** No write persists the runtime's own acknowledgement/compaction string or an
empty body; every write is verified on disk before success; the action log stops cutting
paths; multi-agent workers get the BotPort compaction rule; declared deliverable files get a
size floor and auto-repair of a cut basename.

**Changes.**

- NEW `captain_claw/write_guard.py` (core package, no `flight_deck` imports):
  `PLACEHOLDER_RE = re.compile(r"^[\s\[]*(?:written to disk|use read tool to view|\.\.\. shell command truncated|Written \d+ chars \(\d+ lines\) to\b)", re.I)`
  (anchored; also matches `_compact_shell_tool_call`'s tail);
  `is_placeholder_content(text)`; `EXTENSIONLESS_ALLOW = {Makefile, Dockerfile, LICENSE, README, CHANGELOG, Procfile, CNAME}` + dot-files;
  `requires_extension(path)`; `repair_declared_name(basename, declared: list[str]) -> str|None`
  (unique-prefix match); `verify_readback(file_path, content, append, prev_size) -> {ok, bytes, sha8, reason}`
  (fsync, `stat().st_size` vs `len(content.encode())` (+`prev_size` when appending), sha256 for
  files ≤ 4 MB, length-only above); `guard_enabled()` reads `CLAW_WRITE_GUARD` (unset/`1` → on)
  AND `get_config().tools.write.*`.
- `captain_claw/config.py` `ToolsConfig` (next to `duplicate_call_max`, 543-550): nested
  `write: WriteGuardConfig` = `{reject_placeholder: True, verify_readback: True, verify_retries: 2, require_extension_vfs: True}`
  and `malformed_call_retries: int = 0` (Increment 2).
- `captain_claw/tools/write.py` `WriteTool.execute`:
  - Tier 1 (default-on) gate inserted after `file_path` is resolved and before
    `file_path.parent.mkdir` (278): `content.strip() == ""` or `is_placeholder_content(content[:200])`
    → `ToolResult(success=False, error="placeholder_content_rejected", content="❌ Refused: the content is empty or is a tool acknowledgement/compaction marker, not file text. Re-issue `write` with the FULL text (for long files: write the first part, then append=true for the rest).")`.
    Applies to `append=True` too.
  - Tier 2 (worker-only, `CLAW_WRITE_STRICT=1`): for `is_vfs_path(path)`: no suffix and not
    allowlisted → try `repair_declared_name` against `CLAW_DECLARED_FILES` (JSON list); on a
    unique hit rewrite the path and add `(repaired from <given>)` to `redirect_note`; else
    `error="path_missing_extension"`. If `CLAW_WRITE_MIN_BYTES` is set and the basename is in
    `CLAW_DECLARED_FILES` and `len(content.encode()) < floor` (non-append) →
    `error="content_below_floor"` ("this is a declared deliverable part; a summary is not acceptable").
  - Post-write (after `f.write` at 393-395, before `record_author` 397-400): `f.flush(); os.fsync()`;
    `rb = verify_readback(...)`; retry the open/write up to `verify_retries`; still bad →
    `error="write_verify_failed"` with observed vs expected bytes.
  - Receipt goes into `system_hint`, NOT `result_msg` (421 stays byte-identical — 
    `_parse_written_path_from_tool_output`'s regex `\bto\s+(.+?)(?:\s+\(requested:|\s*$)`
    at `agent_file_ops_mixin.py:216` would capture a same-line receipt as part of the path,
    and a newline-separated receipt makes that parse return `None` — either way the
    receipt must not touch `result_msg`).
    New hint (replaces 428-440): "Saved and verified ({bytes} bytes, sha {sha8}). Your history
    may later show this write as `[written to disk: …]` — that is a compaction marker, NOT
    file content; never pass it as `content`. Use `edit` for targeted changes; use `read`
    only if you need the text back."
- `captain_claw/flight_deck/basna_routes.py` `_summarize_tool_args` (2340-2346): per-key
  cap `200 if k in ("path","file","url","pattern","name") else 40`; outer cap 120 → 260.
- `captain_claw/agent_context_mixin.py` shared-filesystem directive (3086-3095): append
  "After a successful write the system may compact the content in your history to
  `[written to disk: …]`. This is NORMAL — the file IS saved; never re-issue a write with
  that marker as content and never rewrite a file because you see it. Always give files a
  full name with an extension (e.g. `.md`)."
- `captain_claw/flight_deck/vatra_routes.py` `_vatra_env` (352-368): when `quality.write_guard`
  (read from a per-sid `_run_flags[sid]` set at execute entry, popped in the teardown block
  with `_wait_ledger` at 2073) emit `CLAW_WRITE_STRICT=1`, `CLAW_WRITE_MIN_BYTES`,
  `CLAW_DECLARED_FILES`, `CLAW_MY_ARTIFACT` (values from Increment 3). Reporter `extra_env`
  (3096 region) gets the same.

**Knobs.** `CLAW_WRITE_GUARD`, `tools.write.*`, `quality.write_guard` (→ strict worker tier).

**Tests.**
- `tests/test_tools/test_write.py` (extend; pattern `await WriteTool().execute(path=str(tmp_path/'x.md'), content=…)`):
  placeholder body `[written to disk: vfs:p/x.md, 3 lines, 0.1KB — use read tool to view]` →
  `success is False`, `error == "placeholder_content_rejected"`, no file; same for the shell
  marker and for `append=True` against an existing file (unchanged); `content=""`/whitespace
  refused; a file whose SECOND line mentions "written to disk" accepted; `system_hint`
  contains `Saved and verified` and `sha`; `result_msg` still matches
  `_parse_written_path_from_tool_output`; `monkeypatch.setenv("CLAW_WRITE_GUARD","0")` → today's behaviour.
- vfs strict tier: monkeypatch `captain_claw.tools.write.is_vfs_path`→True,
  `write.project_is_readonly`→False, `write.resolve_vfs_path`→`lambda p, create_parents=False: tmp_path/Path(p).name`
  (write.py imports these names at module level), `setenv CLAW_WRITE_STRICT=1`:
  `vfs:proj/eppo-authenticity-pac` → `path_missing_extension`; with
  `CLAW_DECLARED_FILES='["eppo-authenticity-pack.md"]'` → success and `(repaired from` in
  content; `vfs:proj/Makefile` accepted; with `CLAW_WRITE_MIN_BYTES=2000` and a declared
  basename, a 300-char body → `content_below_floor`; without the env → extensionless accepted.
- readback: monkeypatch `write_guard.verify_readback` → `ok=False` twice then `ok=True` → success
  after retries; always False → `write_verify_failed`.
- NEW `tests/test_flight_deck/test_summarize_tool_args.py`: a 43-char `path=` survives
  untruncated; a 200-char `content=` is still cut to 40.
- NEW `tests/test_write_guard.py`: `PLACEHOLDER_RE` matches the exact `compact_ref` format
  built by `agent_session_mixin.py:997-1000` (closes the round trip) and the shell marker;
  does not match prose; `repair_declared_name('eppo-authenticity-pac', [...])` unique-prefix semantics.

**Acceptance (criterion 1).** Every `write` result carries a verified byte count; a
placeholder write appears in the progress `action` feed as
`write ✗ placeholder_content_rejected`; `actions[*].detail` shows full vfs paths.

---

## Increment 2 (P0) — Tool loop: automatic bounded retry of malformed / cut-off tool calls

**Goal.** A truncated JSON argument, a `finish_reason == "length"` response carrying a
`write`/`edit` call, a missing required key, or a write refused by Increment 1 becomes a
structured corrective plus a forced tool call on the next turn — never a partial execution,
never an error the model may ignore. Scoped to ensemble workers.

**Changes.**
- `captain_claw/agent_tool_loop_mixin.py` argument parse (786-790): when `json.loads`
  fails and the tool has required params and `self._malformed_retries_allowed() > 0`, do NOT
  execute; add a tool message "Your `{name}` call arrived with truncated/invalid JSON
  (required: {required}). Re-issue the SAME call in full. For a large `write`: write the
  first part, then call `write` again with `append=true`." and bump
  `self._malformed_retry_count`. Bounded JSON repair is used ONLY to recover `path` for the
  message text — never to execute partial `content`. Keep the `{"raw": …}` fallback for
  tools without required params and when retries are 0.
- Post-execute block (1262-1273): when `result.error in {"placeholder_content_rejected", "path_missing_extension", "content_below_floor", "write_verify_failed"}`,
  count it as a malformed attempt and `discard` the path from `_blind_write_paths` (the
  guard set at 1060-1071) so the corrected re-issue is not blocked; the dup counter needs
  no change (`write` signatures are full-args, 887-892; the exception path already rolls
  back at 1628-1631).
- `captain_claw/agent_orchestration_mixin.py` main loop, beside the stall-retry block
  (2585-2658): if the batch produced a malformed corrective and
  `_malformed_retry_count <= retries`, `setattr(self.provider, "_tool_choice_override", "required")`
  (the provider-side attribute the stall retry already sets at 2656; honoured by
  `LiteLLMProvider` at `llm/__init__.py:2907-2910`). If `response.finish_reason == "length"`
  and `response.tool_calls` contains a `write`/`edit`, skip execution of those calls and
  inject the split-with-append corrective. Reset the counter where `_stall_retry_count` resets.
- `captain_claw/config.py` `tools.malformed_call_retries: int = 0`; the loop uses
  `max(config, int(os.environ.get("CLAW_MALFORMED_CALL_RETRIES") or 0))`.
- `vatra_routes.py` `_spawn_worker` (686-688): every Vatra worker and the reporter get
  `CLAW_MALFORMED_CALL_RETRIES=2` (mirrors the existing `_WORKER_MARKER` injection at 688).

**Knobs.** `tools.malformed_call_retries` (default 0 = today), env `CLAW_MALFORMED_CALL_RETRIES`.

**Tests.** `tests/test_agent/test_tool_calling.py` (scripted `LLMProvider` subclasses driven
with `Agent(provider=…)`): (a) provider emits `ToolCall(name="write", arguments='{"path": "vfs:p/ch6.md", "content": "The archive was')`
then a valid write → with `CLAW_MALFORMED_CALL_RETRIES=2` the first call is not executed,
the tool message contains "truncated/invalid JSON" and "append=true",
`provider._tool_choice_override == "required"` before the second call, the file is written
once; (b) three malformed calls → no third override, loop finalises with the error; (c)
`LLMResponse(finish_reason="length", tool_calls=[write…])` → not executed, corrective present;
(d) retries 0 and no env → byte-identical today (`Error: Missing required argument: path`);
(e) a `placeholder_content_rejected` write followed by a full write to the same path in the
same turn is not blocked by the blind-rewrite guard.

**Acceptance.** Supports criterion 1: a Flash run log shows `malformed call — retry 1/2`
followed by a verified write; no worker ends with an unresolved `Missing required argument`.

---

## Increment 3 (P0) — Deliverable manifest and code-assigned artifact names

**Goal.** Before any owner is dispatched the run knows the target file, its ordered parts,
one owner per part, ranges and the seam owner — as a checkable object in session config;
every subtask has an artifact filename assigned by code and injected into its environment.

**Changes.**
- NEW `captain_claw/flight_deck/deliverable_manifest.py` (pure; no DB/LLM):
  `@dataclass Part{path, order, owner, range, min_bytes, sequential}`,
  `@dataclass Manifest{path, kind, min_bytes, min_sections, section_regex, parts, seam_owner}`;
  `parse(raw, subtasks, vfs_project)` (normalise to `vfs:<project>/<name>`, require an
  extension, drop unknown owners, reject duplicate paths, default `order` from position,
  `sequential=True` when `kind == "fiction"`); `derive(group0_plan, subtasks, vfs_project, kind_hint)`
  (uses planner `produces_file`/`range` when present, else `f"{id}-{slugify(title)[:40]}.md"` in
  depends_on-topological then subtask order; ADDS a `depends_on` edge between consecutive
  ranged parts when the Lead omitted it); `assign_artifacts(subtasks, manifest)`;
  `parse_range("ch6-7")`, `chapters_in(text)` (headings `^#{1,3}\s*(chapter|part)\s+(\d+|one…forty)`),
  `count_sections(text, regex)`, `inputs_for(manifest, subtask_id, subtasks)`,
  `topo_layers(...)` (cycle members share one layer), `to_analysis(...)`.
- Request models: `deliverable: dict | None = None` and `role_tiers: dict | None = None`
  on `ExecuteRequest` (`basna_routes.py:2717`), `VatraStartRequest` (`vatra_routes.py:3729`),
  `VatraPlanApproveRequest` (3912); `dispatch_timeout: float | None = None` on the two Vatra
  models. `start_vatra` (4198-4204) persists `quality`, `deliverable`, `role_tiers`,
  `dispatch_timeout` into the session config and threads them on the `ExecuteRequest`;
  `approve_vatra_plan` resolves `body.deliverable if not None else cfg.get("deliverable")`
  like `quality` at 3976; `_knob_updates` (1088-1105) persists `deliverable`/`role_tiers`
  (tier NAMES only); `_continue_run` inherits `parent_cfg.get("deliverable")`.
- `execute_vatra`, right after the plan/Group 0 map is built: `_manifest = parse(...) or (derive(...) if quality.derive_manifest else None)`;
  `assign_artifacts` → `st["artifact"]`; persist `cfg["deliverable_resolved"]`; progress
  `Deliverable manifest: <path> ← N part(s)`; publish `artifact → subtask` in the
  `_owner_activity` registry (Increment 5).
- `_vatra_env` (352-368): `CLAW_DECLARED_FILES` (JSON list of all part + deliverable
  basenames), `CLAW_MY_ARTIFACT`, `CLAW_WRITE_MIN_BYTES` (part `min_bytes`).
- `_plan_slice_block` (287-313) gains `manifest=None`: "Your deliverable FILE:
  `vfs:<proj>/<part>` — write it with EXACTLY this name. Your range: <range> — nothing
  outside it. Write chapter-by-chapter: first `write`, then `append=true` per chapter; never
  one giant write, never a summary or a pointer." Seam owner: "You own the seam: finish
  <range end> completely — the next part starts from your frozen file." Byte-identical when None.
- Group 0 planner (`_build_group0_prompt` 2752-2820, `_coerce_group0_entries` 2844-2891):
  optional per-agent `produces_file`, `range`; top-level `deliverable {path, kind, sections}`;
  coerced to str with `""` defaults (additive).
- `captain_claw/instructions/vatra/lead.md` (after the `depends_on`/`group` rule near
  line 16): "**One continuous artifact, one writer — or ordered ranges.** A manuscript or
  report body that must read as one piece gets ONE writer subtask. If split, give each part
  a hard non-overlapping `range`, make each later part `depends_on` the previous one, and
  never put two subtasks with the same `owner_archetype_id` on the same range."
  `_normalize_plan` (419-461) copies optional `artifact`/`range` tolerantly.
- `quality_profile.py`: `derive_manifest`, `deliverable_kind`, `deliverable_min_chars`,
  `deliverable_min_sections`, `write_guard`, and the `long_form` preset via a small
  `_PRESET_VALUES` table for string/int members applied before explicit keys.

**Knobs.** `deliverable` (Spark sends `{"path": "fair-measure.md", "kind": "fiction", "min_bytes": 60000, "min_sections": 12, "parts": [{"path": "part-one-ch1-5.md", "order": 1, "range": "ch1-5"}, {"path": "part-two-ch6-10.md", "order": 2, "range": "ch6-10"}], "seam_owner": 1}`;
`owner` may be omitted on `/start` — resend with owners from the polled `route.subtasks` on
`/plan/approve`, or omit `parts` and rely on `derive_manifest`); `quality.derive_manifest`;
`quality.deliverable_kind`. Read-back: `config.deliverable_resolved`, `config.vfs_project`.

**Tests.** NEW `tests/test_flight_deck/test_deliverable_manifest.py` (pure, pattern
`test_vatra_dep_push.py`): normalisation, extension requirement, duplicate rejection,
unknown owner dropped, `parse_range`, `chapters_in` incl. number-words, `topo_layers` on a
chain / cycle, `derive` with and without planner fields, fiction → sequential, derive adds the
missing `depends_on` edge. `test_vatra_group0.py` (extend, fixtures `_SUBTASKS`/`_ARCH`):
new keys pass through and default to `""`; `_plan_slice_block(..., manifest=None)`
byte-identical. `test_vatra_start_tiers.py` (extend `_FakeDB`, record kwargs):
`create_basna_session` config contains `quality` and `deliverable`; captured `ExecuteRequest`
carries both. `test_quality_profile.py`: `long_form` membership; `canon_pass`, `claim_check`,
`block_on_critical`, `gate_blocks_done` off in every preset; new flags absent from `_BOOL_FLAGS`;
`from_dict(None).any_enabled is False`. `test_vatra_groups.py`: add
`test_lead_prompt_teaches_single_artifact_rule` (existing `group`-field pin still passes).

---

## Increment 4 (P0) — Synthesis emits the artifact; done gate

**Goal.** `truth` is the bytes of one assembled file in the run's VFS folder (or a handle
that resolves to it); parts are concatenated by code with deterministic seam findings; the
reporter only smooths seams in place; a pointer/description reply never ships; the run cannot
persist `done` with a missing, undersized or placeholder deliverable.

**Changes.**
- `deliverable_manifest.py`: `part_status(vfs_dir, part, producer_done) -> {exists, bytes, placeholder, sections, chapters, landed, sha8}`
  with `landed = exists ∧ bytes ≥ min_bytes ∧ ¬placeholder ∧ (producer_done ∨ ¬sequential)`;
  `assemble(vfs_dir, manifest) -> {text, parts, seams, findings}` joining parts in `order`
  and emitting findings in the `quality_findings` shape (`source: "assembly"`):
  `duplicate_chapter`, `missing_chapter`, `overlap_heading`, `placeholder_part`, `part_missing`;
  `gate(manifest, text, statuses, min_chars_fallback) -> {ok, reasons, bytes, sections, chapters}`.
- `vatra_routes.py` pure helpers next to `_capture_generated` (793): `_POINTER_RE = r"vfs:[^\s`'\")\]]+"`,
  `_pointer_paths(text)`, `_looks_like_pointer(out, inputs_len)` =
  `len(out) < max(800, 0.25 × inputs_len)` OR (pointer phrases `assembled and complete` /
  `see the file` / `saved to` / `DELIVERABLE_FILE:` AND a `vfs:`/`saved/` payload) OR
  `is_placeholder_content(out)`; `_resolve_deliverable(user_id, project, candidates, min_chars)`
  via `_vfs_resolve_under` (107) choosing the largest resolvable candidate.
- NEW `_capture_vfs_written(vfs_dir, artifact, started, finished, actions)` + step 4
  backfill (2047-2065) under `quality.derive_manifest`/manifest present: the owner's
  declared artifact (exact name, from Increment 3) is read; if `r["output"]` is empty or
  pointer-shaped, `r["output"] = file text`, `r["produced_file"] = True`, `r["vfs_files"] = [...]`;
  file copied into `dest_dir` and listed in `generated_files` with `kind: "generated"` plus a
  `vfs` key so the session `files[]` shows it. (No mtime heuristics — names are exact.)
- `execute_vatra` step 5 (before `_run_reporter` at 2143), `synthesis_emit == "concat_then_smooth"`:
  `asm = assemble(...)`; write `asm.text` to `vfs:<proj>/<manifest.path>` and a backup
  `dest_dir/<path>.pre-smooth`; append `{name, kind: "vfs", path, size, sha8}` to
  `generated_files`; progress `Assembled N parts → vfs:…/<file> (X bytes, Y sections)`; pass
  `smooth_target=_manifest` to `_run_reporter`. Assembly findings are kept for the gate.
- `_run_reporter` (3046-3168) new kwargs `smooth_target`, `resolve_pointer`, `deliverable_kind`:
  - Smooth mode: prompt = module-level `REPORTER_SMOOTH_DIRECTIVE` (reporter.md untouched)
    "The deliverable is ALREADY assembled at vfs:<proj>/<path> (N bytes). Read it. Fix ONLY
    the seams listed below with the `edit` tool (transitions, duplicated sentences, tense/name
    drift). Do NOT rewrite chapters, do NOT summarise, do NOT write any other file. Reply with
    the single word DONE." + seam list. Reporter env carries `CLAW_DECLARED_FILES=[deliverable]`.
    After dispatch re-read the file; collapse guard (`len < 0.9 × pre-smooth` OR
    `count_sections` dropped OR placeholder) → restore from `.pre-smooth` and note it.
    `truth = file bytes`; the chat reply is ignored. (`edit.py` has NO shrink guard — this
    guard is the protection; `write.py`'s shrink guard does not apply.)
  - Rewrite mode with `resolve_pointer` (after `out = …` at 3158): if
    `_looks_like_pointer(out, len(slices_full))` — and only then — resolve
    `[manifest.path] + _pointer_paths(out)`; a resolved file larger than `out` and not
    placeholder → `out = text`, file listed. Else ONE corrective re-dispatch on the same
    reporter (budget-gated: `_budget.can_afford(_retry_est)`): "Your reply was a note ABOUT
    the deliverable. Output the COMPLETE document as your reply, or write the WHOLE document to
    vfs:<proj>/<path> and reply exactly `DELIVERABLE_FILE: vfs:<proj>/<path>`." Re-resolve; if
    still pointer-shaped → progress `Reporter ✗ pointer reply — raw assembly used` and return
    `(fallback, files)` (3075-3077). Existing `if not out and text: out = text` and
    `return (out or fallback), files` remain for the flag-off path.
- `captain_claw/instructions/vatra/reporter.md:28`: replace "save the COMPLETE document to
  your workspace — never a partial file — and keep your reply to a short pointer" with
  "write the COMPLETE document to `vfs:<project>/<name>.md` in the shared project folder —
  never a partial file — and reply with exactly `DELIVERABLE_FILE: vfs:<project>/<name>.md`
  (a reply that merely describes the file is a failed synthesis)". Line 11: add "the team's
  part FILES may also be in vfs:<project>/ — `glob` and `read` them before assembling".
- Done gate (`execute_vatra` between the blocking gate ending at 2321 and the persist at
  2331-2345), under `quality.require_deliverable_file` or a manifest: `g = gate(...)`. On
  failure and `_budget.can_afford(_retry_est)`: ONE bounded repair — for each
  `part_missing`/`placeholder_part` reason re-dispatch that part's owner via
  `_redispatch_owner` (1712; hoist it out of the grouped closure so flat mode can use it)
  with "Your declared file vfs:<part> is missing/empty. Write it IN FULL now
  (chapter-by-chapter, append=true), then reply DONE"; re-assemble (no smoothing) and
  re-gate. Persist `analysis.deliverable = to_analysis(...) | {verdict, reasons}`. Still
  failing → NEW shared `_finish_blocked(sid, user, reason, truth, files, analysis, progress)`:
  persists `status="error"`, `truth` (best assembly), `files`, `analysis.quality_verdict`
  (`"deliverable_missing"`), emits `_progress(sid, "done", f"Deliverable gate FAILED: {reason}", ok=False)`,
  `_progress_done(sid)`, returns the success-shaped dict with `status: "error"`. Never leaves
  `running`; never discards work.
- `quality_profile.py`: `synthesis_emit`, `resolve_pointer_truth`, `require_deliverable_file`
  (parsed like `output_mode` at 224-228 for the string).

**Knobs.** `quality.synthesis_emit`, `quality.resolve_pointer_truth`,
`quality.require_deliverable_file`, `quality.deliverable_min_chars/_min_sections`.
Read-back: `analysis.deliverable.{verdict, reasons, parts[*].{path,bytes,sha8,landed}}`,
`files[]` entries with `vfs`/`kind: "vfs"`, `truth` = file bytes; or
`GET /fd/vfs/read?project=<config.vfs_project>&path=<config.deliverable_resolved.path>`
(`vfs_routes.py:676`; >1 MB truncates — an 18k-word draft is ~110 KB).

**Tests.**
- `test_deliverable_manifest.py` (extend): `assemble` orders by `order`; duplicate
  `# Chapter Six` in two parts → `duplicate_chapter`; range ch6-7 with only ch6 →
  `missing_chapter`; 88-byte marker part → `placeholder_part`; `part_status` sequential +
  `producer_done=False` → `landed False`; `gate` reasons `deliverable_missing` /
  `deliverable_placeholder` / `too_few_sections` / ok.
- NEW `tests/test_flight_deck/test_vatra_reporter.py` (harness: monkeypatch `vr._spawn_worker`
  → `{ok:True, slug:'rep', port:1, auth:'t', message:''}`, `vr._teardown`/`vr._track_worker`/`vr._progress`
  → noop, `captain_claw.flight_deck.server.DATA_DIR` → tmp_path, `vr._vfs_resolve_under` →
  tmp resolver, `vr._dispatch_one` → async canned dict): (1) pointer reply
  `Fair Measure is assembled and complete. File: vfs:p/fair-measure.md`, file absent,
  `resolve_pointer=True` → two dispatches, truth == `fallback`; (2) file present 50 KB → truth
  == file bytes, `files[0]["vfs"]` set; (3) `resolve_pointer=False` → today's behaviour
  (note returned, one dispatch); (4) a 40 KB reply that merely mentions `vfs:p/sources.md` with
  a larger file on disk is NOT replaced; (5) smooth mode: parts pre-written, fake replies
  DONE and truncates the file to 30% → truth == pre-smooth bytes and a restore note; (6)
  smooth mode small edit → truth == edited bytes, prompt contains `Reply with the single word DONE`
  and not `Integrate, don't concatenate`.
- NEW `tests/test_flight_deck/test_vatra_done_gate.py`: extract the gate step as
  `vr._deliverable_gate_step(sid, user, manifest, quality, vfs_dir, dest_dir, redispatch_fn, budget)`;
  a fake `redispatch_fn` that writes the missing part → ok after exactly one repair; a no-op
  fake → verdict `deliverable_missing` and a FakeDB (recording `update_basna_session` kwargs)
  sees `status="error"` with non-empty `truth`. `_finish_blocked` unit test.
- `test_honesty_guard.py`-style prompt capture for `REPORTER_SMOOTH_DIRECTIVE`.

**Acceptance (criteria 2, 3).** `analysis.deliverable.verdict == "ok"`, `files[]` lists
`vfs:vatra-<sid8>/fair-measure.md` with size ≥ floor, `truth` length ≈ file size; assembly
findings `duplicate_chapter`/`missing_chapter` are deterministic; a run whose deliverable is
absent ends `status: error` + `analysis.quality_verdict: deliverable_missing`, never `done`
with a note.

---

## Increment 5 (P1) — Producer liveness, scheduler input gate, honest waits

**Goal.** A consumer with declared inputs never enters `vatra wait` at all — the scheduler
awaits the landed parts before a slot is acquired; for ad-hoc inputs a wait never expires on
a producer that is alive; the tool relays the server's verdict; same-archetype teammates can
see each other; multi-word queries can match; a re-read of a grown file is not "duplicate";
a QA/synthesis owner may say BLOCKED instead of fabricating.

**Changes.**
- NEW module registry `_owner_activity: dict[sid, dict[subtask, {state, started, finished, last_seen, arch, role, title, depends_on, artifact, vfs_files}]]`
  next to `_group_schedule` (154), populated in flat AND grouped mode (separate dict so
  `_sched` truthiness at 1544-1546 is untouched — `schedule_block` returns `""` for empty
  owners anyway, `vatra_groups.py:212-215`). `state="running"` at the top of `_dispatch_owner`
  (1506), `done/failed` at its end and in `_redispatch_owner`; popped in the teardown block
  (2069-2076). Per-part `asyncio.Event`s `_part_events[sid][part.path]` set when
  `part_status(...).landed` after the producer returns.
- `basna_routes.py` `_send_chat_and_collect` / `_dispatch_one` (2670-2674): new
  `on_activity: Callable[[], None] | None`, invoked where `last_activity` is bumped (2600);
  `_owner_callbacks` (1452-1477) passes a lambda stamping `last_seen` (also stamped in
  `_on_action/_on_usage/_on_status`).
- `_dispatch_owner` input gate (`quality.require_inputs`, before the prompt build at
  ~1530 and therefore before `_dispatch_one` acquires the max-parallel gate at 2676-2679):
  `needed = inputs_for(manifest, st["id"], subtasks)` (or, without a manifest, the finished
  artifacts of `depends_on` producers); `await _await_parts(sid, needed, deadline)` on the
  Events with a 2 s VFS-poll fallback, bounded by the producer's own hard deadline
  (`min(3×dispatch_timeout, 3600)` + 30 s); cycles collapse to today's parallel start via
  `topo_layers`. Independent owners keep full parallelism. Progress:
  `⏳ <consumer> waiting for vfs:…/part-one.md (producer s4 running, last activity 12s ago)`.
  Landed parts are attached to the consumer via the existing `file_paths` (1559-1563) —
  copied into `DATA_DIR/<slug>/data/workspace` — with a one-line manifest in the prompt
  ("## Attached inputs from teammates (already delivered)\n- part-one.md (Editor · Part One, 41 KB)");
  `_dep_output_block` inline push stays for outputs ≤ `_DEP_PUSH_CAP`.
- `agent_wait` (3463-3628): `_VatraWaitReq.subtask_id: str = ""`. Resolve producers:
  manifest artifact by `path` basename; `query` via a generalised `vatra_groups.match_owner`
  (extend `match_later_owner`, 247, to all owners); else the caller's `depends_on`. Heartbeat
  (`quality.wait_heartbeat`, stored in `_run_flags[sid]`): at the per-call deadline, if a
  producer is `running` and `now − last_seen ≤ basna_routes._EXTEND_ACTIVITY_S` and
  `rec["alive_waited"] < wait_max_total_s (0 → dispatch_timeout)`, charge `alive_waited`
  only (not `waited`/`attempts`) and return
  `{ready: False, producer_alive: True, can_retry: True, producer: {subtask, role, state, last_seen_s}, note: "<role> is still working on it (last activity Ns ago). Wait again with the same target — do not proceed without it."}`.
  Per-call cap stays `_MAX_WAIT_S` (every call returns within 120 s so the waiter's own
  `_send_chat_and_collect` activity window keeps ticking). Path-ready (3583-3594) additionally
  requires `¬is_placeholder_content` and `bytes ≥ CLAW_WRITE_MIN_BYTES`-equivalent floor;
  otherwise `present_but_placeholder: True`. Apply the later-group guard (3519-3521) to
  `path` waits too. Ledger key = `body.subtask_id or body.owner` (3557).
- `db.py` `list_vatra_board` (1592) / `search_vatra_board` (1612-1627): `exclude_subtask`
  (`AND from_subtask != ?`); tokenised search — tokens ≥ 3 chars minus stopwords, ALL tokens
  as separate `(content LIKE ? OR title LIKE ?)` clauses restricted to
  `kind IN ('output','note','file')` (a narration row must not satisfy a wait), OR the
  existing whole-phrase `LIKE` across all kinds. Endpoints pass `exclude_subtask=body.subtask_id`
  when present, else today's `exclude_owner`.
- `tools/vatra.py`: send `subtask_id` (`_context` 113) on wait/read/search; clamp
  `min(90,…)` → `min(120,…)` (265) and `timeout_seconds` 120 → 150 (49); param doc "0–120"
  (71); not-ready text branches on `producer_alive` ("STILL WORKING … call `vatra`
  action='wait' again with the same target. Do NOT proceed without it."), `present_but_placeholder`,
  `exhausted`/`not_scheduled`/`pulled_forward` (relay server `note`), else today's text.
- `agent_tool_loop_mixin.py` dup signature (874-880): for `read` of a `vfs:` path append
  `|{st_size}:{st_mtime_ns}` via `resolve_vfs_path` (best-effort), so a grown file gets a fresh
  counter; message (938-942) states the observed size instead of "The content has not changed".
- `_build_subtask_prompt` (2704-2706, 2736-2742): replace "(up to 90s)" / "(once)" with
  "it blocks while the producer is still working and tells you whether to wait again". Under
  `require_inputs` for owners with `depends_on`, replace the autonomy sentence (2743-2745)
  with "If a declared input never arrives (the `vatra` tool says the producer stalled or your
  wait budget is spent), reply with exactly `BLOCKED: <artifact>` and stop — do NOT
  reconstruct or invent a substitute." `_dispatch_owner`: `^BLOCKED:` in the first 3 lines →
  `d["ok"]=False`, `d["blocked"]=True`, `kind="gap"` board post, progress `⛔ blocked on …`;
  `_result_of` (1667-1674) carries `blocked` so `usable` (2082) excludes it. If the producer
  finished without landing its artifact, re-dispatch the PRODUCER once with the Increment 4
  corrective, then the consumer once (budget-gated).
- `quality_profile.py`: `require_inputs`, `wait_heartbeat`, `wait_max_total_s`.

**Knobs.** `quality.require_inputs`, `quality.wait_heartbeat`, `quality.wait_max_total_s`;
tokenised search, `exclude_subtask`, mtime-aware re-reads ship default-on (strictly
more-permissive). `execution_groups` is NOT required.

**Tests.**
- NEW `tests/test_flight_deck/test_vatra_wait_heartbeat.py` (FakeDB with async
  `get_basna_session`→`{'id':sid,'config':'{}'}`, `list_vatra_board`→[], `search_vatra_board`→[];
  `vr._resolve_owner`→'u1', `vr.merged_archetypes`→async [], `vr._WAIT_POLL_S=0.01`,
  `vr._WAIT_MIN_S=0`, `vr._MAX_WAIT_S=0.05`, `vr._vfs_resolve_under`→tmp file; seed
  `vr._owner_activity[sid]` and `vr._run_flags[sid]={'wait_heartbeat': True, 'wait_max_total_s': 60}`):
  alive producer + absent file → `ready False`, `producer_alive True`, `_wait_ledger[...]['waited'] == 0`;
  `last_seen` 400 s old → charged; file non-empty on the 3rd poll → `ready True`; 88-byte
  marker file → `present_but_placeholder`; heartbeat off → today's charge (control); path
  wait naming a later-group artifact is refused/pulled like a query wait.
- NEW `tests/test_flight_deck/test_vatra_input_gate.py`: `_await_parts` returns immediately
  when Events are set; file becomes non-placeholder after N polls → landed; producer `failed`
  → `missing` without waiting to the deadline; a recording fake `dispatch` (pattern
  `test_max_parallel.py:39-68`) proves the consumer starts after the producer returns and its
  `file_paths` contains the copied part; `BLOCKED:` reply excluded from `usable`.
- NEW `tests/test_tools/test_vatra_tool_wait.py`: monkeypatch `VatraTool._post`; payload
  includes `subtask_id`; `{producer_alive: True, producer:{role:'Editor', last_seen_s:12}}` →
  "STILL WORKING"/"wait again", never "Proceed with your best alternative"; `{exhausted: True, note: 'X'}` → 'X'.
- NEW `tests/test_flight_deck/test_vatra_board_search.py` (real `Database` on a tmp sqlite
  path; add a fixture if none exists): an `output` row titled 'Story Bible' with 'clue register …
  sealed space … culprit' in the body and a `narration` row with the same words → query
  'Story Bible clue register sealed space culprit' returns the output row only;
  `exclude_subtask='s5'` hides s5's row while a same-archetype s4 row stays visible.
- `tests/test_agent/test_tool_calling.py`: two `read(vfs:p/a.md)` in one turn with the file
  appended between them → second read executes; unchanged → blocked.
- `test_dispatch_resilience.py` (extend): `_dispatch_one(..., on_activity=cb)` forwards.
- Pure: `_build_subtask_prompt(..., require_inputs=True)` contains `BLOCKED:` and not
  "produce your best version"; default call byte-identical (assert against captured output).

**Acceptance (criterion 5).** Progress shows `⏳ … (producer s4 running, last activity Ns ago)`
and a dispatch that starts AFTER `s4 ✓`; zero `⌛ … timed out` events whose
`producer.state == running`; the tool never renders "Proceed with your best alternative"
while the producer is alive.

---

## Increment 6 (P1) — Seam ownership and the gap-closing clarify

**Goal.** Same-wave dependency chains are honoured as start-after-finish (the manifest's
sequential parts, the Lead's `depends_on`); a consumer receives the frozen producer file; a
request naming a missing declared range of a finished provider is granted by code, appended
to the provider's own file; "denied" and "unparseable" are distinguishable; a denial is
recorded as a gap.

**Changes.**
- `vatra_groups.py`: `dep_layers(members)` (Kahn; cycle → one layer) and
  `match_owner_phrase(text, owners)`; `resolve_groups(..., strict_deps=False)` (138) — when on,
  a same-group dependency pushes the DEPENDENT to `eff[dep]+1` (≤ `_MAX_ORD`, `group_lock`
  wins) instead of pulling the producer down (174-178); default off keeps
  `test_vatra_groups.py:171-230` green.
- Grouped loop (1848) and flat round (1942) under `quality.strict_deps`: the Increment 5
  per-owner `_await_parts` gate is the mechanism (finer than layer-gather, holds no slot);
  `dep_layers` is used only to detect cycles and to order `results_by_id` population per
  owner (not after the whole group, cf. 1936-1938) so `_dep_output_block` sees the producer
  as finished.
- `_dep_files_block(st, activity)` appended after 1541 for finished deps: "## Your dependency
  <role> — <title> is FINISHED (frozen). Files: vfs:<proj>/<name> — N bytes. Read them; continue
  exactly where they end; never rewrite or duplicate them." (files > `_DEP_PUSH_CAP` are
  attached via `file_paths` per Increment 5).
- Clarify loop (1879-1915), `quality.clarify_dep_grant`: before `_lead_clarify`,
  `g = deliverable_manifest.gap_request(req, manifest, providers)` (chapter/range mentions
  via `parse_range` + number-words, or a declared part path, against parts owned by roster
  providers that are in the requester's `depends_on` or finished) → approve without the LLM;
  provider instruction "Append the missing <range> to vfs:<part> (append=true), keep the
  existing text untouched, then reply DONE"; progress `Lead (auto): dependency-declared request granted`.
  Roster entries carry `subtask` id; provider match prefers subtask id, then archetype id
  (1907-1908). On deny: strip the `REQUEST:` line from `results_by_id[...]["output"]`, post
  `kind="gap"`, append to a run-level `declared_gaps` merged into `analysis.gaps`. Grants
  count against `CLARIFY_CAP` (`vatra_groups.py:21`), exposed as `quality.clarify_cap` (default 2).
- `vatra_groups.parse_clarify` (335): return `parse_failed: True` on no/invalid JSON (still
  deny); `clarify_prompt` (297) takes the requester's subtask title + `depends_on` and the
  provider's declared part/range and adds rule 0 "If the request names a specific missing
  part/range of a provider's DECLARED artifact and that provider is listed, APPROVE with
  that provider." (additive; existing fragment tests at 107-115 / 347-367 stay true).
- `_lead_clarify` (1730-1745): `default_max` 400 → 800; creds from
  `role_tiers.get("clarify")` (Increment 8) else `"reason"`; log the first 300 chars of the
  raw reply on parse failure so "Lead denied" vs "Lead reply unparseable" are distinguishable.

**Knobs.** `quality.strict_deps`, `quality.clarify_dep_grant`, `quality.clarify_cap`. The
clarify loop exists only in grouped mode (1868-1942), so the grant needs `execution_groups: true`
on `/plan/approve`; `strict_deps` does not.

**Tests.** `test_vatra_groups.py` (extend, pure): `dep_layers` chain/independent/cycle;
`match_owner_phrase('supply Part One Chapters Six and Seven', […])`; `resolve_groups(strict_deps=True)`
moves the dependent up and respects `group_lock`; `parse_clarify('sure thing')['parse_failed'] is True`;
rule-0 text present. `test_deliverable_manifest.py`: `gap_request(...)` → provider s4, range
(6,7), instruction contains `append=true`; vague ask → None. NEW `test_vatra_clarify_gap.py`:
extract `vr._decide_clarify(sp, req, providers, manifest, lead_fn)`; with a manifest match the
async `lead_fn` is never awaited; on deny the output no longer contains `REQUEST:` and
`declared_gaps` has one entry; caplog shows `raw=` on parse failure. `_dep_files_block` pure test.

**Acceptance (criterion 3).** Part Two's dispatch event is timestamped after Part One's
`dispatch ✓`; Part Two's brief lists Part One's file; the 19:09 "supply Chapters Six and
Seven" request lands as `Lead (auto): dependency-declared request granted → append ch6-7`,
not `Lead denied`.

---

## Increment 7 (P1) — Canon QA on the assembled artifact, patch-mode fixes, gate that blocks done

**Goal.** A deterministic continuity checker runs on the resolved deliverable bytes (never a
reconstruction, never `truth[:6000]`), feeds the existing bounded gate, fixes by
quote-anchored patches a Flash reviser can produce, writes its ledger to an audit file, and
can block completion while criticals remain — with QA prose kept out of the deliverable.

**Changes.**
- NEW `captain_claw/flight_deck/research_canon.py` (mirrors `research_consistency.py`):
  `CHUNK_CHARS=24_000`, `chunk_text(text)` on `^#{1,3} ` headings (never mid-sentence),
  `extract_prompt(chunk, n)` → JSON `{entities[{name, fixed:{age, …}, quote}], scenes[{id, order, day, hour, place, present:[{name, whereabouts}], quote}], amounts[{label, paid, attempted, quote}], tallies[{claim, count, items, quote}], alibis[{suspect, alibi, quote}], dated_facts[{scene_id, date_cited, quote}]}`;
  `parse_canon`, `merge` (normalised entity name / scene order), `verify` kinds:
  `fixed_fact`, `whereabouts`, `alibi` (>1 alibi per suspect), `amount` (paid > attempted),
  `tally` (len(items) ≠ count), `chronology` (date_cited > scene date), `duplicate_scene`
  (identical day+hour+place+present in two chunks — the seam-duplication signal);
  `patch_prompt` → `[{find: <exact quote>, replace}]`; `apply_patches(text, patches) -> (text, applied, unapplied)`;
  `run_check(text, *, extract_fn, revise_fn, on_progress)`; `write_audit` → `deliverable.canon.md`
  (pattern `research_consistency.write_audit`, 502). Full re-emission only when `len(text) < 20_000`.
- `quality_findings.py`: `from_canon(findings)` (source `"canon"`); `LOOP_KINDS` (34) becomes
  a per-source map `{"consistency": {identity, relation}, "canon": {fixed_fact, whereabouts, alibi, amount, tally, chronology, duplicate_scene}, "assembly": {duplicate_chapter, missing_chapter}}`;
  `loop_drivers` (85-89) checks by source; `run_gate` (132) gains `canon_recheck_fn` and
  `revise_mode: "full"|"patch"` (patch → `revise_fn` returns patches, gate applies via
  `apply_patches`; unapplied patches stay as remaining findings). Wire the dead
  `from_claim_check` (61) as passengers when `claim_check` ran.
- `vatra_routes.py`: new 5b3 after the consistency block (ends 2258), before the claim
  check: `if quality.canon_pass and truth and _budget.can_afford(2*_retry_est)` → run on
  `truth` (which after Increment 4 IS the merged file); creds `_creds(quality.qa_tier or "fast")`
  for extraction, `qa_tier or "reason"` for patches; audit → `generated_files`;
  `analysis.canon`. Gate assembly (2289-2294): `findings += from_canon(...) + assembly findings`;
  pass `canon_recheck_fn`; `revise_mode="patch"` when `len(truth) ≥ 20_000`. Under
  `quality.gate_blocks_done`, `gate["verdict"] == "critical_findings_remain"` →
  `_finish_blocked(sid, reason="critical_findings_remain", …)` (Increment 4 helper) with
  `analysis.blocking` populated. Skip all passes with `analysis.qa = {skipped: "deliverable_missing"}`
  when the Increment 4 gate failed.
- `deliverable_kind == "fiction"`: do not append `REPORTER_HONESTY_DIRECTIVE`
  (`vatra_routes.py:3135-3136`; text at `quality_profile.py:443-453` requires an in-text
  "Unresolved & assumptions" section whenever anything is unresolved or assumed — 452-453
  omit it only when nothing is, which on an 18k-word Flash draft is never) nor `JUDGMENT_LEDGER_DIRECTIVE` (1549); keep
  `UNVERIFIED_GUARD_DIRECTIVE` for owners; unresolved items go to `analysis.unresolved`. The
  gate checklist's "note the correction" hint is dropped for fiction.
- `quality_profile.py`: `canon_pass`, `gate_blocks_done`, `qa_tier`, `deliverable_kind`
  (explicit-only for the two paid flags).

**Knobs.** `quality.canon_pass`, `quality.gate_blocks_done` (+ existing `block_on_critical`,
`block_max_rounds`, `token_budget`), `quality.qa_tier`, `quality.deliverable_kind`.

**Tests.** NEW `tests/test_flight_deck/test_research_canon.py` (`_seq_fn([...], calls)` async
stubs as in `test_research_consistency.py`): age 34 vs 41 → `fixed_fact` with both quotes;
one character in two places in one scene → `whereabouts`; two alibis → `alibi`;
12 000 > 10 000 → `amount`; "four things" with three → `tally`; fact dated after its scene →
`chronology`; identical scene tuples in two chunks → `duplicate_scene`; `chunk_text` splits on
headings; `merge` unifies 'Ana Kovač'/'Ana'; `apply_patches` applies anchored edits and
reports unapplied; `run_check` on a 120k-char text calls the extractor ≥ 5 times and keeps
length; collapsing patch set rejected. `test_quality_findings.py` (extend): canon `whereabouts`
drives the loop; `revise_mode='patch'` applies stub patches and rechecks; `from_claim_check`
findings ride as passengers; existing driver/passenger tests unchanged. `test_quality_profile.py`:
both paid flags off in every preset incl. `long_form`. `test_honesty_guard.py` (extend,
prompt capture 94-138): fiction → reporter prompt lacks "Unresolved & assumptions"; default
byte-identical. Done-gate smoke (FakeDB `__getattr__` async no-ops, `vr.quality_findings.run_gate`
→ `critical_findings_remain`): `gate_blocks_done` → `status='error'`; off → `status='done'`.

**Acceptance (criterion 4).** `analysis.canon = {input_bytes ≈ deliverable, chunks: N, findings, rounds, verdict}`,
`deliverable.canon.md` in `files[]`, `analysis.quality_verdict == "clean"` on a `done` run,
or `status: error` with `analysis.blocking` listing survivors; no QA section in the prose
when `deliverable_kind == fiction`.

---

## Increment 8 (P2) — Role tiers, weak-tier signal, dispatch wall, small contexts

**Goal.** Spark can put only the Lead, planner, clarify decision, Reporter and QA on a
stronger tier while specialists stay on Flash; a tier entry can be marked weak so the
worker ladders default on; the per-dispatch wall is settable through the Vatra endpoints;
the reporter reads the slices as a file rather than 12 k inline chars.

**Changes.**
- `role_tiers` (added to the models in Increment 3): `_role_tier(role, default)` helper;
  `_creds(_role_tier("lead","reason"))` at 1006 and 1145; `_lead_clarify` 1733 →
  `_role_tier("clarify", …)`; `_run_reporter` 3089 →
  `tier=_role_tier("reporter", arch.get("tier","reason"))`; Group-0 planner spawn →
  `_role_tier("planner", …)`; QA passes (2185, `_claim_check` checker tier) →
  `_role_tier("qa", …)`. Also accept `reporter_archetype` as a request field (today session
  config only, 3062). Record `analysis.tiers_used = {lead, planner, reporter, qa, specialists: [...]}`
  in the cost/learn step.
- `_spawn_worker` (655-663): read optional `weak: bool` on the tier entry → env
  `CLAW_MODEL_WEAK=1`; default `CLAW_MALFORMED_CALL_RETRIES=2` on; trim the worker tool list
  to the archetype's tools ∩ {read, write, edit, glob, vfs, vatra, facts} (+ web tools for
  research archetypes), logging the removed set. No name-based capability classifier
  (`captain_claw/model_capability.py` does not exist; the weak-model plan's rule would call a
  hosted DeepSeek Flash FRONTIER).
- `dispatch_timeout` on `VatraStartRequest`/`VatraPlanApproveRequest` threaded into
  `ExecuteRequest` at `start_vatra` (4198-4204) and `approve_vatra_plan` (3974-3978);
  persisted via `_knob_updates`.
- `_run_reporter` (3117-3118) under `long_form`/manifest: attach `vatra-slices.md`
  (already written to the workspace at 3105-3107) via `file_paths` and inline only a
  per-piece manifest with sizes instead of `_SLICES_INLINE_CHARS` (119).
- Note for Spark: the headless `POST /fd/vatra/agent/start` (`AgentStartReq`,
  `basna_routes.py:804-816`) carries no `tiers`, `quality`, `dispatch_timeout`,
  `agent_max_tokens`, `horizon` or `router_tier` — only the `_AgentReq` identity fields plus
  task/title/max_agents/origin/source_host. `agent_start` builds its `ExecuteRequest` from
  the owner's saved Library tiers, so a per-slot tier map sent on that path is silently
  discarded today. If Spark uses it, mirror the new fields there (open question 1).

**Knobs.** `role_tiers`, `reporter_archetype`, `dispatch_timeout`, `tiers[<name>].weak`.

**Tests.** `test_vatra_start_tiers.py` (extend): `role_tiers`, `dispatch_timeout` reach the
`ExecuteRequest` and the session config. `test_vatra_micro_workers.py` (`spawn_capture`
fixture, 69-86): `{"weak": true}` yields `CLAW_MODEL_WEAK=1`, `CLAW_MALFORMED_CALL_RETRIES=2`
and a trimmed tool list; without it AgentConfig byte-identical. `test_vatra_reporter.py`
(extend): `cfg={'role_tiers': {'reporter': 'reason'}}` → `_spawn_worker` called with
`tier='reason'` although the archetype says balanced. `_lead_clarify` creds captured via a
monkeypatched `vr._provider_call` (pattern `test_honesty_guard.py:94-138`).

---

## Build order & effort

| # | Increment | Effort | Depends on | Why this order |
|---|---|---|---|---|
| 1 | Write boundary + action log + directive parity | S/M | — | Closes the compaction round-trip and the false "truncated path"; default-on bug-fix, shippable alone |
| 2 | Tool-loop malformed-call retry | M | 1 (error codes) | Makes every Increment-1 refusal self-healing on Flash; scoped by worker env |
| 3 | Manifest + artifact names + request fields + `long_form` preset | M | — | The spine every later gate keys on; persists `quality` on `/start` |
| 4 | Synthesis emit + done gate | L | 3, 1 | Criteria 2/3; `truth` becomes the file; `_finish_blocked` shared with 7 |
| 5 | Liveness registry + scheduler input gate + heartbeat waits + tool relay + search/dup fixes | L | 3 (artifact names), 4 (part_status) | Criterion 5; consumers never enter `vatra wait` for declared inputs |
| 6 | strict_deps + gap-closing clarify | M | 5 (registry/Events), 3 (ranges) | Criterion 3 seam ownership; deterministic grant |
| 7 | Canon QA + patch-mode + gate_blocks_done + fiction kind | L | 4 (truth = file), `_finish_blocked` | Criterion 4; reuses `run_gate` |
| 8 | role_tiers, weak tier, dispatch_timeout, slices-as-file | S/M | 3 (fields) | Lowers the base rate; config-only for Spark |
| 9 | FD surface (optional) | S | 3–8 | Quality tab toggles for the new flags; `npm run build` + bundle commit |

Each increment: `pytest tests/test_flight_deck tests/test_tools tests/test_agent/test_tool_calling.py`
subsets green before/after with `quality` absent (regression guarantee: worktree failure-set
diff against the baseline — full `pytest tests/` hangs), plus the increment's own units.
Commit per increment.

## Acceptance test — incident §4 criteria → observable checks

Re-run the same fiction brief on Flash with
`quality: {profile: long_form, deliverable_kind: fiction, canon_pass: true, block_on_critical: true, gate_blocks_done: true, token_budget: <n>}`,
`deliverable: {...}`, `execution_groups: true`, `role_tiers: {lead: reason, reporter: reason, qa: reason}`.

| # | Criterion | Observable check |
|---|---|---|
| 1 | Every write round-trips; no placeholder stub; no truncated path | Every `action` event for `write` shows a full `vfs:` path (no 40-char cut); every write result hint has `Saved and verified (N bytes, sha …)`; zero `placeholder_content_rejected`/`write_verify_failed` that were not followed by a verified write; no file in `vfs:vatra-<sid8>/` matches `PLACEHOLDER_RE` (`analysis.deliverable.parts[*].landed` all true). |
| 2 | Parts assemble into ONE file in the VFS; `truth` is its content | `analysis.deliverable.verdict == "ok"`, `files[]` has `{kind: "vfs", path: "vfs:vatra-<sid8>/fair-measure.md", size ≥ min_bytes}`, `GET /fd/vfs/read` returns it, `len(truth) == size` (± newline). |
| 3 | No duplicated scene, no missing chapter across the seam | No `duplicate_chapter`/`missing_chapter`/`duplicate_scene` in `analysis.blocking`; progress shows Part Two dispatched after `Part One ✓`; any clarify request logged as `Lead (auto): … granted`. |
| 4 | QA/canon ran on the merged draft; no unresolved contradiction on completion | `analysis.canon.input_bytes` ≈ deliverable size, `deliverable.canon.md` in `files[]`, `analysis.quality_verdict == "clean"` with `status == "done"`; otherwise `status == "error"` + `analysis.blocking`. |
| 5 | No wait expired on a live producer | Zero `⌛ … timed out` progress events whose `producer.state == "running"`; declared-input consumers show `⏳ … (producer running …)` then start after the producer's `✓`. |

## Spark-side changes

- Send `quality` (profile `long_form` + explicit paid flags), `deliverable`, `role_tiers`,
  `dispatch_timeout` on `POST /fd/vatra/start` (persisted after Increment 3); resend
  `deliverable` with owners on `/plan/approve` if parts must map to specific subtasks; set
  `execution_groups: true` on `/plan/approve` for the clarify grant.
- Poll `GET /fd/basna/sessions/{id}`; treat `status == "error"` with `analysis.quality_verdict`
  as "do not export" (not a crash); read the deliverable from `truth` or
  `GET /fd/vfs/read?project=<config.vfs_project>&path=<config.deliverable_resolved.path>`.
- Put the stronger model on `tiers.reason` and mark the Flash tier `weak: true`; send
  `output_ctx ≥ 32768` for Flash so a chapter-sized `append` write fits
  (`max_tokens = int(lt.get("output_ctx") or 0) or 32768`, `vatra_routes.py:662`).
- Fiction templates: guarantee `## Chapter N` headings per chapter (chunking, `chapters_in`,
  `min_sections` depend on it) or pass `section_regex` on the manifest.
- Keep the Spark-side delivery contract and continuity-pass prompts; they now have code behind them.

## Rollout & flags

1. Increment 1 ships with `CLAW_WRITE_GUARD` default on; release note + kill-switch documented;
   FD restart. 2. Increments 2–3 are inert for existing callers (retries only in worker env;
   new fields default None). 3. Increments 4–7 are inert unless `quality`/`deliverable` set;
   `long_form` never enables paid loops. 4. Increment 8 additive. All state rides session
   config JSON, `analysis` JSON and VFS-folder files — no migrations. FD bundle rebuild only
   for Increment 9.

## Risks

- **Anchored placeholder regex false positive** — only a file whose first line is the marker
  is refused; kill-switch. **Readback cost** — one fsync + stat (+ hash ≤ 4 MB) per write,
  under the 10 s tool timeout (`write.py:46`). Shell heredoc writes bypass the tool and stay
  unverified.
- **Retry loops** — every corrective is capped (2 per turn / `MAX_STALL_RETRIES` shape,
  one repair per gate, `block_max_rounds`, `CLARIFY_CAP`) and budget-gated.
- **Serialised writers** — sequential parts roughly double wall time for a two-part
  manuscript; only under `strict_deps`/sequential parts. Cycles fall back to parallel.
- **Long heartbeat waits hold a `max_parallel` slot** — mitigated: declared inputs are gated
  before the slot is acquired; ad-hoc waits are capped by `wait_max_total_s`.
- **`status="error"` on a nearly-complete run** reads harsh in the UI — verdict and kept
  truth make it continuable/resumable; flag-gated.
- **Flash extraction recall in the canon pass** — false positives bounded by quote anchoring
  and unanchored patches never applied; recall improves with `qa_tier`.
- **Smoothing via `edit`** can still reword silently — `.pre-smooth` kept for diffing;
  length/section collapse restores the concat.
- **Prompt bloat** — new directives stack only with flags and stay ≤ 10 lines each.

## Deferred / rejected

- Changing the compaction marker text (coordinated BotPort/UI edits; the regex catches it).
- JSON-repairing and EXECUTING a truncated `write` — repair is used only to name the path in
  the corrective.
- Raising `_MAX_WAIT_S` / lifting the tool clamp to minutes — a single blocking call streams
  nothing and would kill the waiter's own dispatch at the 180 s window; un-charged short waits
  instead.
- A new terminal status (`needs_fix`) — Lupa's poll loop and the FD UI only recognise
  done/error; `error` + kept truth is continuable (4253-4255) and resumable (4128).
- Default-on pointer→bytes substitution on any `vfs:` mention — a research reporter citing
  a source file would have its truth replaced; flag-gated and pointer-shape-required.
- VFS file locks/leases — one declared owner per part plus sequencing removes the race; a lock
  a weak model forgets to release deadlocks the run.
- A full DAG scheduler replacing the phase loop — per-owner `_await_parts` gives the guarantee.
- Name-based model capability classifier (weak-model plan Phase 1) — explicit `weak: true`.
- FTS index for the board — tokenised AND-of-LIKE suffices.
- Auto-chaining `/fill-gaps` on a failed gate — re-decomposes and can recreate the two-writer
  split; the bounded in-run repair re-dispatches the specific owner instead.
- Fiction-specific extra posture directives beyond `deliverable_kind` suppression.

## Open questions

1. Which entry does Spark drive — `/start` + `/plan/approve` (Lupa's flow) or the headless
   `/agent/start` (`AgentStartReq` has no tiers/quality)? Mirror the fields there if the latter.
2. Confirm Spark branches on `status`/`analysis.quality_verdict` rather than expecting a new status.
3. Does Spark's template know the part split before planning (explicit manifest) or should
   `derive_manifest` be the default path?
4. Are chapter headings guaranteed in Spark's manuscripts, or is `section_regex` needed per template?
5. Did run 3's Lead plan carry `depends_on` from Part Two to Part One? (Determines how much
   the `derive`-added edge matters.)
6. Is there an existing tmp-sqlite `Database` fixture in `tests/test_flight_deck`? If not,
   Increment 5 adds one.
7. `_owner_activity` should record `time.time()` alongside `time.monotonic()` so any mtime
   comparison is exact — confirm no host drift concerns.

## Touch list

`captain_claw/write_guard.py` (new) · `captain_claw/tools/write.py` · `captain_claw/config.py` ·
`captain_claw/agent_tool_loop_mixin.py` · `captain_claw/agent_orchestration_mixin.py` ·
`captain_claw/agent_context_mixin.py` · `captain_claw/flight_deck/basna_routes.py` ·
`captain_claw/flight_deck/vatra_routes.py` · `captain_claw/flight_deck/vatra_groups.py` ·
`captain_claw/flight_deck/db.py` · `captain_claw/flight_deck/quality_profile.py` ·
`captain_claw/flight_deck/quality_findings.py` · `captain_claw/flight_deck/deliverable_manifest.py` (new) ·
`captain_claw/flight_deck/research_canon.py` (new) · `captain_claw/tools/vatra.py` ·
`captain_claw/instructions/vatra/reporter.md` · `captain_claw/instructions/vatra/lead.md` ·
tests as listed per increment.

## Verification (after all increments)

- `quality` absent: `tests/test_flight_deck` failure set identical to baseline; prompt
  snapshots byte-identical; only `write` refusals of empty/marker bodies and the action-log
  path length differ.
- The five acceptance checks above hold on a real Flash run of the fiction brief.
