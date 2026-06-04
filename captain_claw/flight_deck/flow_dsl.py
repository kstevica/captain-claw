"""Declarative Flow DSL — a 1:1 textual mirror of the Flow JSON spec.

Two deterministic directions (no model involved):
  • compile_dsl(text)  → flow dict   (parser + structured errors)
  • decompile(flow)    → text        (so the click-builder round-trips)
Plus validate_flow(flow) → list[str] of human errors, used to gate BOTH the
deterministic compile and the agent-assisted NL→flow compiler.

Grammar (indentation-light, line-oriented):

    flow "Name"
    description "optional"
    priority 50
    trigger web when contains "introduction" and has image

    step name:
      input
      prompt: "What is your name?"
      timeout: 3600

    step research:
      agent on origin
      prompt: "Research about {{steps.name.output}}"

    step decide:
      branch
      if {{steps.research.output}} contains "error" -> fail
      else -> ok

    step fail:
      emit "Sorry, that failed."
      stop

    step ok:
      emit "{{steps.research.output}}"

    output -> same

Step header line: a bare type keyword (`tool`/`agent`/`vision`/`input`/`emit`/
`branch`), optionally `on <selector>`, optionally a trailing inline `key: value`.
`emit "..."` is shorthand for an emit step with that body. A bare `stop` line
sets the step's stop flag. Branch bodies use `if/elif/else <cond> -> <target>`
where target is a step id or `stop`.
"""

from __future__ import annotations

import re
from typing import Any

VALID_TYPES = {"tool", "agent", "vision", "input", "emit", "branch"}
TRIGGER_CHANNELS = {"any", "whatsapp", "web", "glasses"}
OUTPUT_CHANNELS = {"same", "whatsapp", "web", "glasses", "log"}
_HAS_FLAGS = {"image", "video", "audio", "text", "document"}
_DEFAULT_ON = {"tool": "origin", "agent": "origin", "vision": "capability:vision"}


class DSLError(Exception):
    """A parse error with a 1-based line number."""

    def __init__(self, line: int, msg: str) -> None:
        self.line = line
        self.msg = msg
        super().__init__(f"line {line}: {msg}")


# ── small helpers ──────────────────────────────────────────────────────


def _unquote(s: str) -> str:
    s = s.strip().replace('\\"', '"')  # tolerate over-escaped quotes from models
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _quote(s: str) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


def _strip_comment(line: str) -> str:
    """Drop a trailing ` # comment`, but not a `#` inside quotes."""
    out, in_q, q = [], False, ""
    i = 0
    while i < len(line):
        c = line[i]
        if in_q:
            out.append(c)
            if c == q:
                in_q = False
        elif c in "\"'":
            in_q = True
            q = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


# ── trigger rules ──────────────────────────────────────────────────────


def _parse_trigger(rest: str, lineno: int) -> dict[str, Any]:
    """`<channel> [when <rules>]` / `<channel> always`."""
    parts = rest.split(None, 1)
    if not parts:
        raise DSLError(lineno, "trigger needs a channel (any|whatsapp|web|glasses)")
    channel = parts[0].lower()
    if channel not in TRIGGER_CHANNELS:
        raise DSLError(lineno, f"unknown trigger channel '{channel}'")
    tail = parts[1].strip() if len(parts) > 1 else ""
    trigger: dict[str, Any] = {"on": "message", "channel": channel,
                               "match": {"kind": "rule", "rules": [], "labels": []}}
    if not tail:
        return trigger
    head, _, cond = tail.partition(" ")
    head = head.lower()
    if head == "always":
        trigger["match"] = {"kind": "always", "rules": [], "labels": []}
        return trigger
    if head != "when":
        raise DSLError(lineno, f"expected 'when' or 'always' after channel, got '{head}'")
    rules, mode = _parse_rules(cond.strip(), lineno)
    trigger["match"]["rules"] = rules
    trigger["match"]["mode"] = mode  # 'all' (and) or 'any' (or)
    return trigger


def _split_rules(text: str) -> tuple[list[str], set[str]]:
    """Split a rule list on top-level ` and ` / ` or ` / commas, ignoring those
    words inside quotes. Returns (chunks, operators-used)."""
    chunks: list[str] = []
    ops: set[str] = set()
    buf = ""
    quote = ""
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            buf += c
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "\"'":
            quote = c
            buf += c
            i += 1
            continue
        if c == ",":
            chunks.append(buf); ops.add("and"); buf = ""; i += 1; continue
        if text[i:i + 5].lower() == " and ":
            chunks.append(buf); ops.add("and"); buf = ""; i += 5; continue
        if text[i:i + 4].lower() == " or ":
            chunks.append(buf); ops.add("or"); buf = ""; i += 4; continue
        buf += c
        i += 1
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()], ops


def _parse_rules(text: str, lineno: int) -> tuple[list[str], str]:
    """Normalize a rule list to the router's stored form and a match mode.
    `and`/commas → mode 'all'; `or` → mode 'any'. Mixing the two is rejected."""
    chunks, ops = _split_rules(text)
    if "and" in ops and "or" in ops:
        raise DSLError(lineno, "don't mix 'and' and 'or' in trigger rules — use one (group with separate flows if needed)")
    mode = "any" if "or" in ops else "all"
    rules: list[str] = []
    for c in chunks:
        low = c.lower()
        if low.startswith("has "):
            what = low[4:].strip()
            if what not in _HAS_FLAGS:
                raise DSLError(lineno, f"unknown 'has' flag: {what}")
            rules.append(f"has_{what}")
        elif low.startswith("contains "):
            rules.append(f"contains:{_unquote(c[len('contains '):])}")
        elif low.startswith("from_waid "):
            rules.append(f"from_waid:{_unquote(c[len('from_waid '):])}")
        elif low.startswith("mime "):
            rules.append(f"mime:{_unquote(c[len('mime '):])}")
        elif low.startswith("regex "):
            rules.append(f"regex:{_unquote(c[len('regex '):])}")
        else:
            # Bare quoted/word → substring contains (matches router semantics).
            rules.append(f"contains:{_unquote(c)}")
    return rules, mode


def _rule_to_dsl(rule: str) -> str:
    if rule.startswith("has_"):
        return f"has {rule[4:]}"
    if rule.startswith("contains:"):
        return f'contains {_quote(rule[len("contains:"):])}'
    if rule.startswith("from_waid:"):
        return f'from_waid {_quote(rule[len("from_waid:"):])}'
    if rule.startswith("mime:"):
        return f'mime {_quote(rule[len("mime:"):])}'
    if rule.startswith("regex:"):
        return f'regex {_quote(rule[len("regex:"):])}'
    if rule in (f"has_{f}" for f in _HAS_FLAGS):
        return f"has {rule[4:]}"
    return f"contains {_quote(rule)}"


# ── compile: DSL text → flow dict ──────────────────────────────────────


def compile_dsl(text: str) -> dict[str, Any]:
    """Parse DSL into a flow dict. Raises DSLError(line, msg) on failure."""
    raw_lines = (text or "").splitlines()
    flow: dict[str, Any] = {
        "name": "", "description": "", "enabled": True, "priority": 50,
        "trigger": {"on": "message", "channel": "any",
                    "match": {"kind": "rule", "rules": [], "labels": []}},
        "steps": [], "guardrails": {"max_steps": 20, "timeout_s": 600},
        "output": {"channel": "same", "format": "text"},
    }
    i = 0
    n = len(raw_lines)
    seen_step_ids: set[str] = set()

    def indent(s: str) -> int:
        return len(s) - len(s.lstrip())

    while i < n:
        raw = _strip_comment(raw_lines[i])
        line = raw.strip()
        lineno = i + 1
        if not line:
            i += 1
            continue
        if indent(raw) > 0:
            raise DSLError(lineno, "unexpected indented line outside a step block")

        low = line.lower()
        if low.startswith("flow "):
            flow["name"] = _unquote(line[5:])
            i += 1
        elif low.startswith("description "):
            flow["description"] = _unquote(line[12:])
            i += 1
        elif low.startswith("priority "):
            try:
                flow["priority"] = int(line[9:].strip())
            except ValueError:
                raise DSLError(lineno, "priority must be an integer")
            i += 1
        elif low.startswith("trigger "):
            flow["trigger"] = _parse_trigger(line[8:].strip(), lineno)
            i += 1
        elif low.startswith("output"):
            m = re.match(r"output\s*->\s*(\S+)", line, re.I)
            if not m:
                raise DSLError(lineno, "output must be 'output -> <channel>'")
            ch = m.group(1).lower()
            if ch not in OUTPUT_CHANNELS:
                raise DSLError(lineno, f"unknown output channel '{ch}'")
            flow["output"] = {"channel": ch, "format": "text"}
            i += 1
        elif low.startswith("step "):
            m = re.match(r"step\s+([A-Za-z0-9_]+)\s*:?\s*(.*)$", line)
            if not m:
                raise DSLError(lineno, "step header must be 'step <id>:'")
            sid = m.group(1)
            if sid in seen_step_ids:
                raise DSLError(lineno, f"duplicate step id '{sid}'")
            seen_step_ids.add(sid)
            inline = m.group(2).strip()
            # Gather the indented body lines.
            body: list[tuple[int, str]] = []
            i += 1
            while i < n:
                braw = _strip_comment(raw_lines[i])
                if braw.strip() == "":
                    i += 1
                    continue
                if indent(braw) == 0:
                    break
                body.append((i + 1, braw.strip()))
                i += 1
            step = _parse_step(sid, inline, body, lineno)
            flow["steps"].append(step)
        else:
            raise DSLError(lineno, f"unknown directive: '{line.split()[0]}'")

    if not flow["steps"]:
        raise DSLError(0, "a flow needs at least one step")
    errs = validate_flow(flow)
    if errs:
        raise DSLError(0, "; ".join(errs))
    return flow


def _parse_step(sid: str, inline: str, body: list[tuple[int, str]], header_line: int) -> dict[str, Any]:
    # The type can be on the step header (after the colon) or the first body line.
    lines = ([(header_line, inline)] if inline else []) + body
    if not lines:
        raise DSLError(header_line, f"step '{sid}' is empty")
    first_lineno, first = lines[0]
    toks = first.split(None, 1)
    stype = toks[0].lower()
    remainder = toks[1].strip() if len(toks) > 1 else ""

    # `emit "body"` shorthand.
    if stype == "emit" and remainder and (remainder[0] in "\"'"):
        step: dict[str, Any] = {"id": sid, "type": "emit", "channel": "same", "body": _unquote(remainder)}
        return _consume_fields(step, lines[1:])

    if stype not in VALID_TYPES:
        raise DSLError(first_lineno, f"step '{sid}': unknown type '{stype}'")

    step = {"id": sid, "type": stype}
    if stype in _DEFAULT_ON:
        step["on"] = _DEFAULT_ON[stype]
    if stype == "emit":
        step["channel"] = "same"
    if stype == "branch":
        step["cases"] = []

    # Type line tail is either `on <selector>` or an inline `key: value`.
    if remainder:
        if remainder.lower().startswith("on "):
            if stype != "branch":
                step["on"] = remainder[3:].strip()
        else:
            _apply_field(step, first_lineno, remainder)

    return _consume_fields(step, lines[1:])


def _consume_fields(step: dict[str, Any], lines: list[tuple[int, str]]) -> dict[str, Any]:
    for lineno, ln in lines:
        low = ln.lower()
        if low == "stop":
            step["stop"] = True
            continue
        if step["type"] == "branch" and (low.startswith("if ") or low.startswith("elif ") or low.startswith("else")):
            _apply_branch_line(step, lineno, ln)
            continue
        if low.startswith("on "):
            step["on"] = ln[3:].strip()
            continue
        _apply_field(step, lineno, ln)
    return step


def _apply_field(step: dict[str, Any], lineno: int, ln: str) -> None:
    if ":" not in ln:
        raise DSLError(lineno, f"expected 'key: value' or a known directive, got '{ln}'")
    key, _, val = ln.partition(":")
    key = key.strip().lower()
    val = val.strip()
    t = step["type"]
    if key == "prompt":
        step["prompt"] = _unquote(val)
    elif key == "tool":
        step["tool"] = _unquote(val)
    elif key == "image":
        step["attach"] = _unquote(val)  # vision uses attach
    elif key == "attach":
        step["attach"] = _unquote(val)
    elif key == "body":
        step["body"] = _unquote(val)
    elif key == "channel":
        step["channel"] = _unquote(val).lower()
    elif key == "timeout":
        try:
            step["timeout"] = int(val)
        except ValueError:
            raise DSLError(lineno, "timeout must be an integer (seconds)")
    elif key == "deny":
        step.setdefault("guardrails", {})["deny"] = [x.strip() for x in val.split(",") if x.strip()]
    elif key == "arg":
        # `arg key: value` is handled below via the 'arg ' prefix instead.
        raise DSLError(lineno, "use 'arg <name>: <value>' for tool arguments")
    elif key.startswith("arg "):
        name = key[4:].strip()
        step.setdefault("args", {})[name] = _unquote(val)
    else:
        raise DSLError(lineno, f"unknown field '{key}' for {t} step")


def _apply_branch_line(step: dict[str, Any], lineno: int, ln: str) -> None:
    low = ln.lower()
    if low.startswith("else"):
        rest = ln[4:].strip()
        m = re.match(r"->\s*(\S+)", rest)
        if not m:
            raise DSLError(lineno, "else must be 'else -> <target>'")
        step["default"] = _norm_target(m.group(1))
        return
    kw = "if" if low.startswith("if ") else "elif"
    cond = ln[len(kw):].strip()
    m = re.search(r"->\s*(\S+)\s*$", cond)
    if not m:
        raise DSLError(lineno, f"{kw} must be '{kw} <condition> -> <target>'")
    target = _norm_target(m.group(1))
    when = cond[: m.start()].strip()
    if not when:
        raise DSLError(lineno, f"{kw} needs a condition before '->'")
    step.setdefault("cases", []).append({"when": when, "goto": target})


def _norm_target(t: str) -> str:
    return "__stop__" if t.lower() in ("stop", "__stop__", "end") else t


# ── validate ───────────────────────────────────────────────────────────


def validate_flow(flow: dict[str, Any]) -> list[str]:
    """Structural validation shared by the DSL and the agent compiler."""
    errs: list[str] = []
    steps = flow.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return ["flow has no steps"]
    ids: list[str] = []
    for idx, s in enumerate(steps):
        sid = s.get("id")
        if not sid:
            errs.append(f"step #{idx + 1} has no id")
            continue
        ids.append(sid)
        t = s.get("type")
        if t not in VALID_TYPES:
            errs.append(f"step '{sid}': invalid type '{t}'")
            continue
        if t == "tool" and not s.get("tool"):
            errs.append(f"step '{sid}': tool step needs a 'tool' name")
        if t in ("agent", "vision", "input") and not str(s.get("prompt") or "").strip():
            errs.append(f"step '{sid}': {t} step needs a prompt")
    idset = set(ids)
    if len(idset) != len(ids):
        errs.append("duplicate step ids")
    # Branch targets must exist (or be the stop sentinel / empty=fall through).
    for s in steps:
        if s.get("type") != "branch":
            continue
        targets = [c.get("goto") for c in (s.get("cases") or [])]
        targets.append(s.get("default"))
        for tgt in targets:
            if not tgt or tgt == "__stop__":
                continue
            if tgt not in idset:
                errs.append(f"step '{s.get('id')}': branch goto '{tgt}' is not a step id")
    out_ch = (flow.get("output") or {}).get("channel", "same")
    if out_ch not in OUTPUT_CHANNELS:
        errs.append(f"output channel '{out_ch}' is invalid")
    return errs


# ── decompile: flow dict → DSL text ────────────────────────────────────


def decompile(flow: dict[str, Any]) -> str:
    lines: list[str] = []
    name = flow.get("name") or ""
    lines.append(f"flow {_quote(name)}")
    if str(flow.get("description") or "").strip():
        lines.append(f"description {_quote(flow['description'])}")
    if int(flow.get("priority", 50)) != 50:
        lines.append(f"priority {int(flow['priority'])}")
    lines.append(_trigger_to_dsl(flow.get("trigger") or {}))
    lines.append("")

    for s in (flow.get("steps") or []):
        lines.extend(_step_to_dsl(s))
        lines.append("")

    out = (flow.get("output") or {}).get("channel", "same")
    lines.append(f"output -> {out}")
    return "\n".join(lines).rstrip() + "\n"


def _trigger_to_dsl(trig: dict[str, Any]) -> str:
    ch = trig.get("channel", "any")
    match = trig.get("match") or {}
    kind = match.get("kind", "rule")
    if kind == "always":
        return f"trigger {ch} always"
    rules = match.get("rules") or []
    if not rules:
        return f"trigger {ch}"
    joiner = " or " if str(match.get("mode") or "all").lower() == "any" else " and "
    return f"trigger {ch} when " + joiner.join(_rule_to_dsl(str(r)) for r in rules)


def _step_to_dsl(s: dict[str, Any]) -> list[str]:
    sid = s.get("id", "step")
    t = s.get("type", "tool")
    out = [f"step {sid}:"]
    if t == "branch":
        out.append("  branch")
        cases = s.get("cases") or ([{"when": s.get("when"), "goto": s.get("goto")}]
                                   if s.get("when") or s.get("goto") else [])
        for idx, c in enumerate(cases):
            kw = "if" if idx == 0 else "elif"
            tgt = "stop" if c.get("goto") == "__stop__" else (c.get("goto") or "")
            out.append(f"  {kw} {c.get('when','')} -> {tgt}")
        default = s.get("default") or s.get("else")
        if default:
            out.append(f"  else -> {'stop' if default == '__stop__' else default}")
        return out

    # emit shorthand: `emit "<body>"` when going to the default channel.
    if t == "emit" and s.get("channel", "same") == "same":
        out.append(f"  emit {_quote(s.get('body', ''))}")
        if s.get("stop"):
            out.append("  stop")
        return out

    # header line with selector
    on = s.get("on")
    header = f"  {t}"
    if t in _DEFAULT_ON or on:
        header += f" on {on or _DEFAULT_ON.get(t, 'origin')}"
    out.append(header)
    if t == "tool":
        if s.get("tool"):
            out.append(f"  tool: {s['tool']}")
        for k, v in (s.get("args") or {}).items():
            out.append(f"  arg {k}: {_quote(v)}")
    if s.get("prompt"):
        out.append(f"  prompt: {_quote(s['prompt'])}")
    if s.get("attach"):
        key = "image" if t == "vision" else "attach"
        out.append(f"  {key}: {_quote(s['attach'])}")
    if t == "emit":
        ch = s.get("channel", "same")
        if ch and ch != "same":
            out.append(f"  channel: {ch}")
        if s.get("body"):
            out.append(f"  body: {_quote(s['body'])}")
    if t == "input" and s.get("timeout") and int(s.get("timeout", 0)) != 3600:
        out.append(f"  timeout: {int(s['timeout'])}")
    deny = (s.get("guardrails") or {}).get("deny") or []
    if deny:
        out.append(f"  deny: {', '.join(deny)}")
    if s.get("stop"):
        out.append("  stop")
    return out
