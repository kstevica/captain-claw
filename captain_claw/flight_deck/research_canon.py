"""Narrative-canon continuity check over an ASSEMBLED long-form draft.

Mirrors ``research_consistency`` (extract → deterministic verify → bounded revise →
re-verify → audit file), but the facts are the ones a fiction/long-form deliverable
must keep straight rather than research figures:

- **fixed_fact** — a character's age / fixed detail stated two different ways.
- **whereabouts** — one character in two places within a single scene.
- **alibi** — more than one distinct alibi for a suspect.
- **amount** — a sum paid greater than the sum attempted.
- **tally** — a "four things" list that does not list four.
- **chronology** — a fact dated LATER than the scene that cites it.
- **duplicate_scene** — the same day/hour/place/cast appearing in two chunks
  (the seam-duplication signal a two-writer split produces).

The extractor is an injected LLM call (quote-anchored, so a fix is a precise
find/replace patch a weak reviser can produce); the verifier is pure code. The
audit is written to a separate file and NEVER folded into the deliverable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from captain_claw.logging import get_logger

log = get_logger(__name__)

AUDIT_SUFFIX = ".canon.md"
CHUNK_CHARS = 24_000
FULL_REEMIT_MAX = 20_000  # above this, revise by patches only (never re-emit the whole draft)

CompleteFn = Callable[[str], Awaitable[str]]


# ── chunking ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,3}\s")


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split *text* into chunks on markdown headings, never mid-sentence."""
    body = text or ""
    if len(body) <= max_chars:
        return [body] if body.strip() else []
    # split points at heading starts
    starts = [m.start() for m in _HEADING_RE.finditer(body)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    segments = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(body)
        segments.append(body[s:e])
    # greedily pack segments up to max_chars
    chunks: list[str] = []
    cur = ""
    for seg in segments:
        if cur and len(cur) + len(seg) > max_chars:
            chunks.append(cur)
            cur = seg
        else:
            cur += seg
    if cur.strip():
        chunks.append(cur)
    return chunks


# ── extraction prompt + parse ─────────────────────────────────────────

def extract_prompt(chunk: str, n: int = 60) -> str:
    return (
        "Extract the load-bearing CANON facts from this part of a narrative, so a "
        "continuity check can run. Copy an exact short QUOTE for each so a fix is a "
        "precise find/replace. Output ONLY this JSON (no prose), each list ≤ "
        f"{n} items:\n"
        "{\n"
        '  "entities": [{"name": "<character>", "fixed": {"age": "<n or empty>"}, "quote": "<exact quote stating it>"}],\n'
        '  "scenes": [{"id": "<short label>", "order": <int>, "day": "<day/date or empty>", "hour": "<time or empty>", "place": "<where>", "present": [{"name": "<character>", "whereabouts": "<where they are in this scene>"}], "quote": "<exact quote>"}],\n'
        '  "amounts": [{"label": "<what>", "paid": "<number>", "attempted": "<number>", "quote": "<exact quote>"}],\n'
        '  "tallies": [{"claim": "<e.g. four suspects>", "count": <int>, "items": ["..."], "quote": "<exact quote>"}],\n'
        '  "alibis": [{"suspect": "<name>", "alibi": "<claim>", "quote": "<exact quote>"}],\n'
        '  "dated_facts": [{"scene_id": "<scene id or empty>", "scene_year": "<year of the scene, or empty>", "date_cited": "<a date/year the fact references>", "fact": "<what>", "quote": "<exact quote>"}]\n'
        "}\n\n"
        "TEXT:\n" + chunk
    )


_LIST_KEYS = ("entities", "scenes", "amounts", "tallies", "alibis", "dated_facts")


def parse_canon(output: str) -> dict:
    """Parse the extractor's JSON into a normalised canon dict (best-effort)."""
    empty = {k: [] for k in _LIST_KEYS}
    if not output:
        return empty
    m = re.search(r"\{.*\}", output.strip(), re.DOTALL)
    if not m:
        return empty
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return empty
    if not isinstance(raw, dict):
        return empty
    out = {k: [] for k in _LIST_KEYS}
    for k in _LIST_KEYS:
        v = raw.get(k)
        if isinstance(v, list):
            out[k] = [e for e in v if isinstance(e, dict)]
    return out


def merge(canons: list[dict]) -> dict:
    """Merge per-chunk canon dicts into one (entities unified by normalised name)."""
    out = {k: [] for k in _LIST_KEYS}
    for c in canons:
        for k in _LIST_KEYS:
            out[k].extend(c.get(k) or [])
    return out


# ── verify (pure) ─────────────────────────────────────────────────────

def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _year(s) -> int | None:
    m = re.search(r"\b(1[0-9]{3}|2[0-9]{3})\b", str(s or ""))
    return int(m.group(1)) if m else None


def _num(s) -> float | None:
    t = re.sub(r"[,$£€\s]", "", str(s or ""))
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def verify(canon: dict) -> list[dict]:
    """Deterministic continuity findings over a merged canon dict."""
    findings: list[dict] = []

    # fixed_fact: same entity name, conflicting fixed values
    by_name: dict[str, dict[str, set]] = {}
    quotes: dict[str, list[str]] = {}
    for e in canon.get("entities") or []:
        name = _norm(e.get("name"))
        if not name:
            continue
        fixed = e.get("fixed") if isinstance(e.get("fixed"), dict) else {}
        for attr, val in fixed.items():
            v = _norm(val)
            if not v:
                continue
            by_name.setdefault(name, {}).setdefault(attr, set()).add(v)
            quotes.setdefault(f"{name}|{attr}", []).append(str(e.get("quote") or ""))
    for name, attrs in by_name.items():
        for attr, vals in attrs.items():
            if len(vals) > 1:
                findings.append({"kind": "fixed_fact", "source": "canon", "severity": "critical",
                                 "detail": f"{name}'s {attr} is stated as {sorted(vals)}",
                                 "quotes": quotes.get(f"{name}|{attr}", [])[:2]})

    # whereabouts: one character in two places within a single scene
    # duplicate_scene: identical day+hour+place+cast across scenes
    scene_sigs: dict[str, str] = {}
    for sc in canon.get("scenes") or []:
        present = sc.get("present") if isinstance(sc.get("present"), list) else []
        seen_where: dict[str, str] = {}
        for pr in present:
            if not isinstance(pr, dict):
                continue
            nm = _norm(pr.get("name"))
            wh = _norm(pr.get("whereabouts"))
            if not nm or not wh:
                continue
            if nm in seen_where and seen_where[nm] != wh:
                findings.append({"kind": "whereabouts", "source": "canon", "severity": "critical",
                                 "detail": f"{nm} is in two places in scene "
                                           f"{sc.get('id', sc.get('order', '?'))}: "
                                           f"{seen_where[nm]} vs {wh}",
                                 "quotes": [str(sc.get("quote") or "")]})
            else:
                seen_where[nm] = wh
        sig = "|".join([_norm(sc.get("day")), _norm(sc.get("hour")), _norm(sc.get("place")),
                        ",".join(sorted(_norm(p.get("name")) for p in present if isinstance(p, dict)))])
        if sig.strip("|") and sig in scene_sigs and scene_sigs[sig] != _norm(sc.get("id")):
            findings.append({"kind": "duplicate_scene", "source": "canon", "severity": "critical",
                             "detail": f"a scene with the same day/hour/place/cast appears twice "
                                       f"({scene_sigs[sig]} and {sc.get('id', sc.get('order', '?'))})",
                             "quotes": [str(sc.get("quote") or "")]})
        elif sig.strip("|"):
            scene_sigs[sig] = _norm(sc.get("id"))

    # alibi: >1 distinct alibi per suspect
    alibi_by: dict[str, set] = {}
    for a in canon.get("alibis") or []:
        s = _norm(a.get("suspect"))
        al = _norm(a.get("alibi"))
        if s and al:
            alibi_by.setdefault(s, set()).add(al)
    for suspect, als in alibi_by.items():
        if len(als) > 1:
            findings.append({"kind": "alibi", "source": "canon", "severity": "critical",
                             "detail": f"{suspect} has {len(als)} different alibis: {sorted(als)}"})

    # amount: paid > attempted
    for am in canon.get("amounts") or []:
        paid, att = _num(am.get("paid")), _num(am.get("attempted"))
        if paid is not None and att is not None and paid > att:
            findings.append({"kind": "amount", "source": "canon", "severity": "critical",
                             "detail": f"{_norm(am.get('label')) or 'an amount'}: paid {paid:g} "
                                       f"> attempted {att:g}",
                             "quotes": [str(am.get("quote") or "")]})

    # tally: len(items) != count
    for t in canon.get("tallies") or []:
        items = t.get("items") if isinstance(t.get("items"), list) else None
        cnt = t.get("count")
        try:
            cnt = int(cnt)
        except (TypeError, ValueError):
            cnt = None
        if items is not None and cnt is not None and len(items) != cnt:
            findings.append({"kind": "tally", "source": "canon", "severity": "major",
                             "detail": f"'{_norm(t.get('claim'))}' claims {cnt} but lists {len(items)}",
                             "quotes": [str(t.get("quote") or "")]})

    # chronology: a fact dated later than the scene that cites it
    for df in canon.get("dated_facts") or []:
        sy = _year(df.get("scene_year"))
        cy = _year(df.get("date_cited"))
        if sy is not None and cy is not None and cy > sy:
            findings.append({"kind": "chronology", "source": "canon", "severity": "critical",
                             "detail": f"a fact dated {cy} is cited in a scene set in {sy} "
                                       f"({_norm(df.get('fact'))})",
                             "quotes": [str(df.get("quote") or "")]})
    return findings


# ── patch-mode revision ───────────────────────────────────────────────

def patch_prompt(text: str, findings: list[dict]) -> str:
    lines = []
    for f in findings[:20]:
        q = "; ".join(f.get("quotes") or [])
        lines.append(f"- [{f['kind']}] {f['detail']}" + (f" (quotes: {q})" if q else ""))
    return (
        "A continuity check found these canon contradictions in a narrative. Fix EACH "
        "with the SMALLEST possible edit. Do NOT rewrite the story, do NOT summarise. "
        "Output ONLY a JSON array of anchored find/replace patches — the `find` must be "
        "an EXACT substring of the text:\n"
        '[{"find": "<exact text to replace>", "replace": "<corrected text>"}]\n\n'
        "Contradictions:\n" + "\n".join(lines) + "\n\nTEXT:\n" + text[:FULL_REEMIT_MAX]
    )


def apply_patches(text: str, patches: list[dict]) -> tuple[str, int, list[dict]]:
    """Apply anchored find/replace patches. Returns (new_text, applied, unapplied)."""
    out = text or ""
    applied = 0
    unapplied: list[dict] = []
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        find = str(p.get("find") or "")
        repl = str(p.get("replace") or "")
        if find and find in out:
            out = out.replace(find, repl, 1)
            applied += 1
        else:
            unapplied.append(p)
    return out, applied, unapplied


def parse_patches(output: str) -> list[dict]:
    if not output:
        return []
    m = re.search(r"\[.*\]", output.strip(), re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    return [p for p in arr if isinstance(p, dict)] if isinstance(arr, list) else []


# ── run_check (extract → verify → patch → re-verify) ──────────────────

async def run_check(text: str, *, extract_fn: CompleteFn,
                    revise_fn: CompleteFn | None = None,
                    max_items: int = 60,
                    on_progress: Callable[[str], None] | None = None) -> dict:
    """Chunk → extract each → merge → verify → (patch revise) → re-verify.

    Returns ``{text, revised, findings, initial_findings, chunks, patched, unapplied}``.
    The revision is patch-anchored (never a full re-emit above FULL_REEMIT_MAX) and is
    kept only when it reduces the finding count without collapsing the text.
    """
    def _note(m: str) -> None:
        if on_progress:
            try:
                on_progress(m)
            except Exception:  # noqa: BLE001
                pass

    chunks = chunk_text(text)
    per = []
    for i, ch in enumerate(chunks):
        _note(f"Canon: extracting chunk {i + 1}/{len(chunks)}…")
        per.append(parse_canon(await extract_fn(extract_prompt(ch, max_items)) or ""))
    canon = merge(per)
    findings = verify(canon)
    result = {"text": text, "revised": False, "findings": findings,
              "initial_findings": findings, "chunks": len(chunks),
              "patched": 0, "unapplied": []}
    if not findings or revise_fn is None:
        return result

    _note(f"Canon: {len(findings)} contradiction(s) — revising by patch…")
    patches = parse_patches((await revise_fn(patch_prompt(text, findings))) or "")
    if not patches:
        _note("Canon: no applicable patches — kept the original")
        return result
    revised, applied, unapplied = apply_patches(text, patches)
    collapsed = not revised.strip() or (len(text) > 800 and len(revised) < 0.9 * len(text))
    if collapsed or applied == 0:
        _note("Canon: patches did not apply cleanly — kept the original")
        return result

    canon2 = merge([parse_canon(await extract_fn(extract_prompt(ch, max_items)) or "")
                    for ch in chunk_text(revised)])
    findings2 = verify(canon2)
    if len(findings2) >= len(findings):
        _note("Canon: revision did not reduce contradictions — kept the original")
        return result
    result.update(text=revised, revised=True, findings=findings2,
                  patched=applied, unapplied=unapplied)
    return result


# ── audit + summary (kept OUT of the deliverable) ─────────────────────

def summarize(result: dict) -> dict:
    return {
        "chunks": result.get("chunks", 0),
        "initial": len(result.get("initial_findings") or []),
        "remaining": len(result.get("findings") or []),
        "revised": bool(result.get("revised")),
        "patched": result.get("patched", 0),
        "kinds": sorted({f["kind"] for f in (result.get("findings") or [])}),
    }


def summary_line(result: dict) -> str:
    s = summarize(result)
    line = f"{s['chunks']} chunk(s) · {s['initial']} contradiction(s)"
    if s["revised"]:
        line += f" · after fix: {s['remaining']} ({s['patched']} patch(es))"
    return line


def audit_markdown(result: dict, *, question: str) -> str:
    lines = [f"# Canon check\n\n*Task:* {question[:200]}\n"]
    fnd = result.get("findings") or []
    if not fnd:
        lines.append("\nNo unresolved canon contradictions.\n")
    else:
        lines.append(f"\n{len(fnd)} unresolved contradiction(s):\n")
        for f in fnd:
            lines.append(f"- **{f['kind']}** — {f['detail']}")
    return "\n".join(lines) + "\n"


def write_audit(dest_dir: Path, result: dict, *, question: str,
                base_name: str = "deliverable") -> dict | None:
    """Write the canon audit next to the deliverable (a separate file, NOT the prose).

    Returns a generated-file descriptor, or None when nothing was recorded. The
    audit never enters the deliverable itself."""
    if not result.get("initial_findings"):
        return None
    name = f"{base_name}{AUDIT_SUFFIX}"
    try:
        p = Path(dest_dir) / name
        p.write_text(audit_markdown(result, question=question), encoding="utf-8")
        return {"name": name, "mime": "text/markdown", "size": p.stat().st_size,
                "kind": "generated", "agent": "canon-check"}
    except OSError as e:  # noqa: BLE001 — audit doc is best-effort
        log.warning("canon audit write failed", error=str(e))
        return None
