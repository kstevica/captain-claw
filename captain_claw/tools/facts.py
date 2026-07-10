"""The `facts` tool — the run's shared ledger of load-bearing values.

Workers in a Basna/Vatra run establish numbers, dates, and identifiers that
other pieces build on. This tool reads/writes the machine-readable ledger for
those values (``.facts.db`` in the run's shared VFS folder — see
``captain_claw.flight_deck.facts_ledger``), so a value is stated ONCE and read
everywhere else, instead of each piece re-deriving it from prose.

The folder resolves from the worker's ``CLAW_VFS_PROJECT`` env (the same shared
folder ``vfs:`` writes land in), exactly like the ``researchmap`` tool. Outside
a run with a bound folder the tool returns a clear error, so it is safe to
expose whenever a run arms it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class FactsTool(Tool):
    name = "facts"
    description = (
        "The team's shared ledger of canonical values (numbers, dates, identifiers). "
        "'set' — record a load-bearing value you established: short snake_case `key` "
        "(e.g. total_budget_eur), `value`, optional `unit`, `status` "
        "(verified|derived|estimated|assumed|to_be_completed), `provenance` (URL, file, "
        "or 'derived from <keys>'), `confidence` 0–1, `computed_from`. "
        "'get' — one value by `key`. 'list' — the whole ledger. "
        "If 'set' reports a CONFLICT, the existing value stays canonical — reconcile "
        "with the team (check the source / post an ask) instead of overwriting; "
        "`force=true` only after the team agrees the old value is wrong. "
        "Ledger only values other pieces may reuse — not trivia."
    )
    timeout_seconds = 20.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "get", "list"],
                "description": "'set' a value, 'get' one by key, 'list' the ledger.",
            },
            "key": {"type": "string", "description": "Short snake_case name of the quantity (set/get)."},
            "value": {"type": "string", "description": "The value, plain (set). Numbers without separators."},
            "unit": {"type": "string", "description": "EUR, %, days, FTE, … (set, optional)."},
            "status": {
                "type": "string",
                "description": "verified|derived|estimated|assumed|to_be_completed (set, optional).",
            },
            "provenance": {"type": "string", "description": "URL, file, or 'derived from <keys>' (set, optional)."},
            "confidence": {"type": "number", "description": "0–1 (set, optional)."},
            "computed_from": {"type": "string", "description": "Keys this value derives from (set, optional)."},
            "force": {"type": "boolean", "description": "Replace a conflicting value (set; team-agreed only)."},
        },
        "required": ["action"],
    }

    def _project(self) -> Path | None:
        from captain_claw import vfs
        try:
            return vfs.project_root()
        except Exception as e:  # noqa: BLE001
            log.warning("facts: cannot resolve project root", error=str(e))
            return None

    @staticmethod
    def _fmt(r: dict) -> str:
        unit = f" {r['unit']}" if r.get("unit") else ""
        prov = f" · {r['provenance']}" if r.get("provenance") else ""
        by = f" · by {r['updated_by']}" if r.get("updated_by") else ""
        conf = f" · conf {r['confidence']:.2f}" if r.get("confidence") is not None else ""
        return f"{r['key']} = {r['value']}{unit} [{r['status']}]{conf}{prov}{by}"

    async def execute(self, action: str, key: str = "", value: str = "",
                      unit: str = "", status: str = "", provenance: str = "",
                      confidence: float | None = None, computed_from: str = "",
                      force: bool = False, **kwargs: Any) -> ToolResult:
        from captain_claw import vfs
        from captain_claw.flight_deck import facts_ledger as fl
        project = self._project()
        if project is None:
            return ToolResult(success=False, error="facts: no shared VFS folder bound")
        try:
            if action == "set":
                if not (key or "").strip():
                    return ToolResult(success=False, error="set needs a key")
                if not str(value if value is not None else "").strip():
                    return ToolResult(success=False, error="set needs a value")
                who = ""
                try:
                    who = vfs.agent_label()
                except Exception:  # noqa: BLE001 — label is cosmetic
                    pass
                res = fl.upsert(project, key, value, unit=unit or "",
                                status=status or "", provenance=provenance or "",
                                confidence=confidence, computed_from=computed_from or "",
                                updated_by=who, force=bool(force))
                if res.get("ok"):
                    verb = {"created": "recorded", "merged": "confirmed (metadata merged)",
                            "forced": f"REPLACED (was {res.get('previous')})"}[res["action"]]
                    return ToolResult(success=True,
                                      content=f"{verb}: {self._fmt(res['fact'])}")
                if res.get("reason") == "conflict":
                    # A conflict is information, not a tool failure — return it as
                    # content so the model reads and reconciles instead of retrying.
                    return ToolResult(success=True,
                                      content=f"CONFLICT — {res['message']}")
                return ToolResult(success=False,
                                  error=f"facts set failed: {res.get('message') or res.get('reason')}")

            if action == "get":
                if not (key or "").strip():
                    return ToolResult(success=False, error="get needs a key")
                r = fl.get(project, key)
                if r is None:
                    return ToolResult(success=True,
                                      content=f"'{fl.norm_key(key)}' is not in the ledger. "
                                              "Use action=list to see what is, or establish "
                                              "it yourself and `set` it.")
                return ToolResult(success=True, content=self._fmt(r))

            if action == "list":
                md = fl.dump_markdown(project)
                return ToolResult(success=True,
                                  content=md or "(ledger is empty — `set` the first "
                                                "load-bearing value your piece establishes.)")

            return ToolResult(success=False, error=f"unknown action: {action}")
        except Exception as e:  # noqa: BLE001
            log.warning("facts tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"facts error: {e}")
