# Security Reviewer — review r1

The security review is complete. Here's a summary of my findings:

---

## RealmCraft RTS — Security Review R5

**Scope:** `index.html` (1,775 lines), `test_game.js` (1,392 lines), `plan.md` (36 lines)  
**Overall Risk: LOW** — same as prior reviews

### New Findings (not in prior 9 reports)

| # | CVSS | Title | Location |
|---|------|-------|----------|
| **SR-R5-01** | 2.5 (Low) | O(n²) DoS vector in `findAutoBuildSite` full-map fallback | `index.html:745-766` |
| **SR-R5-02** | 2.0 (Low) | Accidental global `cam` variable — missing `const`/`let` | `index.html:1526` |
| **SR-R5-03** | 1.8 (Low) | Missing zero-division guard on `canvas.clientWidth`/`clientHeight` | `index.html:665-666, 1032-1033, 1701-1704` |
| **SR-R5-04** | 0.0 (Info) | Unused `dpr` variable in `screenToWorld` | `index.html:663` |
| **SR-R5-05** | 0.0 (Info) | Unbounded `autoBuildCooldown` object growth | `index.html:768,784` |

### Reconfirmed Prior Findings (all 9 still unfixed)

The 9 findings from R0–R3 remain present, including:
- **SR-01** (2.8): Unsafe `innerHTML` sinks — 4 locations
- **SR-NEW-01** (3.5): Prototype-pollution via `game.resources[e.carryType]` 
- **SR-03** (2.5): Unbounded entity creation via console

### Auto-Build Chain Analysis

The specific "gather → carry → auto-build" flow has two notable security concerns:
1. The O(n²) fallback search in `findAutoBuildSite` (line 745) — with enough console-created entities, this becomes a DoS
2. The `carryType` string used as dynamic property key on `game.resources` remains the top vulnerability

The full report is saved at **`.reports/SECURITY_REVIEW_R5.md`** with CVSS vectors, remediation code, and regression test guidance.