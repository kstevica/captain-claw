# Security Reviewer — review r3

The security review is complete and saved to `.reports/SECURITY_REVIEW_CURRENT.md`. Here's the executive summary:

---

## Security Review: RealmCraft RTS — CVSS-Ranked Findings

**Overall Risk: LOW** — 0 Critical, 0 High, 0 Medium, 4 Low, 2 Info

RealmCraft is a single-file, zero-dependency, client-side-only RTS game. No backend, no network calls, no text input fields, no storage. The attack surface is vanishingly small. All findings require **browser console access** to exploit — once an attacker has that, the game is already compromised.

### Findings (CVSS-ranked):

| # | CVSS | Severity | Category | Finding | Location |
|---|------|----------|----------|---------|----------|
| **SR-R5-01** | 3.5 | **Low** | Injection | Dynamic property key `game.resources[carryType]` — prototype pollution vector | `index.html:1284`, `:1258` |
| **SR-R5-02** | 2.8 | **Low** | Injection | Unsafe `innerHTML` sinks for game state (3 locations) | `index.html:1363,1364,1498` |
| **SR-R5-03** | 2.0 | **Low** | Defense-in-depth | Mutable definition objects — `BUILDING_DEFS`, `UNIT_DEFS` not frozen | `index.html:103-120` |
| **SR-R5-04** | 2.0 | **Low** | Unsafe input | `Object.entries(BUILDING_DEFS)` iteration exposes prototype pollution | `index.html:1469` |
| **SR-R5-05** | 0.0 | ℹ️ Info | Defense-in-depth | Missing Content-Security-Policy meta tag | `index.html:1-6` |
| **SR-R5-06** | 0.0 | ℹ️ Info | Code quality | Duplicate `attackTarget:null` — dead property (only 1 used) | `index.html:269,276` |

### What's improved since prior reviews:
- ✅ `createUnit` / `createBuilding` now guard against NaN/Infinity (`Number.isFinite`)
- ✅ Auto-build system has per-frame double-spend lock (`_autoBuildThisFrame` Set)
- ✅ Cooldown mechanism prevents resource-exhaustion spam loops
- ✅ Gathering validation now preserves progress when peasant is pushed off tile (walks back to `gatheringNode`)

### Top remediation (low-effort, high-value):
1. **Guard the dynamic property write** on line 1284 with `if (e.carryType === 'gold' || e.carryType === 'wood')` — this already exists on the deposit path but is duplicated without the guard at line 1258
2. **Add a CSP meta tag** in the `<head>` — one line, defense-in-depth
3. **Replace `innerHTML` with `textContent`** on line 1498 where the `stats` string mixes dynamic values — use DOM methods instead