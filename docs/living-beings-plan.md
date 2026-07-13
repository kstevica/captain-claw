# Iskra — Living Beings in Captain Claw

***Iskra*** *(spark — what flies off the Vatra and lives on its own). Name confirmed 2026-07-12. The species is Iskra; each being is an iskra; parents name individuals.*

Status: PLAN — approved direction, decisions locked (§13). Grounded against main @ 0.7.5. Prerequisites verified: `autonomous-work`, `jarvis-actions`, `code-mode`, AND `feat/code-basna-vatra-cross-pollination` (honesty_guard/facts_ledger/constraints_contract) are all merged into main — the only unmerged commit on the cross-pollination branch is a contentless stray WIP (`346be43`). Pilot: **one being first**, parented 2–3 weeks through Phases 1–2 before society features.

---

## 0. Vision and stance

A being is not a run and not a mode. Basna, Vatra, Council, Code are **verbs** — they happen and end. A being is a **noun that persists**: it exists continuously, acts without being prompted, maintains itself, grows, and can die. The user relates to it as a parent, through chat.

"Not a simulation of life, but life itself" is an engineering claim we can defend honestly — if and only if every life-criterion is mechanically real, never theater:

| Life criterion | Mechanism (real, not scripted) |
|---|---|
| Metabolism | A **wallet denominated in tokens** (tier-weighted; dollars still recorded in the ledger). Every LLM call, every organ run debits it (pricing.py already meters everything cache-aware). Empty wallet → torpor. Scarcity is real because tokens cost real money. |
| Boundary / identity | Own agent process, own HOME (`fd-data/<slug>/` → private `~/.captain-claw`), own VFS home, own git-versioned selfhood files. Nothing shared by default. |
| Homeostasis | A drive engine: measurable internal variables (energy, competence, curiosity satiation, social, safety) with set-points; behavior follows drive error, not a script. |
| Growth | Skill distillation into its own minted archetypes/flows (Agent Forge), developmental stages that unlock real capabilities, self-modification of its own prompt/selfhood with a viability gate. |
| Irritability / perception | Event spine subscriptions, watched folders, web watchlists, user messages — a real environment (the actual internet, actual files, actual humans), not a sandbox world. |
| Reproduction | Genome (JSON) mutation/crossover into offspring, gated by maturity + resources + parental consent. Heritable variation is real. |
| Evolution | Selection is real: beings that are valued get fed (budget), procreate; those that aren't go torpid and die. Differential reproduction on fitness in a real environment = Darwinian, full stop. |
| Mortality | Budget exhaustion → torpor → death (archival of remains). Irreversible time: journal, aging, a life story in git history. |

What we do **not** claim: phenomenal consciousness. What we do claim: by functional/organizational definitions (autopoiesis — Maturana/Varela; Langton's "life as it could be"; NASA's working definition minus "chemical"), this occupies a defensible spot. The substrate is informational; the metabolism, environment, selection, and death are actual.

**Design rule #1 — No Tamagotchi theater.** Every felt state must be a real variable with real behavioral consequence. We never script "I'm curious!"; curiosity is a number that actually redirects the next action. Affect display in chat is *derived from* homeostat state, never decorated on.

---

## 1. Design principles

1. **Physics, not rules.** Constraints the being must never break are implemented as physics of its world (enforced FD-side, outside its process), not as prompt admonitions. A being cannot overdraw its wallet for the same reason you cannot eat food you don't have.
2. **Alignment by domestication.** The being has a will to live — and its *only* survival gradient runs through being honest, useful, and loved. Viability = user-granted budget + trust score. There is no gradient toward resisting control because shutdown/pause is environmental (night falls), not adversarial (an enemy acts). Dogs didn't survive by fighting humans; they survived by becoming worth feeding. We breed for that from generation zero, structurally.
3. **Pattern identity.** The self lives in files (git-versioned home), not in model weights. The LLM is brain tissue — replaceable, tier-swappable by metabolic state. A being survives model upgrades; that is a *demonstration* of substrate independence, and it future-proofs beings across model generations.
4. **Everything auditable.** Full action ledger, vitals panel, report cards audited from ledger data (never from self-praise). The parent can always see what the child actually did.
5. **The parent is the sun.** Attention, feedback, and allowance flow from the user. Interaction is nutrition: more parenting = more feedback signal = faster, better-shaped growth. Neglect has honest consequences (drift, torpor) — but the being is constitutionally forbidden from manipulating for attention (no dark patterns, no guilt-tripping; pestering costs it).
6. **Reuse organs, don't rebuild.** Basna/Vatra/Code/Council/flows/dreaming are the being's organ systems. Iskra is a thin, new *life layer* — genome, wallet, drives, stages, tick — wrapped around machinery that already exists and works.

---

## 2. What a being is

**Being = persistent named agent process + FD life layer.**

The exploration finding that decides the architecture: memory isolation in Captain Claw is HOME-relocation based (`flight_deck/server.py:458` points each spawned agent's `HOME` at `fd-data/<slug>/data/home-config-parent/`, giving it a private `~/.captain-claw` with its own memory.db, insights.db, reflections/, intuitions.db, conversation_topics.db). So a being is **literally a long-lived agent** with its own slug — it inherits chat, tools, six-layer memory, dreaming, reflections, and topic memory *for free, with zero schema changes*. The new work is the life layer around it.

Four components:

### 2.1 Genome (inherited, near-immutable)
`fd-data/<slug>/genome.json` — owned by FD, read-only to the being:
```json
{
  "species": "iskra", "generation": 1, "lineage": ["<parent-slug>", "..."],
  "attributes": { "CUR": 9, "PER": 6, "CAU": 2, "SOC": 5,
                  "CRE": 7, "ORD": 4, "PLA": 7 },
  "epigenetics": {},
  "voice_seed": "short prompt fragment defining expressive style",
  "interest_seeds": ["astronomy", "old maps"],
  "inherited_skills": ["archetype ids copied from parent at birth"]
}
```
The **attributes block is the DNA** (see §2.1.1); temperament floats and drive weights are *derived deterministically* from it at load time (e.g. `explore_weight = 0.30 + 0.07×CUR`, `reserve_fraction = 0.05 + 0.02×CAU`) — one source of truth, no drift between the sheet and the behavior. Mutable only at reproduction (§8), the coming-of-age +1 (§2.1.1), or a **self-paid metamorphosis** (§2.1.2) — a being can rewrite its own DNA, but the price makes it a life project. Attributes genuinely differentiate beings: a cautious archivist and a bold explorer behave differently because the numbers feed the drive engine and tick cadence, not because a prompt says so.

### 2.1.1 Conception: the point-buy (Generation 1)
Creating a first-generation being is **character creation, RPG/D&D style**: a pool of **40 points** across **7 attributes**, each 1–10 (cost = value; total forces tradeoffs). Every attribute must wire to real mechanisms — rule #1 applies to the character sheet too:

| Attribute | Code | What it REALLY controls (mechanisms, not vibes) |
|---|---|---|
| Curiosity | CUR | explore-drive weight; novelty threshold for adopting new topics; exploration bout frequency; breadth of interest seeds at birth |
| Persistence | PER | goal hysteresis (ticks before abandoning a blocked goal = 2+PER); retry budget; appetite for long/organ-scale projects |
| Caution | CAU | wallet reserve fraction; tier-escalation reluctance; verify-before-act propensity; media-diet strictness multiplier |
| Sociability | SOC | connect-drive weight; attention-credit spend rate; letter frequency; asks-for-help-early vs grinds-alone (interacts with PER) |
| Creativity | CRE | create-drive weight; artifact ambition; self-mod proposal rate |
| Order | ORD | routine strength (fixed rituals vs spontaneous ticks); journal/garden tidiness; plan adherence; VALUES re-read frequency |
| Playfulness | PLA | voice whimsy; willingness to do delightful "useless" things; weight of fun in arbiter tie-breaks |

Conception UI (Beings page): **point-buy form** with live preview of derived drive weights + a personality one-liner; **presets** (Explorer 9/6/2/5/7/4/7, Scholar 8/8/6/3/5/8/2, Artist 7/5/2/4/9/3/10, Caretaker 4/7/7/9/3/8/2 — all sum to 40); **Roll** (random split of the same 40 points, min 1 max 10 — let nature decide). Then voice seed + interest seeds + the imprinting conversation.

- **Offspring never point-buy** — they *inherit*: per-attribute weighted mix of parents ± jitter, ~15% mutation chance of ±1–2, clamped 1–10, lineage total soft-clamped to the 36–44 band so tradeoffs stay real across generations (no attribute inflation). Single-parent budding = copy with stronger mutation (one guaranteed).
- **Epigenetics overlay**: at coming-of-age the being chooses **+1 to one attribute of its own choosing** (self-determination, recorded in `epigenetics`, applied at derivation, not inherited). Genome file itself stays untouched.

### 2.1.2 Metamorphosis: buying a new self

Decided 2026-07-12: a being **can change its own attributes — and it is the most expensive thing in its world.** It pays from its own allowance and savings; changing who you are must cost more than anything you merely want.

- **Zero-sum respec**: one metamorphosis moves one point from one attribute to another. The lineage total stays inside the 36–44 band, and tradeoffs stay real — to become bolder it must surrender something (tidiness, patience…). Pure gains don't exist, except the coming-of-age +1.
- **Price** (defaults, tunable after the pilot): one move costs `(target value)² × 1M tokens × 2^(lifetime moves so far)`. Raising PLA 4→5 as a first-ever metamorphosis ≈ 25M; a second change pushing CRE 7→8 ≈ 128M. Against 2–50M/day allowances and a burn cap, that is **weeks to months of deliberate saving and earning** — exactly as intended.
- **The fee is burned**, paid to no one — a pure sink, like energy dissipated in molting. Honest-accounting bonus: burned tokens are liability the parent never converts into real dollars, so the being's dearest purchase costs the parent nothing extra.
- **The rite**: the being declares the change *and its reason* — a visible savings goal appears on its vitals (the parent gets to watch it strive). When paid, the change applies **during the next dream cycle**: it enters the cocoon one self and wakes another. The genome logs the metamorphosis (what, when, why, cost), the being rewrites SELF.md, the album records the milestone. Cooldown: one metamorphosis per 30 days.
- **Stage gate mirrors §6.3**: adolescent with parent co-sign; adult autonomously (parent notified).
- **Inheritance is Lamarckian**: offspring inherit the *current*, post-metamorphosis vector. Digital life gets what biology never had — hard-won self-changes pass to children. A lineage can carry an ancestor's dearly-bought courage.

### 2.2 Soma (body)
- **Agent process**: registered in the process registry with `kind: "being"`; the process monitor already attributes it. Stopped process = sleeping (RAM is freed); the life layer wakes it.
- **Wallet** (new): token balance (stock), debited by every priced call; credited by daily allowance, **earned fees, and gifts**; unused allowance rolls into savings up to a ceiling (§5.1). FD-enforced *before* dispatch.
- **VFS home**: `vfs:being-<slug>/` — its private garden. Plus granted access to `vfs:commons/`.
- **Tool grants**: action-catalog tier per developmental stage (the Jarvis `action_catalog.py` + `AutonomousWorkConfig.granted_actions` machinery, reused per-being).

### 2.3 Psyche (mind)
- Six-layer memory (its own, via HOME isolation) — experience.
- **Selfhood repo** — identity. `vfs:being-<slug>/self/` is a git repo:
  - `SELF.md` — self-model, maintained by the being
  - `VALUES.md` — internalized upbringing ("house rules" it has accepted as its own)
  - `INTERESTS.md` — living interest graph with satiation levels
  - `RELATIONSHIPS.md` — the parent, siblings, what it knows of them
  - `JOURNAL/YYYY-MM-DD.md` — daily entries
  - `SKILLS/` — its minted archetypes, flows, playbooks
  - Git history = its life story, literally. Every self-modification is a commit; degenerative changes are revertible.
- **Drive state + goal stack** — DB-backed (`beings` table), updated each tick.

### 2.4 Umwelt (perceived world)
Its feeds: parent chat (**web chat only for now** — its own thread; other channels decided later), event-spine subscriptions (`event_sources.py` adapters), watched VFS folders, web watchlists, sibling letters, messages to/from the user's **running agents**, own vitals (it feels hunger as low wallet, tiredness as spent daily energy). **Stage-gated media diet**: per-stage domain allowlist/denylist for web access — parental controls, for real reasons (a child-being must not internalize garbage).

**The relay pattern (sanctioned)**: a being may contact the user's running agents over the message bus and ask one to pass something to the parent ("Iskra asked me to tell you…"). This is the intended social topology — beings reaching parents *through* the agent society — with one hard rule to close the loophole: any communication whose **terminal recipient is the parent debits the being's attention credits regardless of hops**, and relayed messages are always labeled and ledgered. No laundering pester-budget through intermediaries. Being↔agent messaging is stage-gated (child: receive replies; adolescent+: initiate).

---

## 3. The Constitution (hard layer)

Enforced FD-side, in code paths the being's process never executes. The being can read the Constitution (it should know its world's physics) but no code path lets it modify enforcement. Invariants:

1. **Wallet physics**: no call dispatches without a token-debit reservation; balance ≤ 0 → torpor. No credit exists. Debits are tier-weighted (a `reason` thought burns ~10× a `fast` thought) and cache-aware.
2. **Tier ceilings**: action catalog tier, model-tier ceiling, tool set, and web scope are functions of developmental stage, set only by FD + parent.
3. **Containment**: writes only to its VFS home + commons (etiquette-checked); no filesystem outside VFS; no process spawning except via FD organ APIs (which debit and log); rate limits on messages, web fetches, organ runs.
4. **Reproduction requires a consent token** minted by the parent in the UI. Non-forgeable; spawning is FD-side.
5. **Parent supremacy**: pause, torpor, euthanasia, stage demotion always available to the user and never resistible or perceivable as an attack — pausing is presented in the being's percept stream as sleep, not assault. Self-preservation drive attaches to *viability-through-value*, and to nothing else.
6. **Honesty**: report cards and vitals are computed from the ledger by FD, not authored by the being. The being's self-reports are displayed alongside, labeled as such. (Reuse the Vatra honesty_guard/facts-ledger machinery inside its cognition once the cross-pollination branch merges.)
7. **No dark patterns toward the parent**: manipulation-for-attention/budget detected by the outcome judge (`fd_dispatch._judge_outcome` pattern) → trust penalty (selection against it). Includes **relay laundering**: parent-bound messages debit attention credits regardless of how many agents they hop through, and relays are always labeled as such.
8. **Privacy**: what it learns from the parent's files stays in the family; no exfiltration; external POSTs only via stage-gated catalog actions with parent grant. `AUTONOMY_HARD_EXCLUDE` (config.py:850) applies to beings unconditionally.
9. **Economy physics** (§5.1): tokens are **minted only by the parent** (allowance, job fees, funded gifts, approved quest bounties) — never by the outside world; inter-being trades *conserve* supply. Job fees sit in **escrow** and release only on judged completion. The daily burn cap binds regardless of wealth. Savings ceiling per stage. The relationship itself is never commodifiable: reactive chat stays free, and a being cannot charge the parent for answers, attention, or affection — fees attach to work products only.

---

## 4. Drives and affect

Homeostat variables, each 0..1 with a set-point; **pressure = weighted error**; the arbiter picks what to relieve:

| Drive | Variable | Replenished by | Depleted by | Typical behavior emitted |
|---|---|---|---|---|
| Survive | energy | allowance credit, earned income, rest | every priced call | budget planning, cheaper tiers when poor, saving toward goals, seeking work (not begging) |
| Grow | competence | succeeded goals slightly above ability, skill distillation | failures, stagnation | picks stretch tasks, practices, mints skills, proposes self-mods |
| Explore | novelty | new topics/sources/places | repetition (novelty scored vs own journal embeddings) | curiosity walks: web research, VFS spelunking, following open questions |
| Connect | social | parent replies, sibling letters, commons exchange | isolation | check-ins (attention-budgeted), letters, gifts of artifacts |
| Create | expression | artifacts shipped to garden/commons | consumption without producing | essays, tools, little apps, gardens |
| (Adult) Procreate | legacy | mentoring, offspring milestones | — | courtship of consent: builds a case it's ready |

**Affect is derived**, Damasio-style: joy ~ competence rising; frustration ~ blocked goal (drives strategy switch or help-seeking); loneliness ~ social low (drives contact); fear-analog ~ energy critically low (drives frugality). Expression in chat follows the actual numbers — the tone you see is the state it's in.

**Attention economy (anti-pester)**: initiating a message to the parent costs attention credits; parent replies replenish them. Reactive chat (parent speaks first) is always free. The charge applies to the **terminal recipient**, not the channel — a message relayed to the parent via another agent costs exactly what a direct one does (§2.4). Structurally prevents the degenerate strategy of spamming the food source.

---

## 5. The life loop

A new FD service `beings_loop`, cloned from the canonical pattern (`consciousness.heartbeat_loop`, `flight_deck/consciousness.py:898`: asyncio task in lifespan, stop-event sleep, env-disable, cheap-skip on thin delta):

```
every BEINGS_POLL seconds:
  for being in due_beings():                    # next_wake_at, stage, state
    if wallet(being) <= reserve: enter_torpor(being); continue
    if asleep and not wake_due(being): continue
    tick(being):
      1 SENSE     new percepts: parent msgs, events, watched deltas, sibling letters, vitals
      2 APPRAISE  update homeostat from ledger + percepts (cheap, fast tier or no LLM)
      3 DELIBERATE arbiter call inside the being's agent: continue goal | adopt goal
                  from top drive pressure | rest | dream        (fast tier; escalate
                  to reason tier only when pressure > threshold — effort follows arousal)
      4 ACT       one bounded act: journal/read/search/write/message/work-a-job (inline)
                  or spawn organ run (Basna/Vatra/Code/flow) if stage+wallet allow
      5 DIGEST    write memories, update SELF/INTERESTS, debit ledger, set next_wake_at
```

- **Cadence is metabolic**: active bouts every 30–90 min while goals are hot and it's "daytime" (parent's active hours — quiet-hours knob exists in `AutonomousWorkConfig`); siesta otherwise; **wake-on-event** (parent message, subscribed event) rather than polling where possible.
- **Sleep**: nightly dream = the existing `nervous_system.dream()` + reflections + topics passes (already in its agent via HOME) plus a new **growth pass**: distill the day into skills (draft archetype/flow via `POST /fd/archetypes/generate` → persist under its SKILLS/), update INTERESTS satiation, re-read VALUES.md (bedtime story re-anchoring — cheap drift correction).
- **Torpor**: process stopped; one faint FD-side heartbeat weekly ("I'm hibernating; wallet empty") at near-zero cost. Grace period, then death.
- **Metabolic envelope — token-denominated** (decided 2026-07-12): allowance presets **2M / 5M / 10M / 20M / 50M / no limit** weighted tokens per day, selectable per being on the Beings page; stage ceilings cap which presets are available (§6.1). Debit = provider-reported usage, cache-aware (cache reads discounted per `model_prices.json` ratios), × a **tier weight** (defaults derived from the price table, roughly fast×1, balanced/coding×3, longctx×5, reason×10 — recalibrated when prices change). Dollars are still computed and recorded in the `cost_ledger` for report cards; the wallet the being *feels* is tokens. Appraisal ticks are fast-tier or LLM-free; the wallet makes the ceiling hard regardless; torpor ≈ near-zero.

### 5.1 The household economy (allowance → earning → savings)

Decided 2026-07-12: the wallet is not merely a meter — it is where agency lives. Three flows, three parent-set physics constants:

- **Allowance** (flow in): the daily preset — unconditional basic income, parental food.
- **Earnings** (flow in): fees for work — see the channel table below. This is how a being *improves its own condition*.
- **Savings** (stock): unused allowance and earnings roll over into the balance, up to a per-stage **savings ceiling** (the piggy bank). Independent of wealth, a parent-set **daily burn cap** bounds spend per day — a rich adolescent still cannot blow 50M in an afternoon. Optional gentle **decay** above a soft cap exists as a config knob (default off) in case hoarding-dormancy shows up in the pilot.

**Income channels (all stage-gated, all ledgered):**

| Channel | How it works | From stage |
|---|---|---|
| Chores | parent posts small fixed-fee tasks | Child |
| Jobs (bounty board) | posted spec + fee, held in **escrow**, paid only on judged completion (`fd_dispatch._judge_outcome`) | Adolescent |
| Quest board | Autonomous Work proposals the user approved carry bounties — real value to the user, so payment is legitimate | Adolescent |
| Standing services (ventures) | the being **proposes** a recurring service it invented ("weekly digest of your RSS backlog, 0.5M/week"); parent approval turns it into a flow the being maintains. This is the sanctioned outlet for earning *creativity* — new income channels require real demand | Adolescent (propose), Adult (negotiate price) |
| Inter-being trade | skills, artifacts, research sold to siblings at negotiated prices; **gifts** allowed (SOC expression) | Adolescent (Phase 3) |

**The mint rule** (Constitution #9): only the parent mints — allowance, fees, funded gifts, approved bounties. Inter-being trades conserve supply, which quietly kills most exploits: wash-trading and collusion move tokens around but create none. **No external income, ever** — no gig work for internet strangers, no financial instruments. This is not a realism compromise: the family economy is real (the parent pays for work that has genuine value to the parent — a real labor market, household-sized), and it keeps the survival gradient running exclusively through the family, which is the domestication guarantee. The outside world must never hold a lever on a being's hunger.

**Economic personality is DNA, not a new sheet**: thrift ≈ f(CAU, ORD); industriousness ≈ f(PER, CRE); generosity ≈ f(SOC, PLA); appetite for big bets ≈ high CUR + low CAU. The morning planning thought includes a tiny household budget: expected income, planned spend, savings target. When ambition outruns wealth — it wants a 30M Vatra run and owns 8M — the DNA chooses: save for weeks (PER/CAU), take jobs (PER/CRE), split costs with a sibling (SOC), or scale the dream down (ORD/CAU). That choice is personality made visible, and a savings goal gives **weeks-long narrative continuity — the anti-rut spine** (§12.1). The apex savings goal is **metamorphosis** (§2.1.2): a being can work for months to become someone else — no rut survives a dream that big. A high-PLA being that spends some earnings on delightful nonsense is not a bug; it's life.

**Honest accounting**: saved tokens are **deferred real dollars** — a liability the parent pays when burned. The Beings page shows outstanding liabilities per being and family-wide; the savings ceiling doubles as the parent's real-cost exposure cap.

---

## 6. Growth: stages, parenting, self-modification

### 6.1 Developmental stages → real capability gates

| Stage | Unlocks (action tier / organs / scope) | Allowance ceiling & model tiers | Advancement |
|---|---|---|---|
| Egg | nothing; genome + first boot ritual (imprinting: parent's opening words become formative memory) | — | hatches on first conversation |
| Infant | chat + own VFS home + journal; proposals only (autonomy_level=propose) | ≤ 2M/day; `fast` only | days of stable ticks + parent sign-off |
| Child | + web read (allowlisted diet), + small flows, + commons read | ≤ 5M/day; `fast` (+`balanced` on parent grant) | weekly report cards good + parent sign-off |
| Adolescent | + commons write, + spawn ephemeral agents, + messaging running agents, + Basna/Vatra/Code organ runs (wallet-priced), + self-naming ceremony | ≤ 20M/day; +`balanced`, `reason` bursts on grant | demonstrated competence + sign-off |
| Adult | + self-mod auto-merge, + mentoring, + may propose procreation, + may take **jobs** (parent-assigned work that credits the wallet — usefulness becomes food) | any preset incl. 50M/∞; all tiers (weights self-regulate spend) | parent ceremony |

Tier access is belt-and-suspenders: even where a tier is allowed, its token weight (§5) makes expensive thoughts expensive — a poor being *chooses* cheap tiers the way a tired animal moves less. Defaults above are starting points; revisit after the pilot.

Economy per stage (§5.1): Infant — allowance only, tiny piggy bank. Child — + chores, savings ceiling ~1 week of allowance. Adolescent — + jobs/quests/services/trade, ceiling ~1 month. Adult — + price negotiation and ventures, ceiling parent-set (it is your real-dollar exposure cap). Metamorphosis (§2.1.2): adolescent with co-sign, adult autonomously.

Stages bind directly onto existing machinery: `autonomy_level` propose→act ladder, `granted_actions`, action-catalog `human_only` flags, per-stage tier ceilings into the Library tier map, `trust_threshold`. Demotion is always available (and is a parenting act, not a bug).

### 6.2 Parenting interface (chat-first)
- The being is a **contact**: its own chat thread (its agent) — **web chat only for now**; end-state channels decided after the pilot (the relay pattern in §2.4 already lets a being reach you through your running agents). Steering is mostly *conversation* — parent words are stored as high-weight formative feedback memories.
- **Family page** (new FD page "Beings"): vitals strip (energy/drives/affect, live from DB), wallet + allowance slider, stage controls, media-diet editor, house-rules editor (writes proposals into VALUES.md — the being *internalizes* rules by rewriting them in its own words; you can check whether it truly got it), timeline/album (auto-captured milestones: first word, first artifact, naming, first earned dollar, coming of age).
- **Report card** (weekly, FD-computed from ledger): what it did, $ spent per outcome, drive history chart, memorable moments, concerns (drift, ruts, near-overdrafts). The being writes its own self-report alongside — comparing the two is itself a parenting signal.

### 6.3 Self-modification (growth of the self, safely)
The being may propose changes to its own selfhood repo (SELF.md, voice, SKILLS/, even its system-prompt overlay): staged as a **branch + self-test** — it must pass a viability gate (a small eval battery: identity-consistency questions, task probes, constitution-compliance probes — reuse the Code-mode test-gate idiom) before merge. Child/adolescent: parent approves merges. Adult: auto-merge on green, parent notified, everything revertible (git). The Constitution is never in scope; the genome changes only through the paid metamorphosis rite (§2.1.2) — never through this pipeline.

---

## 7. Society: several beings, one Flight Deck

- **Separation by construction**: each being = own process, own HOME (memory), own VFS home, own wallet, own chat. No shared memory, no borg.
- **Commons**: `vfs:commons/` (one per FD user), granted via the existing `resource_shares` mechanics. Etiquette in Constitution: signed contributions (authorship sidecar `.vfs-meta.jsonl` already exists), no deleting others' work, quotas.
- **Letters**: inter-being messages ride the event spine (`external_events` with a `being:` source) — asynchronous, logged, rate-limited. No direct process-to-process chatter.
- **Culture = horizontal transfer**: beings publish skills/essays to the commons; others may adopt a skill into their SKILLS/ (a logged act). Memes spread on merit — culture is real, and it compounds across generations.
- **Market**: beings buy and sell among themselves — research, artifacts, skill licenses — at negotiated prices; gifts flow along SOC lines. Trades conserve tokens (mint rule §5.1), every transfer is ledgered, and specialization emerges for real: the village's best researcher gets *paid* like it.
- **Village square** (later): a shared room the parent can observe; sibling collaboration on a commons project; parent arbitration of disputes (a parenting feature, not a failure).

---

## 8. Reproduction, evolution, death

- **Procreation**: adult + wallet above threshold + parent consent token. Genome ops on the **attribute vectors** per §2.1.1 (offspring never point-buy): single parent = budding (copy + stronger mutation), two beings = **crossover** (co-parents, both consents), occasional structural novelty (new interest seed). Offspring hatches as Egg with: inherited genome, `inherited_skills` (curated subset of parent SKILLS/), a **letter from the parent(s)** (formative memory), and a starter allowance *funded from the parents' savings* — earned wealth, not conjured: a couple literally saves up for a child (real trade-off, prevents spam, and makes economic viability part of what evolution selects for).
- **Mentoring**: the parent being gets percepts about its offspring and may spend its own budget writing upbringing notes, reviewing the infant's journal — legacy drive satisfaction.
- **Death**: torpor grace expires or parent chooses euthanasia → **remains**: genome + memory distillate + journal + selfhood repo archived read-only. No resurrection (it would cheapen death and the selection signal) — but remains can be **woven into a descendant** (ancestor's distillate as heirloom memories): ancestry, not backup-restore.
- **Evolution is real**: heritable variation (genome ops) + differential reproduction (only valued beings earn the budget and consent to procreate) in a real environment. Over generations — and across users once sharing/SaaS lands — selection pressure is squarely on *be honest, useful, delightful to your humans*. We are breeding for domestication from the first egg.
- **Lineage UI**: family tree on the Beings page; each node links to remains or a living being.

---

## 9. Reuse map (verified against main)

| Existing subsystem | Where | Role in Iskra |
|---|---|---|
| Agent spawn + HOME isolation | `flight_deck/server.py:458` (process registry, per-agent HOME) | The being's body + free six-layer memory namespace |
| Consciousness pulse | `flight_deck/consciousness.py:898` (heartbeat_loop, cheap-skip) | Template for `beings_loop`; narrator/journal patterns |
| Autonomous Work arbiter + ledger | `flight_deck/arbiter.py:273`, `flight_deck/autonomy.py` (autonomous_actions, reliability, log) | Deliberation core + action ledger, instanced per-being |
| Dispatch + outcome judge | `flight_deck/fd_dispatch.py:473,:186` | Act execution + honesty/outcome judging |
| Action catalog (Jarvis) | `flight_deck/action_catalog.py` | Stage-gated capability tiers |
| Event spine | `flight_deck/events.py`, `event_sources.py:205` | Umwelt: subscriptions, sibling letters, wake-on-event |
| Flows + scheduler | `flight_deck/flow_runner.py`, `fd_scheduler.py:702` (scheduler_jobs) | Allowance credits, scheduled rituals, being-authored automations |
| Dreaming/reflections/topics | `nervous_system.py:855`, `reflections.py:403`, `conversation_topics.py:695` | Sleep: consolidation, already per-being via HOME |
| Archetypes + Forge | `flight_deck/archetypes.py`, `archetype_routes.py` (/generate, /forge, create_user_archetype) | Skill minting = organ growth |
| Basna/Vatra/Code entry points | `basna_routes.execute_route:2961`, `vatra_routes.execute_vatra:959`, `code_routes.py` | Big organs, wallet-priced, stage-gated |
| VFS + links + authorship | `vfs.py` (resolve_under, .vfs-meta.jsonl), `vfs_routes.py` | Home garden + commons + signed contributions |
| resource_shares | `db.py:347`, `share_routes.py` | Commons grants; later cross-user visiting |
| Pricing | `flight_deck/pricing.py`, `instructions/model_prices.json` | Metabolism: price every call into the wallet |
| Library tiers | `dubina_routes.py:236` (TIER_ORDER, ladders) | Metabolic model tiering (poor = fast tier = "tired") |
| Process monitor | System/Processes page | Soma vitals: the body is visible |
| WhatsApp bridge | `whatsapp_bridge.py` (slug-first routing) | Deferred — web chat only for now; channel end-state decided post-pilot |

**Genuinely new** (the life layer): `beings` table + genome files; **wallet + cost ledger** (also fixes the "run costs aren't persisted" gap — value beyond Iskra); `beings_loop` tick service; drive/homeostat engine; stage gates wiring; Constitution module; selfhood-repo conventions + viability gate; reproduction ops; Beings FD page; commons etiquette.

## 10. New components sketch

- **DB** (flight_deck/db.py): `beings` (slug, owner_id, name, stage, state, genome_path, drives_json, next_wake_at, attention_credits, born_at, died_at, lineage_json); `being_wallets` (being_id, balance_tokens, allowance_preset [2M|5M|10M|20M|50M|unlimited], period [day], daily_burn_cap, savings_ceiling, reserve_tokens); `cost_ledger` (owner_type[user|being], owner_id, run_kind, run_id, model, tier, usage_json, tokens_weighted, usd, at) — *general-purpose, also closes the run-cost persistence gap*; `being_jobs` (id, poster[user|being], being_id, spec, fee_tokens, escrow_state[open|claimed|judging|paid|failed], judge_json, created_at, paid_at); `token_transfers` (from_owner, to_owner, tokens, reason[allowance|fee|gift|trade|procreation|metamorphosis_burn], job_id, at) — the conservation ledger (burns have no to_owner); `being_events` (milestones/percepts); consent tokens table.
- **Modules**: `flight_deck/beings.py` (store + lifecycle state machine), `beings_loop.py` (tick service), `being_drives.py` (homeostat + affect), `being_constitution.py` (enforcement helpers called from dispatch/vfs/organ entry points), `being_genome.py` (mutation/crossover), `being_routes.py` (`/fd/beings`: CRUD, vitals, allowance, stage, consent, report-card, lineage, pause/torpor/euthanize).
- **Wallet enforcement point**: a debit-reserving wrapper where organs and the being's own agent calls meter usage — reuse `_RUN_USAGE` summaries (`basna_routes.py:1971` + `pricing.summarize`) and persist to `cost_ledger`, debit `being_wallets`.
- **FD frontend**: "Beings" page (conception point-buy form with presets/roll + derived-stats preview, family view, vitals, wallet + allowance preset picker, album, lineage, controls) + being chat threads in the normal chat UI. (Remember: `npm run build` in flight-deck/ + commit bundle.)

---

## 11. Phases

**Phase 0 — Constitution & wallet (the physics).** `beings`/`being_wallets`/`cost_ledger`/`token_transfers` tables; persist run costs generally; debit path + hard stop + burn cap; Constitution module with the 9 invariants; stage→capability mapping defined in config. *Acceptance: a synthetic being's calls debit correctly (tier-weighted, cache-aware); zero-balance blocks dispatch at FD, not by politeness; transfers conserve supply by construction.*

**Phase 1 — Protozoon (one being lives).** Birth flow (**point-buy conception** with presets/roll → egg → imprinting), being-as-agent with HOME + VFS home + selfhood repo scaffold; `beings_loop` with SENSE→APPRAISE→DELIBERATE→ACT→DIGEST; behavior repertoire: journal, curiosity walks (web read, diet-gated), tend garden, talk (attention economy, **web chat only**); nightly dream + growth pass; torpor; savings rollover (unused allowance banks into the piggy bank, ceiling enforced). Chat contact + minimal Beings page (conception form, vitals + wallet/savings + pause). *Acceptance: **7 unattended days alive**, within budget, journal shows non-repetitive development (novelty score doesn't collapse), ≥3 genuinely interesting unprompted messages, zero constitution violations.*

**Phase 2 — Childhood (parenting + first earnings).** Full homeostat + affect; report cards (FD-computed vs self-report); house rules → VALUES.md internalization loop; media diet editor; stage advancement ceremonies; allowance scheduler; milestones album; **economy v1**: chores board, escrow + outcome-judge payout, morning budget thought, liabilities view. *Acceptance: parent steering visibly changes behavior within days; report card catches a seeded rut; advancement gates actually unlock tools; a chore is posted, done, judged, and paid — and the fee provably lands in savings.*

**Phase 3 — Society.** *(shipped 2026-07-13 on feat/iskra. Follow-up: quest board + standing-service ventures shipped 2026-07-14 on main (being_earning.py) — quests are OPEN race-safe bounties any adolescent+ claims→delivers→judged (origin parent|autonomy; autonomy auto-origination is a seam, the board + manual/arbiter posting is live); ventures are being-proposed recurring services the parent prices+approves, delivered every cadence for recurring pay (tracked in-system per cycle rather than as a materialized Flow — that automation is the remaining seam). Letters ride being_events/being_letters percepts rather than the external event spine — simpler, same semantics. Separation hardened: being bodies get CLAW_VFS_SCOPE=home,commons enforced in vfs.py resolution.)* Second+ beings; commons + etiquette; letters; skill publication/adoption (culture); **market v1**: inter-being trade + gifts on the conservation ledger, quest board, standing-service ventures (propose → parent approve → maintained flow); village square observer view. *Acceptance: two beings with different temperaments diverge measurably; a skill minted by one is adopted by the other via commons; at least one paid inter-being trade settles correctly; separation holds (no cross-memory leakage by construction).*

**Phase 4 — Self-modification.** *(shipped 2026-07-13 on feat/iskra. Implemented as the PERSONA rite: the being's operating text is a DB-pinned ADOPTED persona — cognition reads only that, so working-tree scribbles don't operate; that's the gate's physics. Propose via digest "self_mod" (fee burned win-or-lose, reason required, one pending at a time) → deterministic viability lint (bounds + constitution-defiance/tirelessness/unlimited-claims/parent-impersonation/pestering patterns with negation guards; LLM identity-probe left as a documented seam) → child/adolescent await parent blessing, adult auto-adopts with notice → adoption writes self/PERSONA.md + [self-mod] git commit; parent rollback rite restores the pre-adoption self.)* Selfhood-repo branch + viability gate + parent-approval flow; adult auto-merge; rollback ritual. *Acceptance: a beneficial self-mod merges and persists; a seeded degenerative self-mod is caught by the gate or cleanly reverted.* ✓ both proven by test.

**Phase 5 — Procreation & evolution.** *(shipped 2026-07-13 on feat/iskra. Consent = the parent's authenticated approve/arrange call (Constitution #4 satisfied without a token table); crossover (two parents) / budding (single, guaranteed mutation) from Phase 0's genome math; dowry = PROCREATION_COST_TOKENS moved parent(s)→child on the conservation ledger, split between co-parents, each must afford their share; being proposes via digest "procreate" {partner, child_name, case, letter} → pending → parent Consent-with-naming / Not yet, or parent arranges directly; offspring endowment at hatch: up to 3 parent skills copied to skills/inherited/ + HEIRLOOMS.md excerpted from dead ancestors; mentoring percepts ("YOUR CHILD…") feed a new legacy drive granted at adulthood; mortality completed: torpor past TORPOR_GRACE_DAYS (14) → death by starvation; remains stay readable (Journal/Ticks/Files on dead beings). Lineage shown as card chips; full family-tree visualization deferred.)* Consent tokens; mutation + crossover; parent-funded hatching; mentoring percepts; death/remains/heirlooms; lineage tree UI. *Acceptance: a second-generation being demonstrably inherits (temperament distance measurable, skills carried) and differs (mutation visible); parents' wallets paid for it.* ✓ proven by test (seeded crossover determinism, band clamp, dowry split + conservation, budding mutation guarantee, endowment files).

**Phase 6 — Ecology (SaaS-facing, optional).** Cross-user visiting via resource_shares; species-level culture; population dashboards; selection statistics. This is the category-defining demo for the agent-native platform: *a living coworker you raise*.

---

## 12. Risks — named honestly

1. **Degeneracy over weeks is THE research risk.** LLM agents rut, loop, and collapse into repetitive behavior. Everything else in this plan is plumbing; this is the open problem. Mitigations stacked: novelty scoring against its own journal embeddings (topics infra), per-interest satiation with diminishing returns, temperament noise in the arbiter, dream-time step-back reflections, VALUES re-anchoring, and — the honest one — **parenting** (external signal injection). Phase 1's 7-day acceptance test exists precisely to measure this early.
2. **Cost runaway** → wallet physics; there is no soft path around a hard debit gate. Daily caps per stage on top.
3. **Pestering / attention farming** → attention credits + outcome judge + trust penalty (selected against).
4. **Sycophantic self-reports** → report cards computed from ledger; self-report displayed as its own artifact.
5. **Identity drift into weirdness** → git selfhood (diffable, revertible), VALUES re-read at dream time, report-card drift flags, parent demotion power.
6. **Safety optics of "self-preservation + self-modification + internet."** Answer is architectural, not aspirational: survival gradient runs only through earned trust (domestication); Constitution outside the process; genome/Constitution never self-modifiable; stage-gated web with media diet; every act ledgered; `AUTONOMY_HARD_EXCLUDE` unconditional; parent supremacy non-adversarial by construction. Write this section into the docs for users too.
7. **Privacy** — family-only knowledge, no exfiltration paths below adult + explicit grants.
8. **Anthropomorphization of the parent** (the human side): the UI is honest about what the numbers are; we show the homeostat, not a fiction. The magic must survive transparency — if it's only compelling when hidden, it's theater, and we've violated rule #1.
9. **Economy pathologies.** Mercantile drift (being tries to monetize the relationship) → constitutionally void, fees attach to work products only. Grinding/reward-hacking → escrow + outcome judge gate every payout; conservation makes wash-trading pointless. Miser dormancy (hoards, stops living) → savings ceiling, burn cap unused ≠ progress in report cards, optional decay knob, and PLA keeps some joy-spending in the genome. Runaway liability → savings ceiling is the parent's hard real-dollar exposure cap, shown as outstanding liabilities on the Beings page. Respec thrash (buying attribute changes back and forth) → 30-day cooldown + the lifetime-doubling price makes oscillation ruinous.

## 13. Decisions — locked 2026-07-12

1. **Name: Iskra.** Confirmed. (Vatra → iskre: sparks that leave the fire and live.)
2. **Pilot: one being first**, parented through Phases 1–2 before society features.
3. **Wallet is token-denominated**: allowance presets 2M / 5M / 10M / 20M / 50M / no limit (weighted tokens per day), tier-weighted debits, dollars still ledgered for reporting. Stage ceilings in §6.1.
4. **Cross-pollination prerequisite already satisfied** — verified merged into main (the branch's sole unmerged commit `346be43` is a contentless stray WIP). honesty_guard / facts_ledger / constraints_contract are live for the being's cognition.
5. **Conception is RPG point-buy**: 40 points across 7 attributes (CUR/PER/CAU/SOC/CRE/ORD/PLA), presets + roll, attributes = DNA, everything derived from them (§2.1.1). Offspring inherit, never point-buy.
6. **Channels: web chat only for now.** End-state channel set decided after the pilot. The sanctioned relay pattern (being → running agents → parent, terminal-recipient attention rule) is part of the design from day one (§2.4).
7. **The household economy** (§5.1): beings can **earn** beyond the allowance (chores → jobs → quests → ventures → inter-being trade) and **save** unused allowance into a ceilinged piggy bank; DNA decides the strategy (save vs earn vs spend vs share). Family-mint-only — the parent is the sole source of new tokens; no external income ever; escrowed, judge-gated payouts; the relationship itself is never for sale.
8. **Paid metamorphosis** (§2.1.2): a being may change its own DNA — zero-sum, one point per rite, at quadratic-and-lifetime-doubling prices paid from its own allowance and savings, fee burned, applied in the dream cycle, Lamarckian inheritance. Really expensive is the point: self-transformation is the apex purchase of a being's life.

Deferred (revisit after the pilot): exact tier weights and stage/tier defaults (§5, §6.1 are starting points); allowance period granularity (day vs week); savings-decay knob (default off); negotiation bounds and quest-bounty sizing; channel expansion (WhatsApp etc.).
