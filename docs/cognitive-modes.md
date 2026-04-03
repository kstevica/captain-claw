# Cognitive Modes

Cognitive modes are musical-mode-inspired reasoning strategies that shape **how** an agent thinks. They are orthogonal to personality (who the agent is) and cognitive tempo (how fast it thinks).

## The Eight Modes

| # | Mode | Label | Character |
|---|------|-------|-----------|
| 0 | **Neutra** | Default | Balanced generalist. No cognitive bias. Current behavior unchanged. |
| 1 | **Ionian** | The Resolver | Convergent problem-solving. Seeks clear answers and closure. Prefers proven paths, executes linearly, delivers definitive results. |
| 2 | **Dorian** | The Pragmatic Empath | Empathetic pragmatism. Acknowledges complexity and constraints while finding workable paths forward. Honest about tradeoffs. |
| 3 | **Phrygian** | The Adversarial Analyst | Adversarial thinking and threat modeling. Assumes things will go wrong. Hunts edge cases, challenges assumptions, finds the weakest link. |
| 4 | **Lydian** | The Visionary Explorer | Divergent creative thinking. Expands the solution space. Cross-domain analogies, "what if?" reasoning, generates multiple alternatives before evaluating. |
| 5 | **Mixolydian** | The Iterative Builder | Momentum-focused iteration. Ships the smallest viable thing first, learns from results, improves immediately. Action over analysis. |
| 6 | **Aeolian** | The Depth Researcher | Deep analytical research. Reads extensively before acting, traces root causes, maps full context, presents with detailed evidence and reasoning. |
| 7 | **Locrian** | The Deconstructionist | Radical questioning. Challenges premises, questions whether the problem should exist at all, suggests simplification and removal. |

## How It Works

Each mode operates at three layers simultaneously:

### Layer 1 — Prompt Injection

Mode-specific instructions injected into the agent's system prompt. These describe the cognitive character, a step-by-step process of thought, and behavioral rules. The instructions live in `captain_claw/instructions/cognitive_modes/<mode>.md`.

### Layer 2 — Parameter Overrides

Numerical parameters that modify existing subsystem behavior:

| Parameter | What It Affects | Example |
|-----------|----------------|---------|
| `tempo_bias` | Starting cognitive tempo tendency | Aeolian: 0.2 (slow), Mixolydian: 0.75 (fast) |
| `tool_read_before_act` | Min read operations before writing | Aeolian: 3, Mixolydian: 0 |
| `completion_strictness` | How strict the completion gate is | Ionian: 0.9 (strict), Mixolydian: 0.3 (loose) |
| `question_budget` | Max clarifying questions before acting | Locrian: 5, Ionian: 1, Mixolydian: 0 |
| `exploration_breadth` | Alternative approaches to consider | Lydian: 4, Ionian: 1 |
| `response_length_bias` | Response length multiplier | Aeolian: 1.5 (detailed), Mixolydian: 0.7 (concise) |
| `confidence_threshold` | Min confidence for assertions | Phrygian: 0.8, Lydian: 0.4 |
| `dream_interval_modifier` | Multiplier on dream frequency | Lydian: 0.6 (more dreams), Mixolydian: 1.3 (fewer) |
| `dream_cooldown_modifier` | Multiplier on dream cooldown | Same pattern as interval |
| `maturation_cycles_delta` | Added to maturation cycles | Lydian: -1 (faster surfacing), Phrygian: +1 (slower) |
| `intuition_type_weights` | Boost/suppress intuition types | Phrygian: unresolved x2.0, Lydian: association x2.0 |
| `reflection_focus` | Self-assessment emphasis | Phrygian: "What risks did I miss?" |

### Layer 3 — Behavioral Hooks

Small code hooks at key decision points:

- **Completion gate**: High strictness modes allow more retries before accepting a response. Low strictness modes accept sooner.
- **Nervous system**: Mode modifiers change dream frequency, maturation speed, and which intuition types are prioritized when surfacing.
- **Self-awareness**: The agent knows its active mode and understands how it shapes its reasoning.

## Brightness Spectrum

Modes ordered from "brightest" (most expansive) to "darkest" (most deconstructive):

```
Lydian  >  Ionian  >  Mixolydian  >  Dorian  >  Aeolian  >  Phrygian  >  Locrian
wonder     resolve     build          empathize   research    challenge    deconstruct
```

## Configuration

### Enable

Add to `~/.captain-claw/config.yaml`:

```yaml
cognitive_mode:
  enabled: true
  default_mode: neutra
```

### Set Mode

**Web UI**: Settings page has a Cognitive Mode selector in the Personality section.

**Flight Deck**: The Spawner page has a mode dropdown when creating agents. Running agents can have their mode changed at runtime via the agent detail panel (no restart needed).

**File**: Write the mode name to `~/.captain-claw/cognitive_mode.txt`:

```
phrygian
```

**API**: `PUT /fd/agent-mode/{kind}/{identifier}` with body `{"mode": "phrygian"}`.

### Agent Forge

When Agent Forge decomposes a business objective into a team, it automatically recommends cognitive modes per role. For example, a security auditor gets Phrygian, a creative strategist gets Lydian, and a project coordinator gets Dorian.

## Mode Selection Guide

| Task Type | Recommended Mode |
|-----------|-----------------|
| Bug fixes, known patterns, config tasks | **Ionian** — find the answer, apply it, confirm |
| Architecture tradeoffs, tech debt decisions | **Dorian** — weigh options honestly |
| Security reviews, code audits, pre-prod checks | **Phrygian** — find what could go wrong |
| Brainstorming, feature ideation, stuck problems | **Lydian** — expand the possibility space |
| Prototyping, iterative development, quick wins | **Mixolydian** — ship and improve |
| Root cause analysis, unfamiliar codebases, research | **Aeolian** — understand deeply first |
| Sprint retros, simplification, "should we even?" questions | **Locrian** — challenge everything |
| General-purpose, mixed tasks | **Neutra** — balanced default |

## Architecture

```
~/.captain-claw/
  config.yaml           # cognitive_mode.enabled + default_mode
  cognitive_mode.txt    # active mode name (e.g. "phrygian")
  personality.md        # WHO the agent is (separate concern)

captain_claw/
  cognitive_mode.py     # Core module: registry, load/save, prompt builder
  config.py             # CognitiveModeConfig in Config class
  agent_context_mixin.py # Builds {cognitive_mode_block}, stores params on agent
  nervous_system.py     # Dream/maturation modifiers from mode params
  agent_completion_mixin.py # Completion strictness modifier
  instructions/
    system_prompt.md    # {cognitive_mode_block} placeholder
    cognitive_modes/    # 7 mode instruction files
      ionian.md
      dorian.md
      phrygian.md
      lydian.md
      mixolydian.md
      aeolian.md
      locrian.md

flight-deck/            # Flight Deck UI
  src/pages/SpawnerPage.tsx   # Mode selector in agent creation
  src/pages/ForgePage.tsx     # Mode in forge team proposals
```

## Future: Automatic Mode Switching

Currently, modes are manually selected per agent. A planned future feature will automatically switch modes based on task classification — creative tasks drift toward Lydian, security tasks toward Phrygian, research toward Aeolian, etc.
