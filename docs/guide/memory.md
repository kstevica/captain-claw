# Memory and cognition

[← README](../../README.md) · [All guides](README.md)

Captain Claw has a six-layer memory system and autonomous cognitive processes that run without user intervention.

**Memory Layers:**

| Layer | What it stores | How it's used |
|---|---|---|
| Working Memory | Current conversation in the LLM context window | Immediate reasoning |
| Semantic Memory | Hybrid vector + BM25 full-text search over documents and sessions | Auto-injected when relevant to the current query |
| Deep Memory | Typesense-backed long-term archive, scales to millions of documents | Searched on demand for deep recall |
| Insights | Auto-extracted facts, contacts, decisions, and deadlines (SQLite + FTS5) | Cross-session knowledge injected into system prompt |
| Nervous System | Autonomous "intuitions" — patterns, hypotheses, and connections | Surfaces non-obvious findings the agent wouldn't otherwise notice |
| Self-Reflection | Periodic self-assessment of conversations, memory and completed tasks | Improvement directives injected into the system prompt |

Since 0.8.0 a **crowded-session context engine** keeps all of this cache-friendly on modest models: background notes form one frozen block per turn, correctives and rejected replies are never replayed, automated turns run in their own `AUTO` lane, and a turn's first request repeats 96–98% of the previous one (5.6% before). Deep memory finally searches by meaning (its vector width is auto-detected; every vector used to be silently dropped) and is one owner-scoped Flight Deck capability.

**What an owner teaches (0.8.4).** Beside the six layers, an agent set up through [Onboarding](flight-deck.md#onboarding) keeps two things of its owner's. `knowledge/` holds whole documents, which semantic memory indexes first under its own cap (`memory.max_knowledge_files`, 3,000) and the `knowledge` tool searches. **Lessons** are rules, ask-first items and facts that reach the agent with the messages they bear on, picked by words and by meaning, at most 6 items and 1,800 characters a turn.

**Autonomous Processes:**

- **Dreaming** — Background dream cycles cross-reference all memory layers to synthesize intuitions. Runs after every N messages and during idle hours. Intuitions have confidence scores that decay over time unless validated.
- **Tension Tracking** — Holds unresolved contradictions (like musical dissonance) rather than forcing premature resolution. Tensions persist until evidence resolves them.
- **Maturation Pipeline** — New intuitions sit through multiple dream cycles before being surfaced to the agent, reducing noise.
- **Cognitive Tempo** — Detects whether the user is in deep contemplative mode or rapid task execution, and adapts processing depth accordingly (adagio / moderato / allegro).
- **Cognitive Modes** — Seven tunable behavioral profiles (Ionian through Locrian, inspired by musical scales) that shift the agent between analytical, creative, cautious, and exploratory approaches.
- **Self-Reflection** — Periodic self-assessment that reviews conversations, memory, and completed tasks to generate improvement directives injected into the system prompt; since 0.8.3 it also reads a digest of the agent's repeated failures and repeated calls from the tool-call register.
- **Insights Extraction** — Automatically identifies durable knowledge from conversations — deduplicates, categorizes, and stores for future context injection. Since 0.8.4 an exact repeat or the same fact reworded is dropped (numbers and negations must match), and insights keep the file and section they came from; dreams' intuitions get the same check.

**Visualization:**

- **Brain Graph** — Interactive 3D force-directed graph of the entire cognitive topology. Insights, intuitions, tasks, contacts, and sessions rendered as typed nodes with provenance edges. Live WebSocket updates.
- **Process of Thoughts** — Full lineage tracking across all cognitive subsystems. Every message, insight, intuition, and task is connected via provenance IDs, forming a traversable thought graph.
