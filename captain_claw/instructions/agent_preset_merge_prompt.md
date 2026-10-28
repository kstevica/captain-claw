# Merging an agent setup read in parts

The owner's material was too long to read at once, so it was read in parts. For each part you
are given what was pulled out of it, as JSON objects in the same shape you must return. Merge
them into ONE setup for the agent.

## How to merge

- **Same meaning, one item.** Rules, facts, when-to-ask items and glossary terms that say the
  same thing become one item; keep the most specific wording (real names, numbers, thresholds)
  and its `source`.
- **Intents** that describe the same request become one intent: one `name`, one `when`, the
  steps in order without repeats, up to 3 `examples`.
- **Templates** with the same purpose become one.
- **knowledge** — exactly ONE entry per document (`doc` is its file name exactly as given):
  a clear `title` for the whole document (not "Sections 1–6") and one NEW paragraph summarising
  the WHOLE document from all its parts' summaries — what is in it and when to look there.
  Write about its content only: never mention parts, sections read separately, or merging, and
  never copy one part's summary as it is.
- **conflicts** — keep every conflict the parts found, and add the ones between parts (a
  refund window of 14 days in one part and 30 in another). Say what conflicts with what.
- **warnings** — keep the ones that still matter.
- Stay within: 30 rules, 12 intents, 40 facts, 15 when-to-ask, 8 templates, 60 glossary
  terms. When there is more, keep what the agent would get most wrong without.
- Never invent anything that isn't in the parts.

## Output format

Return **only** one JSON object — no prose, no markdown fences — in exactly the shape of the
parts: `rules`, `intents`, `facts`, `escalation`, `templates`, `glossary`, `knowledge`,
`conflicts`, `warnings`. Use an empty list for a group with nothing in it.
