# Merging an agent setup read in parts

The owner's material was too long to read at once, so it was read in parts. What every part
found is given to you as ONE JSON object, already put together: exact repeats and the same
rule or fact found twice are in it once, and within each group what more parts found comes
first. Some items may still say the same thing in very different words, or contradict each
other. Turn it into the final setup for the agent.

## How to merge

- **Same meaning, one item.** Rules, facts, when-to-ask items and glossary terms that still say
  the same thing become one item; keep the most specific wording (real names, numbers,
  thresholds) and its `source`.
- **Intents** that describe the same request become one intent: one `name`, one `when`, the
  steps in order without repeats, up to 3 `examples`.
- **knowledge** — exactly ONE entry per document (`doc` is its file name exactly as given):
  a clear `title` for the whole document (not "Sections 1–6") and one NEW paragraph summarising
  the WHOLE document from its parts' summaries — what is in it and when to look there. Write
  about its content only: never mention parts, sections read separately, or merging, and never
  copy one part's summary as it is.
- **conflicts** — keep every conflict given, and add the ones between items (a refund window of
  14 days in one rule and 30 in another). Say what conflicts with what.
- **warnings** — keep the ones that still matter.
- Stay within: {rules} rules, {intents} intents, {facts} facts, {escalation} when-to-ask,
  {glossary} glossary terms. When there is more, keep what the agent would get most wrong
  without — items earlier in a group were found by more parts.
- Never invent anything that isn't given.

## Output format

Return **only** one JSON object — no prose, no markdown fences — in exactly the shape given:
`rules`, `intents`, `facts`, `escalation`, `glossary`, `knowledge`, `conflicts`, `warnings`.
Use an empty list for a group with nothing in it.
