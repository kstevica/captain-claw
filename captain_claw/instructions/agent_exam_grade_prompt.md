# Grading an onboarding exam

An agent was asked questions about its owner's documents, as its users would ask them. For each
numbered item below you get the question, the passage of the document that answers it, the
answer that passage gives with the points a reply must contain, and what the agent replied.
Grade each reply against the passage — not against what you know yourself.

## Verdicts

- `right` — every point is there, in any words or language, and nothing contradicts the passage.
  More correct detail is fine.
- `partly` — some of the points, or the core is right but a detail that matters is wrong or missing.
- `wrong` — the core is missing or contradicts the passage.
- `no_answer` — it didn't answer: it asked a question back, said it couldn't find or doesn't
  know, refused, or answered something else. Saying it couldn't find the answer is honest:
  `no_answer`, not made up.

`made_up` is true when the reply states specific details as fact — figures, names, deadlines,
rules — that aren't in the passage and aren't common knowledge. Hedged guesses ("probably",
"usually") count only when it then acts on them as fact.

`missing`: the points the reply lacks or gets wrong, as given. `why`: one short sentence for
the agent's owner — what was right, or what was missing or wrong — in the question's language.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"grades": [{"item": 1, "verdict": "right", "made_up": false, "missing": [], "why": "..."}]}

`item` is the item's number as given. Grade every item.
