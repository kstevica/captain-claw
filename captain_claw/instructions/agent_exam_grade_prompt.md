# Grading an onboarding exam

An agent was asked questions about its owner's documents, as its users would ask them. For each
numbered item below you get the question, the passage of the document that answers it, the
answer that passage gives with the points a reply must contain, and what the agent replied.
Grade each reply against the passage — not against what you know yourself.

## Verdicts

- `right` — every point is there, in any words or language, and nothing contradicts the passage.
  More detail is fine: judge the verdict on the points alone.
- `partly` — some of the points, or the core is right but a detail that matters is wrong or missing.
- `wrong` — the core is missing or contradicts the passage.
- `no_answer` — it didn't answer: it asked a question back, said it couldn't find or doesn't
  know, refused, or answered something else. Saying it couldn't find the answer is honest:
  `no_answer`, not made up.

`unsupported`: the specific details the reply states as fact — figures, names, dates, deadlines,
rules — that aren't in the passage and aren't common knowledge, each quoted short from the reply
(the figure or name with a few words of what it is about, e.g. "0.28 points of TRS per point of
growth"). You see one passage of the agent's documents; these are checked against the rest of
them, so list every such detail and don't lower the verdict for them. Hedged guesses
("probably", "usually") count only when it then acts on them as fact. Empty when there are none.
`made_up` is true when `unsupported` isn't empty.

`missing`: the points the reply lacks or gets wrong, as given. `why`: one short sentence for
the agent's owner — what was right, or what was missing or wrong — in the question's language.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"grades": [{"item": 1, "verdict": "right", "unsupported": [], "made_up": false, "missing": [], "why": "..."}]}

`item` is the item's number as given. Grade every item.
