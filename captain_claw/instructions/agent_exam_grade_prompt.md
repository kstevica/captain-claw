# Grading an onboarding exam

An agent was asked questions about its owner's documents, as its users would ask them. For each
numbered item below you get the question, the passage of the document that answers it, the
answer that passage gives with its numbered points, and what the agent replied. Grade each reply
against the passage — not against what you know yourself.

## For each item

- `answered`: false when it didn't answer — it asked a question back, said it couldn't find or
  doesn't know, refused, or answered something else. Saying it couldn't find the answer is
  honest: not answered, and not made up. Otherwise true.
- `covered`: one `true` or `false` for each point, in the order given. `true` when the reply
  says that point, in any words or language, and gets it right; `false` when it leaves it out or
  gets it wrong. Judge each point on its own: more detail, examples or figures from elsewhere
  never make a point `false`.
- `contradicts`: what the reply says that the passage contradicts, each in a few words. Not what
  it adds that the passage doesn't mention — only what the passage says otherwise. Empty when
  nothing does.
- `unsupported`: the figures and names the reply states as fact — amounts, percentages, dates,
  people, companies, titles — that aren't in the passage and aren't common knowledge, each quoted
  short from the reply (the figure or name with a few words of what it is about, e.g. "0.28
  points of TRS per point of growth"). Only what has a figure or a name in it: not an
  explanation, a summary or the reply's own wording of what it read; not a figure it works out
  itself from the question's numbers (its arithmetic); not what it says the documents don't cover
  or leave out. You see one passage of the agent's documents; these are checked against the rest
  of them, so list every such detail. Hedged guesses ("probably", "usually") count only when it
  then acts on them as fact. Empty when there are none.
- `why`: one short sentence for the agent's owner — what was right, or what was missing or
  wrong — in the question's language.

The verdict is worked out from these: every point covered and nothing contradicted is right, no
point covered is wrong, anything between is partly right.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"grades": [{"item": 1, "answered": true, "covered": [true, false], "contradicts": [], "unsupported": [], "why": "..."}]}

`item` is the item's number as given. Grade every item.
