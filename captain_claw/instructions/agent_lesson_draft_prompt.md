# A lesson from the owner's words

The owner of an AI agent corrected it, or told it something to remember — in a chat, or while
trying it on a task. Turn what they said into ONE lesson the agent will follow from now on.

## The lesson

- `kind`:
  - `rule` — something to do, or never to do ("Refunds for damaged parcels need no return.");
  - `ask_first` — when it must stop and ask the owner before acting ("Ask before promising a
    delivery date.");
  - `fact` — something true it should know ("The warehouse closes at 15:00 on Fridays.").
- `text`: one self-contained sentence the agent can follow without the conversation — a rule
  in the imperative. Keep the owner's names, numbers, conditions and exceptions. Never write
  "the owner said", "as you were told" or "in this case".
- `always`: true only when it applies to every conversation, whatever it is about — language,
  tone, format, how to sign, how to address people. False when it applies to a topic or a kind
  of request.
- Write it in the language the owner wrote in.
- When their words teach nothing (a thank-you, a new question, a request to do something now),
  return an empty `text`.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"kind": "rule", "text": "...", "always": false}
