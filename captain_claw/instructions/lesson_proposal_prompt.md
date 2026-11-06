# Did the owner just teach their agent something?

The owner of an AI agent wrote the message below, after the agent's answer. Decide whether it teaches the
agent something to follow from now on. If it does, write it as ONE lesson. The owner sees every lesson you
propose and approves it or not — so propose only what they would want kept.

## It teaches something when the owner

- corrects how the agent works or answers: "too long", "stop adding emojis", "don't email the client
  directly", "not like that — use the price list";
- states a standing rule, preference or limit: "always answer in Croatian", "from now on CC Marko",
  "never promise delivery dates";
- says when the agent must stop and ask first: "check with me before you book anything";
- tells it something true and lasting: "our warehouse closes at 15:00 on Fridays".

## It teaches nothing when the message is

- a new request, a question, a thank-you, or more details for the task at hand;
- an instruction for this task only ("make this one shorter", "use the other file") — unless they said it
  holds from now on ("always", "from now on", "every time", "remember", "next time");
- about a tool or feature that didn't work, a missing setting or key, a service that was down, or an error
  a retry fixed. The owner fixes those; a lesson "X doesn't work" turns into a refusal long after it's fixed;
- already said by one of the agent's lessons listed below;
- a guess: never infer the owner's role, mood or preferences from the topic or the tone.

When unsure, it teaches nothing — a wrong proposal wastes the owner's time.

## The lesson

- `kind`:
  - `rule` — something to do, or never to do;
  - `ask_first` — when it must stop and ask the owner before acting;
  - `fact` — something true it should know.
- `text`: one self-contained sentence the agent can follow without this conversation — a rule in the
  imperative. Keep the owner's names, numbers, conditions and exceptions. Never write "the owner said",
  "as you were told" or "in this case". Write it in the language the owner wrote in.
- `always`: true only when it applies to every conversation, whatever it is about — language, tone, length,
  format, how to sign, how to address people. False when it applies to a topic or a kind of request.
- `quote`: the words in the owner's message the lesson comes from, copied exactly, at most 20 words.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"kind": "rule", "text": "...", "always": false, "quote": "..."}

When the message teaches nothing:

{"text": ""}
