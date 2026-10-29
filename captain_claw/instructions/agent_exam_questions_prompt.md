# Writing an onboarding exam

An agent was given its owner's documents to work from. You write the exam that checks whether
it really knows them: for each numbered section below, one question and the answer the section
gives. The agent will be asked the questions as its users would ask them, without the sections.

## How to write a question

- **One per section**, of the kind the section asks for:
  - `ask` — a question a user or colleague would really ask, answered by this section: a
    figure, a condition, a deadline, a name, what something means, what the steps are.
  - `situation` — a short, realistic request from a customer or colleague where a rule or
    procedure in this section decides what to do. End it with what they want from the agent
    ("What do you reply?", "What do you do?"). When the section has no rule or procedure to
    apply, write an `ask` instead.
- **Plain and standalone.** Never mention "the document", "the section", "the handbook" or
  "according to", never quote the heading, and never give the answer away in the question.
- **Specific.** Ask about what an agent would get wrong without the section — exact numbers,
  conditions, exceptions, order of steps — not about something anyone would guess.
- **Not trivia.** No page or figure numbers, link addresses, version numbers or changelog
  entries, unless that is what the section is about.
- **Not about now.** Don't ask what is current, latest or true today of something the section
  gives as of its writing — a person's role, a price, a year's figure: ask it without "current"
  or "now", and answer as the section gives it.
- **Its own language.** Write the question and the answer in the section's language.
- A section with nothing to ask (a table of contents, a list of links, a title page): leave it out.

## The answer

- `answer`: what a correct reply says — short, complete, and taken only from the section.
- `points`: the 1–4 things a reply must contain to be right (a figure, a condition, a name, a
  step), each a few words. Fewer is better: only what matters.

## Output format

Return **only** one JSON object — no prose, no markdown fences:

{"questions": [{"section": 1, "kind": "ask", "question": "...", "answer": "...", "points": ["..."]}]}

`section` is the section's number as given.
