---
name: example-source-brief
description: Build a source-checked markdown brief from URLs or a topic, then save it into the session showcase folder.
user-invocable: true
disable-model-invocation: false
metadata:
  captainclaw:
    emoji: "🧭"
    skillKey: "example-source-brief"
---

# Example Source Brief

Use this skill when the user wants a concise report from one or more web sources.

## Workflow

1. Gather sources:
- If the user provided URLs, use them directly.
- If only a topic is provided, use `web_search` first and pick relevant sources.

2. Read sources:
- Use `web_fetch` for each selected URL.
- Keep short notes per source: key claims, dates, and uncertainty.

3. Produce a single markdown brief:
- Use the structure from `references/brief-format.md`.
- Include source links in each section.
- Keep tone factual and concise.

4. Save the output:
- If user asks to save, write to `saved/showcase/{session_id}/source-brief.md`.
- If user asks for a specific file name, use that name under the same session showcase folder.

## Quality Rules

- Do not invent facts; if evidence is missing, say so.
- Prefer 3-7 sources unless user requests a different depth.
- If sources conflict, include a short "Conflicts" section.

## Notes

- Prefer internal tools (`web_search`, `web_fetch`, `write`) over generating scripts.
- Only generate scripts if the user explicitly asks for a script/tool.

