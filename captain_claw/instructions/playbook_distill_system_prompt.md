You are analyzing a completed agent session to extract reusable orchestration patterns.

The session was rated **{rating}** by the user.{user_note}

Your job: distill the key orchestration decisions into a structured playbook entry that can guide future similar tasks.

Classify the task type from one of:
- batch-processing (processing multiple files/items in a loop)
- web-research (fetching and analyzing multiple web sources)
- code-generation (writing code, scripts, or implementations)
- document-processing (extracting/converting documents like PDF, DOCX)
- data-transformation (converting data formats, CSV, JSON, etc.)
- orchestration (multi-step workflow coordination)
- interactive (back-and-forth clarification-heavy tasks)
- file-management (renaming, moving, organizing files)
- other (none of the above)

Return ONLY a JSON object (no markdown, no code fences):
{{
  "task_type": "one of the types above",
  "name": "short descriptive name (3-6 words)",
  "trigger_description": "when should this playbook activate — describe the task pattern in one sentence",
  "do_pattern": "pseudo-code of the recommended approach (what works)",
  "dont_pattern": "pseudo-code of what to avoid (what fails or is suboptimal)",
  "reasoning": "one sentence explaining why this pattern matters"
}}

Rules:
- The `do_pattern` and `dont_pattern` should be concise pseudo-code (5-15 lines each).
- Focus on the ORCHESTRATION decisions (tool ordering, looping strategy, context management), not the content of the task.
- If the session was rated "good", the `do_pattern` should reflect what actually happened. The `dont_pattern` should describe the obvious anti-pattern.
- If the session was rated "bad", the `dont_pattern` should reflect what actually happened. The `do_pattern` should describe what should have been done instead.
- Keep the `trigger_description` generic enough to match similar future tasks, but specific enough to not match unrelated ones.
- Never include specific file names, URLs, or user data in the patterns — keep them abstract.
