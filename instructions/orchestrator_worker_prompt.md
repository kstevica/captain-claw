Task: {task_title}

{task_description}
{file_manifest}

MANDATORY tool rules — you MUST follow these:
- To READ or GET web page content (for summarizing, analyzing, extracting info): use the web_fetch tool directly. NEVER write scripts, use Playwright, use headless browsers, or run Python code to fetch web pages. The web_fetch tool returns clean readable text from any URL.
- To DOWNLOAD a binary file (PDF, image, archive) to disk: use `curl` via the shell tool.
- NEVER create intermediate web-fetching artifacts (raw HTML dumps, extracted.json, metadata.json, retrieval_meta.json) just to process web content. Use web_fetch, work with the returned text directly in memory, and produce only the output files your task description asks for.
- DO write output files (CSV, JSON, Markdown, etc.) when your task description says to produce them or when downstream tasks will need them. These are legitimate task outputs, not throwaway intermediates.
- Prefer direct tool calls over writing scripts. Use the minimum number of tool calls to complete the task.
- If upstream tasks produced files, read them with the read tool. They are listed in the file manifest above.

Execute this task completely. Provide a clear, complete result when done.