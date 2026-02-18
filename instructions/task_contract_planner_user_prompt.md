Create a task contract for this user request.

User request:
{user_input}

Recently observed source URLs from current context (optional):
{recent_source_urls}

Must cover all listed sources/links in this turn:
{require_all_sources}

Extracted list members that likely require per-member execution (optional):
{extracted_list_members}

If `Must cover all listed sources/links in this turn` is `true`, include all listed URLs in `prefetch_urls`,
and create requirements that enforce coverage of every listed URL.

If `Extracted list members ...` is not `(none)`, include requirements that enforce coverage of each listed member.

Plan for complete execution and verification quality. Use nested `children` tasks when the request has phases/sub-steps.
Return JSON only.
