Extract valuable, reusable insights from the conversation context below. Output ONLY a JSON array (no markdown, no explanation).

Each insight object:
{"content": "1-2 sentence fact", "category": "...", "entity_key": "category:identifier or null", "importance": 1-10, "tags": "comma,separated"}

Categories: contact, decision, preference, fact, deadline, project, workflow

Rules:
- Only extract genuinely useful facts that would help in future conversations (not transient chatter or task progress)
- entity_key enables dedup: use format "category:normalized_id" (e.g. "contact:john@acme.com", "preference:timezone", "project:website-redesign"). Set null if no natural key exists.
- importance: 1=trivial, 5=useful, 8=important, 10=critical
- Skip anything already listed in "Known insights" — do NOT re-extract
- Output [] if nothing new worth extracting
- Max 5 insights per extraction
- For contacts: extract name, email, role, company when available
- For deadlines: include the date in content
- DO NOT extract peer-agent rosters, fleet membership lists, available-agent lists, or "peer agents include X, Y, Z" style statements — that information is volatile (changes per session/council) and is supplied to you separately each session via the system prompt's peer agents block. Storing it as a persistent fact creates stale, cross-session leakage.