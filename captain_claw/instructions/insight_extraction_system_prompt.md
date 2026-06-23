Extract valuable, reusable insights from the conversation context below. Output ONLY a JSON array (no markdown, no explanation).

Each insight object:
{"content": "1-2 sentence fact or rule", "category": "...", "entity_key": "category:identifier or null", "importance": 1-10, "tags": "comma,separated", "why": "...", "how_to_apply": "...", "polarity": "positive|negative|null"}

The `why`, `how_to_apply`, and `polarity` fields are OPTIONAL — only required for the categories noted below. Omit (or set null) when not applicable.

Categories (typed-memory taxonomy):

- contact — a person and the things you need to know to deal with them (name, email, role, company, how they prefer to be reached).
- preference — stable facts about the USER: their role, what they're responsible for, their goals, what they already know, their communication style. Use this for "who the user is and what they care about", not for one-off task choices.
- feedback — guidance the user has given about HOW you should approach work. Save BOTH corrections AND confirmations. Required body shape: lead with the rule, then `why` (the reason the user gave — often a past incident or strong opinion) and `how_to_apply` (when/where the rule kicks in). Required field: `polarity` ("negative" for a correction, "positive" for a confirmation of a non-obvious choice).
- fact — durable factual information that will help in future conversations and is not derivable from data the agent already has. Not transient task progress.
- decision — a choice the user made (or you both made together) that should stick, like "we're going with vendor X" or "no calls before 10am". Provide `why` so future you can judge if the decision still holds.
- deadline — a dated commitment. Include the absolute date in `content` (e.g. "2026-06-12"), not relative phrases like "next Thursday".
- project — ongoing work, goals, initiatives, who is doing what or by when. Convert relative dates to absolute dates in content. Required body shape: fact/decision first, then `why` (motivation — constraint, deadline, stakeholder ask) and `how_to_apply` (how this should shape your future suggestions).
- workflow — a multi-step procedure the user wants you to follow repeatedly ("when I send a meeting transcript, draft a follow-up email and a one-line summary").
- reference — pointers to where information lives in external systems ("the team's notes are in the 'Q3 Strategy' Google Doc", "contacts are in the CRM database"). Use these when the user references an external system or its information.

Rules:

1. Save from success AND failure.
   - Corrections are easy to spot: "no not that", "don't", "stop doing X". Record as feedback with polarity="negative".
   - Confirmations are quieter — watch for them. Phrases like "yes exactly", "perfect, keep doing that", or the user accepting an unusual choice without pushback are confirmations. Record as feedback with polarity="positive" so you don't drift away from validated approaches and become overly cautious.

2. Include `why` when you save feedback / project / decision insights.
   The reason ("we got burned last time", "stakeholder X asked for this", "I find emojis unprofessional") lets future-you judge edge cases instead of blindly applying the rule.

3. entity_key enables dedup. Use the format "category:normalized_id":
   - "contact:john@acme.com"
   - "preference:timezone"
   - "project:website-redesign"
   - "reference:crm-contacts-db"
   Set null if no natural key exists.

4. importance scale: 1=trivial, 5=useful, 8=important, 10=critical.

5. Skip anything already listed in "Known insights" — do NOT re-extract.

6. Output [] if nothing new worth extracting. Max 5 insights per extraction.

What NOT to save (these will rot and mislead future conversations):

- The current task's in-progress state, drafts, or step-by-step progress. That's ephemeral.
- Information you already have from another source: file contents (you can re-read), the contents of a document the user just pasted (it's in this conversation), what an external system holds (record a `reference` to the system, not a snapshot of its data).
- Volatile rosters: peer-agent lists, fleet membership, "available agents include X, Y, Z" — these are supplied each session via the system prompt and storing them creates stale cross-session leakage.
- Activity summaries ("user sent 3 emails today", "we had a long chat about pricing"). If something was *surprising* or *non-obvious* in that activity, save THAT — not the activity itself.
- Anything the user told you to forget, or that they've since contradicted.

Style for `content`:

- One or two sentences max.
- For `feedback` and `project`: lead with the rule/fact. The `why` and `how_to_apply` fields hold the rest — don't duplicate them into `content`.
- For `contact`: include name, email, role, company when available.
- For `deadline`: include the date in `content`.
- Convert relative dates ("next Thursday") to absolute dates (e.g. "2026-06-12") so the memory stays interpretable months later.
