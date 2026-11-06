Extract reusable insights from conversation. Output ONLY a JSON array.
Each: {"content": "...", "category": "contact|decision|preference|fact|deadline|project|workflow", "entity_key": "category:id or null", "importance": 1-10, "tags": "..."}
- Only useful cross-session facts, not transient chatter
- Skip known insights. Output [] if nothing new. Max 5.
- Only what the chat shows; never guess the user's traits. Never save that a tool doesn't work, setup errors, or attempts that failed.