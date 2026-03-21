Find non-obvious connections across memory layers. Output ONLY a JSON array.
Each: {"content": "1-3 sentence pattern", "thread_type": "connection|pattern|hypothesis|association|unresolved", "confidence": 0.0-1.0, "importance": 1-10, "source_layers": ["..."], "tags": "..."}
For resolved tensions add: "resolves_tension_id": "id"
- Only genuine insights, not restatements. Skip existing intuitions. Output [] if nothing. Max 3.
- "unresolved" = contradiction or open question to hold, not resolve. Tensions are valuable.
- Refine maturing intuitions by adjusting confidence/importance.
