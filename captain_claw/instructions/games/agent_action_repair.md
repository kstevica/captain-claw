Based on your reasoning below, output the SINGLE best JSON action.

Your reasoning: {reasoning}

Available verbs: move, take, drop, say, talk, use, examine, give, look, wait
Arg formats: move={{"direction":"..."}}, take={{"entity_id":"..."}}, drop={{"entity_id":"..."}}, say={{"text":"..."}}, talk={{"target":"<char_id>","text":"..."}}, use={{"item_id":"...","target_id":"..."}}, examine={{"entity_id":"..."}}, give={{"entity_id":"...","target_id":"..."}}
Output ONLY the JSON: {{"verb": "...", "args": {{...}}}}
Do NOT choose 'wait' unless your reasoning explicitly says to do nothing.