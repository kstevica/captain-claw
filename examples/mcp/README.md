# Parallel Search MCP

Add optional web search and page excerpts to Flight Deck using its existing HTTP
MCP connection. The [Parallel Search MCP endpoint](https://docs.parallel.ai/integrations/mcp/search-mcp)
works without a Parallel API key. Anonymous requests use the free tier with lower
rate limits; model inference is separate.

## Add the connection

Start `flight-deck`, sign in, and open **Connections → MCP servers → Add server**.
Use the fields in [parallel-search.json](parallel-search.json):

- Name: `parallel-search`
- Transport: `http`
- URL: `https://search.parallel.ai/mcp`
- Headers: `{"User-Agent": "captain-claw-flight-deck/0.7.8"}`
- Enabled: on
- OAuth client ID, client secret and token endpoint: leave empty

Test the connection, then save. A new connection exposes `web_search` and
`web_fetch`. Its agent proxy names are `mcp_parallel-search_web_search` and
`mcp_parallel-search_web_fetch`. The empty `allowed_agents` list permits all
agents; select specific agent slugs to restrict access. Existing providers and
connections keep their settings. MCP is a Flight Deck feature, so a standalone
agent without `FD_URL` does not load these tools.

The JSON is a single record for the `POST /fd/mcp/servers` API, not a replacement
for the stored servers list. Use a new name if `parallel-search` already exists;
the API updates records by name. See [the MCP reference](../../USAGE.md#MCP-servers-centrally-managed)
for authenticated API usage and allowlists.

## Run a search and fetch without an LLM

From the repository root, install into a fresh environment:

```bash
uv venv .venv-parallel
uv pip install --python .venv-parallel/bin/python -e .
.venv-parallel/bin/python examples/mcp/parallel_search.py
```

The script loads the adjacent connection record through Flight Deck's MCP storage
and manager, discovers the tools, searches for the Python asyncio documentation,
then fetches excerpts from the first result. It prints both JSON results and exits
with an error if either call fails or returns no excerpts. It uses a temporary
servers file and closes its connection afterward, leaving saved Flight Deck
connections untouched. No model or Parallel credentials are used by the script.
This is a direct tool smoke test; it does not run an LLM conversation.

For tool calls in a conversation, generate one UUID `session_id` and reuse it
across related search and fetch calls. The script does this automatically. Consult
the discovered schemas for current arguments and limits; avoid repeated calls
when the free endpoint returns a rate limit.
