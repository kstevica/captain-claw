# Publishing pages with deed.page

[deed.page](https://deed.page) turns HTML or static files that an agent writes
into a public `*.deed.page` URL. Its HTTP MCP endpoint needs no API key or
account. This optional example uses Flight Deck's existing MCP manager; it does
not change built-in tools, saved connections, or model settings.

From this repository root, install into a fresh environment and run the smoke
test:

```bash
uv venv .venv-deed-page
uv pip install --python .venv-deed-page/bin/python -e .
.venv-deed-page/bin/python examples/mcp/deed_page.py
```

The script loads [deed-page.json](deed-page.json) into a temporary servers file,
discovers the tools, publishes a sample page that expires after ten minutes,
fetches the public URL, then deletes the page with the owner token returned by
the publish call. Pass `--file report.html` to publish your own HTML, and
`--keep` to keep it with no expiry; the script then prints the token you need to
update or delete it later. No model is used; this is a direct tool smoke test.

To use it in conversations, open **Connections → MCP servers → Add server**
with the fields from [deed-page.json](deed-page.json):

- Name: `deed-page` (choose a new name if it already exists)
- Transport: `http`
- URL: `https://deed.page/mcp`
- Headers: `{"User-Agent": "captain-claw-flight-deck/0.8.3"}`
- Allowed agents: select the agents that should be able to publish

Test and save the connection. The proxy tools include
`mcp_deed-page_publish_site` and `mcp_deed-page_delete_site`. The first publish
returns an owner token; agents should keep it, because updates and deletion
need it. Set `ttl_seconds` for pages that should expire.

Published pages are public to anyone with the URL. Restrict the connection to
agents that should share output, and do not publish secrets or private files.
An empty agent allowlist permits every agent. Limits and the full tool list are
in the [deed.page documentation](https://deed.page/docs).
