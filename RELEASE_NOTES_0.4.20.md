# Captain Claw v0.4.20 Release Notes

**Release title:** Sign in with ChatGPT, Scoped Google OAuth, and On-Device Gemma

**Release date:** 2026-04-08

## Highlights

Captain Claw 0.4.20 rewires the auth story for the two biggest model providers and adds a new on-device runtime. You can now sign into OpenAI through **"Sign in with ChatGPT"** (reusing the Codex CLI's OAuth tokens — no API key needed, uses your ChatGPT Plus/Pro/Team plan). Google OAuth has been **tightened for security and unblocked for unverified apps**: the bundled client secret is gone, users supply their own Cloud OAuth client, and scopes are now picked individually with sensitivity badges so a fresh install defaults to non-sensitive scopes only. Captain Claw also ships a production-grade **LiteRT subprocess worker** so on-device Gemma models (`.litertlm`) run in an isolated child process that can crash and respawn without losing your WebSocket session.

## Sign in with ChatGPT (OpenAI OAuth / Codex)

Captain Claw can now authenticate to OpenAI using the same "Sign in with ChatGPT" flow the Codex CLI uses, reusing the tokens the Codex CLI caches in `~/.codex/auth.json`. No `OPENAI_API_KEY` required — the agent talks directly to `chatgpt.com/backend-api/codex/responses` on your ChatGPT plan.

**New: `ChatGPTResponsesProvider`**

- Direct connection to the ChatGPT Responses API (SSE streaming, `input` items instead of `messages`, per-request `session_id`).
- Speaks the full Responses API protocol: function calling via `response.output_item.added` / `response.function_call_arguments.delta` / `.done` events, reconstructed transparently into the agent's tool-call pipeline.
- Shape-tolerant output parser — walks any output node shape to recover text from `response.output_text.delta`, `output_text`, `text.value`, or nested `content` arrays. The Codex backend's `response.completed` shape differs slightly from the strict Responses API spec, and the parser now handles both.
- Empty-turn detection raises a clear `LLMAPIError` instead of silently finalizing, so a rejected request can never look like a successful but blank response.

**New: `CodexAuthManager`**

Single source of truth for resolving Codex OAuth tokens inside a running captain-claw process:

- **Flight Deck mode** — when `FD_URL` is set, pulls tokens from `GET /fd/codex/access_token`, which re-reads `~/.codex/auth.json` on demand. Every sub-agent spawned by Flight Deck shares one centrally-managed connection, even across hosts.
- **Local mode** — reads `~/.codex/auth.json` directly for standalone installs.
- **Auto-refresh** — decodes the JWT `exp` claim with a 60-second safety margin. When a token is stale or a 401 comes back mid-request, `_ensure_fresh_auth(force=True)` re-reads the file and retries exactly once. Transparent recovery in both `complete()` and `complete_streaming()`.

**New: Flight Deck Connections card**

- **GET `/fd/codex/status`** — returns `configured`, `connected`, `email`, `plan`, `account_id`, `expires_at`, `seconds_until_expiry`, `stale`, `access_token_preview`.
- **POST `/fd/codex/reimport`** — forces a re-read of `~/.codex/auth.json` (useful after running `codex login` again).
- **GET `/fd/codex/access_token`** — agent-facing endpoint, gated by loopback or `X-Agent-Secret` header (same trust model as Google).
- UI shows a **ChatGPT (Codex)** card next to the Google card: email, plan, expiry countdown, token preview, and a **"Reimport from Codex CLI"** button. Red "Not configured" badge when `~/.codex/auth.json` is missing, amber "Token stale" when expired.

**Model aliasing**

The Codex backend only serves a specific set of Codex-family models (not the full OpenAI API catalog). Captain Claw now:

- Maintains `_CODEX_BACKEND_SUPPORTED_MODELS` with the known-good list (`gpt-5`, `gpt-5-codex`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.3-codex`, `codex-mini-latest`).
- Automatically remaps common non-codex aliases — `gpt-5-mini`, `gpt-5.1-mini`, `gpt-5.2-mini`, `gpt-5.3-mini`, `gpt-5.4-mini` — to `gpt-5.1-codex-mini`, logging the remap.
- Warns at provider construction when the normalized model isn't in the supported set so you don't silently hit empty-response territory.

**Provider auto-selection**

Setting `provider: openai` in config now auto-activates the ChatGPT path when the requested model is a Codex-family model *or* when explicit `extra_headers` are supplied. All other OpenAI models continue to go through the standard LiteLLM path, so existing `api.openai.com/v1` setups are untouched.

**Task naming skip**

The cosmetic background task-naming LLM call used to fail loudly on the ChatGPT/Codex path because LiteLLM has no mapping for `chatgpt.com/backend-api/codex/responses`. Task naming is now skipped for the `ChatGPTResponsesProvider` class with a single info-level log line, matching the existing skip for the `litert/` provider.

## Google OAuth Tightening

Captain Claw no longer ships with a bundled Google OAuth client. This is a **security hardening** change: previously the client secret lived in the repo, meaning every user shared the same OAuth app, and a secret scanner on GitHub would flag the commit. Now each deployment must create its own OAuth 2.0 Client ID in Google Cloud Console and paste the Client ID + Client Secret into the Connections page.

**What changed**

- `_BUNDLED_CLIENT_ID` and `_BUNDLED_CLIENT_SECRET` removed from `flight_deck/google_oauth_routes.py`.
- `GoogleAuthMode` is now only `'custom'` — no fallback.
- `/fd/google/status` returns `configured: false` until the user saves their credentials.
- `/fd/google/login` responds with a 400 HTML page instead of silently using the bundled client.
- `POST /fd/google/config` accepts `{clear: true}` to wipe saved credentials, tokens, and scopes in one shot.
- UI: credentials form is always visible, with a **"Remove credentials"** button (red) and a "Not configured" red pill until credentials are saved.

**Per-scope picker with sensitivity tiers**

Google blocks unverified apps from requesting sensitive/restricted scopes, and the "App is blocked" screen users were hitting is caused precisely because the default scope list requested `cloud-platform`, `drive`, `calendar`, `gmail.readonly`, and `gmail.compose` — all sensitive or restricted. Captain Claw now:

- Defaults to **non-sensitive scopes only** — `openid`, `email`, `profile`, `drive.file`. A fresh install can connect an unverified Google OAuth client out of the box without any "App is blocked" screen.
- Ships a **scope catalog** of 13 selectable scopes (`SCOPE_CATALOG`), each tagged with a sensitivity tier:
  - `none` (green badge) — freely usable by unverified apps.
  - `sensitive` (amber badge) — Google flags these; unverified apps need to add the user as a Test user on the consent screen.
  - `restricted` (red badge) — requires security review + verification in production.
- New **scope picker** in the Connections page: checkboxes grouped by area (identity / Drive / Gmail / Calendar / Cloud), with label, sensitivity badge, and human description for each scope.
- Selected scopes are persisted in `system_settings` and sent through `build_authorization_url`.
- `sanitize_scopes()` filters unknown entries and always forces `openid` + `email` so `/userinfo` works.
- Changing the scope set **invalidates stored tokens** (same as rotating the client_id) — OAuth refresh tokens are bound to their original scope grant, and Google requires fresh consent for any scope change.
- `GET /fd/google/config` returns `scopes`, `default_scopes`, and `scope_catalog` so the UI can render everything from one response.
- New `GET /fd/google/scope_catalog` endpoint exposes the catalog separately.

**Vertex AI reporting**

`supports_vertex` is now only reported as `true` when the user has actually ticked the `cloud-platform` scope **and** set a `project_id`, so the UI doesn't promise Vertex access on a connection that can't deliver it.

## LiteRT / On-Device Gemma Support

Captain Claw 0.4.20 ships a production-grade runtime for **litert-lm** models — Google's on-device `.litertlm` format for running Gemma (and other models) locally without a network round-trip. This is a new local-first path alongside Ollama.

**Why a subprocess worker?**

litert-lm's C++ engine has two failure modes that take down the host process when run inline:

1. **KV-cache overflow** — if a long conversation exceeds the model file's baked-in max-seq-length, the C++ side calls `LOG(FATAL)` or hangs forever holding the GIL. The parent agent loses its WebSocket clients and looks "disconnected" to Flight Deck even though the process is still bound to its port.
2. **GPU context exhaustion** — multiple `Engine` objects in the same process fight over the Metal / Vulkan device.

Running the engine inline made captain-claw crash-prone. 0.4.20 isolates both failure modes in a dedicated child process.

**New: `captain_claw/llm/litert_worker.py`**

- **`worker_main`** — child-process entry point. Owns the `litert_lm.Engine` and answers `send_message` requests over a pair of multiprocessing queues. Uses stdlib logging only, so the child doesn't pull captain-claw config initialization.
- **`LiteRTWorkerClient`** — parent-side RPC client. Wraps the child process, enforces a wall-clock timeout on every call, detects crashes (timeout / queue-EOF / child died), and transparently respawns a fresh child on the next call.
- **`get_or_create_litert_worker`** — process-wide registry keyed by `(abs_model_path, backend, max_num_tokens)`. Multiple `LiteRTProvider` instances in the same parent share one child process so the model is mmap'd once and the GPU context is grabbed once.

**Gemma tool calling**

Gemma local models have no structured function-calling bridge, so captain-claw ships a lightweight text-based tool protocol:

- **`_litert_build_tool_manifest`** — turns the tool registry into a plain-text manifest that's injected into the system prompt.
- **`_litert_parse_gemma_args`** — parses Gemma's JSON-ish argument emission.
- **`_litert_parse_pyish_args`** — fallback parser for when Gemma emits Python-style keyword arguments (`tool_name(arg1="foo", arg2=42)`).

Result: Gemma local models now support the full captain-claw tool ecosystem end-to-end.

**Provider normalization**

`litert`, `litert-lm`, `litertlm`, and `gemma-local` all normalize to the same `litert` provider key — you can use whichever name makes sense in your config.

## Gmail Reply Threading (follow-up)

`google_mail.create_draft` gained a `reply_to_message_id` parameter back in an earlier 0.4.19 build. 0.4.20 polishes the reply flow:

- When `reply_to_message_id` is set, the tool fetches the original message's headers (`Message-ID`, `References`, `From`, `To`, `Cc`, `Subject`), builds a proper RFC 5322 reply with `In-Reply-To` + `References`, and posts the draft with `threadId` so Gmail nests it under the original thread.
- Drafts now use `EmailMessage(policy=email.policy.SMTP)` from construction **and** always attach a multipart/alternative body (text/plain + auto-generated HTML via `_text_to_html`). Gmail's web composer is HTML-first for threaded reply drafts and drops text-only bodies, which caused earlier drafts to appear empty.
- Tool description now has a `ROUTING:` prologue and the list/search actions append a `_FOLLOWUP_HINT` showing exact follow-up recipes — this prevents agents from reaching for filesystem tools when a user says "read the one from X".

## Flight Deck Connections Page

The Connections page got a design pass to accommodate the new ChatGPT card and the scope picker:

- Page wrapper is now `overflow-y-auto` with bottom padding — the whole page scrolls independently when content overflows (the scope picker alone adds ~500px).
- Both **Google** and **ChatGPT (Codex)** cards are **collapsible**. Click the header row to toggle; a chevron rotates 90° to indicate state. The bottom border on the header disappears when collapsed so the card becomes a tight pill.
- Status pills use `/15` opacity with bordered backgrounds so they look right in both light and dark themes.

## Minor Improvements

- `ChatGPT Responses API stream parsed` info log prints raw-line count, event count, and sorted event-type set — useful diagnostics when debugging a new Codex backend behavior.
- `ChatGPT Codex: remapped model alias` and `ChatGPT Codex: model not in known-supported set` warnings at provider construction.
- `_is_codex_family_model()` helper correctly identifies GPT-5 / Codex family names so the provider factory activates the ChatGPT path when appropriate.
- `FD_URL` / `FD_AGENT_SHARED_SECRET` environment variables now drive both the Google **and** the Codex token-fetching paths identically.
- Cleaned up google OAuth history: the previously-bundled secret was squashed out of local git history before being pushed to GitHub's secret scanner.

## Breaking Changes

- **Google OAuth requires user-supplied credentials.** Any existing deployment that relied on the bundled OAuth client must now create their own Google Cloud OAuth 2.0 Client ID and enter it on the Connections page. The `mode: "bundled"` option has been removed; passing `mode` in `POST /fd/google/config` is accepted for backwards compatibility but ignored. The Connect Google button stays disabled until credentials are saved.
- **Default Google scopes have shrunk** to the non-sensitive set (`openid`, `email`, `profile`, `drive.file`). Previously-saved deployments will keep their stored scope selection, but fresh installs no longer request Gmail / Drive / Calendar / Vertex by default — tick them in the scope picker and re-consent.
- **`DEFAULT_SCOPES` in `captain_claw/google_oauth.py` changed.** Code that imports it will now get the non-sensitive list. Use the `SCOPE_CATALOG` to enumerate all available scopes.
- **`ChatGPTResponsesProvider` is now selected automatically** when `provider: openai` is combined with a Codex-family model name or with explicit `extra_headers`. If you previously relied on getting the LiteLLM path for a Codex-family name, either pick a different model or pass a custom provider configuration.

## Upgrade Notes

1. **Update your Google OAuth credentials.** If you were using the (now-removed) bundled client, create an OAuth 2.0 Client ID in [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (type: Web application), add `http://localhost:25080/fd/google/callback` as the redirect URI, and paste the Client ID + Client Secret in **Flight Deck → Connections → Google**.
2. **If you want Gmail / Drive / Calendar access**, tick those scopes in the picker and **add your Google account as a Test user** on the OAuth consent screen (Google Cloud Console → APIs & Services → OAuth consent screen → Test users). Otherwise Google will block the connection with "App is blocked" for sensitive scopes until you verify the app.
3. **To use Sign in with ChatGPT**, install the Codex CLI and run `codex login` on the same host that runs Flight Deck. Then open **Flight Deck → Connections → ChatGPT (Codex)** and click **Reimport from Codex CLI**. Select a Codex-family model like `gpt-5-codex` or `gpt-5.1-codex-mini`.
4. **To use on-device Gemma**, download a `.litertlm` model file and configure your session with `provider: litert`, `model: /absolute/path/to/model.litertlm`.
