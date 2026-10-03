# Google subscription through Antigravity CLI

`antigravity-cli` uses Google's official local `agy` executable and its cached
Google sign-in. It is separate from `gemini` / `google`, which use the Gemini
API. Captain Claw does not read, import, distribute, or store the CLI's OAuth
tokens. There is no API fallback.

## Setup

1. Install [Antigravity CLI](https://antigravity.google/docs/cli/install/),
   version 1.2.1 or newer, on the machine running Captain Claw.
2. Run `agy` interactively under the same OS account and sign in with Google.
   Gemini CLI no longer supports individual subscription sign-in; see Google's
   [migration announcement](https://github.com/google-gemini/gemini-cli/discussions/28017).
3. In Antigravity settings, disable **Use G1 Credits**, or select **Never** for
   **AI Credit Overages**. Remove `modelProvider` from
   `~/.gemini/antigravity-cli/settings.json`; `modelProvider: "gemini"` selects
   API-key authentication. Save this file as UTF-8 without a BOM.
4. In Flight Deck, an admin can open **Connections → Google (Antigravity)**
   and check the connection and quota. The check calls `/usage` and `models`;
   it does not generate a model response or import credentials.
5. Choose `antigravity-cli` in Library model tiers or the local process Spawner.
   Use a Gemini model ID reported by `agy models`, for example
   `gemini-3.8-flash-low` on the version tested. Leave API key and Base URL empty.

```yaml
model:
  provider: antigravity-cli
  model: gemini-3.8-flash-low
```

If `agy` is outside PATH, set `ANTIGRAVITY_CLI_PATH` to its native executable.
On Windows the official install location in `%LOCALAPPDATA%/agy/bin/agy.exe`
is also checked. No shell is used; conversation data goes through UTF-8 stdin.

## Billing and limits

The adapter rejects API keys, endpoint overrides, custom auth headers,
`modelProvider`, and enabled `useG1Credits`. It removes API-key, Vertex,
ADC-project and alternate gateway variables from the child process.
The native CLI omits false/default settings when saving, so a missing
`useG1Credits` field is treated as the CLI's default `false`.

Google's [plan quotas](https://antigravity.google/docs/plans/) still apply.
Quota exhaustion or a missing login is an error; Captain Claw does not activate
extra credits or switch to the paid API. Keep overages disabled in Antigravity;
do not change its settings while requests are running. A successful quota
check establishes access, not the plan name or a Google One invoice audit.
Captain Claw's token-based cost estimates are not Google billing records.

## Scope of this adapter

- **Text only.** Captain Claw suppresses tool definitions for providers declaring
  `supports_tools = False`. Direct calls that pass tools return a clear error.
  Use another provider for workflows requiring agent tools.
- Each request uses a fresh temporary workspace and a custom Markdown profile
  with `excludeDefaultComponents: true`, empty tools, disabled inherited
  customizations/MCP, and command execution off. See the official CLI
  [1.2.1 release notes](https://github.com/google-antigravity/antigravity-cli/blob/main/CHANGELOG.md).
  This profile is not installed into the user's global agent configuration.
- History is supplied as a role-labelled JSON transcript for each new request;
  native CLI conversations are not resumed or shared between Captain Claw chats.
- Streaming/callback responses are buffered: the complete text arrives after
  the CLI finishes. Sampling and output length are controlled by Antigravity;
  Captain Claw's `temperature` and `max_tokens` cannot enforce CLI limits.
- Requests time out after 120 seconds by default. Cancellation terminates the
  child CLI and its language-server process tree; no retry through another billing route takes place.
- The host sign-in is shared by local processes under that OS account. It is
  not per-Flight-Deck-user authentication. Diagnostics are admin-only.
- Docker/remote agents need their own CLI, settings and sign-in. Captain Claw
  does not forward host credentials into containers or browsers.

## Validation

Unit tests mock the CLI; they cover billing guards, malformed/BOM settings,
factory aliases, API separation, UTF-8 transcripts, errors, cancellation,
timeouts, buffered callbacks, and admin-only quota diagnostics. They require no
Google account, network access or subscription quota.

```sh
pytest tests/test_llm/test_antigravity_provider.py tests/test_flight_deck_antigravity.py tests/test_llm/test_provider_factory.py tests/test_llm/test_claude_cli_provider.py
cd flight-deck && npm ci && npm run build
```

An opt-in live check should use a brief text prompt with overages disabled,
then inspect `/usage` and `/credits`. Never exhaust quota as a billing test.
Sign out in `agy` to stop using the cached account, or select another provider
in Captain Claw to disable this integration.
