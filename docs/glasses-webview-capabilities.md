# Meta Ray-Ban Display — third-party webview capabilities

Empirical reference compiled May 2026 from running an instrumented probe
page inside the Display's launcher webview. Goal: tell future-Stevica
exactly what works, what doesn't, and *why* — so we don't burn another
afternoon re-deriving it whenever Meta ships a firmware update.

## TL;DR

The Display loads third-party "apps" as URLs inside an **Android WebView**
hosted by Meta's launcher. The launcher decides at WebView-creation time
which native bridges to expose. Meta-owned origins get a bridge called
`MetaGlassSDK`; our apps don't, and there is **no in-page mechanism to
acquire it**.

| Surface | Status |
|---|---|
| HTML / CSS / JS rendering | ✅ Works (Chrome 146 / Android WebView 146) |
| WebSocket | ✅ Works |
| `fetch()` over HTTPS | ✅ Works |
| Intra-app `location.href` navigation | ✅ Works |
| `<input>`, `<textarea>`, contentEditable — focus | ✅ Works (cursor appears) |
| Click / pointer / touch events | ✅ Works |
| **Text input — system keyboard summoned by focus** | ❌ Blocked at WebView host |
| **`webkitSpeechRecognition`** | ❌ `service-not-allowed` / `"MetaGlassSDK dictation not available"` |
| **`getUserMedia({ audio })` / `({ video })`** | ❌ Blocked |
| **Neural Band handwriting → focused input** | ❌ Not delivered to our webview as text |
| **Scheme links (`tel:`/`sms:`/`mailto:`/`intent://`)** | ❌ Click event fires, system handler never launches |
| **Page-declared permission meta tags** | ❌ Parsed but ignored (9 variants tested) |
| `navigator.clipboard` (existence) | ✅ Constructor present; not exhaustively tested |

## Webview identity

```
User-Agent:
  Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv)
  AppleWebKit/537.36 (KHTML, like Gecko)
  Version/4.0 Chrome/146.0.7680.177 Safari/537.36

navigator.userAgentData.brands:
  [{ brand: "Chromium",          version: "146" },
   { brand: "Not-A.Brand",       version: "24"  },
   { brand: "Android WebView",   version: "146" }]
navigator.userAgentData.platform: "Android"
navigator.userAgentData.mobile:   false
```

Key reads:
- `"; wv)"` in the UA and `"Android WebView"` in `userAgentData` confirm we
  are inside `android.webkit.WebView`, **not** Chrome proper. Capability
  decisions are taken by the **hosting Android app** (Meta's launcher),
  not by Chromium itself.
- `Greatwhite` is the Display's internal codename.
- `mobile: false` is amusing — Meta presents the glasses as a non-mobile
  Android device. Don't rely on `mobile`-flag responsive logic.

## The gating model (with high confidence)

Android `WebView` exposes a `WebView.addJavascriptInterface(obj, name)` API
that lets the **native host app** bind a Java/Kotlin object onto the
loaded page as a global JS object. The host app decides which interfaces
to bind for which loaded URL. **All capability gating happens here, at
WebView instantiation time, in the launcher's native code.**

Concretely:

- A WebView loading `https://web.whatsapp.com/...` gets bound a `MetaGlassSDK`
  native interface (and probably several others). WhatsApp's JS can then
  call `MetaGlassSDK.startDictation()` and equivalent.
- A WebView loading our tunnel URL gets **no `MetaGlassSDK` binding**.
  The error message `"MetaGlassSDK dictation not available"` is emitted by
  whatever shim Meta installed for `webkitSpeechRecognition` — that shim
  checks whether the native bridge is present and refuses if not.

**Implications:**
- Page-level declarations (meta tags, manifest fields, HTTP headers) cannot
  influence the decision — by the time our page is parsed, the WebView is
  already instantiated with whatever interfaces the launcher chose.
- The decision key is the **URL/origin** that the launcher follows when
  the user taps our app's icon. Meta-owned origins get the bridges; ours
  doesn't. There is no documented or undocumented way for a third party
  to opt into a Meta-owned origin without being Meta.
- An ordinary user adding an app via the mobile Meta-AI companion app is
  effectively just registering the URL; the *capabilities* of the resulting
  webview are decided by hardcoded origin checks in the launcher.

This is consistent with Meta's playbook everywhere else they ship (Quest,
Instagram Effects, Facebook apps): server-side review + identity-based
gating.

## The one Meta-injected global we found

A scan of `window`, `navigator`, and `document` for property names
matching `/meta|glass|ray|sdk|fb|messenger|whatsapp|cortex|wearable/i`
returned exactly one Meta-specific property:

```
window.__fbAndroidBridgeAuthToken :: string
```

Read but **not poked**. The name implies it's an opaque auth token for an
"FB Android Bridge" — presumably consumed when Meta-owned pages call
native bridge methods. We have the token but no native interface to spend
it against (no `fb`/`bridge` namespace object exists).

Forensically this is the strongest single piece of evidence for the gating
model:

- Meta is uniformly injecting the token into **every** loaded page (ours
  included), which means the launcher's WebView setup is the same code
  path for everyone.
- The token is per-page (different on every navigation).
- The differentiator between Meta apps and ours is **whether the native
  interfaces that accept the token are bound** — not the token itself.

Do **not** use this token for anything. It's an internal credential and
poking at Meta's server-side endpoints with it would almost certainly
violate TOS.

## What does **not** work (and why each was tested)

### Text input fields
`<input>`, `<textarea>`, and `contentEditable` divs all take focus when
tapped — `focus`/`blur` events fire and a visual cursor appears. **No
system keyboard is summoned.** The IME bridge from Android WebView to the
OS-level input method is severed by the launcher. This is policy, not a
missing feature: Android WebView normally surfaces the keyboard
automatically on focused input.

### Web Speech API (`webkitSpeechRecognition`)
The constructor exists. Calling `.start()` reliably produces:

```
onerror.error   = "service-not-allowed"
onerror.message = "MetaGlassSDK dictation not available"
```

`"service-not-allowed"` is the W3C-standard error meaning "the user agent
refuses to provide this service" — i.e. the *user agent's* decision, not
a missing user permission. The `message` is custom Meta wording.

Tested with `en-GB`; behaviour identical across languages.

### `getUserMedia`
`navigator.mediaDevices` exists as a property but a call that actually
asks for mic/camera will be refused at the platform layer. Not separately
tested in this round (the speech API failure is the more granular signal).

### Scheme launches
`tel:+10000000000`, `sms:+10000000000?body=hi`, `mailto:hi@example.com`,
and a synthesized `intent://...;scheme=https;package=com.android.chrome;end`
were placed as `<a>` tags. **Every click event fired in JS, but no native
handler launched** — no dialer, no SMS composer, no mail client, no Chrome
intent dispatch. The launcher's WebView is configured with a permissive
`shouldOverrideUrlLoading` (or equivalent) that silently swallows non-http
schemes.

### Permission meta tags (the load-bearing falsification)
Nine candidate `<meta>` shapes were injected server-side via
`?perms=1`:

```html
<meta name="meta-glasses-permissions"  content="dictation microphone camera">
<meta name="meta-glass-permissions"    content="dictation microphone camera">
<meta name="meta-permissions"          content="dictation microphone camera">
<meta name="MetaGlassSDK-permissions"  content="dictation microphone camera">
<meta name="x-meta-glasses-permissions" content="dictation microphone camera">
<meta name="permissions"               content="dictation microphone camera">
<meta name="capabilities"              content="dictation microphone camera">
<meta http-equiv="Permissions-Policy"  content="microphone=*, camera=*, speaker-selection=*">
<meta http-equiv="Feature-Policy"      content="microphone *; camera *">
```

Baseline page transferSize: **29331 bytes**.
Declared page transferSize: **30013 bytes** (+682, confirming delivery).

`webkitSpeechRecognition.start()` produced the **identical error string**
in both runs. No new globals appeared. No behavioural change of any kind.

**Conclusion: the WebView host does not consult any page-declared
permission convention.** This kills the "we just need the right meta tag"
hypothesis cleanly.

## A genuinely interesting adjacent finding

The declared-perms run logged a `keydown` with `key="Unidentified"` on
each focused input — events that did **not** appear in the baseline run.
Almost certainly the Neural Band firing a gesture into the focused input
field. Android maps unknown hardware keycodes to `"Unidentified"`.

So while Neural-Band **glyphs** (handwriting letters) are not delivered to
third-party webviews today, Neural-Band **gesture events** *do* propagate
to focused webview inputs as `keydown` events. If Meta ever maps the
glyph stream onto standard `KeyA`/`KeyB`/… `keydown` events for
third-party apps, the existing input fields would Just Work without any
code change on our side. Worth watching for in future firmware notes.

## Things that **do** work and that we rely on

- **WebSocket** (used by `glasses_view.html` to subscribe to the channel
  bus) — fully functional, including reconnect behaviour.
- **`fetch()` over HTTPS** — used by every probe action, no surprises.
- **Intra-app navigation** (`location.href` to another route on the same
  origin) — `/glasses/view` → `/glasses/input` works, including back-link
  with preserved query params.
- **Click / tap / pointer events** — fire reliably with usable
  coordinates. Long-press and drag not exhaustively tested.
- **Render quality** — opacity, gradients, and small fonts render
  cleanly. The Display projector is dim, so high-contrast palettes (HUD
  green / cyan on near-black) work much better than subtle greys.

## How to re-run the experiment

The probe page is intentionally permanent so it stays useful as a canary
when Meta ships firmware updates.

1. From your phone bridge (or any text-entry surface), send one message
   to the channel — this establishes the agent binding so the probe page
   doesn't need a port.
2. On the glasses, open `/glasses/view?c=<channel>`. Tap the `✎` icon
   in the HUD header — that navigates to `/glasses/input?c=<channel>`.
3. Run the six numbered sections top to bottom. Each has a status line
   that confirms what happened.
4. Tap **Scan globals** in Test 7 — fills the globals-log block with the
   live `window` / `navigator` / `document` Meta-property dump.
5. Tap **Save to FD /tmp** in the debug-log toolbar. The full page log +
   the globals dump is POSTed to `/glasses/input-log` and written to
   `/tmp/glasses-input-<UTC-ts>-ch_<channel>.log` on the FD box.
6. Tap **Reload with perm tags** and repeat steps 3–5 to produce a
   declared-mode comparison file.
7. On the dev box, `diff` the two files. Anything different between
   baseline and declared is news.

## What to watch for in future updates

- **Any change in the `MetaGlassSDK dictation not available` error
  message.** Wording change, error-code change, or appearance of an
  `onresult` event would be the first signal.
- **New globals appearing in the Test-7 scan.** Particularly anything
  matching `/meta|glass|ray|sdk|fb/i`. If a `MetaGlassSDK` namespace
  appears alongside the existing `__fbAndroidBridgeAuthToken`, the gate
  has opened.
- **Neural-Band handwriting glyphs arriving as `KeyA`/`KeyB`-style
  `keydown` events** instead of `Unidentified`. That would mean Meta has
  opened the glyph stream to third-party webviews.
- **Different gating per origin** — e.g. if you ever get a Meta-issued
  app slot with a vanity origin (e.g. `*.metawearapps.com`), capabilities
  may differ from a tunnel URL.

## References

- [Meta — Wearables Device Access Toolkit](https://wearables.developer.meta.com/docs/develop/dat) (overview only; capability surfaces gated behind program access)
- [Meta — CES 2026 announcement: Display teleprompter & EMG handwriting](https://www.meta.com/blog/ces-2026-meta-ray-ban-display-teleprompter-emg-handwriting-garmin-unified-cabin-university-of-utah-tetraski/)
- [Meta — Neural handwriting help page](https://www.meta.com/help/ai-glasses/866944989643926/)
- [Meta — Live captions help page](https://www.meta.com/help/ai-glasses/23879220601763496/)
- [Android Developers — `WebView.addJavascriptInterface`](https://developer.android.com/reference/android/webkit/WebView#addJavascriptInterface\(java.lang.Object,%20java.lang.String\))
- W3C Web Speech API — [`SpeechRecognitionErrorEvent.error`](https://wicg.github.io/speech-api/#enumdef-speechrecognitionerrorcode) (`service-not-allowed`)
- Captain Claw probe page: [`captain_claw/flight_deck/static/glasses_input.html`](../captain_claw/flight_deck/static/glasses_input.html)
- Captain Claw probe routes: [`captain_claw/flight_deck/glasses_bridge.py`](../captain_claw/flight_deck/glasses_bridge.py) — search for `glasses_input_page` / `glasses_input_log` / `_PERM_META_CANDIDATES`
