# Meta Ray-Ban Display — third-party webview capabilities

> **This document has two parts.**
> The [October 2026 update](#october-2026-update) below is the current picture:
> firmware v127 and v129, Meta's published web-app docs, and Meta's toolkits.
> The [May 2026 probe record](#may-2026-probe-record-firmware-v125) further down
> is the original probe write-up. It is kept as the historical record, and several
> of its "blocked" findings no longer hold.
>
> For the glasses frontend built on these findings, see [glasses-hud.md](glasses-hud.md).

## October 2026 update

*Re-checked on 2026-10-10 against Meta's published web-app docs, Meta's
GitHub toolkits and community on-device reports. We have not re-run our own
probe on v129 yet; see the [re-probe checklist](#re-probe-checklist-firmware-v129).*

**How this was sourced.** Meta's doc pages (`wearables.developer.meta.com`,
`developers.meta.com`, `meta.com`) could not be fetched from our research
sandbox. Their content here comes from search-engine extracts of those pages,
so the wording may be paraphrased. Meta's public GitHub repos were cloned and
read directly:
[facebook/meta-wearables-webapp](https://github.com/facebook/meta-wearables-webapp)
and
[facebook/meta-ray-ban-display-ui-toolkit-web](https://github.com/facebook/meta-ray-ban-display-ui-toolkit-web).

Each claim carries one of these confidence labels:

| Label | Meaning |
|---|---|
| **official** | Read verbatim in a Meta-owned repo or doc |
| **official (extract)** | On an official Meta page, seen only through a search extract |
| **community** | Third-party repo or article; on-device claims we have not reproduced |
| **inferred** | Our own reasoning; no source states it |
| **our probe** | Our May 2026 measurements (see the record below) |

### What changed

Since firmware **v127** (with Meta AI app **v272+**), a standard text field opens
Meta's system **composer** when the wearer focuses it **and pinches** it. The
composer offers handwriting and voice dictation, and since **v129** an on-screen
keyboard too. That reverses May's "no text input" finding. Meta also made other
changes:

- It now requires `<meta name="mrbd-web-app-capable" content="yes">`. None of
  the nine tags we tried in May was this one.
- It documents the Back gesture as `history.back()`, and documents
  `speechSynthesis`.
- It added an optional voice-control path, WebMCP (`document.modelContext`).

Meta's docs still say web apps get no microphone, no camera and no
`SpeechRecognition`.

May's per-origin [gating model](#the-gating-model-with-high-confidence) still
explains why our pages get no `MetaGlassSDK` bridge. The composer and WebMCP
work differently: firmware version and device settings gate them, not the
page's origin (inferred).

### Timeline

| Date | Change | Source |
|---|---|---|
| 2026-05-14 | Web Apps developer preview launches. It needs firmware **v125** and Meta AI app **v272**, and offers motion and orientation, GPS from the phone, Neural Band and touchpad input, and local storage. Our May probe ran in this window. | official (extract): [Meta blog](https://developers.meta.com/blog/build-for-display-glasses/) |
| 2026-06-05 | Meta adds `mrbd-web-app-capable` to every template ("required to identify MRBD compatible webapp"). Commit `de9ebda`. | official: [meta-wearables-webapp](https://github.com/facebook/meta-wearables-webapp) |
| 2026-07-27 | Firmware **v127** consumer release (Muse Spark, Threads). Web-app text input "requires glasses firmware v127+". | official (extract): [v127 post](https://www.meta.com/blog/meta-ray-ban-display-glasses-v127-muse-spark-threads/), [Build page](https://wearables.developer.meta.com/docs/develop/webapps/build/) |
| 2026-08-06 | Toolkit plugin 127.0.0 adds the `add-text-input` (composer), `add-gestures` and `add-offline` skills. On-screen Back buttons are dropped in favour of the Back gesture. Commit `24d7bfc`. | official |
| 2026-09-11 | Docs corrected from 60 fps to **30 Hz** ("the panel is hard-locked at 30Hz"). Commit `ca5fb95`. | official |
| 2026-09-23/24 | Connect 2026: web-app text can come from "dictation, handwriting, or an on-screen keyboard that all feed one text composer". Rollout from 2026-09-30. | official (extract): [Connect recap](https://developers.meta.com/blog/meta-connect-recap-ai-glasses/) |
| 2026-09-24 | The UI Toolkit (`@wearables-ui-toolkit/mrbd` 129.0.0) goes public, with `InputTextView`, a focus engine and Back handling. | official: [toolkit repo](https://github.com/facebook/meta-ray-ban-display-ui-toolkit-web) |
| 2026-10-01 | WebMCP skill: Meta AI can call tools the page registers on `document.modelContext`. It is "off by default and enabled per device". Commit `e617d92`. | official |
| v129 | The on-screen keyboard inside the composer "needs firmware v129+". We found no dated Display v129 release note, and v129 was still rolling out gradually in early October. | official (extract): Build page; community: [Android Authority](https://www.androidauthority.com/meta-glasses-navigation-audio-update-3719444/) |

### Status table (October 2026)

| Surface | May 2026 (v125, our probe) | October 2026 | Confidence |
|---|---|---|---|
| HTML / CSS / JS (Chrome 146 WebView), WebSocket, `fetch()` over HTTPS | ✅ | ✅ | our probe |
| Text entry into `<input>` / `<textarea>` | ❌ focus only, no keyboard | ✅ system composer opens on **focus + pinch** (fw v127+, Meta AI app v272+) | official |
| On-screen keyboard | ❌ | ✅ inside the composer, raised by swiping down (fw v129+); limited symbols | official (extract) |
| `<input type="password">` | ❌ | ❌ never opens the composer | official |
| `contentEditable` | ❌ | ⚠️ listed as eligible only in the v127 toolkit skill; verify | official (v127 skill); untested |
| Neural Band handwriting into a field | ❌ not delivered | ✅ through the composer (needs the Neural Band) | official (extract) |
| `webkitSpeechRecognition` | ❌ `service-not-allowed` | ❌ assumed unchanged: no doc mentions it, and Meta says "no speech recognition" | inferred |
| `getUserMedia` microphone / camera | ❌ (not separately tested) | ❌ officially "not yet supported"; community reports conflict | official (extract) + community |
| `speechSynthesis` (text-to-speech) | not tested | ✅ one en-US voice, through the glasses speaker | official (extract) |
| Back gesture | not characterised | ✅ the shell calls `history.back()`, giving `popstate`; system menu at the root | official (extract) |
| D-pad / pinch input | `keydown key="Unidentified"` on focused inputs | ✅ `ArrowUp/Down/Left/Right` + `Enter`; `Unidentified` is still ambiguous | official |
| Required meta tag | not tested (not among our 9) | `mrbd-web-app-capable` is required | official |
| Page-declared permission tags | ❌ ignored | ❌ still no permission-tag mechanism | our probe; inferred |
| Voice control by Meta AI | ❌ | ⚠️ WebMCP (`document.modelContext`): off by default, enabled per device | official |
| `MetaGlassSDK` bridge | ❌ not bound | ❌ still not documented for third parties | official (absence) |
| Scheme links (`tel:` / `sms:` / `mailto:` / `intent:`) | ❌ swallowed | not re-tested | our probe |
| Service Worker + Cache API | not tested | ✅ documented (HTTPS only) | official (extract) |
| `localStorage` / `sessionStorage` | not tested | ✅ documented; cookies and IndexedDB are **not** documented | official (extract) |
| Motion / orientation; geolocation (from the phone) | not tested | ✅ documented; call `requestPermission()` from a user gesture | official (extract) |

### Text entry: the on-glasses composer

- **How it opens.** The wearer focuses a standard text field and **pinches** it.
  - "Programmatic `.focus()` does not open composer." Meta's device skill says the same: "programmatic focus does not open the composer". So `autofocus` does not help either.
  - Activating a focused text field opens the composer "instead of dispatching a click to your page".
  - Sources: official (extract) Build page; official `ai-glasses-webapp-device/SKILL.md`.
- **Eligible fields.** `<textarea>` and `<input type="text|search|email|url|tel|number">`. The v127 skill also lists `contenteditable`.
  - `type` only controls eligibility. `inputmode` and `enterkeyhint` are ignored.
  - **`type="password"` never opens the composer.** Neither do `date`, `checkbox` or `radio`.
  - Source: official, commit `24d7bfc`.
- **How text arrives.** The composer keeps its own buffer. When the wearer confirms, the finished string lands in **one** update: `input`, then `change`. There are no per-key `keydown` / `keypress` / `keyup` events. Source: official (extract).
  - Unknown: whether `beforeinput` or `composition*` events fire, and whether a commit replaces or appends to the field's existing value.
- **Modes.** Handwriting (needs the Neural Band), dictation and the keyboard. Source: official.
  - The page cannot choose a mode and cannot open the keyboard directly.
  - The keyboard cannot be raised while the composer is in dictation mode.
- **Keyboard (v129+).** The wearer raises it by swiping down in the composer. Source: official (extract).
  - Layout: a four-row grid with a letters layer and a numbers-and-symbols layer (including `@` and `.`). Shift latches caps lock on and off.
  - It **cannot type** these characters: apostrophe `'`, double quote `"`, underscore `_`, backslash `\`, angle brackets `< >`, square brackets `[ ]`, braces `{ }`, pipe `|`, tilde `~`, caret `^` and backtick `` ` ``.
  - Meta advises against requiring these characters in email or URL fields.
- **Availability.** Meta: "On some firmware builds, the composer may be unavailable. Design your app to remain functional without it." Also: "Do not build a custom keyboard." Source: official.
- **Gotchas seen on device.** Source: community, [Glasscast](https://github.com/handzlikchris/Glasscast).
  - The pinch that closes the composer can arrive late, as an Enter or a click on whatever is focused next.
  - The glasses may reset focus to the first control when the composer closes.
- **Sign-in.** Meta: "have users sign in before they reach the glasses", because username and password entry on a 600×600 display is poor. Source: official (extract), [agent-tools page](https://wearables.developer.meta.com/docs/develop/webapps/agent-tools/).

### Required head tags

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="mrbd-web-app-capable" content="yes">
<meta name="description" content="What this app does, in one sentence.">
<link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png">
```

- **`mrbd-web-app-capable`.** Source: official.
  - Meta's game core contract: "Without this the device will not route D-pad / EMG input to the page."
  - The Build page says the tag identifies the page as Display-compatible for "upcoming discovery surfaces".
  - **None of the nine tags in our May probe was this one**, so the May input results were measured without it.
- **`description`.** Make it app-specific. WebMCP passes it to Meta AI as context. Source: official.
- **Viewport.** The current toolkit uses `width=device-width`; older Meta samples hard-code `width=600, height=600`. Treat 600×600 as "a validation target, not a fixed layout size". Sources: official; official (extract), [Test page](https://wearables.developer.meta.com/docs/develop/webapps/test/).
- **Icons.** PNG at least 52×52 px, or Unicode symbols. "SVGs are not supported." Source: official (extract), [Troubleshooting](https://wearables.developer.meta.com/docs/develop/webapps/troubleshooting/).
- **Our pages.** The HUD (`flight-deck/hud.html`) carries the tag, and so does the probe page (`glasses_input.html`).

### Input mapping

There is no pointer, hover, physical keyboard or touchscreen.

| Wearer does | Page receives | Notes and confidence |
|---|---|---|
| Swipe (thumb along the index finger on the Neural Band, or the temple touchpad) | `keydown` `ArrowUp` / `ArrowDown` / `ArrowLeft` / `ArrowRight` | official. Either own the arrows (`preventDefault()` and move focus yourself, as Meta's toolkit does) or leave them to native spatial navigation, never both. |
| Pinch (index finger to thumb) | `keydown` `Enter` on `document.activeElement`. Sometimes **also** a `click`, in either order, within about 500 ms. | `Enter`: official (game core contract). The double fire: community (Glasscast). Dedupe so one pinch means one action. Meta: "Pinch is not a positioned pointer event." |
| Pinch on a focused text field | The composer opens; the page gets no click | official (extract) |
| Ambiguous Neural Band events | `keydown key="Unidentified"` | official: "ambiguous; the input layer ignores it". This explains the `Unidentified` events in our May probe. |
| Any key | `KeyboardEvent.code` is empty | official: Meta's `PointerKeyboardInput.ts` says "Prefer `key`". Use `.key`. |
| Back (middle finger to thumb) | No key event; the shell calls `history.back()` | See [Back and history](#back-and-history) |
| `Escape` | Desktop and Simulator only | community: a Meta engineer says Escape is "not delivered on-device", although Meta's toolkit still listens for it. On the device, handling Escape **and** `popstate` would go back twice. |
| Middle tap | The universal web-app menu (Restart, Resume, Permissions) | official (extract) |
| Pinch-drag | Pointer events, only when `body { touch-action: none }` is in the initial CSS. Pointer Lock is unsupported. | official |

Wheel, wrist-rotate and double-tap gestures have no documented DOM mapping;
treat them as reserved by the system (inferred).

**Open conflict.** Newer Build chapters, quoted in a
[community PR](https://github.com/gregmarra/mbta-nearby/pull/1), say wearable
input "is not a page-level `ArrowUp` … or `Enter` key-event contract".
Meta's own toolkit and game templates still rely on those keydowns.

### Back and history

- **What Back does.** The shell checks `navigation.canGoBack`. Source: official (extract), Build page.
  - If it is true, the shell calls `history.back()`, and the page gets `popstate` (or a Navigation API `navigate` event).
  - Otherwise the shell shows the native system menu. Back while that menu is visible exits the app.
- **The 5-entry cap.** A session allows five history entries, counting the first page. Once `navigation.entries().length` reaches 5, further `pushState()` calls replace the current entry instead.
  - Source: official (extract); also community, [mrbd-ui-kit](https://github.com/michaelcummingsofficial/mrbd-ui-kit). We have not seen the wording verbatim.
- **How to use history.** Source: official.
  - Seed the first entry with `replaceState()` and push one entry per drill-down.
  - Restore the view in `popstate`, and never push while handling Back.
  - "Do not add an in-app Back button."
- **Focus after Back.** The glasses reset focus after Back, so restore it yourself. Source: community, Glasscast.

### Speech and audio

- **Text-to-speech works.** `window.speechSynthesis` needs no permission prompt and plays through the glasses speaker. Source: official (extract).
  - There is one "Default" en-US voice, and `getVoices()` returns it immediately.
  - `volume` is ignored, `boundary` events never fire, and pitch tops out at 2.
  - `cancel()` fires `end`, not `error`.
  - Listeners on the `speechSynthesis` object never fire; use the utterance's events.
- **Not supported: microphone, camera, `getUserMedia` and `SpeechRecognition`.** The Build page says "Web Apps do not yet support: Camera, Microphone, Notifications". Community reports conflict:
  - [ggoonnzzaallo/meta_display](https://github.com/ggoonnzzaallo/meta_display) (August 2026) reports `getUserMedia` audio and video working after a pinch.
  - [amanshah0729/vision](https://github.com/amanshah0729/vision) (September 2026), Glasscast (October 2026) and the [rokid-lumen docs](https://github.com/beyondlevi/rokid-lumen) report `NotFoundError` or no microphone.
  - Do not build on it; re-probe instead.
- **Web Audio works,** but the `AudioContext` must be created or resumed on the first pinch. Source: official.

### Voice control: WebMCP

- **How it works.** The page registers tools with `document.modelContext.registerTool({name, description, inputSchema, execute, annotations}, {signal})`, and Meta AI calls them when the wearer speaks. Source: official, `ai-glasses-webapp-webmcp/SKILL.md` (2026-10-01).
  - Meta: "The browser installs `document.modelContext` before page scripts run." Feature-detect it; don't polyfill it.
- **Off by default.** It is "enabled per device, by Developer Mode or by rollout", and eligibility is fixed when the app launches. Sources: official; W3C [implementation status](https://github.com/webmachinelearning/webmcp) ("coming soon").
- **Limits.** Source: official.
  - Parameters must be scalar. `enum`, `minimum`, `maximum` and `default` are dropped.
  - About 57 characters of a tool name are effective.
  - Descriptions are cut near 256 characters (parameters) and 1024 (tools).
  - `execute` has a 10 s deadline.
  - `isError` is dropped, so return JSON that includes a `next_action`.
  - Reserved names: `openUrl`, `goBack`, `goForward`, `reload`, `getCurrentUrl`, `getPageTitle`, `getPageText`.
- **The D-pad path must stay complete.** Meta: "The app must stay fully usable by D-pad with no assistant at all." Source: official.

### Runtime envelope and budgets

| | Value | Source |
|---|---|---|
| Viewport | 600 × 600 CSS px, DPR 1 | official (toolkit `AGENTS.md`) |
| Display | Additive waveguide: `#000` is transparent. Surfaces `#0a0a0f`–`#1C1E21`; text `#FFF` / `#E4E6EB` / `#B0B3B8`. Nothing interactive under 14 px; body text 16 px or more; 8 px safe margin. | official |
| Panel | 30 Hz, so a 33 ms frame budget. "Never add a 60 fps loop." | official |
| Link | About 500 Kbps down ("1 KB ≈ 16 ms"); about 150 ms per round trip | official |
| CPU | About 12× slower than a modern laptop core | official |
| First load | First paint under 1 s; usable under 5 s; **under 300 KB** transferred; **fewer than 15** initial requests | official (`ai-glasses-webapp-build`) |
| Warm launch | Under 2 s with 0 bytes transferred (Service Worker) | official |
| Heap | Under 128 MB. A page over the budget gets its renderer killed. | official |
| Hosting | Public HTTPS URL; HTTP is not supported ("The glasses runtime requires HTTPS for every Web App URL it loads") | official (extract) |
| Debugging | No console and no tethering; log remotely | official |

### Detecting the glasses (User-Agent)

- **No documented UA.** Meta does not document a User-Agent. Ours, recorded in May:
  `Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv) … Chrome/146.0.7680.177 …`
- **Meta's own check.** Meta's toolkit tests `userAgent.includes('Greatwhite') && userAgent.includes('; wv)')` (`PageTransition.tsx`). Our HUD and the `/` → `/hud/` redirect use the same check. Source: official.
  - Meta's game template uses a looser test (`wv` plus `Android`). It matches every Android in-app browser, so don't use it.
- **Route on the UA, never lay out on it.** Meta's toolkit says "Never hardcode device or viewport dimensions, or branch on them". The registered URL (`/hud/`) is the real switch.
- **Risk: Android 17 UA reduction.** Android WebView's default UA "will be reduced starting with Android 17" to `Mozilla/5.0 (Linux; Android 10; K; wv) …`. Source: official (extract), [Android Developers blog](https://android-developers.googleblog.com/2024/12/user-agent-reduction-on-android-webview.html).
  - That would drop `Greatwhite`; the `wv` token stays.
  - The Display runs Android 14 today.
  - The `Sec-CH-UA-Model` client hint (WebView 116+) may still report the model. It is untested on the Display.

### Re-probe checklist (firmware v129)

Run this on glasses with firmware v129 or later and Meta AI app v272 or later.
The probe page
[`glasses_input.html`](../captain_claw/flight_deck/static/glasses_input.html)
now carries `mrbd-web-app-capable`. It also has two new sections: Test 7 logs
composer events and Test 8 is a capability readout.

1. Write down the firmware and Meta AI app versions (Meta AI app → Devices → your glasses).
2. Open `/glasses/input?c=<channel>`. If `FD_GLASSES_BRIDGE_TOKEN` is set, add `&t=<token>` so that Save works.
3. **Swipes and pinches.** Swipe and pinch around, then check the log:
   - Do swipes now arrive as `ArrowUp`/`ArrowDown`/… instead of `Unidentified`?
   - Does a pinch produce `Enter`, a `click`, or both?
   - Is `code` empty?
   - Does focus reach the native `<button>`s (Copy / Send / Clear)?
4. **Test 7 · Composer.** Focus the textarea and pinch it. Does the composer open? Dictate something and confirm, then record:
   - the event order: is there a `beforeinput`? `composition*` events? `input` → `change`?
   - whether the value was replaced or appended to;
   - where the closing pinch lands (an Enter or click on the next focused element? a focus reset to the first control?).

   Repeat with the `type=text` input.
5. **Keyboard.** In the composer, swipe down. Does the keyboard appear? Try `' " _ [ ] ~` to confirm the symbol limits.
6. **Tests 1–3.** Does pinching `<input>`, `<textarea>` or the contentEditable div open the composer? Does focus alone, without a pinch, do anything?
7. **Test 4.** Is `webkitSpeechRecognition` still `service-not-allowed` / "MetaGlassSDK dictation not available"?
8. **Test 8 · Capabilities.** Record:
   - `document.modelContext` (is WebMCP on?);
   - the `speechSynthesis` voices;
   - `userAgent` (still `Greatwhite … ; wv)`?) and `userAgentData.model`;
   - `viewport` (600x600 @1?);
   - `serviceWorker` and `speechRecognition`;
   - the list of Meta-looking globals.

   Then tap **Try getUserMedia(audio)**. Does it succeed, or fail with `NotFoundError` / `NotAllowedError`?
9. **Test 6.** Do scheme links still do nothing?
10. **Back.** Does the Back gesture produce `popstate` with no key event? (`Escape` should not appear.)
11. **Save the log.** Tap **Save to FD /tmp** to write `/tmp/glasses-input-<UTC-ts>-ch_<channel>.log`. Diff it against the May files and update this document.
12. **Checks on `/hud/`.** The probe page can't cover two things:
    - After pairing, close the app and relaunch it. Does the httpOnly `fd_refresh` cookie survive?
    - Does the 5-entry history cap behave as described?

### Sources (October 2026 update)

**Meta, official.** Pages marked "extract" were read through search-engine extracts only.
- [Web Apps: Build](https://wearables.developer.meta.com/docs/develop/webapps/build/), [Test](https://wearables.developer.meta.com/docs/develop/webapps/test/), [Setup](https://wearables.developer.meta.com/docs/develop/webapps/setup/), [Agent tools / WebMCP](https://wearables.developer.meta.com/docs/develop/webapps/agent-tools/), [Troubleshooting](https://wearables.developer.meta.com/docs/develop/webapps/troubleshooting/), [Wearables FAQ](https://developers.meta.com/wearables/faq/) (extract)
- [Build for Display glasses (May 2026)](https://developers.meta.com/blog/build-for-display-glasses/), [Connect 2026 recap](https://developers.meta.com/blog/meta-connect-recap/), [Connect 2026 AI-glasses recap](https://developers.meta.com/blog/meta-connect-recap-ai-glasses/), [v127 release post](https://www.meta.com/blog/meta-ray-ban-display-glasses-v127-muse-spark-threads/), [Neural Band gestures help](https://www.meta.com/help/ai-glasses/764536076119235/) (extract)
- [facebook/meta-wearables-webapp](https://github.com/facebook/meta-wearables-webapp):
  - `AGENTS.md`
  - the skills `ai-glasses-webapp-device`, `ai-glasses-webapp-build`, `ai-glasses-webapp-optimize-performance` and `ai-glasses-webapp-webmcp`
  - the game plugin's `docs/core-contract.md` and `PointerKeyboardInput.ts`
  - commits `de9ebda`, `24d7bfc`, `ca5fb95`, `e617d92`

  All read directly.
- [facebook/meta-ray-ban-display-ui-toolkit-web](https://github.com/facebook/meta-ray-ban-display-ui-toolkit-web): `FocusNavigationProvider.tsx`, `BackNavigation.ts`, `InputTextView.tsx`, `PageTransition.tsx` (read directly)

**Other official sources.**
- [webmachinelearning/webmcp implementation status](https://github.com/webmachinelearning/webmcp)
- [Android Developers: User-Agent reduction on Android WebView](https://android-developers.googleblog.com/2024/12/user-agent-reduction-on-android-webview.html)

**Community.**
- [GlassKit platform audit (June 2026)](https://github.com/GlassKitApp/glasskit-ui/blob/main/docs/platform-audit-2026-06.md)
- [mrbd-ui-kit](https://github.com/michaelcummingsofficial/mrbd-ui-kit)
- [Glasscast](https://github.com/handzlikchris/Glasscast)
- [ggoonnzzaallo/meta_display](https://github.com/ggoonnzzaallo/meta_display)
- [amanshah0729/vision](https://github.com/amanshah0729/vision)
- [beyondlevi/rokid-lumen](https://github.com/beyondlevi/rokid-lumen)
- [gregmarra/mbta-nearby PR #1](https://github.com/gregmarra/mbta-nearby/pull/1)
- ["Add a Web App" missing (issue #9)](https://github.com/facebook/meta-wearables-webapp/issues/9)
- [MIXED: WebMCP off by default](https://mixed-news.com/en/meta-ray-ban-display-webmcp-voice-control-web-apps-off-by-default/)

---

## May 2026 probe record (firmware v125)

Empirical reference compiled May 2026 from running an instrumented probe
page inside the Display's launcher webview. Goal: tell future-Stevica
exactly what works, what doesn't, and *why* — so we don't burn another
afternoon re-deriving it whenever Meta ships a firmware update.

## TL;DR (May 2026, firmware v125)

The Display loads third-party "apps" as URLs inside an **Android WebView**
hosted by Meta's launcher. The launcher decides at WebView-creation time
which native bridges to expose. Meta-owned origins get a bridge called
`MetaGlassSDK`; our apps don't, and there is **no in-page mechanism to
acquire it**.

| Surface | Status (May 2026, v125) |
|---|---|
| HTML / CSS / JS rendering | ✅ Works (Chrome 146 / Android WebView 146) |
| WebSocket | ✅ Works |
| `fetch()` over HTTPS | ✅ Works |
| Intra-app `location.href` navigation | ✅ Works |
| `<input>`, `<textarea>`, contentEditable — focus | ✅ Works (cursor appears) |
| Click / pointer / touch events | ✅ Works |
| **Text input — system keyboard summoned by focus** | ❌ Blocked at WebView host · *May 2026 (v125)*; since v127 a focus + pinch opens the composer |
| **`webkitSpeechRecognition`** | ❌ `service-not-allowed` / `"MetaGlassSDK dictation not available"` · *May 2026 (v125)*; still undocumented (assume blocked) |
| **`getUserMedia({ audio })` / `({ video })`** | ❌ Blocked · *May 2026 (v125)*; still officially unsupported |
| **Neural Band handwriting → focused input** | ❌ Not delivered to our webview as text · *May 2026 (v125)*; since v127 through the composer |
| **Scheme links (`tel:`/`sms:`/`mailto:`/`intent://`)** | ❌ Click event fires, system handler never launches · *May 2026 (v125)*; not re-tested |
| **Page-declared permission meta tags** | ❌ Parsed but ignored (9 variants tested) · *May 2026 (v125)*; the now-required `mrbd-web-app-capable` was not among them |
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

> **Superseded (October 2026).** Since firmware v127, focusing a field **and
> pinching it** opens Meta's composer. See
> [Text entry: the on-glasses composer](#text-entry-the-on-glasses-composer).
> The finding below is what we measured on v125, on a page without
> `mrbd-web-app-capable`.

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

> **October 2026.** The probe page now also has Test 7 (composer event log)
> and Test 8 (capability readout, including the scan for Meta-looking
> globals). For firmware v129, follow the
> [re-probe checklist](#re-probe-checklist-firmware-v129) at the top.


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
