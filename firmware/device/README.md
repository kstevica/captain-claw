# Captain Claw · handheld voice client (M5Stack StickS3, StopWatch and PaperColor)

Press the talk button and talk; press it again, or just stop talking for
2.5 s, to send (on the StopWatch: hold it, talk, let go). Flight Deck transcribes it, sends it to
one of your agents, and the device shows the reply, reads it aloud, and shows
any picture the agent made.

Server side: `captain_claw/flight_deck/device_bridge.py` (`/device/ws`).

One source tree builds both devices (`src/board.h` holds what differs):

| | StickS3 | StopWatch | PaperColor |
|---|---|---|---|
| screen | 240×135 LCD (`ui_stick.cpp`) | 466×466 round AMOLED (`ui_round.cpp`) | 4" colour e-paper, 400×600, six inks (`ui_paper.cpp`) |
| talk button | front (A) | yellow (A) | top (C) |
| other button | side (B) | the second button (B) | A: next · status · hold = Options |
| third button | — | — | B: back · home · hold = Wi-Fi & settings |
| extras | — | vibration (buzz before speech, on talk), clock | two status lights, room temperature + humidity, keeps its screen asleep |
| build | `pio run -e sticks3` | `pio run -e stopwatch` | `pio run -e papercolor` |

The sections below say "front"/"side" for A/B.

## What the agent learns about the device

Every turn goes up with the device's readings at that moment
(`telemetry.cpp`): battery (%, volts, charging or on USB), Wi-Fi (network and
signal), how it's held and whether it's moving (StickS3/StopWatch IMU: lying
flat / tilted / held upright · still / moving / shaking), the room's
temperature and humidity (PaperColor SHT40), and the local time. Flight Deck
turns them into one line of context (`device_context_line()`), so "how's my
battery?" or "is it warm in here?" just work. The agent doesn't recite them
unless asked, or unless the battery is nearly empty. They are never stored as
the user's words.

## Offline voice notes

Out of Wi-Fi, or Flight Deck down? Talk as usual:
the recording (up to 60 s) is kept as a voice note in the device's own flash.
The screen says "Saved offline". Notes are kept at telephone quality —
8 kHz, 4-bit IMA ADPCM (`src/adpcm.h`), 4 KB a second, an eighth of the
16 kHz PCM the mic records — which is as low as speech goes and still
transcribes well. The StopWatch's and PaperColor's 3.4 MB flash partition
holds about 13 minutes of notes (it held 1½ minutes as WAV), the StickS3's
1.5 MB about 5½. The status page shows how many are waiting and the room
left.

Once the device is back online, it asks first. The deck says "There are
saved voice notes. Would you like to send them to the agent?" and the screen
shows the question. The talk button sends them; the side button (PaperColor:
B) keeps them for later. A "later" holds until a new note is recorded or the
device starts cold; on the StickS3 and StopWatch, Menu → *Send voice notes*
sends them any time. Sent notes go to Flight Deck, which transcribes them (as 8 kHz PCM)
and gives the agent all of them as one message, oldest first, each with when
it was recorded, marked as recorded offline. The agent handles them and
answers once. A note is deleted from the device only after the deck has it;
one Soniox couldn't transcribe stays for a later try. Notes kept by older
firmware (16 kHz WAV) still go up as they are.

Without Wi-Fi at switch-on they carry on offline (and keep looking for it)
instead of opening the setup hotspot. They look first: when none of the
saved networks is in reach, they're note takers within a couple of seconds
instead of trying for 30. Setup is a switch-on with the talk
button held, or Menu → *Wi-Fi & settings* (PaperColor: hold B). Holding B
(the side button) on the setup screen leaves it without saving.

## Talk from sleep

Asleep, hold the talk button and start speaking at once. The device records
while it wakes up and reconnects (a few seconds), then sends all of it as if
you had spoken just now. A short press only wakes it. On the StickS3 and
PaperColor, press the talk button and talk, then press it again (or stop
talking) to send; a wake-up press that's followed by silence sends nothing. (On the StopWatch and PaperColor,
a recording that can't reach Flight Deck is kept as an offline voice note.)

## PaperColor (colour e-paper)

Every redraw of its Spectra 6 panel takes 15–30 s (there is no faster or
partial refresh worth having), and the paper keeps its picture with no power
at all. So it's voice-first and redraws when you put it down:

- **While you use it, the paper doesn't change:** you talk (press the top
  button, C, talk, press it again — or just stop: 2.5 s of quiet sends it),
  hear the answer, and the status lights show what's happening. About 30 s after you
  stop, the paper draws the newest answer once. Waking up and falling asleep
  don't redraw when it already shows that answer. Pages, the status page and
  B (the home screen, with an answer still waiting) draw at once.
- **Text only, in colour:** the six inks (black, white, red, yellow, blue,
  green) for the text and marks, pictures dithered into them. A redraw takes
  the same time in black and white, so the colours stay. (`kMono` in
  `ui_paper.cpp` switches to black and white.)
- **Status lights:** red blinking while it records you (brighter when you're
  louder); while the agent works, a blink for every step it takes — light
  blue for a tool, dark blue for narration and progress, purple for the model
  thinking; green blinking with the voice; a soft white glow while the paper
  redraws; white blinks as the cue before speech. Starting up (a switch-on
  or a wake), both are yellow; then the left one turns green when Wi-Fi is up
  (red if there's none) and the right one green when the agent is reached
  (red if it isn't); both green stay for 6 s, then go out. Idle, a short blink
  every 10 s says how near sleep is: green for the first 30% of the wait,
  blue for the next 50%, yellow for the last 20%. Off / low / bright in the
  settings.
- **Options instead of a menu:** hold A for a QR code. Open it on a phone on
  the same Wi-Fi: every setting is there (the link carries a one-time key and
  closes when you leave the screen). On that screen the top button checks for
  (and installs) a software update.
- **Home screen:** the date, the room (temperature, humidity), whether it's
  ready, and the newest answer.
- **Asleep** (after 1 / 3 / 10 idle minutes) the paper keeps what it shows. It
  redraws it every 30 min / 1 h / 3 h without Wi-Fi (settings), for the date,
  the room and the battery. The last answer lives in flash, so it survives
  the sleep; press B after waking to read it again.
- **Battery empty:** it leaves "Battery empty" on the paper and switches off.

| | C (top) | A | B |
|---|---|---|---|
| click | talk, click again to send · while it speaks: stop · on Options: update | home: status · answer: next page (after the last, the first) · Options: done | answer: home screen · home: the answer · back |
| hold | | Options (QR code; hold again closes) | Wi-Fi & settings (the setup hotspot; hold B again to leave) |
| double-click | hear the last answer again | (on an answer) hear it again | |

To see every paper screen without the device (Mac, SDL2 from Homebrew):

```bash
pio run -e preview_paper
uv run --no-project --with pillow python preview/render.py paper   # → preview/out/paper/sheet.png
```

## StopWatch screen

The round screen follows M5Stack's own StopWatch look: black, a glowing badge
in the middle with a name under it, page dots and chevrons for lists. The ring
round the edge shows what the watch is doing — red while it listens (filling
toward the 110 s recording limit), amber while the agent works (a tick per
tool step). While it speaks, a short cyan line under the text follows the voice
(only that strip redraws, so it costs next to nothing). The clock sits at the top between the Wi-Fi
(left) and agent (right) dots; its time zone is set in the setup page.

Swipe an answer to page it: left or up for the next page, right or down for
the one before — and from the first page back to its picture, when it came
with one. Swipes stop at either end. A double tap (off the record button)
goes to the home screen from an answer, a picture, the menu or a notice; a
swipe there brings the answer back. With a background picture set, a double
tap on the home screen shows the picture alone, nothing over it; a double tap
there (or the other button) goes back home, a swipe to the answer.

The menu, the agent and character pickers and the status pages take touch
too: swipe left for the next row or page, right for the one before, tap a
chevron for the same, tap the badge between them to choose. The "send voice
notes?" question has two pills to tap. A picture from the agent fills the
circle; a single tap shows all of it, scaled down, and another fills again.
The menu has two levels: what you reach for in the moment at the top (Replay,
Status, Agent, Volume, Tap to talk, Software update, Wi-Fi) and the rest under
*Settings*. The watch answers touch with a vibration vocabulary: a tick for a
step or a page, a double tick at the end, a longer one for a choice, three
for something that went wrong. The home screen's "Hold the yellow button to
talk" goes away after the first five recordings.

Screens move (0.14.1): a page, a picker row or a status page slides in from
the side it came from (four eased frames, ~200 ms); going somewhere else —
home, the menu, a notice, the picture alone, the status — fades through black
(the panel's brightness, so it costs no drawing). An answer arriving still
cuts in, so nothing delays it. While the agent works, the ring's track
breathes under the moving head.

The text follows the voice (0.14.2). While an answer is spoken, the sentence
being said is lit and the rest dimmed, the page turns by itself when the
voice reaches the next one (until you page by hand), and in character mode
the sentence shows as a subtitle under the face. No word timings are
needed: the watch knows how far into the speech the speaker is (the mouth
uses the same estimate); once the speech has all arrived, that fraction of
the text is the place, and before that the pace learned from the last answer
carries it (the deck says how many characters it speaks: `"spoken"` in the
reply). After the speech, a small replay button sits under the text — tap it
to hear the answer again.

*Tap to talk* (Menu, or the settings page; off by default) puts a record
button on the screen: under the name at home, under an answer or the
character. Tap it and talk; the button turns into a stop square, and a second
tap sends the recording (offline it's kept as a voice note). Holding the
yellow button works as before, and its press also ends a tapped recording.
A tapped recording also ends by itself after 2.5 s of quiet (and is sent);
with nothing said at all it's dropped.

While an answer is spoken, the button under it is a **pause** (two bars) —
whether or not Tap to talk is on — and a **play** while it's held; the bar
says *paused*. The rest of the answer keeps arriving meanwhile, so play goes
on from the word it stopped at. A held answer counts as idle: the screen
dims and the watch sleeps on its usual timers (a touch wakes it, with the
play button still there until it sleeps). The answer always leaves room for
the button, so its text never jumps when one comes or goes.
Text uses Inter (`fonts/`, made by `tools/make_fonts.py`; SIL OFL, see
`fonts/OFL.txt`), laid out to the circle.

**Background picture** (0.13.0). Send your agent a picture and ask for it as the
watch background: its `watch_background` tool hands it to Flight Deck
(`flight_deck/device_backgrounds.py`), which crops it to fill the circle and
keeps it. Every screen is drawn on it, darkened so the text stays readable,
instead of black — except a picture the agent shows, which stays on black. A
double tap on the home screen shows the picture alone, as it is (0.13.1); a
double tap there goes back home. The watch keeps it in flash (`/background.jpg`),
names it in `prefs` (`"bg": id`), and Flight Deck sends the owner's
(`{"type":"background","id","size"}`, then 0x04 parts and a 0x05 last part)
whenever it's another one; size 0 clears it. 0.13.0 darkens nothing itself, so it
gets the picture darkened by Flight Deck. Ask the agent to clear it to go back to
black.

To see every screen without the watch (Mac, SDL2 from Homebrew):

```bash
pio run -e preview
uv run --no-project --with pillow python preview/render.py   # → preview/out/sheet.png
```

## Speech chime

With *Beeps around speech* on (the default), the device plays a short rising
chime (C6 → G6) before it speaks and the same two tones falling (G6 → C6)
once the answer has played to its end, so you know it's done. A stopped
answer or a replay gets neither.

*Talk sounds* (on by default) are single notes, so they can't be mistaken for
those: a short high blip (A6) when you press, with recording starting a
quarter second later (talk after the blip), and a lower one (A5) when you let
go and it's sent or kept offline. A tap that only stops speech, or a
press too short to send, gets no second blip.

## Buttons

StickS3:

| | Front (A) | Side (B) |
|---|---|---|
| click | talk, click again to send (or 2.5 s of quiet) · while it speaks: stop | next page · back (a picture: its text) |
| hold | | menu (hold again closes) |
| double-click | hear the last answer again | (on an answer) hear it again |
| in the agent list | choose | move down |

StopWatch: hold the yellow button to talk and let go to send; a tap stops
speech or flips picture ↔ text; the other button works like the StickS3's
side button.

**Replay.** Double-click the side button on an answer (its text or its picture), or
choose Menu → *Replay answer* (the first row, while there's an answer with sound), to
hear the last answer again, without the cue. The device keeps the whole answer in
memory (its speech, then any sound the agent made, up to ~66 s), so a replay plays
at once and needs no network. If part of the answer never arrived (you stopped it
early, or the link dropped), the device asks Flight Deck to send it again instead
(`{"type":"replay"}`; the speech is synthesized afresh). Powering off forgets it.
A single side click on an answer now waits 0.35 s to make sure it isn't a double-click.

**A link that drops mid-turn** (weak Wi-Fi). The device stays on the working screen
with *Connection lost — reconnecting…*, awake and retrying, for up to 2 minutes.
Flight Deck keeps the turn and its answer (10 minutes): after the reconnect the answer
comes as usual, or the device shows *Thinking…* again while the agent is still at it.
If the link stays away, or Flight Deck no longer has the turn (it restarted), the
device says *Answer lost* — ask again. The heartbeat allows three missed 10 s pong
windows (it was two of 5 s) before it counts the link as gone.

In the side-button list, *Volume* (25/50/75/100%) and *Boost* (off, +6, +10 dB —
extra speech gain with a soft limiter) change with each front-button press and
beep at the new level. Above 75% volume on battery the stick may restart on loud
peaks (M5Stack's note); on USB power it's fine.

Display → *icons* shows an animated character instead of text: Captain, Bot, Bird, Car,
Matrix, Orc, Elf, Dwarf, C64, Amiga, Cat, Dog, Dinosaur, Alien, Caveman, HAL 9000, KITT or
Pac-Man. Menu → *Character »* previews each one cycling through its poses (side button =
next, front = pick, side hold = back).

Menu → *Text size* (small / medium / large) sets the reading text: replies, the working
screen, notices, the menu and the status page (which then spans pages: B = next page, after
the last one back to the main screen). The top bar stays compact.

Hold the front button while switching on to reopen setup. With no Wi-Fi
saved, setup opens by itself; with none in reach for 30 s, it carries on
offline and keeps voice notes. (Pressing it to wake from sleep is talking,
not setup. See *Talk from sleep*.)

## Setup

1. The device opens a hotspot `Captain-StickS3-xxxx` / `Captain-StopWatch-xxxx` / `Captain-PaperColor-xxxx`
   (scan the QR code on screen).
2. Open http://192.168.4.1 and fill in:
   - one or more Wi-Fi networks (up to 5; it joins whichever is in reach)
   - Flight Deck address: `ws://<mac-ip>:25080` on your network, or `wss://<your deck domain>`
   - access token: Flight Deck → Connections → *Agent access for Claude (MCP)* → Generate token
     (not needed on a deck with auth off)
   - optional: agent slug, spoken-reply language (`en`, `hr`, …), volume, screen-off (5/10/15 s)
     and power-off (15/30/60 s) timeouts; on the StopWatch also buzz and time zone; on the
     PaperColor orientation, status lights, key clicks, reading the menu aloud, power-off
     (1/3/10 min), redrawing while asleep and time zone
3. Save — the device restarts and connects.

The deck needs `SONIOX_API_KEY` (speech in, voice out). A deck reached
through a domain needs that domain in `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL`.

## Build and flash

```bash
uv tool install platformio
cd firmware/device
pio run -e sticks3 -t upload      # or: -e stopwatch / -e papercolor
```

If the upload can't connect: hold the side button until the green LED
blinks (download mode), upload, then press the side button once to restart.
StopWatch: with USB in, hold the power button ~2 s until the green LED lights.
PaperColor: with USB in, hold the side reset button ~3 s for download mode.
Back up the factory firmware before the first upload:
`esptool read-flash 0 0x1000000 stopwatch-factory.bin` (or `papercolor-factory.bin`).

Serial log: open the port with DTR/RTS released — asserting them resets the
StickS3 into download mode. `pio device monitor --dtr 0 --rts 0`.

## Updates over the air

From 0.8.0 the devices update over Wi-Fi from Flight Deck (design:
`docs/device-ota-plan.md`). Bump `FW_VERSION` in `src/main.cpp`, then:

```bash
uv run --no-project --with cryptography python tools/publish.py sticks3 \
    --deck https://glasses.captain-claw.com --token cc_pat_… --notes "what changed"
```

It builds, signs (ECDSA P-256, key in `~/.captain-claw/firmware-signing/` — back it
up; without it no device accepts a new build) and uploads (admin token). Connected
devices of that kind get the offer at once, others on their next connect:
Menu → *Software update* (or by itself with *Auto update*, when idle on USB). A new
version that can't reach Flight Deck within 90 s rolls back to the previous one.
The key pair was made once with `publish.py keygen`; `src/ota_pubkey.h` is its
public half.

The StickS3 screens use trimmed copies of efontJA (`src/efont_subset.*`, made by
`tools/subset_fonts.py`) — run it again if a screen needs a glyph that isn't kept.

## Back to stock firmware

M5Burner → StickS3 / StopWatch / PaperColor → UIFlow2 (or Xiaozhi), or write back a full flash dump
with `esptool write-flash 0 <dump>.bin` in download mode.
