# Captain Claw · handheld voice client (M5Stack StickS3, StopWatch, PaperColor; Waveshare ePaper 1.54)

Press the talk button and talk; press it again, or on most devices just stop
talking for 2.5 s, to send. Flight Deck transcribes it, sends it to
one of your agents, and the device shows the reply, reads it aloud, and shows
any picture the agent made.

Server side: `captain_claw/flight_deck/device_bridge.py` (`/device/ws`).

One source tree builds them all (`src/board.h` holds what differs; the Waveshare boards have
[their own section](#waveshare-esp32-s3-epaper-154-epaper154-epaper154_touch)):

| | StickS3 | StopWatch | PaperColor |
|---|---|---|---|
| screen | 240×135 LCD (`ui_stick.cpp`) | 466×466 round AMOLED (`ui_round.cpp`) | 4" colour e-paper, 400×600, six inks (`ui_paper.cpp`) |
| talk button | front (A) | yellow (A) | top (C) — on the B build (`papercolor_b`): A |
| other button | side (B) | the second button (B) | A: next · status · hold = Options |
| third button | — | — | B: back · home · hold = Wi-Fi & settings |
| extras | — | vibration (buzz before speech, on talk), clock | two status lights, room temperature + humidity, keeps its screen asleep |
| build | `pio run -e sticks3` | `pio run -e stopwatch` | `pio run -e papercolor` (A) or `-e papercolor_b` (B) |

The PaperColor's other buttons above are the A build's; the B build's are in
[PaperColor the "B" way](#papercolor-the-b-way-papercolor_b-versions-1x).
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
  blue for the next 50%, yellow for the last 20%. Offline, for the first
  minute, one light blinks blue (half a second on, half off): the left one
  when there's no Wi-Fi, the right one when there's Wi-Fi but Flight Deck or
  the agent can't be reached (before 1.0.13 both blinked, whichever it was).
  Off / low / bright in the
  settings.
- **Options instead of a menu:** hold A for a QR code. Open it on a phone on
  the same Wi-Fi: every setting is there (the link carries a one-time key and
  closes when you leave the screen). On that screen the top button checks for
  (and installs) a software update.
- **Home screen:** the date, the room (temperature, humidity), whether it's
  ready, and the newest answer.
- **Asleep** (after 1, 3, 10 or 30 idle minutes, 1 or 2 hours, or never — the
  phone page) the paper keeps what it shows. It
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

### PaperColor the "B" way (`papercolor_b`, versions 1.x)

The same paper built as if it had no screen: `src/ui_paper_b.cpp` replaces
`ui_paper.cpp` (the "A" way above stays, env `papercolor`; both are the
`papercolor` kind for updates, and the deck offers whichever version is
newest, so B is published as 1.x). The lights carry listening, working,
speaking and trouble exactly as above; the voice carries the answers:

- **A dashboard, every five minutes** (1.0.11; in blocks of colour since
  1.0.24): a black block out to the paper's edges, round only at the bottom,
  with what waits in Flight Deck, the agent and the Wi-Fi and agent dots on the
  right, a big clock as of the drawing and the date.
  Under it three blocks of the same width: the room in blue (temperature, humidity), the power
  in green, yellow under 40 %, red under 20 % (percent, and how long it has or
  until it's full, with a bolt while charging), the notes in yellow, red when
  under 2 minutes of room are left (how many, and the room: minutes in flash,
  GB on a card). Then a big tile: the mic and Captain, online or offline
  (1.0.27). The links (Wi-Fi, deck,
  agent) are on the options screen (1.0.26), under the code; the dots at the
  top still show them at a glance. Text on a colour stays between its two inks, so its edges don't
  snap to a third. Redrawn in full on every five-minute mark
  (:00, :05 … :55), awake or asleep (the timer wake lands just after the
  mark, draws it without Wi-Fi and sleeps again), and after an offline voice
  note is kept.
- **No legend: what a button does is drawn beside it** (1.0.27): on every
  screen, an icon in a white circle with a black ring (1.0.28) where the button
  is — A and B on the left edge, C on top, in the middle. On the dashboard A is the mic, B Captain (the last answer), C a
  gear (the options). On an answer A is the mic, B an arrow down (the next
  page) or a house (the last page: the dashboard), C the gear. On the options A
  is an arrow back, B Captain, C the update. In the setup hotspot A is the
  arrow back. Words beside A's and B's icons start clear of them, and take the
  whole width below.
- **Fast, whole refreshes** (1.0.10): every refresh starts with a reset pulse
  to the panel's controller (EPD reset is G12). With the paper keeping its
  picture through restarts, M5GFX starts it without one, and refreshes were
  several times slower; the 1.0.5–1.0.9 partial clock refresh (gone in
  1.0.11) also left the controller in its window without it.
- **A white pass, hourly and at power-on** (1.0.18; only these two since
  1.0.25): six-ink paper that only ever goes from one picture to the next
  loses colour and keeps faint shadows of earlier pictures. Once an hour (the
  first dashboard on its mark in a new hour, normally :00) and when it's
  powered on (with the first dashboard after that) the paper goes all white
  first, then draws the dashboard. An update or a crash doesn't get one, nor
  does a dashboard someone asked for (back from the answer, a press), and with
  no clock there are no hours to go by. Another screen asked for during the
  white pass cuts it short, as above, and it's done with the next dashboard.
- **Quick screen changes** (1.0.12): the refresh is the firmware's own
  (`PaperPanel::refresh()`, the same steps as M5GFX's `display()`), so it
  knows when the panel is in its waveform. Asking for another screen then —
  the answer, its next page, the options, back — cuts that waveform short
  with the same reset pulse and draws the new one at once, instead of
  waiting for the old one to finish. (The same screen again, or the answer
  redrawn once it's whole, waits.) Opening the options no longer waits for a
  2–4 s Wi-Fi scan first (it runs in the background for the phone page's
  network list), and A on the options goes back to the page it came from —
  the answer or the dashboard.
- **The answer** (1.0.11; 1.0.12 only from the answer page): when the
  answer page is on the paper, a new answer goes there as its voice starts,
  at the largest of four sizes that fits it on one screen. Since 1.0.25 the
  page reads as a conversation: the question in a yellow bubble (up to three
  lines, first page only), a row with Captain, the agent and the time,
  a picture if there is one, then the answer, with a line for how far through
  the pages it is. It's drawn again once it's whole
  and the voice is over (it's spoken sentence by sentence as it's written);
  with speaking off, when it arrives. From the dashboard the paper stays on
  the dashboard: B shows the answer. Longer answers page at the smallest size; B
  pages on, and after the last page B brings the dashboard back. It lives in
  flash: B after a sleep shows it. Hold B for the dashboard now.
- **A talks** (1.0.11; C before): press, talk, press again — or 2.5 s of
  quiet sends it; a double press plays the last answer again. With no deck to
  send it to, it's a note: only a press ends it (1.0.27).
- **Note-taking mode** (1.0.16): hold A (1 s) to turn it on or off — rising
  tones on, falling off, and the dashboard redraws (since 1.0.24 the top block
  turns red and the tile black, with a yellow notepad, the count on its corner,
  "Hold [the mic] to send", the notes' length and the room left). While it's on, every recording is kept as a
  voice note, online too (press A, talk, press A; no 2.5 s quiet stop — notes
  have their pauses); turning it off sends them all to the agent as one
  turn. The press that holds on begins a recording, which is dropped.
- **C: the options code** (1.0.11; A before) — the phone settings page; with
  no Wi-Fi it opens the setup hotspot instead. On it, A goes back (to the
  answer or the dashboard, where it was opened from) and C installs the
  update on offer. Hold C for the hotspot (*Wi-Fi & settings*)
  at any time; a short press of A there leaves it.
- **No asking:** voice notes kept offline go to the agent by themselves on
  the next link, and *Auto update* defaults to on (an update installs itself
  when idle on USB) — both can be changed on the phone page. Notices have
  nowhere to go: the lights say what matters, and go back to their idle
  blink once a recording is over (1.0.11: after an offline note they stayed
  on the recording lights).
- **Recording comes first** (1.0.14): while it records — talking, an
  offline note, or the recording a wake press starts — nothing redraws by
  itself; the five-minute dashboard (or the offline one) waits until the
  recording ends, and after an offline note the dashboard is drawn then.
- **Offline** (1.0.13): once Wi-Fi, Flight Deck or the agent has been
  down for 30 s, the paper is redrawn with it — the top block's dots in red
  (an answer being read keeps its page; the options code stays) — and again
  once they've been back for 10 s. The offline blink says which: the left
  light for Wi-Fi, the right for Flight Deck or the agent.
- **Recording with no link** (1.0.27): a note needs no link, so the link going
  doesn't touch one, and a recording on its way to the deck when the link goes
  carries on as a note (what went up before that is lost). With Wi-Fi known to
  be down, A starts a note straight away. Trying for the deck no longer holds
  the main loop (its name can take 15 s to fail, the connection 5 s more): a
  task of its own knocks first, and the link is only worked on for 15 s after
  the deck answered, never while a note is written.
- **Moving it does nothing** (1.0.1): motion is never read as use, and what
  the paper shows, with when the dashboard was drawn, is kept in flash as
  well as across deep sleep — a restart, a brownout or the battery jolted
  loose doesn't redraw it. Each connect tells the deck why the device booted
  (`boot=power|restart|crash|watchdog|brownout|button|timer` in its log).
- **Downloading an update** (1.0.2): the lights blink purple left, then
  right, one a second, until it restarts into the new version (or gives up).
- **Key beeps** (1.0.4): every press of C and B, and of A on the options
  code, gives a short beep (the options button higher, B lower); in the
  setup hotspot every press beeps. "Beep off" or *Key beeps: off* on the
  phone page silences them.
- **microSD card** (1.0.5; 1.0.15 put in or out while awake): with a card
  in the slot, voice notes recorded offline are kept on it (hours of room;
  the dashboard says how much is free) and notes already in flash move onto
  it; with none, they stay in flash (about 14 min) as before. It's found at
  every start and wake, within a minute while the paper is left alone, and
  right before an offline note is kept; taken out, the notes go to flash
  again. FAT32 (or FAT16) only — this SD library has no exFAT, so a card
  over 32 GB usually needs reformatting. The notes live in `captain-notes/`
  (`claw-notes/` before 1.0.15 — moved over at start-up), and only files
  named the way the device names them (`<epoch>-<8 hex>.adp`) are ever read,
  sent to the agent or removed: anything else on the card is left alone.
  The card shares the panel's bus, so it waits while the paper redraws.
- **No length limit on a note** (1.0.16): a note made awake (note mode, or
  offline) is written to the card or flash as it's recorded — coded on the
  main loop (`adpcm::StreamCoder`, the same bytes as coding it whole:
  `test/host/adpcm_stream.cpp`), written by a task of its own so a slow
  write never stalls the mic — up to the room left (about 14 min in flash,
  hours on a card). A recording begun by the press that wakes the paper is
  still held in memory first (up to 30 s): it starts before the storage is.
- **Recording lights** (1.0.7): the left light blinks red while it records
  (brighter as you speak); the right one, steady, says how long it's been —
  green for the first 20 s, yellow for the next 20 s, red after that.
- **Power off** (1.0.3): after 1, 3, 10 or 30 idle minutes, 1 or 2 hours, or
  never (phone page). Awake it stays on Wi-Fi, so it answers at once but the
  battery lasts hours rather than days.

| | A | C (top) | B |
|---|---|---|---|
| click | talk, click again to send · while it speaks: stop · on Options: back · in the hotspot: leave | Options (QR code) · on Options: update · no Wi-Fi: the setup hotspot | the last answer · next page · after the last: the dashboard |
| hold | | Wi-Fi & settings (the setup hotspot) | the dashboard now (on letting go) |
| hold 4 s | | | switch off: the capsized robot, about 20 s under a white glow, then off |
| double-click | hear the last answer again | | |

The power button switches it on again. Pressed twice, it cuts the power at once
and the paper keeps whatever it shows.

```bash
pio run -e papercolor_b -t upload
pio run -e preview_paper_b && uv run --no-project --with pillow python preview/render.py paper_b   # → preview/out/paper_b/
```

## Waveshare ESP32-S3-ePaper-1.54 (`epaper154`, `epaper154_touch`)

Two boards, one build each: SKU 32298 (`pio run -e epaper154`) and the Touch
version, SKU 34212 (`pio run -e epaper154_touch`). Same board otherwise:
ESP32-S3-PICO-1-N8R8 (the StickS3's module), a 200×200 black-and-white panel
(SSD1681), ES8311 codec with mic and speaker (the StickS3's too), SHTC3
temperature and humidity (not used: see below), PCF85063 clock, microSD, BOOT and PWR buttons, one
green LED. **Only the V2 board works** ("V2" on the back or top-left of the
PCB); a V1 has 4 MB flash and 2 MB PSRAM, too small for this firmware.

It's written from Waveshare's schematic and examples. Both boards have run it,
and the measurements below are the non-touch board's. M5Unified
doesn't know this board, so `hw_epaper154.cpp` skips
`M5.begin()` (its detection would probe these pins) and sets up M5Unified's
speaker, mic, button and I2C classes by hand. `ui_epaper154.cpp` is the screen,
`epd_ssd1681.cpp` the panel driver.

**The screen.** Far faster than the PaperColor's, so this one behaves like the
stick: a drawn menu, live status and working screens, answers that page with
the voice. Screens draw to a canvas at once; a background task sends the latest
one to the panel, only if it differs, as a partial refresh (no flash, 0.6 s). A
full refresh (1.8 s, flashes) clears what those leave behind:
after power-up, for QR codes, going to sleep after 5 or more partials, on the
home screen left alone for a minute after 20 or more, and on the next change
after 60 at the latest. The frame on the panel stays in RTC memory, so a wake
goes on from it with no flash. The serial log times every refresh
(`[paper] partial refresh done, 580 ms`).

**The PaperColor's trick.** Every full refresh starts from a reset pulse and
the controller's whole set-up. What it doesn't take over is cutting a running
waveform short for a new screen: the panel is left mid-change, so a full refresh
has to follow, and at 1.8 s that's no quicker than letting the first one finish.

| | `epaper154` | `epaper154_touch` |
|---|---|---|
| talk | BOOT: press, a blip, and 0.5 s later it listens; press again to send, and 0.5 s later a lower blip | the same, or the Talk button on the screen |
| wake | BOOT or PWR: only wakes it; the next BOOT press talks | the same |
| cancel | PWR while recording: dropped, nothing sent | the same |
| next | PWR: next row or page · hold = menu | the same, or swipe; in a list, tap the left or right third to move, the middle to choose |
| home | PWR: double-click, from any screen (not while it records or the agent works) | the same, or a double tap off the Talk button |
| replay | BOOT: double press, or Menu → *Replay answer* | the same |
| off | PWR held 4 s (on USB: deep sleep, since USB keeps it powered) | the same |
| Wi-Fi & settings | Menu → *Wi-Fi & settings* | the same |

There's no 2.5 s silence stop on this board: its mic's level ended recordings
mid-sentence, so only a press ends one, or the length limits (110 s online, 60 s
as an offline note).

The home screen is the clock and one big tile saying what BOOT does: a microphone
and "Boot to talk", or until it's ready, what it's waiting for. Hold BOOT for 1 s
for note mode: the tile turns black, with a notepad, "Boot to send", how many notes
there are on the notepad's corner and the recording time left under the words. The
turn in or out of note mode is a full refresh, so the old tile doesn't show through.
The line under the tile, the battery's time left or to full, follows the text size.

Holding the talk button at power-on can't open setup here as it does elsewhere:
BOOT held at power-on is the chip's download mode. No pin reports USB power or
charging, so they're inferred from USB traffic and the battery voltage (a plain
charger shows only once the battery is near full).

**The lights.** The red one belongs to the charger chip: its STAT pin drives it
directly, with no GPIO on the way, so it's on while charging and the firmware can
neither change nor read it. The green one is ours (G3, PWM). With one colour the
rhythm carries the meaning, and it's the one thing here that answers at once and
tells awake from asleep, since the panel keeps its picture either way:

| green light | means |
|---|---|
| on while a button is down | the press landed |
| steady, brighter with your voice | recording |
| breathing, a flick per step | the agent is working |
| flickering with the voice | speaking |
| two blinks every 5 s | something waits in Flight Deck |
| one blink every 10 s | awake, nothing to do |
| dark | asleep or off |
| three blinks every 5 s | battery under 15% |
| slow on and off, 1 s each | no Wi-Fi (its first minute) |
| two blinks every 2 s | Wi-Fi, but no Flight Deck or agent (its first minute) |
| quick blinks, then steady for 6 s | starting up, then connected |
| fast flicker | downloading an update |

"Low" under Status lights is half of "Bright".

**Asleep.** The sleeping screen keeps the clock going: a timer wakes the board
just after each minute turns, it redraws without Wi-Fi and sleeps again (Menu →
More → *Redraw asleep*, or the phone page: every minute, 2, 5, 10 or 30 min, 1 h,
3 h, never; at 2, 5 or 10 the time on screen is the last round one, like 19:25).
Under 10% battery it goes hourly.

Falling asleep, it goes to the home screen for 5 s, draws the sleeping screen and
waits 10 s with Wi-Fi still on (a press at either stage just goes back), and doesn't
go down within 5 s of a redraw. Status → *Woke* lists the last
boots that weren't the clock's usual tick, newest first, with the seconds since it
went to sleep ("PWR 240s  timer 12s  restart"). From the schematic, holding G17 high feeds
T2's base through 1K, so about 2.6 mA flows whenever it sleeps on battery,
whatever the clock does. The current hasn't been measured.

The picture doesn't depend on power, which is why the sleeping screen stays. On
battery the board stays on only while G17 is high; it's latched through deep
sleep, and cleared to switch off.

Measured on the non-touch board: the codec, clock and sensor all answer; the
clock keeps time; a full refresh takes 1803 ms, a partial one 580 ms; with one
tone playing, the mic hears the speaker (4600 against 1 with the amplifier off).
The mic's quiet peaks at about 450, under the 2400 that counts as a voice.

The temperature sensor isn't used. The board warms it: in a room of about 24 °C, on
USB, it read 28.8 °C after a long sleep and 32.7 °C awake, and a correction from
those numbers still read way off. So neither the screen nor the agent gets a room
reading (`kHasClimate` in `board.h`).

The power latch holds on battery: the sleeping clock keeps going there, and OTA
updates install. Still to check: which way up the touch is (Menu → More →
*Orientation*); ghosting (`kMaxPartials` and friends in `ui_epaper154.cpp`); current
in deep sleep. Put the factory image aside first: `esptool.py read_flash 0 0x800000
factory.bin`.

Screens on a Mac, both builds:

```bash
pio run -e preview_epaper154 -e preview_epaper154_touch
```

```bash
uv run --no-project --with pillow python preview/render.py epaper154
```

## StopWatch screen

The round screen is Captain's look (more below): black, a badge in the middle
with a name under it, page dots for lists. The ring
round the edge shows what the watch is doing — coral while it listens (filling
toward the 110 s recording limit), gold while the agent works (a tick per
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

The menu, the agent picker and the status pages take touch too: swipe left
for the next row or page, right for the one before, tap either side for the
same, tap the badge between them to choose. The "send voice
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
voice reaches the next one (until you page by hand), and with Captain on the
screen the sentence shows as a subtitle under him. No word timings are
needed: the watch knows how far into the speech the speaker is (the mouth
uses the same estimate); once the speech has all arrived, that fraction of
the text is the place, and before that the pace learned from the last answer
carries it (the deck says how many characters it speaks: `"spoken"` in the
reply). From 0.15.2 each sound also says which characters of the shown text
it covers (`"from"`/`"to"` on `audio_start`), so the lit sentence is exact at
every sentence and by its length within, and the pace is learned from each
sentence as it finishes — the estimate above only serves an older deck. After the speech, a small replay button sits under the text — tap it
to hear the answer again.

The answer comes as it's written (0.15.0). Flight Deck follows the model's
text as it streams and, each time a sentence completes, shows the answer so
far (`reply_part`) and reads that sentence aloud, so the first words are heard
while the rest is still being written; the complete answer (`reply`) replaces
the text in place and only what wasn't read yet is spoken. The sounds of one
answer follow each other without a cue (`"cont"` on `audio_start`), and the
closing chime waits for `speech_end`. Text the model writes before a tool call
isn't the answer: `speech_reset` takes the watch back to the working screen.
While the agent works, the button under the timer is a **stop**: the agent
stops after its current step.

A watch first (0.15.1): idle, the screen is a clock face — the date, the time
large, the agent's badge small with its name, the hint while it's new, the
battery as an arc at the bottom of the ring. **Raise to wake**: a wrist raise
lights a dark or dimmed screen, lowering the wrist on an idle screen puts it
out (the watch still sleeps on its *Power off* timer — set it to 3 min to stay
wakeable longer; a raise can't wake it from sleep, only the buttons can). The
background behind text screens is **frosted** — a light blur, then darkened —
so a busy photo doesn't fight the words; the picture-only screen keeps it
sharp.

0.16.0: an answer's page dots sit under the clock and its text starts lower,
clear of the buttons; the status is three
pages of gauges — the ring as the Wi-Fi signal with the deck and agent under
it, the ring as the battery with uptime and use, then the firmware, voice,
memory and notes; a replay, or play after a pause, goes to the page the
voice is on and follows it again.

0.16.1: on the StickS3 too, the page turns with the voice (no lit sentence on
its small screen), and a replay or play after a pause goes to the voice's
page; a text-size change re-wraps the answer and keeps the voice's page. A
recording's silence stop now measures "quiet" against the recording's own
quietest and loudest blocks, not a fixed level — the stick's mic sits above
that level even in a quiet room, so it never went quiet.

0.16.2: on the stick a press that wakes it only wakes it — the next press
talks (`board::kTalkFromSleep`; the watch still talks from sleep while the
button is held, the paper on its top button). Flight Deck now counts the
voice's places in UTF-8 bytes, as the firmware does, so letters like č and
đ no longer put the lit sentence and the page turns behind the voice.

0.17.0: the battery's time. There's no fuel gauge to ask, so the watch and the
stick keep the level over the clock — through sleeps too, in RTC memory, a
sample a minute awake and one before sleep — and once the window spans five
minutes and two per cent, say "≈ 2 h 10 min left" (or "full in ≈ 45 min"
while charging). The watch shows it under the date on the clock face
in place of the agent's name; the stick in its status, the watch on its
Power page too. A change of charging state starts the window over.

*Tap to talk* (Menu, or the settings page; off by default) puts a record
button on the screen: on the clock face, under an answer or under Captain. Tap it and talk; the button turns into a stop square, and a second
tap sends the recording (offline it's kept as a voice note). Holding the
yellow button works as before, and its press also ends a tapped recording.
From 0.20.1 nothing ends a recording by itself but the length limit (110 s):
you press or tap when you're done, however long you pause.

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

The look (0.20.0) is Captain's, from `build_support/brand/bot`: black, gold for what
leads, cyan for his eyes and voice, coral for the microphone, navy for the surfaces.

- **Clock face:** Captain above the time, on a dial of minute dots with the seconds going
  round in gold. His face and lamp are the agent's state: asleep, eyes shut with a blue
  lamp; not ready, sad; the date sits between the Wi-Fi and agent dots.
- **Listening:** Captain with wide eyes over the voice as bars, the ring in coral.
- **Working:** Captain's eyes scanning and his lamp pulsing, what you said, what the agent
  is doing, and the time and steps in a pill, the ring in gold.
- **An answer:** the pages are an arc on the right instead of dots.
- **Menu:** a navy disc with a gold rim, gold for the one in use, the next rows peeking in
  from the sides. The agent called Captain shows him in place of initials.
- **Notices and sleep:** Captain over the words, sad for a warning, asleep before sleep.

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

## Agent asleep or switched off

From 0.19.2 (paper B 1.0.22) the agent's dot says why its link is down.
Flight Deck adds `"power"` to `bound` and to the `connected` / `agent_offline`
states — `running`, `asleep` (Flight Deck put it to sleep for being idle) or
`stopped` (switched off, or no agent to bind) — and sends `agent_offline`
again when that changes while the link is down (`bus_client.py`).

- **Asleep: blue.** The agent dot on the paper, the watch and the stick, and
  the agent's name on the paper (the dashboard's *Agent* row, "· asleep") and
  on the watch's info page. The device counts as ready: talking wakes the
  agent, so no offline blink, no offline redraw and no link renewal for it.
  The paper shows it on its next five-minute dashboard.
- **Stopped: red.** The same places, "· stopped", and the watch's home says
  "The agent is stopped"; it counts as offline as before (the offline blink,
  the paper redrawn after 30 s).
- A deck that doesn't send `power`: yellow, as before.

## Waiting in Flight Deck

From 0.19.0 (paper B 1.0.20) the device tells you when something waits for
you in Flight Deck: a proposal from Autonomous Work waiting for your approval,
or a nudge. Flight Deck sends `{"type":"waiting","count":n,"text":"…","beep":true}`
(`owner_notify.py`) to every connected device of the owner who has *Devices*
switched on under Connections → Notifications. What came is also in Flight
Deck's bell, where a proposal is approved or rejected (in the kiosk too).

- **The beep**: two rising tones wider apart than the speech chime (E6 → C7),
  a double buzz on the StopWatch, two seconds of magenta on the PaperColor's
  lights, and a dark screen lit. Not over a recording, the voice or a turn
  under way: it waits for them to end. *Beeps around speech* (and "beep off")
  silences the tones, *Buzz* the motor. In quiet hours (the autonomy config's,
  UTC) the count comes without a beep.
- **The banner** (0.19.1, watch and stick): "Flight Deck", the line that came
  and "Open the bell", at once on the home screen (not over an answer, a menu
  or a question); a press goes back.
- **The count**: a magenta badge on every screen (0.19.1) — after the link dots
  in the watch's top row, before the battery in the stick's status bar — plus
  "n waiting in Flight Deck" in place of the home hint and a *Waiting* row. The
  B paper lists it with its links (red ink) and asleep under how to wake it;
  while proposals wait its idle blink is magenta on the right. Approving,
  rejecting or expiring them updates it without a beep; 0 clears it.
- **Asleep**: a device in deep sleep isn't connected, so it hears about it when
  it next wakes and connects — once, for things from the last 24 hours.

## Buttons

StickS3:

| | Front (A) | Side (B) |
|---|---|---|
| click | talk, click again to send (or 2.5 s of quiet) · while it speaks: stop | next page · back (a picture: its text) |
| hold | | menu (hold again closes) |
| double-click | hear the last answer again | (on an answer) hear it again |
| in the agent list | choose | move down |

StopWatch (0.18.0): press the yellow button once, talk, and press it again —
or just stop: 2.5 s of quiet sends it, as the on-screen record button does (it
was hold-to-talk). A press while it speaks stops the speech; a double press
plays the last answer again; a press that wakes the watch — from sleep or a
dark screen — only wakes it (0.18.1), the next one talks. The clock face is
the middle of three pages (0.18.1): swipe right for the background picture
alone, swipe left for the info page — the voice notes waiting and the room
for more, the Wi-Fi network and its signal, the Flight Deck address, the
agent — and swipe back the other way; up or down goes to the last answer. Hold the yellow button (1 s) for **note-taking
mode**, as on the PaperColor B: every recording is kept as a voice note (the
screen says "Note", the ring goes round once a minute, no quiet stop), the
clock face shows a red arc and "Note mode · hold yellow to send", and holding
it again turns the mode off and sends the notes to the agent as one turn (no
asking). Notes — in note mode or offline — are written to flash as they're
recorded, so there's no 1-minute limit, only the room left. The other button
works like the StickS3's side button.

**Replay.** Double-click the side button on an answer (its text or its picture), or
choose Menu → *Replay answer* (the first row, while there's an answer with sound), to
hear the last answer again, without the cue. The answer's sound (its speech, then
any sound the agent made) plays through a ring in memory (~68 s at 24 kHz) with no
limit on its length (0.18.2 / paper B 1.0.17; it used to stop at ~66 s): the device
tells Flight Deck to hold (`audio_hold`) while the ring has as much still to play as
it can keep — a long answer arrives faster than it's spoken, and a pause holds it —
and to go on (`audio_go`) once some has played; the deck keeps taking the voice from
TTS meanwhile. An answer that fits the ring replays at once from it, with no network;
a longer one — or one that never all arrived (you stopped it early, or the link
dropped) — is sent again by Flight Deck (`{"type":"replay"}`; the speech is
synthesized afresh). A deck that doesn't know `audio_hold` sends it all: the device
plays as much as the ring holds and ends there, as before. Powering off forgets it.
(Mac check: `test/host/audio_ring.cpp` runs `src/audio.cpp` on a fake speaker.)
Flight Deck still speaks at most ~1500 characters of an answer (`MAX_SPOKEN_CHARS`);
the rest is shown, not read.
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

Display → *Captain* shows Captain, the robot from the app icon, instead of text: he blinks,
listens, thinks and speaks with his grille (`src/face.cpp`).

Menu → *Text size* (small / medium / large) sets the reading text: replies, the working
screen, notices, the menu and the status page (which then spans pages: B = next page, after
the last one back to the main screen). The top bar stays compact.

Hold the front button while switching on to reopen setup. With no Wi-Fi
saved, setup opens by itself; with none in reach for 30 s, it carries on
offline and keeps voice notes. (Pressing it to wake from sleep is talking,
not setup. See *Talk from sleep*.)

## Sound by voice

From 0.17.1 (paper B 1.0.1) a recording that is only a sound command changes
the device instead of going to the agent — Flight Deck spots it
(`sound_command()` in `device_bridge.py`, for devices that connect with
`cmds=sound`) and the device says the result back at the new volume:

- **"volume up" / "volume down"** (or "louder", "quieter", "pojačaj", "tiše"…):
  a quarter at a time, 25 % to 100 %.
- **"beep on" / "beep off"**: every beep at once — before speech, talking,
  key clicks.

Anything longer ("turn up the volume on the kitchen speaker") still goes to the
agent.

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
pio run -e sticks3 -t upload      # or: -e stopwatch / -e papercolor / -e papercolor_b
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
up; without it no device accepts a new build) and uploads (admin token). The
`papercolor_b` env publishes as the `papercolor` kind (`--hw` to say otherwise):
the paper takes whichever version is newest, so the B builds are 1.x. Connected
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
