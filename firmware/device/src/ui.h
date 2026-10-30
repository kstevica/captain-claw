// The screen. Each device has its own implementation of this interface:
//   ui_stick.cpp  StickS3: 240×135 landscape LCD, efont, trimmed efontJA (has č ć š ž đ)
//   ui_round.cpp  StopWatch: 466×466 round AMOLED, smooth Inter fonts
//   ui_paper.cpp  PaperColor: 400×600 colour e-paper, Inter; slow to redraw, so
//                 show…() only asks for a screen and tick() draws it once settled
//   ui_epaper154.cpp  Waveshare 1.54": 200×200 black-and-white e-paper, efont; a
//                 background task refreshes only what changed
// All draw into an off-screen canvas and push it whole (no flicker).
//
// Colours are RGB565 (uint16_t) throughout — M5GFX reads a uint32_t colour
// as RGB888, so passing TFT_* constants through uint32_t draws wrong colours.
#pragma once

#include <Arduino.h>

#include <vector>

#include "face.h"

namespace ui {

constexpr uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) {
  return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

// Health of a link, shown as a colour block (stick) or dot (round, paper).
// The agent's link also reads Asleep (blue: Flight Deck put it to sleep for
// being idle, the next word wakes it) or Off (red as Down: it's switched off).
enum Health : uint8_t { Down = 0, Partial = 1, Up = 2, Asleep = 3, Off = 4 };

void begin();
void brightness(uint8_t level);

// Every loop pass: a screen that draws later (e-paper) does it from here.
void tick();
// Until what was asked last is on the screen (e-paper: a redraw, up to ~30 s).
void flush();
// The screen is redrawing (e-paper), for the status lights.
bool busy();
// When the device was last in use (millis(): a press, talking, an answer
// playing). A slow screen (e-paper) holds its redraws until it has been
// left alone for a while.
void idleSince(uint32_t ms);
// The user asked to see something (a page, the answer, the status): the
// next screen asked for is drawn without waiting for a quiet spell.
void drawNow();
// A screen is waiting for the device to be left alone (e-paper).
bool waiting();
// Screen orientation 0..3 (e-paper; the others have one way up).
void setRotation(uint8_t rotation);

// True when screens animate on their own (the round screen's ring and
// timers): the main loop then redraws them at frame rate.
bool animated();

// Full-screen notices.
void showSetup(const char* reason, const char* apName, const char* ip);
void showMessage(const char* title, const String& body, uint16_t color = 0xFFFF);
// A yes/no question: the talk button answers `yes`, the side (or back)
// button `no` — the screen names them.
void showAsk(const char* title, const String& body, const char* yes, const char* no);
// An address to open on a phone: a QR code, the address and a note.
void showLink(const char* title, const String& url, const String& note);
// What the screen shows while the device sleeps (e-paper keeps it, with the
// character on it when there's no answer to keep).
void showSleeping(const String& hint, uint8_t character);

// Status bar: Wi-Fi + agent health, agent name, state, battery (redrawn with every view).
void setBar(const String& agent, const char* state, uint16_t stateColor);
void setHealth(Health wifi, Health agent);

// One labelled line on the status page / idle panel.
struct Row {
  String label;
  String value;
  uint16_t color;
};
// Draws page `page` of the rows (in the chosen text size); returns the page
// count. The title shows on the first page only.
int showPanel(const char* title, uint16_t titleColor, const std::vector<Row>& rows, int page = 0);

// "2 h 10 min", "45 min", "3 d 4 h".
inline String timeSpan(int minutes) {
  if (minutes < 60) return String(max(1, minutes)) + " min";
  if (minutes < 24 * 60) {
    const int h = minutes / 60, m = minutes % 60;
    return m ? String(h) + " h " + String(m) + " min" : String(h) + " h";
  }
  const int d = minutes / (24 * 60), h = (minutes / 60) % 24;
  return h ? String(d) + " d " + String(h) + " h" : String(d) + " d";
}

// What the status screens show. The round screen draws it as gauges over
// three pages; the others as rows (statusRows).
struct Status {
  bool wifi = false;
  String ssid, ip;
  int rssi = 0;
  bool deck = false;
  String deckHost;
  int rttMs = -1;
  String agent;
  bool agentLinked = false;
  bool agentAsleep = false, agentOff = false;  // its link down: asleep (wakes for a word) / switched off
  int batteryPct = -1, batteryMv = 0;
  bool charging = false, usb = false;
  int minutesLeft = -1, minutesToFull = -1;  // from the level over the clock (-1: not known yet; 0 to full: full)
  String uptime;
  String wake;  // why it last woke, where the board can tell ("" = not shown)
  int turns = 0, lastAnswerMin = -1;
  bool climate = false;
  float tempC = 0, humidity = 0;
  String voice;
  int volumePct = 0, boostDb = 0;
  bool speak = true;
  uint8_t textSize = 0;
  String deviceId, fw;
  uint32_t freeKb = 0;
  int notes = 0, notesRoomMin = 0;
  String notesLength;
  bool notesSending = false;
  String update;  // a newer version on offer ("" none)
};
int showStatus(const Status& s, int page);  // returns the page count

inline std::vector<Row> statusRows(const Status& s) {
  const uint16_t green = rgb(40, 200, 90), yellow = rgb(250, 200, 40), red = rgb(230, 50, 50),
                 text = rgb(230, 230, 230), cyan = rgb(60, 210, 240), blue = rgb(70, 150, 255);
  std::vector<Row> rows;
  rows.push_back({"Wi-Fi", s.wifi ? s.ssid + "  " + String(s.rssi) + " dBm" : String("not connected"),
                  s.wifi ? (s.rssi < -80 ? yellow : green) : red});
  rows.push_back({"IP", s.wifi ? s.ip : String("-"), text});
  rows.push_back({"Deck", s.deckHost + (s.deck ? (s.rttMs >= 0 ? "  " + String(s.rttMs) + " ms" : String("  ok")) : String("  offline")),
                  s.deck ? green : red});
  rows.push_back({"Agent", (s.agent.length() ? s.agent : String("none")) +
                               (s.agentLinked ? "" : s.agentAsleep ? "  (asleep)" : s.agentOff ? "  (stopped)" : "  (waiting)"),
                  s.agentLinked ? green : s.agentAsleep ? blue : s.agentOff ? red : (s.deck ? yellow : red)});
  char volts[16];
  snprintf(volts, sizeof(volts), "%.2f V", s.batteryMv / 1000.0f);
  String time = s.charging ? (s.minutesToFull == 0 ? String("  full") : s.minutesToFull > 0 ? "  full in ~" + timeSpan(s.minutesToFull) : String(""))
                           : (s.minutesLeft >= 0 ? "  ~" + timeSpan(s.minutesLeft) + " left" : String(""));
  rows.push_back({"Battery", String(s.batteryPct) + "%  " + volts + (s.charging ? "  charging" : s.usb ? "  on USB" : "") + time,
                  s.batteryPct >= 40 ? green : s.batteryPct >= 20 ? yellow : red});
  if (s.climate) {
    char room[32];
    snprintf(room, sizeof(room), "%.1f °C  %d%% humidity", s.tempC, (int)(s.humidity + 0.5f));
    rows.push_back({"Room", room, text});
  }
  rows.push_back({"Voice", s.voice + "  vol " + String(s.volumePct) + "%" +
                               (s.boostDb ? "  +" + String(s.boostDb) + " dB" : String("")) + (s.speak ? "" : "  muted"),
                  text});
  rows.push_back({"Usage", String(s.turns) + " turn" + (s.turns == 1 ? "" : "s") +
                               (s.lastAnswerMin >= 0 ? "  last " + String(s.lastAnswerMin) + "m ago" : String("")) +
                               "  up " + s.uptime,
                  text});
  if (s.wake.length()) rows.push_back({"Woke", s.wake, text});
  rows.push_back({"Device", s.deviceId + "  fw " + s.fw + "  " + String(s.freeKb) + "k free", text});
  if (s.notes) {
    rows.push_back({"Offline notes", String(s.notes) + " waiting  " + s.notesLength + "  room " + String(s.notesRoomMin) +
                                         " min" + (s.notesSending ? "  sending…" : ""),
                    yellow});
  }
  if (s.update.length()) rows.push_back({"Update", s.update + " available (menu → Software update)", cyan});
  return rows;
}

// The main screen when nothing is going on (text mode). The stick shows the
// title and rows as a panel; the round screen shows the agent's badge with
// the hint under it.
struct Home {
  bool ready;
  const char* title;       // "All good" / "Getting ready"
  uint16_t color;          // the title's / hint's colour
  String hint;             // "hold the yellow button to talk", "waiting for agent"…
  std::vector<Row> rows;   // device, Wi-Fi, deck, agent
  uint8_t character = 0;   // e-paper: shown when there's no answer to keep
  String battery;          // the round screen: "≈ 2 h 10 min left" / "full in ≈ 45 min" under the character
  bool noteMode = false;   // the B paper: note-taking mode is on
};

void showHome(const Home& home);
#if defined(BOARD_STOPWATCH)
// The watch's info page — swipe left from the clock face (ui_round.cpp): a
// tile a row, its value by a coloured dot, what it is under it.
void showInfo(const std::vector<Row>& rows);
#endif
#if defined(PAPER_B)
// The B paper (ui_paper_b.cpp): seconds until its dashboard is next due
// (0: now) — the sleep timer is set to land on it.
uint32_t dashboardDueInSec();
// The answer's voice has started (or it came with speaking off): on the
// paper now. Then it's whole and the voice is over: kept, and redrawn if it
// grew since.
void answerStarted();
void answerEnded();
// A on the options: back to the page they were opened from.
void leaveOptions();
// The links' state for the dashboard (its LINKS rows), without drawing.
void setHome(const Home& home);
// The links have been down a while (or came back after that): what the
// paper shows is drawn again with them.
void linksChanged();
// A recording is going on: nothing is redrawn by itself (the five-minute
// dashboard waits) until it ends.
void hold(bool recording);
// Where the notes are kept changed (a microSD card put in or taken out).
void notesChanged();
#endif

#if defined(BOARD_EPAPER154)
// The panel's controller to sleep, before the device's deep sleep (the
// picture stays; ui_epaper154.cpp keeps it, for the next refresh to start from).
void panelSleep();
// Switched off: the capsized robot, and how to switch it back on.
void showSwitchedOff(const String& hint);
#endif

// How many proposals wait in Flight Deck, as last heard (0: none): a magenta
// badge on every screen — the watch's top row, the stick's status bar — and
// on the B paper's dashboard (asleep, under how to wake it).
void setWaiting(int count);

// Conversation views (text mode).
void showListening(int level, uint32_t elapsedMs);
// While the agent works: what it heard, its latest activity lines (oldest
// first), the last narration / phase line, and the time and tool steps so far.
void showWorking(const String& heard, const std::vector<String>& activity, const String& status,
                 uint32_t elapsedMs, int steps);
void setReply(const String& text);  // wraps into pages (shown by showReply)
// What the user asked, shown over the answer where there's room (e-paper).
void setQuestion(const String& text);
// Keep the current answer on the home screen (e-paper), or let it go.
bool canPin();
bool pinned();
void pin(bool on);
// The answer kept in flash from before a sleep (e-paper) becomes the current
// one again; false when there's none (or the device keeps none).
bool restoreAnswer();
// Reading text size: 0 small, 1 medium, 2 large (re-wraps the current reply).
void setTextSize(uint8_t size);
void showReply();                   // redraw the current page
bool nextPage();                    // false when it wrapped back to page 1
bool prevPage();                    // false on page 1 (stays there)
bool onLastPage();                  // the current page is the answer's last
bool onFirstPage();
// The voice's place in the answer, as a character index into its text (-1:
// not speaking): the sentence there is lit and the rest dimmed, the
// character's subtitle shows it, and the page turns to follow it unless the
// user paged by hand. True when the screen needs drawing again (the
// sentence or the page changed).
bool setSpoken(int at);
// The page turns with the voice again, from where it is now (a replay, or
// play after a pause and paging by hand).
void followVoice(int at);
// More of the answer (it streams in): the text so far, keeping the page and
// the voice's place.
void growReply(const String& t);
// A small replay button under an answer that has sound; tap = hear it again.
void setReplayable(bool on);
bool replayHit(int x, int y);
// How the next screen drawn comes in (the round screen; the others cut):
// sliding on from the right (forward) or back from the left, or fading
// through black. One-shot: the next push uses it.
void slide(bool forward);
void fade();
bool hasReply();

// While an answer plays (text mode), every frame or so: a light indicator
// that follows the voice, `level` 0..100 (round: a short line at the bottom,
// only that strip redrawn). The others show nothing.
void showSpeaking(int level);

// Icon mode: the character in a pose.
void showFace(uint8_t kind, face::Pose pose, int level);
// Menu → Character: the character being previewed, `index` of KindCount.
void showCharPick(uint8_t kind, face::Pose pose, int level, int index);

// Pictures (full screen, with a small health/battery overlay on the stick).
// Takes the bytes over (they were ps_malloc'ed) and frees the last picture.
void storeImage(uint8_t* jpg, size_t len);
bool hasImage();
void showStoredImage();
// A picture bigger than the round screen fills it; a tap shows all of it,
// scaled down, and the next tap fills again.
void toggleImageFit();
// A picture behind every screen (the round one; the others ignore it): a JPEG
// of the screen's size, drawn darkened under the screens. nullptr / 0 clears it.
void setBackground(const uint8_t* jpg, size_t len);
bool hasBackground();
// The background picture alone, as it is, nothing over it.
void showBackdrop();

// One entry of a menu or list. The stick shows `label` as one line; the
// round screen shows a card: `badge` big in a glowing badge (a value like
// "75%"), or a glyph — Initials turns `badge` (a name) into its initials —
// with `title` under it, then `note`.
enum class Glyph : uint8_t { None, Initials, Wifi, Back, Info, Replay, Gear };
struct Item {
  String label;
  String title;
  String badge;
  String note;
  Glyph glyph = Glyph::None;
  bool on = true;  // false: a switched-off setting (a dim badge)
};
void showPicker(const std::vector<Item>& items, int selected, int current, const char* title = "");

// Attention flash before speech: on = the screen (stick) / ring (round) is white.
void flashFrame(bool on);

// Tap to talk (touch screens, an option): a record button on the main
// screens (home, answer, character), a stop button while a recording that
// was tapped on listens.
// Talk / Stop: Tap to talk's record and stop buttons. Pause / Play: an
// answer's sound, on the answer screen.
enum class TapButton : uint8_t { Off, Talk, Stop, Pause, Play };
void setTapButton(TapButton button);
// A touch at (x, y) lands on the button on the screen now.
bool tapButtonHit(int x, int y);
// The yes/no pills of showAsk: 1 = yes, 0 = no, -1 = neither (or no ask showing).
int askHit(int x, int y);

}  // namespace ui
