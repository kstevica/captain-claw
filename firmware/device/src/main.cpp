// Captain Claw voice client for the M5Stack StickS3, StopWatch and PaperColor
// (board.h).
//
//   Talk      (StickS3 front, StopWatch yellow, PaperColor C on top)
//             hold = talk, release = send · tap = stop speaking / flip picture↔text
//             in the menu: choose / change the row
//   Side      (StickS3 side, StopWatch the other one, PaperColor A)
//             click = next page / next row · hold = menu (hold again = close)
//             double-click on an answer = hear it again (also Menu → Replay answer)
//   Back      (PaperColor B only)
//             click = an answer: the home screen · home: the answer · else back
//             hold = Wi-Fi & settings (the setup hotspot; hold B again there to leave)
//   StickS3 and PaperColor: the talk button isn't held — a press starts the
//   recording and the next press (or 2.5 s of quiet) sends it; a quick double
//   press replays.
//   Hold the talk button while switching on (or no Wi-Fi saved) = setup hotspot
//   Asleep: hold the talk button and talk at once — it records while it reconnects
//   (StickS3, PaperColor: press it and talk)
//
// Talks to Flight Deck's /device/ws (captain_claw/flight_deck/device_bridge.py):
// 16 kHz PCM up, with the device's readings (telemetry.h) on each turn; reply
// text, 24 kHz PCM speech and JPEG pictures down.

#include <ArduinoJson.h>
#include <LittleFS.h>
#include <M5Unified.h>
#include <WebSocketsClient.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <driver/gpio.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>
#include <esp_timer.h>

#include "audio.h"
#include "board.h"
#include "face.h"
#include "leds.h"
#include "notes.h"
#include "ota.h"
#include "portal.h"
#include "settings.h"
#include "telemetry.h"
#include "ui.h"

#if defined(BOARD_PAPERCOLOR)
#include <M5PM1.h>
#endif

#define FW_VERSION "0.14.2"

// The loop task's stack (Arduino's default is 8 KB). The smooth fonts on the
// round and paper screens draw each glyph through a stack buffer — a timer
// digit is ~5 KB — and screens are often drawn from inside the WebSocket
// library's nested callbacks, already deep: 8 KB overflowed on the StopWatch.
SET_LOOP_TASK_STACK_SIZE(20 * 1024);

// The talk, next and back buttons (board.h: the PaperColor talks on its top
// button, C; the others on A).
static m5::Button_Class& m5Button(char k) { return k == 'C' ? M5.BtnC : k == 'B' ? M5.BtnB : M5.BtnA; }
static m5::Button_Class& talkBtn() { return m5Button(board::kTalkM5); }
static m5::Button_Class& sideBtn() { return m5Button(board::kSideM5); }
static m5::Button_Class& backBtn() { return m5Button(board::kBackM5); }  // only where kBackPin >= 0

// Least time between two frames of an animation. The round screen's frames
// cost ~50 ms each: at the stick's pace the loop would only ever draw, and
// buttons and the speech stream would wait.
static uint32_t frameGap(uint32_t ms) { return board::kRound ? max<uint32_t>(ms, 120) : ms; }

#ifdef CLAW_PERF
// Timing, one serial line a second (a build flag, not in normal builds).
namespace ui {
extern uint32_t g_perfDrawUs, g_perfDrawMax, g_perfPushUs, g_perfPushMax, g_perfFrames;
}
namespace audio {
uint32_t g_perfUnderruns = 0;
}
using audio::g_perfUnderruns;
using ui::g_perfDrawMax;
using ui::g_perfDrawUs;
using ui::g_perfFrames;
using ui::g_perfPushMax;
using ui::g_perfPushUs;
static uint32_t pfLoops = 0, pfLoopMax = 0, pfWsMax = 0, pfUpdMax = 0, pfTelMax = 0, pfGapMax = 0, pfLastPoll = 0,
                pfPrintMs = 0;
#define PERF_T(var, stmt)              \
  do {                                 \
    uint32_t _t = micros();            \
    stmt;                              \
    uint32_t _d = micros() - _t;       \
    if (_d > var) var = _d;            \
  } while (0)
static void perfReport(uint32_t loopUs) {
  ++pfLoops;
  pfLoopMax = max<uint32_t>(pfLoopMax, loopUs);
  if (millis() - pfPrintMs < 1000) return;
  pfPrintMs = millis();
  uint32_t f = max<uint32_t>(1, g_perfFrames);
  Serial.printf("[perf] loops %u max %u ms | frames %u draw avg %u max %u | push avg %u max %u | ws %u upd %u tel %u"
                " | feed gap %u ms | dry %u | playing %d\n",
                pfLoops, pfLoopMax / 1000, g_perfFrames, g_perfDrawUs / f / 1000, g_perfDrawMax / 1000,
                g_perfPushUs / f / 1000, g_perfPushMax / 1000, pfWsMax / 1000, pfUpdMax / 1000, pfTelMax / 1000,
                pfGapMax / 1000, g_perfUnderruns, (int)audio::playing());
  pfLoops = pfLoopMax = pfWsMax = pfUpdMax = pfTelMax = pfGapMax = 0;
  g_perfDrawUs = g_perfDrawMax = g_perfPushUs = g_perfPushMax = g_perfFrames = g_perfUnderruns = 0;
}
#else
#define PERF_T(var, stmt) stmt
#endif

// Home: the PaperColor's home screen even while there's an answer (C hold).
// Link: the "settings on your phone" address.
// Ask: a yes/no question (send the voice notes kept offline?).
// Backdrop: the background picture alone (a double tap on the home screen).
enum class Mode { Idle, Listening, Waiting, Reply, Image, Menu, Agents, CharPick, Status, Notice, Home, Link, Ask,
                  Backdrop };

static WebSocketsClient ws;
static WiFiMulti wifiMulti;
static bool wsUp = false;
static bool agentLinked = false;  // Flight Deck's link to the agent is up
static ota::Offer offer;           // a newer build Flight Deck offers (docs/device-ota-plan.md)
static String offerAnnounced;      // version the user has been told about
static bool offerAutoTried = false;
static bool probation = false;     // this boot is a fresh update that hasn't reached the deck yet
static bool checkingUpdate = false;
static Mode mode = Mode::Idle;
static String agentName, agentSlug;
static String heard, statusLine;
static std::vector<String> activity;  // what the agent is doing, oldest first
static int toolSteps = 0;
static uint32_t workStartMs = 0;
static bool workDirty = false;
// The link dropped while the agent was working: Waiting stays, as Flight Deck
// keeps the turn and its answer for the reconnect — until kTurnHoldMs without
// the link, or kResumeMs after a reconnect that doesn't pick the turn up.
static constexpr uint32_t kTurnHoldMs = 120000;
static constexpr uint32_t kResumeMs = 8000;
static uint32_t turnDropMs = 0, turnRelinkMs = 0;  // 0 = no drop / not back yet
static uint32_t recStartMs = 0, lastActivityMs = 0, lastDrawMs = 0, menuOpenedMs = 0;
static bool dimmed = false, screenOff = false;
static int menuSel = 0;
static std::vector<String> agentNames, agentSlugs;
static int agentSel = 0, agentCur = -1;
static bool agentsLoading = false;
static uint8_t charSel = 0;  // character being previewed
static int statusPage = 0, statusPages = 1;
static uint32_t flashStartMs = 0;
static bool flashing = false;
static int rttMs = -1;
static uint32_t lastPingMs = 0, lastReplyMs = 0, turns = 0;
static uint8_t* imgPart = nullptr;  // a picture arriving in parts (PSRAM)
static size_t imgPartLen = 0;
// The background picture (round screens): Flight Deck sends the owner's when
// the one the watch names in "prefs" isn't it (flight_deck/device_backgrounds.py).
static constexpr const char* kBgFile = "/background.jpg";
static constexpr const char* kBgIdFile = "/background.id";
static String bgId;                 // the one shown ("" none)
static String bgIncomingId;         // the one arriving in parts (0x04 … 0x05)
static size_t bgIncomingSize = 0;
static uint8_t* bgPart = nullptr;
static size_t bgPartLen = 0;
static uint32_t talkTapMs = 0;      // when the talk button was last tapped (double-click = replay)
static uint32_t buzzOffMs = 0;       // vibration motor runs until then (0 = off)
// More pulses after the one running (a double tick, the error's three).
static uint8_t buzzMore = 0;
static uint16_t buzzOnMs = 0, buzzGapMs = 0;
static uint32_t buzzAgainMs = 0;
static uint32_t pendingTapMs = 0;    // a single tap on a picture acts once it can't become a double tap
// Following the voice through the answer's text: how much of it the deck
// speaks, how long the text is, and the speaking pace learned from the last
// answer whose speech had all arrived (characters a second).
static int spokenChars = 0, replyChars = 0;
static float charsPerSec = 14.0f;
static bool paceLearned = false;
static constexpr uint16_t kHintTurns = 5;  // the home screen's "hold … to talk" goes after this many
static uint32_t turnBusyMs = 0;      // last moment a turn was under way (Wi-Fi stays fully awake)
static bool wifiAwake = false;
static bool deckReplay = false;      // asked the deck to send the last answer again
static uint32_t deckReplayMs = 0;
static bool quietSpeech = false;     // replayed speech comes without the cue
static uint32_t closeAtMs = 0;       // the closing chime, a moment after the answer's last sound (0 = none)
static uint32_t sideClickMs = 0;     // a side click on an answer, waiting for a second one (0 = none)
static Mode sideClickMode = Mode::Idle;
static bool sideClickWoke = false;   // that click only woke the screen

// Talking from sleep: the talk button woke the device and is still held. The
// mic records into PSRAM at once (M5Unified fills the buffer by itself) while
// Wi-Fi and the deck connect; then it all goes up, as if spoken just now.
static constexpr uint32_t kPrerollMaxMs = 30000;   // a wake-to-talk recording
static constexpr uint32_t kOfflineMaxMs = 60000;   // a voice note kept offline (notes.h)
static uint32_t prerollMaxMs = kPrerollMaxMs;      // this recording's cap
static bool prerollOffline = false;                // recorded knowing the deck can't be reached
static constexpr uint32_t kPrerollMinMs = 1000;    // less than this recorded is a press that woke it
static constexpr int kPrerollQuietPeak = 900;      // a short recording that never got louder is too
static int16_t* preroll = nullptr;
static size_t prerollCap = 0, prerollSent = 0;  // in samples
static uint32_t prerollStartMs = 0;
static volatile uint32_t prerollEndMs = 0;     // when the button was let go (0 = still held)
static bool prerollOn = false, prerollStreaming = false, prerollChecked = false;
static bool prerollEndSounded = false;  // its "let go" blip has played

// Tap to talk: the recording going on was started on the screen (or, where
// the talk button toggles, by a press of it), so it ends with a second tap
// or press, not by letting go of a button.
static volatile bool tapRec = false;
static volatile bool talkWasDown = true;  // the button sampler's last look at the talk button

// Voice notes kept offline are asked about once the deck is back (aloud and
// on the screen) and go up on a yes. A no holds until there are more notes
// or a cold start; Menu → Send voice notes still sends them.
static bool upAllowed = false;  // said yes, on this link
static bool upAsked = false;    // asked, on this link
static uint32_t askedMs = 0;
RTC_DATA_ATTR static int notesDeclined = -1;  // how many were waiting at the no

// A menu row to read aloud (e-paper), once the presses settle.
static String sayText;
static uint32_t sayAtMs = 0;
static bool asideOpen = false;  // the deck is sending a line spoken aside

// The agent's name survives a deep sleep for the sleeping screen.
RTC_DATA_ATTR static char rtcAgent[48] = "";

static constexpr uint32_t kTapMs = 300;      // shorter press = tap, not talk
static constexpr uint32_t kDoubleMs = 350;   // a second click within this = double-click
static constexpr uint32_t kDimLeadMs = 2000;  // dims this long before the screen goes off
static constexpr uint8_t kBright = 128;
static constexpr uint32_t kFlashMs = 700;    // white / normal, every 100 ms
static constexpr uint16_t kGreen = ui::rgb(40, 200, 90), kRed = ui::rgb(230, 50, 50);
static constexpr uint16_t kYellow = ui::rgb(250, 200, 40), kCyan = ui::rgb(60, 210, 240);
static constexpr uint16_t kOrange = ui::rgb(255, 150, 40), kText = ui::rgb(230, 230, 230);
static constexpr size_t kMaxImage = 256 * 1024;
static constexpr uint64_t kButtonPins = (1ULL << board::kTalkPin) | (1ULL << board::kSidePin) |
                                        (board::kBackPin >= 0 ? (1ULL << board::kBackPin) : 0);

// ── helpers ──────────────────────────────────────────────────────────

static String urlEncode(const String& s) {
  String o;
  const char* hex = "0123456789ABCDEF";
  for (uint8_t c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      o += (char)c;
    } else {
      o += '%';
      o += hex[c >> 4];
      o += hex[c & 15];
    }
  }
  return o;
}

static void touch() {
  lastActivityMs = millis();
  if (dimmed || screenOff) {
    ui::brightness(kBright);
    dimmed = screenOff = false;
  }
}

// A short buzz of the vibration motor (StopWatch); the loop switches it off.
static void buzz(uint16_t ms, uint8_t level = 200) {
  if (!board::kHasMotor) return;
  M5.Power.setVibration(level);
  buzzOffMs = millis() + ms;
  if (!buzzOffMs) buzzOffMs = 1;
  buzzMore = 0;
  buzzAgainMs = 0;
}

// The haptic vocabulary — one motor, so it's all timing: a tick for a step
// or a page turn, a double tick at the end of a list, a longer one for a
// choice, three for something that went wrong.
static void buzzTicks(uint8_t n, uint16_t onMs, uint16_t gapMs) {
  if (!board::kHasMotor || !n) return;
  buzz(onMs);
  buzzMore = n - 1;
  buzzOnMs = onMs;
  buzzGapMs = gapMs;
}
static void hapticTick() { buzzTicks(1, 20, 0); }
static void hapticBump() { buzzTicks(2, 20, 70); }
static void hapticConfirm() { buzzTicks(1, 45, 0); }
static void hapticError() { buzzTicks(3, 30, 80); }

static void sendJson(const JsonDocument& doc) {
  String out;
  serializeJson(doc, out);
  ws.sendTXT(out);
}

static void sendType(const char* type) {
  JsonDocument d;
  d["type"] = type;
  sendJson(d);
}

static void sendPrefs() {
  JsonDocument d;
  d["type"] = "prefs";
  d["voice"] = settings.femaleVoice ? "female" : "male";
  d["speak"] = settings.speak;
  if (board::kRound) d["bg"] = bgId;  // the background it shows: the deck sends the owner's if it differs
  sendJson(d);
}

static String serverHost() {
  String h = settings.server;
  int p = h.indexOf("://");
  if (p >= 0) h = h.substring(p + 3);
  int slash = h.indexOf('/');
  return slash < 0 ? h : h.substring(0, slash);
}

static String uptime() {
  uint32_t m = millis() / 60000;
  return m >= 60 ? String(m / 60) + "h " + String(m % 60) + "m" : String(m) + "m";
}

static ui::Health wifiHealth() {
  if (WiFi.status() != WL_CONNECTED) return ui::Down;
  return WiFi.RSSI() < -80 ? ui::Partial : ui::Up;
}

static ui::Health agentHealth() {
  if (!wsUp) return ui::Down;
  return agentLinked ? ui::Up : ui::Partial;
}

// "12s · 3 steps" while the agent runs.
static const char* workLabel() {
  static char buf[32];
  uint32_t secs = (millis() - workStartMs) / 1000;
  if (toolSteps) {
    snprintf(buf, sizeof(buf), "%lus · %d step%s", (unsigned long)secs, toolSteps, toolSteps == 1 ? "" : "s");
  } else {
    snprintf(buf, sizeof(buf), "working %lus", (unsigned long)secs);
  }
  return buf;
}

static void startWork() {
  activity.clear();
  toolSteps = 0;
  workStartMs = millis();
}

static void setBarState() {
  ui::setHealth(wifiHealth(), agentHealth());
  const char* st;
  uint16_t col;
  if (!wsUp) {
    st = "offline", col = kRed;
  } else if (mode == Mode::Listening) {
    st = "listening", col = kRed;
  } else if (audio::paused()) {
    st = "paused", col = kText;
  } else if (audio::playing()) {
    st = "speaking", col = kCyan;
  } else if (mode == Mode::Waiting) {
    st = workLabel(), col = kYellow;
  } else if (mode == Mode::CharPick) {
    static char buf[24];
    snprintf(buf, sizeof(buf), "%d/%d  A=pick", charSel + 1, (int)face::KindCount);
    ui::setBar(face::name(charSel), buf, kCyan);
    return;
  } else if (mode == Mode::Menu || mode == Mode::Agents) {
    st = "menu", col = kText;
  } else if (mode == Mode::Status) {
    static char buf[16];
    if (statusPages > 1) snprintf(buf, sizeof(buf), "status %d/%d", statusPage + 1, statusPages);
    else snprintf(buf, sizeof(buf), "status");
    st = buf, col = kText;
  } else {
    st = agentLinked ? "ready" : "no agent", col = agentLinked ? kGreen : kYellow;
  }
  ui::setBar(agentName, st, col);
}

// ── status page / idle panel ─────────────────────────────────────────

// "1:24"
static String clockLength(uint32_t secs) {
  char len[12];
  snprintf(len, sizeof(len), "%lu:%02lu", (unsigned long)(secs / 60), (unsigned long)(secs % 60));
  return len;
}

// "3 voice notes"
static String notesCount() {
  const int n = notes::count();
  return String(n) + (n == 1 ? " voice note" : " voice notes");
}

// The tap button the screens draw: none, record, or stop (a tapped recording).
// The button on the screen (StopWatch): pause / play under an answer while
// its sound plays, else Tap to talk's record / stop button.
static ui::TapButton tapButtonKind() {
  if (!board::kHasTouch) return ui::TapButton::Off;
  if ((mode == Mode::Reply || mode == Mode::Idle) && audio::playing() && !tapRec) {
    return audio::paused() ? ui::TapButton::Play : ui::TapButton::Pause;
  }
  if (!settings.tapTalk) return ui::TapButton::Off;
  return tapRec ? ui::TapButton::Stop : ui::TapButton::Talk;
}

static void syncTapButton() { ui::setTapButton(tapButtonKind()); }

static std::vector<ui::Row> statusRows() {
  std::vector<ui::Row> rows;
  bool wifiOk = WiFi.status() == WL_CONNECTED;
  rows.push_back({"Wi-Fi", wifiOk ? WiFi.SSID() + "  " + String(WiFi.RSSI()) + " dBm" : String("not connected"),
                  wifiOk ? (WiFi.RSSI() < -80 ? kYellow : kGreen) : kRed});
  rows.push_back({"IP", wifiOk ? WiFi.localIP().toString() : String("-"), kText});
  rows.push_back({"Deck", serverHost() + (wsUp ? (rttMs >= 0 ? "  " + String(rttMs) + " ms" : String("  ok")) : String("  offline")),
                  wsUp ? kGreen : kRed});
  rows.push_back({"Agent", (agentName.length() ? agentName : String("none")) + (agentLinked ? "" : "  (waiting)"),
                  agentLinked ? kGreen : (wsUp ? kYellow : kRed)});
  int pct = M5.Power.getBatteryLevel();
  int mv = M5.Power.getBatteryVoltage();
  bool chg = M5.Power.isCharging() == m5::Power_Class::is_charging;
  rows.push_back({"Battery", String(pct) + "%  " + String(mv / 1000.0f, 2) + " V" + (chg ? "  charging" : ""),
                  pct >= 40 ? kGreen : pct >= 20 ? kYellow : kRed});
  float tC, rh;
  if (telemetry::climate(&tC, &rh)) {
    rows.push_back({"Room", String(tC, 1) + " °C  " + String((int)lroundf(rh)) + "% humidity", kText});
  }
  rows.push_back({"Voice", String(settings.femaleVoice ? "female" : "male") + "  vol " +
                               String((settings.volume * 100 + 127) / 255) + "%" +
                               (settings.boostDb ? "  +" + String(settings.boostDb) + " dB" : String("")) +
                               (settings.speak ? "" : "  muted"),
                  kText});
  rows.push_back({"Usage", String(turns) + " turn" + (turns == 1 ? "" : "s") +
                               (lastReplyMs ? "  last " + String((millis() - lastReplyMs) / 60000) + "m ago" : String("")) +
                               "  up " + uptime(),
                  kText});
  rows.push_back({"Device", deviceId() + "  fw " FW_VERSION "  " + String(ESP.getFreePsram() / 1024) + "k free", kText});
  if (board::kOfflineNotes && notes::count()) {
    rows.push_back({"Offline notes", String(notes::count()) + " waiting  " + clockLength(notes::seconds()) + "  room " +
                                         String(notes::room(audio::kMicRate) / audio::kMicRate / 60) + " min" +
                                         (upAllowed && wsUp ? "  sending…" : ""),
                    kYellow});
  }
  if (offer.valid()) rows.push_back({"Update", offer.version + " available (menu → Software update)", kCyan});
  return rows;
}

static void showIdle() {
  if (settings.iconMode) {
    ui::showFace(settings.character, agentLinked ? face::Pose::Idle : face::Pose::Sad, 0);
    return;
  }
  bool wifiOk = WiFi.status() == WL_CONNECTED;
  bool ok = wsUp && agentLinked && wifiOk;
  ui::Home home;
  home.ready = ok;
  home.title = ok ? "All good" : "Getting ready";
  home.color = ok ? kGreen : kYellow;
  // Offline, a device that keeps notes still takes one.
  const bool keeps = board::kOfflineNotes && notes::ready();
  home.hint = ok        ? (settings.talks < kHintTurns ? String(board::kTalkHint) : String(""))
              : !wifiOk ? (keeps ? String("Offline · talk to keep a note") : String("No Wi-Fi"))
              : !wsUp   ? String("Connecting to Flight Deck…")
                        : String("Waiting for the agent");
  String noun = board::kNoun;
  noun.setCharAt(0, toupper(noun[0]));
  home.rows.push_back({noun, ok ? String(board::kTalkToggles ? "ready — press " : "ready — hold ") + board::kTalkButton
                               : String("not ready"), ok ? kGreen : kYellow});
  home.rows.push_back({"Wi-Fi", wifiOk ? WiFi.SSID() : String("not connected"), wifiOk ? kGreen : kRed});
  home.rows.push_back({"Deck", wsUp ? serverHost() : String("connecting…"), wsUp ? kGreen : kRed});
  home.rows.push_back({"Agent", agentLinked ? agentName : (wsUp ? String("waiting for agent") : String("-")),
                       agentLinked ? kGreen : kYellow});
  home.character = settings.character;
  ui::showHome(home);
}

// ── menu ─────────────────────────────────────────────────────────────

enum MenuRow { MReplay, MPin, MStatus, MNotes, MAgent, MVolume, MBoost, MVoice, MDisplay, MTapTalk, MMore,
               MTextSize, MCharacter, MBeep, MTalkSnd, MFlash, MBuzz, MLead, MScreenOff, MPowerOff, MRotation,
               MLeds, MClicks, MSayMenu, MAsleep, MPhone, MUpdate, MAutoUpdate, MSetup, MBack, MCount };

// Two levels (not the paper's): what you reach for in the moment at the
// top, the rest under Settings. menuMore: the Settings level is open.
static bool menuMore = false;
static bool topRow(int row) {
  switch (row) {
    case MReplay: case MPin: case MStatus: case MNotes: case MAgent: case MVolume: case MTapTalk:
    case MMore: case MUpdate: case MSetup:
      return true;
    default:
      return false;
  }
}

// The last answer can be heard again: from memory, or (part of it never
// arrived) from the deck.
static bool canReplay() { return audio::hasClip() || (audio::clipCut() && wsUp); }

// Rows this device has now (only a device with a motor can buzz; Replay only
// while there's an answer to replay; the paper has no backlight to switch
// off and can't animate a character, but can turn and has lights).
static bool rowShown(int row) {
  // The PaperColor's menu is read aloud, so it keeps two rows: Options (a QR
  // code for the settings page on the phone, where everything else is set)
  // and the software update.
  if (board::kEpaper) return row == MPhone || row == MUpdate;
  if (row == MMore) return !menuMore;
  if (row == MBack) return menuMore;
  if (menuMore == topRow(row)) return false;  // the other level's
  switch (row) {
    case MReplay: return canReplay();
    case MPin: return ui::canPin();
    case MBuzz: return board::kHasMotor;
    case MNotes: return board::kOfflineNotes && notes::count() > 0;  // kept offline: send them now
    case MTapTalk: return board::kHasTouch;
    case MDisplay:
    case MCharacter:  // the paper is text only
    case MScreenOff: return !board::kEpaper;
    case MRotation:
    case MLeds:
    case MClicks:
    case MAsleep: return board::kEpaper;
    case MSayMenu: return false;  // the paper's menu is only ever read aloud
    default: return true;
  }
}

static const char* const kRotations[] = {"portrait", "landscape", "portrait, flipped", "landscape, flipped"};
static const char* const kLedLevels[] = {"off", "low", "bright"};

static String asleepLabel() {
  if (!settings.asleepRefreshMin) return "never";
  return settings.asleepRefreshMin >= 60 ? "every " + String(settings.asleepRefreshMin / 60) + " h"
                                         : "every " + String(settings.asleepRefreshMin) + " min";
}

static String powerOffLabel() {
  return board::kEpaper ? String(settings.powerOffMin) + " min" : String(settings.powerOffS) + " s";
}

static constexpr uint8_t kVolumeSteps[] = {64, 128, 191, 255};  // 25 / 50 / 75 / 100 %
static constexpr uint8_t kBoostSteps[] = {0, 6, 10};

static String menuLabel(int row) {
  switch (row) {
    case MReplay: return "Replay answer";
    case MMore: return "Settings »";
    case MBack: return "Back";
    case MPin: return ui::pinned() ? "Unpin answer" : "Pin answer to home";
    case MStatus: return "Status";
    case MNotes: return "Send voice notes (" + String(notes::count()) + ")";
    case MTapTalk: return String("Tap to talk: ") + (settings.tapTalk ? "on" : "off");
    case MAgent: return "Agent: " + (agentName.length() ? agentName : String("none")) + " »";
    case MVolume: return "Volume " + String((settings.volume * 100 + 127) / 255) + "%";
    case MBoost: return settings.boostDb ? "Boost +" + String(settings.boostDb) + " dB" : String("Boost off");
    case MVoice: return String("Voice: ") + (settings.femaleVoice ? "female" : "male");
    case MDisplay: return String("Display: ") + (settings.iconMode ? "icons" : "text");
    case MTextSize: {
      static const char* sizes[] = {"small", "medium", "large"};
      return String("Text size: ") + sizes[settings.textSize];
    }
    case MCharacter: return String("Character: ") + face::name(settings.character) + " »";
    case MBeep: return String("Beeps around speech: ") + (settings.cueBeep ? "on" : "off");
    case MTalkSnd: return String("Talk sounds: ") + (settings.talkSounds ? "on" : "off");
    case MFlash:
      return String(board::kEpaper ? "Light" : "Flash") + " before speech: " + (settings.cueFlash ? "on" : "off");
    case MBuzz: return String("Buzz before speech: ") + (settings.cueBuzz ? "on" : "off");
    case MLead: return "Cue lead: " + String(settings.cueLeadS) + " s";
    case MScreenOff: return "Screen off: " + String(settings.screenOffS) + " s";
    case MPowerOff: return "Power off: " + powerOffLabel();
    case MRotation: return String("Orientation: ") + kRotations[settings.rotation & 3];
    case MLeds: return String("Status lights: ") + kLedLevels[settings.ledLevel % 3];
    case MClicks: return String("Key clicks: ") + (settings.keyClicks ? "on" : "off");
    case MSayMenu: return String("Read menu aloud: ") + (settings.speakMenu ? "on" : "off");
    case MAsleep: return "Redraw asleep: " + asleepLabel();
    case MPhone: return board::kEpaper ? "Options »" : "Settings on phone »";
    case MUpdate: return offer.valid() ? "Update to " + offer.version + " »" : String("Software " FW_VERSION " · check");
    case MAutoUpdate: return String("Auto update: ") + (settings.autoUpdate ? "on" : "off");
    default: return "Wi-Fi & settings…";
  }
}

// A menu row as a card: the round screen shows the value big in a badge.
static ui::Item menuItem(int row) {
  static const char* sizes[] = {"small", "medium", "large"};
  auto onOff = [](bool on) { return String(on ? "on" : "off"); };
  ui::Item it;
  it.label = menuLabel(row);
  it.note = "yellow = change";
  switch (row) {
    case MReplay: it.title = "Replay answer"; it.glyph = ui::Glyph::Replay; it.note = "yellow = hear it again"; break;
    case MMore: it.title = "Settings"; it.glyph = ui::Glyph::Gear; it.note = "voice, display, sounds, timers…"; break;
    case MBack: it.title = "Back"; it.glyph = ui::Glyph::Back; it.note = "to the menu"; break;
    case MPin:
      it.title = ui::pinned() ? "Unpin answer" : "Pin answer";
      it.badge = ui::pinned() ? "pinned" : "";
      it.note = ui::pinned() ? "the home screen shows the newest again" : "keeps it on the home screen, asleep too";
      break;
    case MStatus: it.title = "Status"; it.glyph = ui::Glyph::Info; it.note = "Wi-Fi, deck, battery…"; break;
    case MNotes:
      it.title = "Send voice notes";
      it.badge = String(notes::count());
      it.note = clockLength(notes::seconds()) + " kept offline · yellow = send";
      break;
    case MTapTalk:
      it.title = "Tap to talk";
      it.badge = onOff(settings.tapTalk);
      it.on = settings.tapTalk;
      it.note = "a record button on the screen";
      break;
    case MAgent:
      it.title = "Agent";
      it.badge = agentName.length() ? agentName : String("?");
      it.glyph = agentName.length() ? ui::Glyph::Initials : ui::Glyph::None;
      it.note = agentName.length() ? agentName + " · change" : String("none · choose one");
      break;
    case MVolume: it.title = "Volume"; it.badge = String((settings.volume * 100 + 127) / 255) + "%"; break;
    case MBoost:
      it.title = "Speech boost";
      it.badge = settings.boostDb ? "+" + String(settings.boostDb) + " dB" : String("off");
      it.on = settings.boostDb > 0;
      break;
    case MVoice: it.title = "Voice"; it.badge = settings.femaleVoice ? "female" : "male"; break;
    case MDisplay: it.title = "Display"; it.badge = settings.iconMode ? "icons" : "text"; break;
    case MTextSize:
      it.title = "Text size";
      it.badge = String("Aa");
      it.note = String(sizes[settings.textSize]) + " · change";
      break;
    case MCharacter: it.title = "Character"; it.badge = face::name(settings.character); it.note = "yellow = browse"; break;
    case MBeep:
      it.title = "Beeps around speech";
      it.badge = onOff(settings.cueBeep);
      it.on = settings.cueBeep;
      it.note = "a chime before it speaks, the reverse after";
      break;
    case MTalkSnd:
      it.title = "Talk sounds";
      it.badge = onOff(settings.talkSounds);
      it.on = settings.talkSounds;
      it.note = "a blip when it listens, a lower one when you let go";
      break;
    case MFlash:
      it.title = board::kEpaper ? "Light before speech" : "Flash before speech";
      it.badge = onOff(settings.cueFlash);
      it.on = settings.cueFlash;
      break;
    case MBuzz: it.title = "Buzz before speech"; it.badge = onOff(settings.cueBuzz); it.on = settings.cueBuzz; break;
    case MLead: it.title = "Cue lead"; it.badge = String(settings.cueLeadS) + " s"; it.note = "how early the cue comes"; break;
    case MScreenOff: it.title = "Screen off"; it.badge = String(settings.screenOffS) + " s"; it.note = "after this long idle"; break;
    case MPowerOff: it.title = "Power off"; it.badge = powerOffLabel(); it.note = "after this long idle"; break;
    case MRotation: it.title = "Orientation"; it.badge = kRotations[settings.rotation & 3]; it.note = "turns the paper"; break;
    case MLeds:
      it.title = "Status lights";
      it.badge = kLedLevels[settings.ledLevel % 3];
      it.on = settings.ledLevel > 0;
      it.note = "red records, blue works, green speaks";
      break;
    case MClicks: it.title = "Key clicks"; it.badge = onOff(settings.keyClicks); it.on = settings.keyClicks; break;
    case MSayMenu:
      it.title = "Read menu aloud";
      it.badge = onOff(settings.speakMenu);
      it.on = settings.speakMenu;
      it.note = "hear each row before the paper shows it";
      break;
    case MAsleep:
      it.title = "Redraw asleep";
      it.badge = asleepLabel();
      it.on = settings.asleepRefreshMin > 0;
      it.note = "date, room and battery on the sleeping paper";
      break;
    case MPhone:
      it.title = board::kEpaper ? "Options" : "Settings on phone";
      it.glyph = ui::Glyph::Wifi;
      it.note = "every setting, from your phone";
      break;
    case MUpdate:
      it.title = "Software update";
      it.badge = offer.valid() ? offer.version : String(FW_VERSION);
      it.note = offer.valid() ? String("yellow = install") : String("up to date · yellow = check");
      break;
    case MAutoUpdate:
      it.title = "Auto update";
      it.badge = onOff(settings.autoUpdate);
      it.on = settings.autoUpdate;
      it.note = "installs when idle on USB";
      break;
    default: it.title = "Wi-Fi & settings"; it.glyph = ui::Glyph::Wifi; it.note = "opens the setup hotspot"; break;
  }
  return it;
}

// The rows this device shows, and where the selected one is among them.
static std::vector<ui::Item> menuItems(int* shownSel) {
  std::vector<ui::Item> items;
  for (int i = 0; i < MCount; ++i) {
    if (!rowShown(i)) continue;
    if (i == menuSel) *shownSel = items.size();
    items.push_back(menuItem(i));
  }
  return items;
}

static std::vector<ui::Item> agentItems() {
  std::vector<ui::Item> items;
  for (size_t i = 0; i < agentNames.size(); ++i) {
    ui::Item it;
    it.label = it.title = agentNames[i];
    bool back = !agentsLoading && i + 1 == agentNames.size();
    if (agentsLoading) {
      it.badge = "…";
    } else if (back) {
      it.title = "Back";
      it.glyph = ui::Glyph::Back;
      it.note = "to the menu";
    } else {
      it.badge = agentNames[i];
      it.glyph = ui::Glyph::Initials;
      it.note = (int)i == agentCur ? "current agent" : "yellow = choose";
    }
    items.push_back(it);
  }
  return items;
}

template <size_t N>
static uint8_t nextStep(const uint8_t (&steps)[N], uint8_t now) {
  for (uint8_t v : steps) {
    if (v > now) return v;
  }
  return steps[0];
}

// Reads a row aloud a moment after the last press (e-paper: the paper shows
// it ~20 s later). Quick presses say only where they stopped.
static void sayLater(String text) {
  if (!board::kEpaper || !settings.speakMenu) return;
  text.replace(" »", "");
  text.replace(":", ",");  // "Volume, 75%" reads better than "Volume: 75%"
  sayText = text;
  sayAtMs = millis() + 350;
  if (!sayAtMs) sayAtMs = 1;
}

static void sayRow() {
  if (mode == Mode::Menu) sayLater(menuLabel(menuSel));
  else if (mode == Mode::Agents && !agentsLoading && agentSel < (int)agentNames.size())
    sayLater(agentSel + 1 == (int)agentNames.size() ? String("Back") : agentNames[agentSel] + (agentSel == agentCur ? ", current" : ""));
  else if (mode == Mode::CharPick) sayLater(face::name(charSel));
}

static void backToMain();
static void startInstall();
static void connectServer();
static void dropBackgroundPart();
static void leaveBackdrop();
static void setBackground(const uint8_t* jpg, size_t len, const String& id);
static void sendNotesNow();
static void replayAnswer();
static String phoneUrl;  // "settings on phone" address while it's open

static void redraw() {
  setBarState();
  syncTapButton();
  ui::setReplayable((mode == Mode::Reply || mode == Mode::Idle) && canReplay() && !audio::playing());
  switch (mode) {
    case Mode::Listening:
      if (settings.iconMode) ui::showFace(settings.character, face::Pose::Listening, 0);
      else ui::showListening(0, millis() - recStartMs);
      break;
    case Mode::Waiting:
      if (settings.iconMode) ui::showFace(settings.character, face::Pose::Thinking, 0);
      else ui::showWorking(heard, activity, statusLine, millis() - workStartMs, toolSteps);
      break;
    case Mode::Reply:
      if (settings.iconMode && audio::playing()) {
        ui::showFace(settings.character, audio::paused() ? face::Pose::Idle : face::Pose::Speaking,
                     audio::outputLevel());
      } else {
        ui::showReply();
      }
      break;
    case Mode::Image: ui::showStoredImage(); break;
    case Mode::Menu: {
      int sel = 0;
      auto items = menuItems(&sel);
      ui::showPicker(items, sel, -1, "Menu");
      break;
    }
    case Mode::Agents: ui::showPicker(agentItems(), agentSel, agentCur, "Agent"); break;
    case Mode::CharPick: {
      // Cycle through the poses so the preview shows everything it does.
      static const face::Pose poses[] = {face::Pose::Idle, face::Pose::Listening, face::Pose::Thinking,
                                         face::Pose::Speaking, face::Pose::Sad};
      uint32_t t = millis();
      face::Pose pose = poses[(t / 2500) % 5];
      int lvl = pose == face::Pose::Speaking || pose == face::Pose::Listening
                    ? 45 + (int)(40 * sinf(t / 90.0f)) : 0;
      ui::showCharPick(charSel, pose, lvl, charSel);
      break;
    }
    case Mode::Status: {
      auto rows = statusRows();
      int n = ui::showPanel("", kText, rows, statusPage);
      if (n != statusPages || statusPage >= n) {  // the bar showed a stale "x/y"
        statusPages = n;
        statusPage = min(statusPage, n - 1);
        setBarState();
        ui::showPanel("", kText, rows, statusPage);
      }
      break;
    }
    case Mode::Notice: break;  // stays until the next event
    case Mode::Link:
      if (board::kEpaper) {
        String title = offer.valid() ? "Options · update ready" : "Options · " FW_VERSION;
        ui::showLink(title.c_str(), phoneUrl, "Scan it with your phone (same Wi-Fi): every setting is there.");
      } else {
        ui::showLink("Phone settings", phoneUrl,
                     "Scan it with your phone (same Wi-Fi). Saving restarts the " + String(board::kNoun) + ".");
      }
      break;
    case Mode::Home: showIdle(); break;
    case Mode::Backdrop: ui::showBackdrop(); break;
    case Mode::Ask:
      ui::showAsk("Saved voice notes",
                  notesCount() + " recorded offline (" + clockLength(notes::seconds()) + "). Send " +
                      (notes::count() == 1 ? "it" : "them") + " to the agent?",
                  "send", "later");
      break;
    case Mode::Idle:
      if (ui::hasReply()) ui::showReply();
      else showIdle();
      break;
  }
}

static void notice(const char* title, const String& body, uint16_t color) {
  ui::fade();
  mode = Mode::Notice;
  setBarState();
  ui::showMessage(title, body, color);
}

static void backToMain() {
  mode = ui::hasReply() ? Mode::Reply : Mode::Idle;
  redraw();
}

// From the picture alone back to the home screen (even with an answer kept).
static void leaveBackdrop() {
  ui::fade();
  mode = ui::hasReply() ? Mode::Home : Mode::Idle;
  redraw();
}

static void openMenu() {
  menuMore = false;
  menuSel = board::kEpaper ? MPhone : canReplay() ? MReplay : MStatus;
  menuOpenedMs = millis();
  mode = Mode::Menu;
  ui::fade();
  redraw();
  sayRow();
}

// Options: the settings page on the phone, its address as a QR code (on
// the PaperColor this is the whole menu: every setting lives there).
static void openOptions() {
  phoneUrl = phoneSettingsBegin();
  if (phoneUrl.isEmpty()) {
    ui::drawNow();
    notice("Not connected", "Join Wi-Fi first, or use Wi-Fi & settings.", kRed);
    return;
  }
  menuOpenedMs = millis();
  mode = Mode::Link;
  ui::drawNow();  // asked to see it
  redraw();
}

// Software update: install the build Flight Deck offers, or ask it for one.
static void updateSoftware() {
  ui::drawNow();  // the answer to a press: shown at once
  if (offer.valid()) {
    startInstall();
  } else if (wsUp) {
    checkingUpdate = true;
    sendType("update_check");
    notice("Software update", "Checking for a newer version…", kText);
  } else {
    notice("Offline", "Connect to Flight Deck to check for updates.", kRed);
  }
}

static void chooseMenuRow() {
  menuOpenedMs = millis();
  switch (menuSel) {
    case MReplay:
      ui::fade();
      backToMain();
      replayAnswer();
      return;
    case MMore:
    case MBack:
      menuMore = menuSel == MMore;
      ui::slide(menuMore);  // into Settings, or back out
      menuSel = menuMore ? 0 : MMore;
      while (!rowShown(menuSel)) ++menuSel;  // the level's first row
      redraw();
      sayRow();
      return;
    case MStatus:
      statusPage = 0;
      mode = Mode::Status;
      ui::drawNow();  // asked to see it
      ui::fade();
      redraw();
      return;
    case MAgent:
      if (!wsUp) {
        notice("Offline", "Can't list agents until the " + String(board::kNoun) + " is connected.", kRed);
        return;
      }
      agentNames = {"Loading agents…"};
      agentSlugs = {""};
      agentSel = 0;
      agentCur = -1;
      agentsLoading = true;
      mode = Mode::Agents;
      sendType("agents");
      redraw();
      return;
    case MSetup:
      ui::flush();
      runSetupPortal("Settings");
    case MPin:
      ui::pin(!ui::pinned());
      mode = board::kEpaper ? Mode::Home : Mode::Idle;  // the home screen shows what it keeps now
      redraw();
      return;
    case MPhone:
      openOptions();
      return;
    case MRotation:
      settings.rotation = (settings.rotation + 1) & 3;
      ui::setRotation(settings.rotation);
      break;
    case MLeds:
      settings.ledLevel = (settings.ledLevel + 1) % 3;
      leds::setLevel(settings.ledLevel);
      break;
    case MClicks: settings.keyClicks = !settings.keyClicks; break;
    case MSayMenu: settings.speakMenu = !settings.speakMenu; break;
    case MAsleep:
      settings.asleepRefreshMin = settings.asleepRefreshMin == 0    ? 30
                                  : settings.asleepRefreshMin == 30 ? 60
                                  : settings.asleepRefreshMin == 60 ? 180
                                                                    : 0;
      break;
    case MVolume:
      settings.volume = nextStep(kVolumeSteps, settings.volume);
      audio::setVolume(settings.volume);
      break;
    case MBoost:
      settings.boostDb = nextStep(kBoostSteps, settings.boostDb);
      audio::setBoost(settings.boostDb);
      break;
    case MVoice:
      settings.femaleVoice = !settings.femaleVoice;
      if (wsUp) sendPrefs();
      break;
    case MDisplay: settings.iconMode = !settings.iconMode; break;
    case MTextSize:
      settings.textSize = (settings.textSize + 1) % 3;
      ui::setTextSize(settings.textSize);
      break;
    case MCharacter:
      charSel = settings.character;
      mode = Mode::CharPick;
      ui::slide(true);
      redraw();
      sayRow();
      return;
    case MBeep: settings.cueBeep = !settings.cueBeep; break;
    case MTalkSnd: settings.talkSounds = !settings.talkSounds; break;
    case MFlash: settings.cueFlash = !settings.cueFlash; break;
    case MBuzz:
      settings.cueBuzz = !settings.cueBuzz;
      if (settings.cueBuzz) buzz(160);
      break;
    case MLead: settings.cueLeadS = settings.cueLeadS == 1 ? 2 : 1; break;
    case MScreenOff: settings.screenOffS = settings.screenOffS == 5 ? 10 : settings.screenOffS == 10 ? 15 : 5; break;
    case MPowerOff:
      if (board::kEpaper) settings.powerOffMin = settings.powerOffMin == 1 ? 3 : settings.powerOffMin == 3 ? 10 : 1;
      else settings.powerOffS = settings.powerOffS == 15 ? 30 : settings.powerOffS == 30 ? 60 : 15;
      break;
    case MUpdate:
      updateSoftware();
      return;
    case MAutoUpdate: settings.autoUpdate = !settings.autoUpdate; break;
    case MNotes:
      sendNotesNow();
      return;
    case MTapTalk: settings.tapTalk = !settings.tapTalk; break;
  }
  settings.save();
  redraw();
  if (menuSel == MVolume || menuSel == MBoost) audio::beep();
  if (menuSel == MBeep && settings.cueBeep) audio::cue();
  if (menuSel == MTalkSnd && settings.talkSounds) {  // what it sounds like
    audio::talkStart();
    audio::talkEnd();
  }
  sayRow();  // the new value
}

static void chooseAgent() {
  if (agentsLoading) return;
  if (agentSel == (int)agentNames.size() - 1) {  // "« Back"
    mode = Mode::Menu;
    redraw();
    return;
  }
  settings.agent = agentSlugs[agentSel];
  settings.save();
  JsonDocument d;
  d["type"] = "select";
  d["agent"] = settings.agent;
  sendJson(d);
  backToMain();
}

// ── microphone → server ──────────────────────────────────────────────

static void micSink(const int16_t* samples, size_t count) {
  if (wsUp) ws.sendBIN((const uint8_t*)samples, count * sizeof(int16_t));
}

static void startOfflineNote();

static void startTalking() {
  if (!wsUp) {
    // Kept as a voice note (sent once the deck is back) where the device can.
    if (board::kOfflineNotes && notes::room(audio::kMicRate) > 3 * audio::kMicRate) {
      startOfflineNote();
      return;
    }
    notice("Offline", "Not connected to Flight Deck yet.", kRed);
    return;
  }
  flashing = false;
  quietSpeech = false;
  ui::setQuestion("");
  if (settings.talkSounds) audio::talkStart();  // the blip, then (0.25 s on) the mic listens
  audio::micStart();  // also stops any playback
  JsonDocument d;
  d["type"] = "rec_start";
  d["rate"] = audio::kMicRate;
  sendJson(d);
  recStartMs = millis();
  mode = Mode::Listening;
  buzz(35);
  redraw();
}

// The end of a recording goes up with the device's readings (battery,
// Wi-Fi, motion, room climate, clock): the agent gets them as context.
static void sendRecEnd() {
  JsonDocument d;
  if (settings.talks < kHintTurns) {  // the home screen's hint has done its job after a few
    ++settings.talks;
    settings.save();
  }
  d["type"] = "rec_end";
  telemetry::fill(d["ctx"].to<JsonObject>());
  sendJson(d);
}

static void stopTalking(bool send) {
  tapRec = false;
  audio::micStop(send ? micSink : [](const int16_t*, size_t) {});
  if (send) sendRecEnd();
  else sendType("rec_cancel");
  if (send && settings.talkSounds) audio::talkEnd();  // not for a tap: nothing was sent
  if (send) {
    heard = "";
    statusLine = "Transcribing…";
    startWork();
    mode = Mode::Waiting;
    redraw();
  } else {
    backToMain();
  }
}

// ── speech cue: beep + flash, then the speech ────────────────────────

static void startSpeech(uint32_t rate) {
  bool quiet = quietSpeech;  // a replay the user asked for needs no cue
  bool cue = !quiet && (settings.cueBeep || settings.cueFlash);
  bool buzzCue = !quiet && board::kHasMotor && settings.cueBuzz;
  bool goesOn = audio::playBegin(rate, (cue || buzzCue) ? settings.cueLeadS * 1000UL : 0);
  touch();
  if (goesOn || quiet) return;  // goesOn: the answer's next sound, straight after the last
  if (settings.cueBeep) audio::cue();
  if (buzzCue) buzz(160);
  if (settings.cueFlash && board::kEpaper) {
    leds::flash();  // the paper can't flash; the lights can
  } else if (settings.cueFlash) {
    M5.Display.setBrightness(255);
    flashing = true;
    flashStartMs = millis();
  }
}

// ── talking from sleep ───────────────────────────────────────────────

// Woken by the talk button, and it's still held: they're talking already.
// (Where the button toggles, its press is enough.)
static bool wokenToTalk() {
  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  bool byTalk = cause == ESP_SLEEP_WAKEUP_EXT0 ||
                (cause == ESP_SLEEP_WAKEUP_EXT1 && (esp_sleep_get_ext1_wakeup_status() & (1ULL << board::kTalkPin)));
  return byTalk && (board::kTalkToggles || gpio_get_level(board::kTalkPin) == 0);
}

static void startPreroll(uint32_t maxMs = kPrerollMaxMs) {
  // As long as memory allows, halving down to 10 s — but always one try,
  // however short the cap (anything unwritten is silence).
  for (prerollMaxMs = maxMs;; prerollMaxMs /= 2) {
    prerollCap = (size_t)prerollMaxMs * audio::kMicRate / 1000;
    preroll = (int16_t*)ps_calloc(prerollCap, sizeof(int16_t));
    if (preroll || prerollMaxMs / 2 < 10000) break;
  }
  if (!preroll) return;
  M5.Speaker.end();
  M5.Mic.begin();
  if (!M5.Mic.record(preroll, prerollCap, audio::kMicRate)) {  // fills in the background, in real time
    M5.Mic.end();
    free(preroll);
    preroll = nullptr;
    return;
  }
  prerollStartMs = millis();
  prerollEndMs = 0;
  prerollSent = 0;
  prerollOn = true;
  prerollStreaming = prerollChecked = prerollOffline = prerollEndSounded = false;
  leds::set(leds::State::Listening, 50);
}

static void stopPreroll() {
  M5.Mic.end();
  free(preroll);
  preroll = nullptr;
  prerollOn = prerollStreaming = prerollOffline = false;
  tapRec = false;
}

static size_t prerollReady(uint32_t atMs);

// Talking while the deck can't be reached: the same background recording,
// kept as a voice note when it's let go (keepOffline).
static void startOfflineNote() {
  size_t room = notes::room(audio::kMicRate);
  uint32_t maxMs = (uint32_t)min<uint64_t>(kOfflineMaxMs, (uint64_t)room * 1000 / audio::kMicRate);
  flashing = false;
  audio::playStop();
  if (settings.talkSounds) audio::talkStart();
  startPreroll(maxMs);
  if (!prerollOn) {
    notice("Offline", "Not connected to Flight Deck, and no memory free to record.", kRed);
    return;
  }
  prerollOffline = true;
  recStartMs = prerollStartMs;
  ui::setQuestion("");
  mode = Mode::Listening;
  buzz(35);
  redraw();
}

// The mic's level now, from what it last wrote (for the screen and lights).
static int prerollLevel() {
  size_t end = prerollReady(millis() - 40), from = end > 800 ? end - 800 : 0;
  int peak = 0;
  for (size_t i = from; i < end; ++i) peak = max(peak, abs((int)preroll[i]));
  return min(100, peak * 100 / 12000);
}

static String localTimeText(time_t t) {
  if (t < 1700000000) return "";
  struct tm tm;
  localtime_r(&t, &tm);
  char buf[40];
  strftime(buf, sizeof(buf), "%a %Y-%m-%d %H:%M (%z)", &tm);
  return buf;
}

// The deck wasn't reached: keep the recording as a voice note, sent with a
// word to the agent that it was made offline once the deck is back.
static void keepOffline(uint32_t stopAt) {
  const size_t n = prerollReady(stopAt);
  M5.Mic.end();  // nothing more lands in the buffer
  time_t when = telemetry::clockSet() ? time(nullptr) - (time_t)((millis() - prerollStartMs) / 1000) : 0;
  if (!board::kEpaper) ui::showMessage("Saving", "Keeping the voice note offline…", kText);
  bool kept = board::kOfflineNotes && n >= audio::kMicRate && notes::save(preroll, n, audio::kMicRate, when);
  if (kept) notesDeclined = -1;  // something new to ask about
  bool knewOffline = prerollOffline;
  stopPreroll();
  if (kept) {
    String waiting = notes::count() > 1 ? " (" + String(notes::count()) + " waiting)" : String("");
    notice("Saved offline", "Kept on the " + String(board::kNoun) + waiting + ". Back online, it asks to send " +
                                (notes::count() > 1 ? "them" : "it") + " to the agent.", kGreen);
  } else if (board::kOfflineNotes) {
    notice("Not kept", knewOffline ? "Offline, and no room left to keep it." : "Couldn't reach Flight Deck, and no room to keep it.", kRed);
  } else {
    notice("Not sent", "Couldn't reach Flight Deck in time — try again.", kRed);
  }
}

// The samples the mic has surely written by `atMs` (it runs in real time).
static size_t prerollReady(uint32_t atMs) {
  if ((int32_t)(atMs - prerollStartMs) <= 0) return 0;
  return min(prerollCap, (size_t)((atMs - prerollStartMs) * (uint64_t)audio::kMicRate / 1000));
}

// A send can find the link gone: the disconnect is handled inside it (onWs),
// which starts this recording over for the next link — so stop at once.
static void prerollSend(size_t upTo) {
  while (wsUp && prerollSent < upTo) {
    size_t at = prerollSent, n = min(upTo - at, (size_t)2048);
    ws.sendBIN((const uint8_t*)(preroll + at), n * sizeof(int16_t));
    if (!wsUp) return;
    prerollSent = at + n;
  }
}

// A press that only woke it: too little recorded, or a short stretch that
// never got louder than the room (the press itself began before the mic).
static bool prerollWokeOnly(uint32_t end, bool wholeCap = false) {
  uint32_t ms = end - prerollStartMs;
  if (ms < kPrerollMinMs) return true;
  if (ms > 4000 && !wholeCap) return false;  // (ran to the cap untouched: silent all along?)
  size_t n = prerollReady(end);
  int peak = 0;
  for (size_t i = audio::kMicRate / 10; i < n; i += 2) peak = max(peak, abs((int)preroll[i]));  // past the mic's start
  return peak < kPrerollQuietPeak;
}

// The recording is over (let go, or full): the mic stops — what it wrote
// stays in the buffer — and the "let go" blip plays, once.
static void prerollEndSound() {
  if (prerollEndSounded) return;
  prerollEndSounded = true;
  M5.Mic.end();
  if (settings.talkSounds) audio::talkEnd();
}

// From loop(): once the deck is reached, what was said goes up — and the
// rest follows live until the button is let go (or 30 s are used).
static void pumpPreroll() {
  if (!prerollOn) return;
  const uint32_t end = prerollEndMs;  // before `now`: the sampler may set it any moment
  const uint32_t now = millis();
  const bool full = (int32_t)(now - prerollStartMs) >= (int32_t)prerollMaxMs;
  const uint32_t stopAt = end ? end : full ? prerollStartMs + prerollMaxMs : 0;  // 0 = still talking
  // Let go, or full (a press that only woke a toggling button runs to the cap).
  if (stopAt && !prerollChecked && (int32_t)(now - stopAt) >= 150) {
    prerollChecked = true;  // once: it reads the whole recording
    // Pressed on purpose (awake, offline): only too short counts as nothing.
    bool nothing = prerollOffline ? stopAt - prerollStartMs < kPrerollMinMs : prerollWokeOnly(stopAt, !end);
    if (nothing) {
      bool sent = prerollStreaming, offline = prerollOffline;
      stopPreroll();
      if (sent) sendType("rec_cancel");  // nothing worth hearing went up after all
      if (offline) notice("Too short", "Nothing kept: talk a little longer.", kOrange);
      else if (sent || mode == Mode::Listening) backToMain();
      return;
    }
    prerollEndSound();  // a real recording: let go is heard at once, whatever the link does
  }
  if (!wsUp) {
    // Let go and still no deck: keep it offline. A wake on Wi-Fi expects the
    // link any moment; without Wi-Fi (or recorded offline) there's no point waiting.
    bool noLink = prerollOffline || WiFi.status() != WL_CONNECTED;
    if (stopAt && (int32_t)(now - stopAt) > (noLink ? 1500 : 25000)) {
      prerollEndSound();  // (full while held: no release played it)
      keepOffline(stopAt);
    }
    return;
  }
  if (!prerollStreaming) {
    prerollStreaming = true;
    JsonDocument d;
    d["type"] = "rec_start";
    d["rate"] = audio::kMicRate;
    sendJson(d);
    recStartMs = prerollStartMs;
    mode = Mode::Listening;
    redraw();
  }
  if (!stopAt) {
    prerollSend(prerollReady(now - 120));  // stay behind the mic
    return;
  }
  if ((int32_t)(now - stopAt) < 150) return;  // let the last of it land
  prerollEndSound();
  prerollSend(prerollReady(stopAt));
  if (!wsUp) return;  // dropped mid-send: it all goes again (or is kept offline)
  sendRecEnd();
  if (!wsUp) return;
  stopPreroll();
  heard = "";
  statusLine = "Transcribing…";
  startWork();
  mode = Mode::Waiting;
  redraw();
}

// ── offline voice notes → Flight Deck ─────────────────────────────────

// Once the deck is back, the device is idle and the user said yes, the
// waiting notes go up (oldest first, up to 10 a time): notes_begin, then each
// as note_start + PCM + note_end, then notes_end with the readings. The deck
// transcribes them into one turn for the agent, marked as recorded offline,
// and answers notes_done with the ids it has; only those are deleted here.
static std::vector<notes::Note> upNotes;  // the batch going up
static size_t upIndex = 0;
static bool upOpen = false;               // the current note's file is open, its note_start sent
static bool upAwaitAck = false;           // all sent: waiting for notes_done
static uint32_t upAtMs = 0, upRetryAtMs = 0;
static bool upSentAny = false;            // this batch has a note the deck took

// The question, spoken by the deck and shown here; the talk button says yes.
static void askNotes() {
  upAsked = true;
  askedMs = millis();
  touch();
  buzz(60);
  ui::fade();
  mode = Mode::Ask;
  ui::drawNow();  // the e-paper too: a question waits for nobody
  redraw();
  JsonDocument d;
  d["type"] = "say";
  d["text"] = "There are saved voice notes. Would you like to send them to the agent?";
  sendJson(d);
}

// A yes (here, or Menu → Send voice notes): they go up from the next pass.
static void sendNotesNow() {
  if (!wsUp || !agentLinked) {
    notice("Offline", "The notes go up once the " + String(board::kNoun) + " reaches the agent.", kRed);
    return;
  }
  upAllowed = upAsked = true;
  notesDeclined = -1;
  upRetryAtMs = millis();
  notice("Sending voice notes", notesCount() + " going to the agent. Its answer comes as usual.", kGreen);
}

static void answerAsk(bool yes) {
  ui::drawNow();
  if (yes) {
    sendNotesNow();
  } else {
    notesDeclined = notes::count();
    backToMain();
  }
}

static void abortUpload(bool tellDeck) {
  if (upNotes.empty() && !upAwaitAck) return;
  if (upOpen) notes::closeRead();
  if (tellDeck && wsUp && !upAwaitAck) sendType("notes_cancel");
  upNotes.clear();
  upIndex = 0;
  upOpen = upAwaitAck = upSentAny = false;
  upRetryAtMs = millis() + 20000;
}

// After the batch's last note: the deck transcribes what it took.
static void finishBatch(uint32_t now) {
  if (!upSentAny) {  // nothing readable went up
    sendType("notes_cancel");
    upNotes.clear();
    upRetryAtMs = now + 20000;
    return;
  }
  JsonDocument e;
  e["type"] = "notes_end";
  telemetry::fill(e["ctx"].to<JsonObject>());
  sendJson(e);
  if (!wsUp) return;  // dropped: abortUpload has reset everything
  upAwaitAck = true;
  upAtMs = now;
}

static void pumpUpload() {
  if (!board::kOfflineNotes) return;
  const uint32_t now = millis();
  if (mode == Mode::Ask && now - askedMs > 60000) backToMain();  // no answer: asked again on the next link
  if (upAwaitAck) {
    if (now - upAtMs > 300000) {  // no answer (a deck that doesn't know notes): try much later
      upAwaitAck = false;
      upNotes.clear();
      upRetryAtMs = now + 600000;
    }
    return;
  }
  if (!wsUp) {
    abortUpload(false);
    return;
  }
  if (mode == Mode::Listening || mode == Mode::Waiting || prerollOn || audio::micActive()) {
    abortUpload(true);  // they're talking: the notes wait
    return;
  }
  if (audio::playing()) return;  // never over a voice
  // Every send below can find the link gone: onWs → abortUpload runs inside
  // it and clears the batch, so each one is followed by a check.
  if (upNotes.empty()) {
    bool idle = mode == Mode::Idle || mode == Mode::Reply || mode == Mode::Home || mode == Mode::Image ||
                mode == Mode::Notice || mode == Mode::Backdrop;
    if (!notes::count() || !agentLinked || !idle || (int32_t)(now - upRetryAtMs) < 0) return;
    if (!upAllowed) {
      if (!upAsked && notes::count() != notesDeclined) askNotes();
      return;
    }
    upNotes = notes::list(10);
    if (upNotes.empty()) return;
    upIndex = 0;
    upOpen = upSentAny = false;
    JsonDocument d;
    d["type"] = "notes_begin";
    d["count"] = upNotes.size();
    sendJson(d);
    if (!wsUp || upNotes.empty()) return;
  }
  const String id = upNotes[upIndex].id;
  if (!upOpen) {
    const notes::Note n = upNotes[upIndex];
    uint32_t rate = audio::kMicRate;
    if (!notes::openRead(n, &rate)) {  // not a readable note: set aside, never sent again
      notes::quarantine(id);
      if (++upIndex >= upNotes.size()) finishBatch(now);
      return;
    }
    upOpen = true;
    JsonDocument d;
    d["type"] = "note_start";
    d["id"] = id;
    d["rate"] = rate;
    d["seconds"] = (float)n.samples / rate;
    if (n.when) {
      d["epoch"] = (uint32_t)n.when;
      d["recorded"] = localTimeText(n.when);
    }
    sendJson(d);
    if (!wsUp || upNotes.empty()) return;
  }
  static uint8_t chunk[4096];
  for (int i = 0; i < 8; ++i) {  // ~32 KB a pass keeps the loop lively
    size_t got = notes::read(chunk, sizeof(chunk));
    if (got) {
      ws.sendBIN(chunk, got);
      if (!wsUp || upNotes.empty()) return;
      continue;
    }
    notes::closeRead();
    upOpen = false;
    JsonDocument d;
    d["type"] = "note_end";
    d["id"] = id;
    sendJson(d);
    if (!wsUp || upNotes.empty()) return;
    upSentAny = true;
    if (++upIndex >= upNotes.size()) finishBatch(now);
    break;
  }
}

static void pollFlash() {
  if (!flashing) return;
  uint32_t el = millis() - flashStartMs;
  static int lastPhase = -1;
  int phase = el / 100;
  if (el >= kFlashMs) {
    flashing = false;
    lastPhase = -1;
    ui::brightness(kBright);
    redraw();
    return;
  }
  if (phase != lastPhase) {
    lastPhase = phase;
    if (phase % 2 == 0) ui::flashFrame(true);
    else redraw();
  }
}

// ── replay: the last answer again ────────────────────────────────────

// From memory, at once. A picture stays up while it plays, except in icon
// mode, where the character speaks.
static void replayLocal() {
  if (!audio::hasClip()) {
    notice("Nothing to replay", "The last answer had no sound.", kText);
    return;
  }
  audio::replay();
  if (settings.iconMode && mode == Mode::Image) mode = Mode::Reply;
  if (!flashing) redraw();
}

static void replayAnswer() {
  if (audio::clipCut() && wsUp) {
    // Part of it never arrived (stopped early, or the link dropped): the
    // deck sends the whole answer again.
    deckReplay = true;
    deckReplayMs = millis();
    quietSpeech = true;
    sendType("replay");
  } else {
    replayLocal();
  }
}

// ── server → device ──────────────────────────────────────────────────

static void onText(const char* payload, size_t len) {
  JsonDocument doc;
  if (deserializeJson(doc, payload, len)) return;
  String type = doc["type"] | "";

  if (type == "bound") {
    agentName = doc["agent"] | "";
    strlcpy(rtcAgent, agentName.c_str(), sizeof(rtcAgent));
    agentSlug = doc["slug"] | "";
    agentLinked = doc["linked"] | true;
    if (mode == Mode::Agents || mode == Mode::Idle || mode == Mode::Notice) backToMain();
    else redraw();
  } else if (type == "agents") {
    agentNames.clear();
    agentSlugs.clear();
    agentCur = -1;
    String current = doc["current"] | "";
    for (JsonObject a : doc["items"].as<JsonArray>()) {
      String slug = a["slug"] | "";
      if (slug == current) agentCur = agentNames.size();
      agentNames.push_back(a["name"] | slug);
      agentSlugs.push_back(slug);
    }
    agentNames.push_back("« Back");
    agentSlugs.push_back("");
    agentSel = max(0, agentCur);
    agentsLoading = false;
    if (mode == Mode::Agents) {
      redraw();
      sayRow();
    }
  } else if (type == "state") {
    String st = doc["state"] | "";
    if (st == "connected") {
      agentLinked = true;
    } else if (st == "agent_offline") {
      agentLinked = false;
    } else if (st == "transcribing") {
      statusLine = "Transcribing…";
      leds::action(leds::Step::Model);  // speech to text
    } else if (st == "thinking") {
      statusLine = "Thinking…";
      turnDropMs = turnRelinkMs = 0;  // (after a dropped link: the deck still has the turn)
      leds::action(leds::Step::Model);
      if (mode != Mode::Waiting && mode != Mode::Listening) {
        startWork();  // a typed or resent turn: no recording came first
        mode = Mode::Waiting;
      }
    } else if (st == "ready" && mode == Mode::Waiting) {
      mode = ui::hasReply() ? Mode::Reply : Mode::Idle;
    }
    if (mode != Mode::Listening && !flashing) redraw();
  } else if (type == "transcript") {
    heard = doc["text"] | "";
    ui::setQuestion(heard);
    statusLine = "Thinking…";
    workDirty = true;
    leds::action(leds::Step::Narration);  // what it heard
  } else if (type == "status") {
    statusLine = doc["text"] | "";
    workDirty = true;
    leds::action(leds::Step::Narration);
  } else if (type == "activity") {
    String line = doc["text"] | "";
    String tool = doc["tool"] | "";
    if (line.length()) {
      activity.push_back(line);
      if (activity.size() > 8) activity.erase(activity.begin());
      bool isTool = tool.length() && tool != "progress";
      if (isTool) ++toolSteps;
      workDirty = true;
      leds::action(isTool ? leds::Step::Tool : leds::Step::Narration);
    }
  } else if (type == "reply") {
    touch();
    ++turns;
    Serial.printf("[claw] reply; loop stack never below %u bytes free\n", (unsigned)uxTaskGetStackHighWaterMark(nullptr));
    lastReplyMs = millis();
    String said = doc["text"] | "";
    spokenChars = doc["spoken"] | 0;
    replyChars = said.length();
    paceLearned = false;
    ui::setReply(said);
    face::setText(said);
    audio::newClip();  // this answer's sound replaces the last one's
    deckReplay = quietSpeech = false;
    mode = Mode::Reply;
    if (!flashing) redraw();
  } else if (type == "audio_start" && (doc["aside"] | false)) {
    asideOpen = true;  // a menu row read aloud: apart from the answer
    audio::asideBegin(doc["rate"] | 24000);
  } else if (type == "audio_end" && asideOpen) {
    asideOpen = false;
    audio::asideEnd();
  } else if (type == "audio_start") {
    asideOpen = false;  // an answer's sound: whatever came before is over
    if (deckReplay) {  // the deck's copy of the answer replaces what arrived
      deckReplay = false;
      audio::newClip();
    }
    startSpeech(doc["rate"] | 24000);
    setBarState();
  } else if (type == "audio_end") {
    audio::playEnd();
  } else if (type == "replay_none") {
    if (deckReplay) {  // the deck no longer has it: play what did arrive
      deckReplay = quietSpeech = false;
      replayLocal();
    }
  } else if (type == "update") {
    ota::Offer o;
    o.version = doc["version"] | "";
    o.size = doc["size"] | 0;
    o.sha256 = doc["sha256"] | "";
    o.sig = doc["sig"] | "";
    o.url = doc["url"] | "";
    o.notes = doc["notes"] | "";
    if (o.valid() && ota::compareVersions(o.version, FW_VERSION) > 0) {
      offer = o;
      bool asked = checkingUpdate;
      checkingUpdate = false;
      if ((asked || offerAnnounced != o.version) && !settings.autoUpdate &&
          (mode == Mode::Idle || mode == Mode::Reply || mode == Mode::Notice || mode == Mode::Menu)) {
        offerAnnounced = o.version;
        notice(("Update " + o.version + " available").c_str(),
               (o.notes.length() ? o.notes + " — " : String("")) + "Menu → Software update.", kCyan);
      }
    }
  } else if (type == "update_none") {
    if (checkingUpdate) {
      checkingUpdate = false;
      notice("Up to date", String("This is the latest version (" FW_VERSION ")."), kGreen);
    }
  } else if (type == "notes_done") {
    // The deck has these (and has sent them to the agent): gone from here.
    for (JsonVariant id : doc["ids"].as<JsonArray>()) notes::remove(id.as<String>());
    upAwaitAck = false;
    upNotes.clear();
    upRetryAtMs = millis() + 3000;  // more waiting: the next batch shortly
  } else if (type == "notes_failed") {
    upAwaitAck = false;
    upNotes.clear();
    upRetryAtMs = millis() + 600000;
    if (mode == Mode::Idle || mode == Mode::Reply || mode == Mode::Home || mode == Mode::Backdrop) {
      notice("Offline notes not sent", String(doc["text"] | "Flight Deck couldn't take them.") + " They stay on the " +
                                           board::kNoun + ".", kOrange);
    }
  } else if (type == "pong") {
    if (doc["t"].is<uint32_t>()) rttMs = millis() - doc["t"].as<uint32_t>();
  } else if (type == "background") {
    dropBackgroundPart();
    bgIncomingId = doc["id"] | "";
    bgIncomingSize = doc["size"] | 0;
    if (!bgIncomingSize) setBackground(nullptr, 0, "");  // cleared: back to black
  } else if (type == "error") {
    touch();
    hapticError();
    leds::alert();
    notice("Hmm", doc["text"] | "Something went wrong.", kOrange);
  }
}

static void dropImagePart() {
  free(imgPart);
  imgPart = nullptr;
  imgPartLen = 0;
}

// A picture: 0x03 frames are parts with more to come, 0x02 is the last (or
// only) part. A WebSocket frame here carries at most ~15 KB.
static void dropBackgroundPart() {
  free(bgPart);
  bgPart = nullptr;
  bgPartLen = 0;
}

// Shown, and kept in flash for the next boot (nullptr: back to black).
static void setBackground(const uint8_t* jpg, size_t len, const String& id) {
  ui::setBackground(jpg, len);
  bgId = len ? id : "";
  if (!len && mode == Mode::Backdrop) mode = ui::hasReply() ? Mode::Home : Mode::Idle;  // nothing left to show
  if (notes::ready()) {
    if (len) {
      File f = LittleFS.open(kBgFile, "w");
      const bool ok = f && f.write(jpg, len) == len;
      f.close();
      File idf = LittleFS.open(kBgIdFile, "w");
      if (idf) idf.print(ok ? id : "");
    } else {
      LittleFS.remove(kBgFile);
      LittleFS.remove(kBgIdFile);
    }
  }
  if (mode != Mode::Listening && !flashing) redraw();
}

// At boot: the background kept from last time.
static void loadBackground() {
  if (!board::kRound || !notes::ready() || !LittleFS.exists(kBgFile)) return;
  File f = LittleFS.open(kBgFile, "r");
  const size_t len = f ? f.size() : 0;
  uint8_t* buf = len && len <= kMaxImage ? (uint8_t*)ps_malloc(len) : nullptr;
  if (buf && f.read(buf, len) == len) {
    ui::setBackground(buf, len);
    File idf = LittleFS.open(kBgIdFile, "r");
    bgId = idf ? idf.readString() : String();
    bgId.trim();
  }
  free(buf);
}

static void backgroundData(const uint8_t* data, size_t len, bool last) {
  const size_t need = bgPartLen + len;
  uint8_t* grown = bgIncomingSize && need <= kMaxImage ? (uint8_t*)ps_realloc(bgPart, need) : nullptr;
  if (!grown) {
    dropBackgroundPart();
    return;
  }
  memcpy(grown + bgPartLen, data, len);
  bgPart = grown;
  bgPartLen = need;
  if (!last) return;
  if (bgPartLen == bgIncomingSize) setBackground(bgPart, bgPartLen, bgIncomingId);
  dropBackgroundPart();
  bgIncomingSize = 0;
}

static void imageData(const uint8_t* data, size_t len, bool last) {
  size_t need = imgPartLen + len;
  uint8_t* grown = need <= kMaxImage ? (uint8_t*)ps_realloc(imgPart, need) : nullptr;
  if (!grown) {
    dropImagePart();
    return;
  }
  memcpy(grown + imgPartLen, data, len);
  imgPart = grown;
  imgPartLen = need;
  if (!last) return;
  touch();
  ui::storeImage(imgPart, imgPartLen);  // the screen keeps the bytes
  imgPart = nullptr;
  imgPartLen = 0;
  if (board::kEpaper && ui::hasReply()) {
    // The answer card has room for it beside the words.
    if (mode == Mode::Reply || mode == Mode::Idle) redraw();
  } else if (!(settings.iconMode && audio::playing())) {
    mode = Mode::Image;
    if (!flashing) redraw();
  }
}

static void onBinary(const uint8_t* data, size_t len) {
  if (len < 2) return;
  if (data[0] == 0x01) {
    if (asideOpen) audio::asideFeed(data + 1, len - 1);
    else audio::playFeed(data + 1, len - 1);
  } else if (data[0] == 0x02 || data[0] == 0x03) {
    imageData(data + 1, len - 1, data[0] == 0x02);
  } else if (board::kRound && (data[0] == 0x04 || data[0] == 0x05)) {
    backgroundData(data + 1, len - 1, data[0] == 0x05);
  }
}

static uint32_t wsEvents = 0;  // counts everything the WebSocket delivers

static void onWs(WStype_t type, uint8_t* payload, size_t length) {
  ++wsEvents;
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("[claw] flight deck connected");
      wsUp = true;
      rttMs = -1;
      if (turnDropMs) turnRelinkMs = max<uint32_t>(millis(), 1);
      sendPrefs();
      redraw();
      break;
    case WStype_DISCONNECTED:
      if (wsUp) {
        Serial.println("[claw] flight deck disconnected");
        wsUp = false;
        agentLinked = false;
        if (audio::micActive()) audio::micStop([](const int16_t*, size_t) {});
        audio::playStop();
        dropImagePart();
        dropBackgroundPart();  // sent again in full on the next link (prefs names the old one)
        bgIncomingSize = 0;
        deckReplay = false;
        asideOpen = false;
        if (prerollOn) {  // the deck dropped that recording: send it all again on the next link
          prerollStreaming = false;
          prerollSent = 0;
        }
        abortUpload(false);  // the notes go up again next time — asked first
        upAllowed = upAsked = false;
        if (!prerollOn) tapRec = false;  // a tapped recording that streamed: gone with the mic
        if (mode == Mode::Waiting) {  // the deck keeps the answer: wait for the link (holdTurn)
          if (!turnDropMs) turnDropMs = max<uint32_t>(millis(), 1);
          turnRelinkMs = 0;
          statusLine = "Connection lost — reconnecting…";
          workDirty = true;
        } else if (mode == Mode::Listening || mode == Mode::Ask) {
          mode = ui::hasReply() ? Mode::Reply : Mode::Idle;
        }
        redraw();
      }
      break;
    case WStype_TEXT:
      onText((const char*)payload, length);
      break;
    case WStype_BIN:
      onBinary(payload, length);
      break;
    default:
      break;
  }
}

static void connectServer() {
  String url = settings.server;
  bool tls = url.startsWith("wss://");
  url = url.substring(tls ? 6 : 5);
  int slash = url.indexOf('/');
  String hostport = slash < 0 ? url : url.substring(0, slash);
  String prefix = slash < 0 ? "" : url.substring(slash);
  int colon = hostport.lastIndexOf(':');
  String host = colon < 0 ? hostport : hostport.substring(0, colon);
  uint16_t port = colon < 0 ? (tls ? 443 : 80) : hostport.substring(colon + 1).toInt();
#ifdef OTA_TEST_NO_DECK
  // Rollback test build (PLATFORMIO_BUILD_FLAGS=-DOTA_TEST_NO_DECK): Wi-Fi works,
  // Flight Deck is never reached, so an update to this build must roll back.
  host = "deck.invalid";
#endif

  int sw = board::kScreenW, sh = board::kScreenH;
  if (settings.rotation & 1 && board::kEpaper) std::swap(sw, sh);  // turned paper
  String path = prefix + "/device/ws?device=" + urlEncode(deviceId()) + "&sw=" + String(sw) + "&sh=" + String(sh);
  if (board::kRound) path += "&round=1";               // fit pictures inside the circle
  if (board::kEpaper) path += "&paper=1";              // pictures at their own shape, on white
  if (board::kChunkedImages) path += "&img=chunks";    // pictures bigger than one frame
  if (settings.token.length()) path += "&token=" + urlEncode(settings.token);
  if (settings.agent.length()) path += "&agent=" + urlEncode(settings.agent);
  if (settings.lang.length()) path += "&lang=" + urlEncode(settings.lang);
  path += String("&voice=") + (settings.femaleVoice ? "female" : "male");
  path += String("&hw=") + board::kHw + "&fw=" FW_VERSION;
  if (!settings.speak) path += "&speak=0";

  if (tls) {
    ws.beginSSL(host.c_str(), port, path.c_str(), "", "");  // no pinning (v1)
  } else {
    ws.begin(host.c_str(), port, path.c_str(), "");
  }
  // The library sends "Origin: file://" by default — Flight Deck's origin guard
  // (rightly) refuses that as a cross-site browser. A device sends no Origin.
  ws.setExtraHeaders("");
  ws.onEvent(onWs);
  ws.setReconnectInterval(3000);
  // Weak Wi-Fi holds a pong back for seconds (retransmits): three missed
  // 10 s windows, not two of 5 s, before the link counts as gone.
  ws.enableHeartbeat(15000, 10000, 3);
}

// ── over-the-air updates (docs/device-ota-plan.md) ───────────────────

static String httpBase() {
  String b = settings.server;
  if (b.startsWith("wss://")) b = "https://" + b.substring(6);
  else if (b.startsWith("ws://")) b = "http://" + b.substring(5);
  while (b.endsWith("/")) b.remove(b.length() - 1);
  return b;
}

static void installProgress(size_t done, size_t total) {
  static int lastPct = -1;
  int pct = total ? (int)(done * 100 / total) : 0;
  if (pct == lastPct || (pct % 2 && pct != 100)) return;
  lastPct = pct;
  ui::showMessage(("Updating to " + offer.version).c_str(),
                  String(pct) + "%  ·  " + String(done / 1024) + " / " + String(total / 1024) +
                      " KB. Keep it on.",
                  kCyan);
}

static void startInstall() {
  if (!offer.valid()) return;
  if (mode == Mode::Listening || mode == Mode::Waiting || audio::playing()) {
    notice("Not now", "Finish the conversation first.", kOrange);
    return;
  }
  int pct = M5.Power.getBatteryLevel();
  if (!telemetry::onUsbPower() && pct >= 0 && pct < 30) {
    notice("Battery low", "Charge to 30% (or plug in) to update.", kOrange);
    return;
  }
  touch();
  audio::playStop();
  ws.disconnect();  // one TLS connection at a time; we restart anyway
  wsUp = false;
  agentLinked = false;
  ui::setHealth(wifiHealth(), ui::Partial);
  ui::setBar(agentName, "updating", kCyan);
  ui::showMessage(("Updating to " + offer.version).c_str(), String("Downloading… keep it on."), kCyan);
  leds::set(leds::State::Updating);
  leds::tick();
  ui::flush();  // e-paper: on the paper before the download takes over
  String err = ota::install(offer, httpBase(), settings.token, installProgress);
  if (err.isEmpty()) {
    ui::showMessage("Update installed", "Restarting into " + offer.version + "…", kGreen);
    delay(1200);
    ESP.restart();
  }
  lastActivityMs = millis();
  notice("Update failed", err + ". Still on " FW_VERSION ".", kOrange);
  connectServer();
}

// ── power ────────────────────────────────────────────────────────────

static volatile uint32_t buttonSeenMs = 0;

static String sleepHint() {
  return board::kEpaper ? String(board::kTalkHint) : String("Press the talk button to wake.");
}

#if defined(BOARD_PAPERCOLOR)
// The PaperColor's power chip (charging, and the status lights' supply).
static M5PM1 pm1;
static bool pm1Up = false;

// The two RGB lights run off the PMIC's 5 V boost converter (M5Stack's own
// PaperColor firmware switches it on at start), which starts off: without it
// the lights get their colours but no power.
static void paperLights(bool on) {
  if (pm1Up) pm1.setBoostEnable(on);
}
#endif

// Deep sleep. The talk button wakes it (on the paper any button does, and
// a timer redraws the sleeping screen now and then).
[[noreturn]] static void sleepNow() {
  leds::off();
#if defined(BOARD_PAPERCOLOR)
  paperLights(false);  // nothing to show asleep
#endif
  ui::brightness(0);
  // The paper's panel powers itself down after every redraw and keeps its
  // picture; its controller's deep sleep would need a reset pulse to wake,
  // which a boot that keeps the picture doesn't send.
  if (!board::kEpaper) {
    M5.Display.sleep();
    M5.Display.waitDisplay();
  }
  if (board::kEpaper) {
    for (gpio_num_t pin : {board::kTalkPin, board::kSidePin, board::kBackPin}) {
      if (pin < 0) continue;
      rtc_gpio_pullup_en(pin);
      rtc_gpio_pulldown_dis(pin);
    }
    esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
    esp_sleep_enable_ext1_wakeup(kButtonPins, ESP_EXT1_WAKEUP_ANY_LOW);
    if (settings.asleepRefreshMin) esp_sleep_enable_timer_wakeup(settings.asleepRefreshMin * 60ULL * 1000000ULL);
  } else {
    esp_sleep_enable_ext0_wakeup(board::kTalkPin, 0);  // the talk button pulls low
  }
  esp_deep_sleep_start();
}

static void goToSleep() {
  audio::playStop();
  uint32_t t0 = millis();
  if (board::kEpaper) {
    ui::setBar(agentName, "asleep", kText);
    ui::showSleeping(sleepHint(), settings.character);  // waits for the paper (~20 s)
    if ((int32_t)(buttonSeenMs - t0) > 0) {  // pressed meanwhile: stay awake
      touch();
      backToMain();
      return;
    }
  } else {
    ui::showSleeping(String("Press the ") + board::kTalkButton + " to wake.", settings.character);
    delay(1200);
  }
  ws.disconnect();
  WiFi.disconnect(true);
  if (board::kHasMotor) M5.Power.setVibration(0);
  sleepNow();
}

// "Nothing happening" = no button, no new reply, and not listening, speaking,
// flashing or waiting for the agent: then the screen goes off, later the
// stick powers off (deep sleep; the front button wakes it).
static void handleIdlePower() {
  if (mode == Mode::Listening || mode == Mode::Waiting || mode == Mode::Link || audio::sounding() || flashing ||
      probation || prerollOn || !upNotes.empty() || upAwaitAck) {  // (an upload ends by the ack timeout at worst)
    lastActivityMs = millis();
    return;
  }
  uint32_t idle = millis() - lastActivityMs;
  uint32_t offMs = settings.screenOffS * 1000UL;
  if (!board::kEpaper) {  // paper has no backlight: what it shows costs nothing
    if (!dimmed && idle > offMs - min(offMs / 2, kDimLeadMs)) {
      ui::brightness(24);
      dimmed = true;
    }
    if (!screenOff && idle > offMs) {
      ui::brightness(0);
      screenOff = true;
    }
  }
  if (idle > settings.powerOffMs() && !ui::busy()) goToSleep();
}

// The buttons are sampled every 20 ms off the main loop, so any press (or a
// held button) counts as activity even when a loop pass is slow — drawing,
// TLS, saving settings — and a quick click would slip between two passes.
// It also notes when a talk-from-sleep press ends, even while setup() waits
// for Wi-Fi.
static void sampleButtons(void*) {
  bool talk = gpio_get_level(board::kTalkPin) == 0;
  if (talk || gpio_get_level(board::kSidePin) == 0 || (board::kBackPin >= 0 && gpio_get_level(board::kBackPin) == 0)) {
    buttonSeenMs = millis();
  }
  if (prerollOn && !prerollEndMs) {
    // Let go (held to talk), or pressed again (a toggled recording — which
    // talk-from-sleep is too, before the loop runs).
    bool ends = tapRec ? board::kTalkToggles && talk && !talkWasDown : !talk;
    if (ends) {
      uint32_t t = millis();
      prerollEndMs = t ? t : 1;
      tapRec = false;
    }
  }
  talkWasDown = talk;
}

static void noteButtonActivity() {
  uint32_t seen = buttonSeenMs;
  if ((int32_t)(seen - lastActivityMs) > 0) {
    lastActivityMs = seen;
    if (dimmed && !screenOff) {  // brighten back; a dark screen wakes on the press itself
      ui::brightness(kBright);
      dimmed = false;
    }
  }
}

// ── side button ──────────────────────────────────────────────────────

// An answer on screen: a side double-click there replays it, so a single
// click waits kDoubleMs to see whether a second follows.
static bool onAnswer() { return mode == Mode::Reply || mode == Mode::Image; }

static void noteSideClick(bool wokeOnly) {
  sideClickMs = millis();
  if (!sideClickMs) sideClickMs = 1;
  sideClickMode = mode;
  sideClickWoke = wokeOnly;
}

// One side click: next row / page, or back to the main screen.
static void sideClick() {
  ui::drawNow();  // a page, the status: asked to see it (the e-paper draws it now)
  if (mode == Mode::Menu || mode == Mode::CharPick || (mode == Mode::Agents && !agentsLoading) ||
      (mode == Mode::Status && statusPage + 1 < statusPages) ||
      (mode == Mode::Reply && !(audio::playing() && settings.iconMode))) {
    ui::slide(true);  // a step on
  } else if (mode == Mode::Ask || mode == Mode::Status || mode == Mode::Image || mode == Mode::Notice ||
             mode == Mode::Link || mode == Mode::Idle || mode == Mode::Home || mode == Mode::Backdrop) {
    ui::fade();  // another place
  }
  if (mode == Mode::Ask) {
    answerAsk(false);
  } else if (mode == Mode::Menu) {
    do menuSel = (menuSel + 1) % MCount;
    while (!rowShown(menuSel));
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::CharPick) {
    charSel = (charSel + 1) % face::KindCount;
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::Agents) {
    if (!agentsLoading) agentSel = (agentSel + 1) % agentNames.size();
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::Status) {
    if (statusPage + 1 < statusPages) {
      ++statusPage;
      redraw();
    } else {
      backToMain();  // after the last page
    }
  } else if (mode == Mode::Image || mode == Mode::Notice || mode == Mode::Link) {
    backToMain();
  } else if (mode == Mode::Reply) {
    if (audio::playing() && settings.iconMode) {
      audio::playStop();  // the character gives way to the text
      sendType("stop");
      redraw();
    } else {
      ui::nextPage();
    }
  } else if (mode == Mode::Idle || mode == Mode::Home) {
    statusPage = 0;
    mode = Mode::Status;
    redraw();
  } else if (mode == Mode::Backdrop) {
    leaveBackdrop();
  }
}

// The PaperColor's C button: one step back — the row or page before, else
// out to the main screen; from the home screen, to the answer.
static void backClick() {
  ui::drawNow();  // back, the answer, the home screen: asked to see it
  if (mode == Mode::Menu || mode == Mode::CharPick || (mode == Mode::Agents && !agentsLoading) ||
      (mode == Mode::Status && statusPage > 0)) {
    ui::slide(false);  // a step back
  } else if (mode == Mode::Ask || mode == Mode::Status || mode == Mode::Reply || mode == Mode::Image ||
             mode == Mode::Notice || mode == Mode::Link || (mode == Mode::Home && ui::hasReply())) {
    ui::fade();
  }
  if (mode == Mode::Ask) {
    answerAsk(false);
  } else if (mode == Mode::Menu) {
    do menuSel = (menuSel + MCount - 1) % MCount;
    while (!rowShown(menuSel));
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::CharPick) {
    charSel = (charSel + face::KindCount - 1) % face::KindCount;
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::Agents) {
    if (!agentsLoading) agentSel = (agentSel + agentNames.size() - 1) % agentNames.size();
    menuOpenedMs = millis();
    redraw();
    sayRow();
  } else if (mode == Mode::Status) {
    if (statusPage > 0) {
      --statusPage;
      redraw();
    } else {
      backToMain();
    }
  } else if (mode == Mode::Reply) {
    // The home screen, from any page (A pages on); it shows the newest
    // answer too, so one still waiting for a quiet spell is drawn now.
    mode = Mode::Home;
    redraw();
  } else if (mode == Mode::Home || (mode == Mode::Idle && !ui::hasReply())) {
    if (ui::hasReply() || ui::restoreAnswer()) {  // the answer from before a sleep, too
      mode = Mode::Reply;
      redraw();
    }
  } else if (mode == Mode::Image || mode == Mode::Notice || mode == Mode::Link) {
    backToMain();
  }
}

// ── status lights ────────────────────────────────────────────────────

static int micLevel = 0;
static uint32_t linkChangeMs = 0;  // when the deck link last came or went

static void updateLeds() {
  using S = leds::State;
  static int batteryPct = 100;
  static uint32_t batteryAt = 0;
  if (millis() - batteryAt > 30000 || !batteryAt) {
    batteryAt = millis();
    batteryPct = telemetry::onUsbPower() ? 100 : M5.Power.getBatteryLevel();
  }
  if (prerollOn || mode == Mode::Listening) leds::set(S::Listening, micLevel);
  else if (audio::playing()) leds::set(S::Speaking, audio::outputLevel());
  else if (mode == Mode::Waiting) leds::set(S::Working);
  else if (leds::startingUp()) leds::set(S::Startup);
  else if (!wsUp && millis() - linkChangeMs < 60000) leds::set(S::Connecting);  // not forever
  else if (ui::busy()) leds::set(S::Redrawing);
  else if (batteryPct >= 0 && batteryPct < 15) leds::set(S::LowBattery);
  else leds::setIdle(millis() - lastActivityMs, settings.powerOffMs());
  // Since start-up (this boot or wake): Wi-Fi gets the 30 s setup gives it,
  // the agent a few seconds once the deck is reached.
  using L = leds::Link;
  const L wifi = WiFi.status() == WL_CONNECTED ? L::Up : millis() > 30000 ? L::Down : L::Trying;
  const L agent = wsUp && agentLinked ? L::Up
                  : wifi == L::Down || millis() > 60000 || (wsUp && millis() - linkChangeMs > 8000) ? L::Down
                                                                                                   : L::Trying;
  leds::links(wifi, agent);
  leds::tick();
}

// ── PaperColor power ─────────────────────────────────────────────────

#if defined(BOARD_PAPERCOLOR)
// Charging on (M5Stack's own firmware does this through the PMIC); an empty
// battery leaves "charge me" on the paper — which costs nothing to keep —
// and switches off.
static void paperPower() {
  pm1Up = pm1.begin(&M5.In_I2C, M5PM1_DEFAULT_ADDR, M5PM1_I2C_FREQ_100K) == M5PM1_OK;
  if (pm1Up) pm1.setChargeEnable(true);
  paperLights(settings.ledLevel > 0);
  int mv = M5.Power.getBatteryVoltage();
  if (mv > 0 && mv < 3300 && !telemetry::onUsbPower()) {
    ui::showMessage("Battery empty", "Charge it over USB-C, then press the power button.", kRed);
    ui::flush();
    M5.Power.powerOff();
  }
}
#endif

// A timer woke the paper: redraw the sleeping screen (date, room, battery)
// without Wi-Fi and sleep again — unless a button was pressed meanwhile.
static void asleepRedraw() {
  uint32_t t0 = millis();
  ui::setBar(rtcAgent, "asleep", kText);
  ui::showSleeping(sleepHint(), settings.character);
  if ((int32_t)(buttonSeenMs - t0) > 0 || gpio_get_level(board::kTalkPin) == 0) return;  // awake after all
  sleepNow();
}

// Back onto Wi-Fi without stalling the loop: WiFiMulti scans synchronously
// (2–3 s a call, every pass while offline). Instead an async scan every
// 15 s, then the strongest saved network in reach.
static void keepWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  static uint32_t lastScanMs = 0;
  int found = WiFi.scanComplete();
  if (found == WIFI_SCAN_RUNNING) return;
  if (found >= 0) {
    int best = -1, bestRssi = -1000;
    for (int i = 0; i < found; ++i) {
      for (int k = 0; k < settings.netCount; ++k) {
        if (WiFi.SSID(i) == settings.nets[k].ssid && WiFi.RSSI(i) > bestRssi) {
          best = k;
          bestRssi = WiFi.RSSI(i);
        }
      }
    }
    WiFi.scanDelete();
    if (best >= 0) WiFi.begin(settings.nets[best].ssid.c_str(), settings.nets[best].pass.c_str());
    return;
  }
  if (!lastScanMs || millis() - lastScanMs > 15000) {
    lastScanMs = millis();
    // Refused while the driver keeps retrying the last network (it never
    // gives up): stop that, so the next try can scan for another one.
    if (WiFi.scanNetworks(true) == WIFI_SCAN_FAILED) {
      WiFi.disconnect();
      lastScanMs = millis() - 13000;
    }
  }
}

// ── setup / loop ─────────────────────────────────────────────────────

// At start-up, one scan (~2 s): the strongest saved network in reach (its
// index in settings.nets), -1 when none is, -2 when the scan failed.
static int strongestSavedNetwork() {
  const int n = WiFi.scanNetworks();
  if (n < 0) return -2;
  int best = -1, bestRssi = -1000;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < settings.netCount; ++k) {
      if (WiFi.SSID(i) == settings.nets[k].ssid && WiFi.RSSI(i) > bestRssi) {
        best = k;
        bestRssi = WiFi.RSSI(i);
      }
    }
  }
  WiFi.scanDelete();
  Serial.printf("[claw] wi-fi scan: %d networks, %s\n", n,
                best >= 0 ? settings.nets[best].ssid.c_str() : "none of the saved ones");
  return best;
}

void setup() {
  auto cfg = M5.config();
  cfg.internal_mic = true;
  cfg.internal_spk = true;
  if (board::kEpaper) cfg.clear_display = false;  // the paper keeps what it showed while asleep
  M5.begin(cfg);
  Serial.begin(115200);
  settings.load();
  setenv("TZ", settings.tz.c_str(), 1);  // local time before NTP (the RTC keeps UTC)
  tzset();
  audio::begin(settings.volume, settings.boostDb);
  // Woken by holding the talk button: they're speaking already, so the mic
  // starts before anything slow (it records while Wi-Fi connects).
  bool talking = settings.configured() && wokenToTalk();
  if (talking) {
    tapRec = board::kTalkToggles;  // ends with the next press, not on letting go
    startPreroll();
  }
  sideBtn().setHoldThresh(800);
  if (board::kBackPin >= 0) backBtn().setHoldThresh(800);
  esp_timer_create_args_t sampler = {};
  sampler.callback = sampleButtons;
  sampler.name = "buttons";
  esp_timer_handle_t samplerTimer;
  if (esp_timer_create(&sampler, &samplerTimer) == ESP_OK) esp_timer_start_periodic(samplerTimer, 20000);

  ui::begin();
  ui::setRotation(settings.rotation);
  ui::brightness(kBright);
  ui::setTextSize(settings.textSize);
  telemetry::begin();
  leds::begin();
  leds::setLevel(settings.ledLevel);
#if defined(BOARD_PAPERCOLOR)
  paperPower();
#endif
  if (board::kEpaper && esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_TIMER) {
    asleepRedraw();  // returns only when a button was pressed meanwhile
    // Pressed during the redraw: awake, and the next press talks (a recording
    // started now would begin long after that press).
    if (settings.configured() && !board::kTalkToggles && gpio_get_level(board::kTalkPin) == 0) {
      talking = true;
      startPreroll();
    }
  }
  notes::begin();
  loadBackground();
  probation = ota::onProbation();
  Serial.printf("[claw] boot %s fw %s%s%s: %d network(s), server '%s'\n", deviceId().c_str(), FW_VERSION,
                probation ? " (on probation)" : "", talking ? " (talking)" : "", settings.netCount,
                settings.server.c_str());

  M5.update();
  if ((!talking && talkBtn().isPressed()) || !settings.configured()) {
    runSetupPortal(settings.configured() ? "Setup" : "First-time setup");
  }

  ui::setHealth(ui::Down, ui::Down);
  ui::setBar("", "wi-fi…", kYellow);
  leds::startup();  // both yellow; then left = Wi-Fi, right = the agent
  if (!prerollOn) leds::set(leds::State::Startup);
  leds::tick();
  // The paper keeps the sleeping screen while it connects (a redraw would
  // take longer than connecting); the lights say how it's going.
  if (!board::kEpaper) ui::showMessage("Connecting to Wi-Fi", settings.nets[0].ssid, kText);
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(deviceId().c_str());
  for (int i = 0; i < settings.netCount; ++i) {
    wifiMulti.addAP(settings.nets[i].ssid.c_str(), settings.nets[i].pass.c_str());
  }
  // A device that keeps voice notes carries on offline when no Wi-Fi is in
  // reach (it keeps looking); the others open setup. Setup stays a cold
  // power-on with the talk button held, or Menu → Wi-Fi & settings.
  const bool goesOffline = board::kOfflineNotes && notes::ready();
  bool offline = false;
  uint32_t t0 = millis();
  // Look first: with none of the saved networks in reach, a device that
  // keeps notes is a note taker at once (keepWifi() goes on looking) instead
  // of trying for 30 s; with one, it joins the strongest straight away. (A
  // fresh update on probation takes the slow way: it may need to roll back.)
  if (!probation) {
    const int net = strongestSavedNetwork();
    if (net == -1 && goesOffline) {
      offline = true;
    } else if (net >= 0) {
      WiFi.begin(settings.nets[net].ssid.c_str(), settings.nets[net].pass.c_str());
      while (WiFi.status() != WL_CONNECTED && millis() - t0 < 12000) {
        M5.update();
        delay(50);
      }
    }
  }
  while (!offline && WiFi.status() != WL_CONNECTED && wifiMulti.run(5000) != WL_CONNECTED) {
    M5.update();
    // A fresh update that can't even reach Wi-Fi goes back to the old version
    // (which may well be fine) rather than into setup.
    if (probation && millis() - t0 > 30000) ota::rollBack();
    if (probation) continue;
    // A talk-from-sleep recording already let go: no point waiting the full 30 s.
    bool spoken = prerollOn && prerollEndMs && millis() - t0 > 8000;
    if (goesOffline && (millis() - t0 > 30000 || spoken)) {
      offline = true;
      break;
    }
    if (!goesOffline && ((!prerollOn && talkBtn().wasPressed()) || millis() - t0 > 30000)) {
      if (prerollOn) stopPreroll();
      runSetupPortal("No Wi-Fi in reach");
    }
  }
  leds::links(offline ? leds::Link::Down : leds::Link::Up, offline ? leds::Link::Down : leds::Link::Trying);
  leds::tick();
  if (offline) {
    Serial.println("[claw] no wi-fi in reach: carrying on offline (voice notes are kept)");
  } else {
    Serial.printf("[claw] wifi %s ip %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
    ui::setHealth(wifiHealth(), ui::Down);
    if (!board::kEpaper) ui::showMessage("Wi-Fi connected", WiFi.SSID() + "  " + WiFi.localIP().toString(), kGreen);
  }
  configTzTime(settings.tz.c_str(), "pool.ntp.org", "time.google.com");  // clocks; the RTC follows
  connectServer();  // it connects by itself once Wi-Fi is back
  lastActivityMs = millis();
  if (offline) redraw();  // "Offline · talk to keep a note"
}

// A tapped (or toggled) recording dropped: a double press.
static void cancelTapRecording() {
  tapRec = false;
  if (prerollOn) {
    stopPreroll();
    backToMain();
  } else {
    stopTalking(false);
  }
}

// A recording that was tapped on ends: a second tap (or the talk button).
static void endTapRecording() {
  if (prerollOn) {  // offline: pumpPreroll keeps it
    uint32_t t = millis();
    prerollEndMs = t ? t : 1;
    tapRec = false;
  } else if (mode == Mode::Listening) {
    stopTalking(true);
  }
}

// A tapped or toggled recording (no button held: StopWatch Tap to talk, the
// PaperColor) ends by itself after 2.5 s without a voice and goes on as if
// tapped again; with nothing said at all, it's dropped. A held button ends
// its own recording.
static constexpr uint32_t kSilenceStopMs = 2500;

static void stopOnSilence() {
  static uint32_t watching = 0;  // the recording's start (a new one resets the rest)
  static uint32_t voiceMs = 0;   // when a voice was last heard in it (0 = not yet)
  static size_t scanned = 0;     // background-recording samples looked at so far
  static int loudRun = 0;        // …and its loud blocks in a row (kVoiceRun = a voice)
  if (!tapRec) return;
  const bool live = !prerollOn && mode == Mode::Listening && audio::micActive();
  if (!live && !(prerollOn && !prerollEndMs)) return;
  const uint32_t start = prerollOn ? prerollStartMs : recStartMs;
  if (start != watching) {
    watching = start;
    voiceMs = 0;
    loudRun = 0;
    scanned = audio::kMicRate * 3 / 10;  // past the mic's start-up
  }
  const uint32_t now = millis();
  if (live) {
    if (audio::micVoiceMs()) voiceMs = audio::micVoiceMs();
  } else {
    // The background recording, block by block like the live mic — all of
    // it, also what was said while the device woke and connected.
    const size_t B = audio::kVoiceBlock;
    for (const size_t ready = prerollReady(now - 40); scanned + B <= ready; scanned += B) {
      int peak = 0;
      for (size_t i = scanned; i < scanned + B; ++i) peak = max(peak, abs((int)preroll[i]));
      loudRun = peak >= audio::kVoicePeak ? loudRun + 1 : 0;
      if (loudRun >= audio::kVoiceRun) voiceMs = prerollStartMs + (uint32_t)((scanned + B) * 1000ULL / audio::kMicRate);
    }
  }
  if ((int32_t)(now - (voiceMs ? voiceMs : start)) < (int32_t)kSilenceStopMs) return;
  Serial.printf("[claw] %.1f s quiet: recording %s\n", kSilenceStopMs / 1000.0f, voiceMs ? "ends" : "dropped (nothing said)");
  if (voiceMs) {
    endTapRecording();
    return;
  }
  cancelTapRecording();
  leds::alert();
  hapticError();
  if (!board::kEpaper) notice("Nothing heard", "It stopped after 2.5 s of quiet. Tap and talk again.", kOrange);
}

// A swipe on an answer: the next (forward) or previous screen of it — the
// picture, if it came with one, then the text's pages. No wrapping round.
static void swipeAnswer(bool forward) {
  if (mode == Mode::Image) {
    if (forward) {  // on to the text
      ui::slide(true);
      backToMain();
    }
    return;
  }
  if (settings.iconMode && audio::playing()) {  // the character speaking: the text, as a side click does
    sideClick();
    return;
  }
  if (forward) {
    if (ui::onLastPage()) {
      hapticBump();  // nothing after this
    } else {
      ui::slide(true);
      ui::nextPage();
      hapticTick();
    }
  } else if (!ui::onFirstPage()) {
    ui::slide(false);
    ui::prevPage();
    hapticTick();
  } else if (ui::hasImage()) {  // before the first page: the picture
    ui::slide(false);
    mode = Mode::Image;
    redraw();
    hapticTick();
  } else {
    hapticBump();
  }
}

// A double tap off the tap button: the home screen, from an answer, a
// picture, the menu or a notice (not while it records or the agent works);
// on the home screen, the background picture alone, and from it home again.
static void goHome() {
  if (mode == Mode::Listening || mode == Mode::Waiting || mode == Mode::Ask) return;
  if (mode == Mode::Backdrop) {
    leaveBackdrop();
  } else if (mode == Mode::Home || (mode == Mode::Idle && !ui::hasReply())) {
    if (!ui::hasBackground()) return;  // home already, and nothing behind it to show
    ui::fade();
    mode = Mode::Backdrop;
    redraw();
  } else {
    ui::fade();
    mode = Mode::Home;
    redraw();
  }
  hapticTick();
}

// The talk button in a picker: this row, agent or character.
static void pickCharacter() {
  settings.character = charSel;
  settings.save();
  mode = Mode::Menu;
  menuOpenedMs = millis();
  ui::slide(false);
  redraw();
}
static void chooseHere() {
  if (mode == Mode::Menu) chooseMenuRow();
  else if (mode == Mode::Agents) chooseAgent();
  else if (mode == Mode::CharPick) pickCharacter();
}

// The touch screen (StopWatch). Swipes page an answer: left or up goes on,
// right or down goes back (on the home screen, back to the answer). A double
// tap goes home. With Tap to talk on, a tap on the button on the screen
// starts a recording and a second tap sends it. A touch on a dark screen
// only wakes it.
static void pollTouch(bool dark) {
  if (!M5.Touch.isEnabled()) return;
  static bool woke = false;  // this touch began on a dark screen
  auto t = M5.Touch.getDetail();
  if (t.wasPressed()) {
    woke = dark;
    touch();
  }
  if (!t.wasReleased() || woke) return;
  const int dx = t.distanceX(), dy = t.distanceY();
  const int dist = max(abs(dx), abs(dy));
  const bool picker = mode == Mode::Menu || mode == Mode::Agents || mode == Mode::CharPick;
  if (dist >= 45) {
    const bool sideways = abs(dx) > abs(dy);
    if (mode == Mode::Reply || mode == Mode::Image) {
      swipeAnswer(sideways ? dx < 0 : dy < 0);
    } else if ((mode == Mode::Home || mode == Mode::Backdrop) && ui::hasReply()) {
      backToMain();  // home → the answer again
    } else if ((picker || mode == Mode::Status) && sideways) {
      hapticTick();  // left: the next row or page; right: the one before
      if (dx < 0) sideClick();
      else backClick();
    }
    return;
  }
  if (dist > 25) return;
  const int x = t.base_x, y = t.base_y;
  if (picker) {  // the chevrons, and the badge between them to choose
    if (x < board::kScreenW / 3) {
      hapticTick();
      backClick();
    } else if (x > 2 * board::kScreenW / 3) {
      hapticTick();
      sideClick();
    } else {
      hapticConfirm();
      chooseHere();
    }
    return;
  }
  if (mode == Mode::Status) {
    hapticTick();
    if (x < board::kScreenW / 2) backClick();
    else sideClick();
    return;
  }
  if (mode == Mode::Ask) {
    const int hit = ui::askHit(x, y);
    if (hit < 0) return;
    if (hit) hapticConfirm();
    else hapticTick();
    answerAsk(hit == 1);
    return;
  }
  if ((mode == Mode::Reply || mode == Mode::Idle) && ui::replayHit(x, y)) {
    hapticConfirm();
    replayAnswer();
    return;
  }
  const ui::TapButton button = tapButtonKind();
  if (button == ui::TapButton::Off || !ui::tapButtonHit(x, y)) {
    static uint32_t lastTapMs = 0;  // taps on the button act at once, so only these count
    if (lastTapMs && millis() - lastTapMs < kDoubleMs + 100) {
      lastTapMs = 0;
      pendingTapMs = 0;
      goHome();
    } else {
      lastTapMs = millis();
      if (!lastTapMs) lastTapMs = 1;
      if (mode == Mode::Image) pendingTapMs = lastTapMs;  // a single tap, unless a second follows
    }
    return;
  }
  if (button == ui::TapButton::Pause || button == ui::TapButton::Play) {
    if (button == ui::TapButton::Pause) audio::pause();
    else audio::resume();
    buzz(25);
    redraw();
    return;
  }
  if (tapRec) {
    endTapRecording();
    return;
  }
  if (mode != Mode::Idle && mode != Mode::Home && mode != Mode::Reply) return;
  bool wasSpeaking = audio::playing();
  tapRec = true;  // before it starts: the button sampler would end an offline recording at once
  startTalking();
  if (mode != Mode::Listening) {  // offline with no room, say
    tapRec = false;
    return;
  }
  if (wasSpeaking) sendType("stop");
}

// Where the voice is in the answer's text, as a character index (-1: not
// speaking). Once the speech has all arrived its length is known and the
// place is the played fraction of it; before that, the pace of the last
// answer (learned here) carries the estimate.
static int spokenCharAt() {
  if (!audio::playing()) return -1;
  const size_t total = audio::speechTotal(), at = audio::speechAt();
  const int chars = spokenChars > 0 ? min(spokenChars, replyChars) : replyChars;
  if (audio::speechEnded() && total) {
    if (!paceLearned) {
      paceLearned = true;
      charsPerSec = constrain(chars * (float)audio::playRateHz() / total, 6.0f, 30.0f);
    }
    return (int)((uint64_t)chars * at / total);
  }
  return (int)(at * charsPerSec / audio::playRateHz());
}

// A turn the link dropped in the middle of (see turnDropMs): over when the
// answer or "thinking" comes back, or lost when the link stays away too long
// or comes back without the deck picking the turn up (it restarted, say).
static void holdTurn() {
  if (!turnDropMs) return;
  if (mode != Mode::Waiting) {  // an answer, an error or a button ended it
    turnDropMs = turnRelinkMs = 0;
    return;
  }
  uint32_t now = millis();
  if (turnRelinkMs ? now - turnRelinkMs < kResumeMs : now - turnDropMs < kTurnHoldMs) return;
  turnDropMs = turnRelinkMs = 0;
  touch();
  notice("Answer lost", "The connection dropped while the agent was working. Ask again.", kOrange);
}

void loop() {
#ifdef CLAW_PERF
  uint32_t loopT0 = micros();
#endif
  PERF_T(pfUpdMax, M5.update());
  noteButtonActivity();
  bool wasUp = wsUp;
  // Everything that's waiting, not one frame: the library reads one per
  // call, and when a pass is slow (the round screen's frames take ~50 ms)
  // one 4 KB speech frame per pass is slower than the voice plays.
  for (int i = 0; i < 16; ++i) {
    uint32_t seen = wsEvents;
    PERF_T(pfWsMax, ws.loop());
    if (wsEvents == seen) break;
  }
  if (wsUp != wasUp) linkChangeMs = millis();
  holdTurn();
  keepWifi();
  PERF_T(pfTelMax, telemetry::poll());
  pumpPreroll();
  pumpUpload();
  if (phoneSettingsActive()) {
    if (mode == Mode::Link) phoneSettingsPoll();
    else phoneSettingsEnd();
  }
  bool inMenu = mode == Mode::Menu || mode == Mode::Agents || mode == Mode::Status || mode == Mode::CharPick;
  // On a dark screen a press only wakes it — except the front button on the
  // main screen, which starts talking right away.
  bool dark = screenOff;

  // The paper changes slowly: a click says the press was heard.
  if (board::kEpaper && settings.keyClicks && !prerollOn &&
      (sideBtn().wasPressed() || backBtn().wasPressed() ||
       (talkBtn().wasPressed() && (inMenu || mode == Mode::Link || mode == Mode::Ask)))) {
    audio::click();
  }

  // Front button: hold to talk, tap for quick actions; in the menu, choose.
  // (Not while a talk-from-sleep recording holds it — but a press ends one
  // that was tapped on or toggled; looked at first, as a press below may start one.)
  const bool endsPreroll = talkBtn().wasPressed() && prerollOn && tapRec;
  if (talkBtn().wasPressed() && !prerollOn) {
    touch();
    if (dark && inMenu) {
      // woken only
    } else if (mode == Mode::Menu || mode == Mode::Agents || mode == Mode::CharPick) {
      chooseHere();
    } else if (mode == Mode::Status) {
      openMenu();
    } else if (mode == Mode::Link) {
      if (board::kEpaper) updateSoftware();  // the Options screen: the talk button updates
      else backToMain();
    } else if (mode == Mode::Ask) {
      answerAsk(true);
    } else if (tapRec && board::kTalkToggles && millis() - recStartMs < kDoubleMs + 100) {
      cancelTapRecording();  // a double press: no recording — the last answer again
      if (canReplay()) replayAnswer();
    } else if (tapRec) {
      endTapRecording();  // the second press (or, tapped on the screen, the button too)
    } else if (talkTapMs && millis() - talkTapMs < kDoubleMs + 100 && canReplay()) {
      talkTapMs = 0;  // a double-click on the talk button: hear the last answer again
      replayAnswer();
    } else if (board::kTalkToggles && audio::playing()) {
      audio::playStop();  // a press while it speaks only stops it; the next one talks
      sendType("stop");
      talkTapMs = millis();  // (or, straight after, replays it)
    } else {
      bool wasSpeaking = audio::playing();
      tapRec = board::kTalkToggles;  // before it starts: the sampler would end an offline one on letting go
      startTalking();
      if (mode != Mode::Listening) tapRec = false;
      if (wasSpeaking) sendType("stop");
    }
  }
  if (endsPreroll && tapRec) endTapRecording();  // (unless the sampler saw the press first)
  if (talkBtn().wasReleased() && mode == Mode::Listening && !prerollOn && !tapRec) {
    bool tap = millis() - recStartMs < kTapMs;
    stopTalking(!tap);
    talkTapMs = tap ? millis() : 0;  // a second press soon after is a double-click
    if (tap) {
      sendType("stop");  // a tap also silences the rest of the answer
      if (ui::hasImage() && mode == Mode::Reply) {
        mode = Mode::Image;
        redraw();
      }
    }
  }

  if (board::kHasTouch) pollTouch(dark);
  stopOnSilence();

  // Side button: page / next row · hold = menu (hold again = close).
  if (dark && (sideBtn().wasHold() || sideBtn().wasClicked())) {
    touch();  // woken only — though a second click makes it a double-click
    if (sideBtn().wasClicked() && onAnswer()) noteSideClick(true);
  } else if (sideBtn().wasHold()) {
    touch();
    if (mode == Mode::Ask) {
      answerAsk(false);
    } else if (mode == Mode::CharPick) {  // back to the menu, keeping the old character
      mode = Mode::Menu;
      menuOpenedMs = millis();
      redraw();
    } else if (inMenu || (board::kEpaper && mode == Mode::Link)) {
      ui::fade();
      backToMain();
    } else if (mode != Mode::Listening) {
      if (board::kEpaper) openOptions();  // the paper's whole menu
      else openMenu();
    }
  } else if (sideBtn().wasClicked()) {
    touch();
    if (sideClickMs && mode == sideClickMode) {  // the second click: hear the answer again
      sideClickMs = 0;
      replayAnswer();
    } else if (onAnswer()) {
      noteSideClick(false);  // it may be the first of two
    } else {
      sideClick();
    }
  }
  if (sideClickMs && millis() - sideClickMs > kDoubleMs) {
    sideClickMs = 0;  // just the one click
    if (!sideClickWoke && mode == sideClickMode) sideClick();
  }

  // Third button (PaperColor B): back · hold = Wi-Fi & settings, the setup
  // hotspot (the paper has no menu row for it; Options needs Wi-Fi already).
  if (board::kBackPin >= 0) {
    if (backBtn().wasHold()) {
      touch();
      if (mode == Mode::Ask) {
        answerAsk(false);
      } else if (mode != Mode::Listening && !prerollOn) {
        ui::flush();
        runSetupPortal("Wi-Fi & settings");  // saving (or holding B again) restarts it
      }
    } else if (backBtn().wasClicked()) {
      touch();
      backClick();
    }
  }

  // Menus close by themselves after a while — on paper much later, since
  // each step takes a redraw to show.
  const uint32_t menuIdle = board::kEpaper ? 180000 : 20000;
  if ((mode == Mode::Menu || mode == Mode::Agents) && millis() - menuOpenedMs > menuIdle) backToMain();
  if (mode == Mode::CharPick && millis() - menuOpenedMs > menuIdle * 3) backToMain();
  if (mode == Mode::Link && millis() - menuOpenedMs > 600000) backToMain();

  // The menu row to read aloud, once the presses have settled.
  if (sayAtMs && (int32_t)(millis() - sayAtMs) >= 0) {
    sayAtMs = 0;
    bool inList = mode == Mode::Menu || mode == Mode::Agents || mode == Mode::CharPick;
    if (wsUp && inList && !audio::playing()) {
      JsonDocument d;
      d["type"] = "say";
      d["text"] = sayText;
      sendJson(d);
    }
  }

  // A deck too old to replay never answers: play what did arrive.
  if (deckReplay && millis() - deckReplayMs > 4000) {
    deckReplay = quietSpeech = false;
    replayLocal();
  }

  // Keep a round-trip time for the status page.
  if (wsUp && millis() - lastPingMs > 15000) {
    lastPingMs = millis();
    JsonDocument d;
    d["type"] = "ping";
    d["t"] = millis();
    sendJson(d);
  }

  // Keep the mic stream and the speaker fed.
  if (mode == Mode::Listening) {
    // Live (mic → deck) or the background recording (talk from sleep, offline).
    int level = micLevel = prerollOn ? prerollLevel() : audio::micPoll(micSink);
    if (millis() - lastDrawMs > frameGap(80)) {
      lastDrawMs = millis();
      setBarState();
      syncTapButton();
      if (settings.iconMode) ui::showFace(settings.character, face::Pose::Listening, level);
      else ui::showListening(level, millis() - recStartMs);
    }
    if (!prerollOn && millis() - recStartMs > 110000) stopTalking(true);  // the server caps at 120 s
  }
  if (buzzOffMs && (int32_t)(millis() - buzzOffMs) >= 0) {
    M5.Power.setVibration(0);
    buzzOffMs = 0;
    if (buzzMore) {  // the next pulse of a pattern, after its gap
      buzzAgainMs = millis() + buzzGapMs;
      if (!buzzAgainMs) buzzAgainMs = 1;
    }
  }
  if (buzzAgainMs && (int32_t)(millis() - buzzAgainMs) >= 0) {
    const uint8_t more = buzzMore - 1;
    buzz(buzzOnMs);  // (clears the pattern: set it again)
    buzzMore = more;
  }
  if (pendingTapMs && millis() - pendingTapMs > kDoubleMs + 100) {
    pendingTapMs = 0;
    if (mode == Mode::Image) {  // a single tap on a picture: all of it, or the middle again
      ui::toggleImageFit();
      hapticTick();
      ui::fade();
      redraw();
    }
  }
  // Wi-Fi power saving holds incoming packets until the radio's next wake-up
  // (100–300 ms), which stalls the speech stream: during a turn — from the
  // press until the answer has played — the radio stays fully awake.
  if (mode == Mode::Listening || mode == Mode::Waiting || audio::playing()) turnBusyMs = millis();
  bool awake = turnBusyMs && millis() - turnBusyMs < 5000;
  if (awake != wifiAwake && WiFi.status() == WL_CONNECTED) {
    wifiAwake = awake;
    WiFi.setSleep(!awake);
  }
  pollFlash();
  bool wasPlaying = audio::playing();
#ifdef CLAW_PERF
  if (audio::playing() && pfLastPoll) pfGapMax = max<uint32_t>(pfGapMax, micros() - pfLastPoll);
  pfLastPoll = audio::playing() ? micros() : 0;
#endif
  audio::playPoll();
  // The answer's sound played to its end: the cue again, the other way round
  // — a moment later, so an agent-made sound that follows comes first.
  if (audio::takeFinished() && settings.cueBeep && !quietSpeech) {
    closeAtMs = millis() + 350;
    if (!closeAtMs) closeAtMs = 1;
  }
  if (closeAtMs && (int32_t)(millis() - closeAtMs) >= 0) {
    closeAtMs = 0;
    if (!audio::playing() && !audio::micActive() && !prerollOn) audio::cueEnd();
  }
  if (wasPlaying != audio::playing() && (mode == Mode::Reply || mode == Mode::Idle) && !flashing) {
    if (!audio::playing()) ui::setSpoken(-1);  // done: the whole text lit again
    redraw();
  }

  if (!flashing && !screenOff) {
    uint32_t since = millis() - lastDrawMs;
    if (mode == Mode::Waiting) {
      // Changes at most ~6×/s; the timer (and the icon's dots) once in a while.
      static uint32_t lastSecond = 0;
      uint32_t secs = (millis() - workStartMs) / 1000;
      if ((workDirty && since > 150) || secs != lastSecond ||
          (settings.iconMode && since > frameGap(face::frameMs(settings.character))) ||
          (ui::animated() && since > frameGap(60))) {
        lastSecond = secs;
        lastDrawMs = millis();
        workDirty = false;
        redraw();
      }
    } else if (settings.iconMode && mode == Mode::Reply && audio::sounding() &&
               since > frameGap(min<uint32_t>(60, face::frameMs(settings.character)))) {
      lastDrawMs = millis();
      setBarState();
      ui::setSpoken(spokenCharAt());  // the subtitle
      ui::showFace(settings.character, face::Pose::Speaking, audio::outputLevel());
    } else if (!settings.iconMode && ui::animated() && mode == Mode::Reply && audio::sounding() && since > 80) {
      lastDrawMs = millis();
      if (ui::setSpoken(spokenCharAt())) redraw();  // the sentence or the page changed: the whole screen
      else ui::showSpeaking(audio::outputLevel());  // else the speaking line follows the voice (its strip only)
    } else if (settings.iconMode && mode == Mode::Idle && !ui::hasReply() && since > frameGap(face::frameMs(settings.character))) {
      lastDrawMs = millis();  // blinking, rain, bouncing…
      redraw();
    } else if (mode == Mode::CharPick && since > frameGap(face::frameMs(charSel))) {
      lastDrawMs = millis();
      redraw();
    } else if ((mode == Mode::Status && since > 1000) || (ui::animated() && mode == Mode::Idle && since > 1000) ||
               since > 30000) {
      lastDrawMs = millis();  // live status page; battery/health in the bar elsewhere
      if (mode != Mode::Listening && mode != Mode::Notice && mode != Mode::Image) redraw();
    }
  }

  // A fresh update proves itself by reaching Flight Deck within 90 s;
  // otherwise the bootloader goes back to the previous version.
  if (probation) {
    if (wsUp) {
      ota::markGood();
      probation = false;
    } else if (millis() > 90000) {
      ota::rollBack();
    }
  }
  if (wsUp && settings.lastFw != FW_VERSION) {  // first run of this version
    bool updated = settings.lastFw.length() > 0;
    settings.lastFw = FW_VERSION;
    settings.save();
    if (updated && (mode == Mode::Idle || mode == Mode::Reply)) {
      notice("Updated", "Now running " FW_VERSION ".", kGreen);
    }
  }
  if (settings.autoUpdate && offer.valid() && !offerAutoTried && !audio::playing() &&
      (mode == Mode::Idle || mode == Mode::Reply || mode == Mode::Image) && telemetry::onUsbPower()) {
    offerAutoTried = true;
    startInstall();
  }

  ui::idleSince(lastActivityMs);  // the e-paper redraws once it's been left alone
  ui::tick();
  updateLeds();
  noteButtonActivity();
  handleIdlePower();
#ifdef CLAW_PERF
  perfReport(micros() - loopT0);
#endif
  delay(1);
}
