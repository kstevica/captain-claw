// PaperColor, the "B" way: a device with two lights and a voice, and a
// screen only because it's there. The lights carry listening, working,
// speaking and trouble (leds.cpp), the deck's voice carries the answers. The
// screen is a dashboard — the clock, the room, the battery, the voice notes
// kept offline, the links — redrawn on every five-minute mark (awake, or on
// the timer wake from sleep) and when asked for. An answer goes on the paper
// as its voice starts, at the largest size that fits it on one screen
// (paging only when it must), and again once it's whole; B pages it. A
// talks, C shows the options code. A redraw can't be interrupted, so the main
// loop never waits for one (a background task runs it); each starts from a
// reset panel controller, which also makes it several times faster.
//
// What the paper shows is remembered across deep sleep and restarts, as is
// when the dashboard was last drawn, so a wake doesn't redraw. The last
// answer lives in flash (LittleFS), so B shows it after a sleep too.
#include <M5Unified.h>
#include <math.h>
#include <time.h>

#include <functional>
#include <initializer_list>
#include <utility>

#include "board.h"
#include "fonts.h"
#include "telemetry.h"
#include "ui.h"

#ifndef PREVIEW
#include <ArduinoJson.h>
#include <LittleFS.h>
#include <Preferences.h>
#include <lgfx/v1/panel/Panel_ED2208.hpp>

#include "battery.h"
#include "notes.h"
#else
namespace battery {  // the preview stands these in (preview/paper_b.cpp)
int minutesLeft();
int minutesToFull();
}  // namespace battery
namespace notes {
bool ready();
bool onCard();
uint64_t freeBytes();
int count();
uint32_t seconds();
size_t room(uint32_t rate);
}  // namespace notes
#endif

static constexpr uint32_t kNoteRate = 16000;  // the mic's rate (audio::kMicRate): what the notes are kept at

namespace ui {

// ── inks ───────────────────────────────────────────────────────────────

struct InkDef {
  uint8_t r, g, b;
};
static constexpr InkDef kInkDefs[6] = {
    {0, 0, 0}, {255, 255, 255}, {255, 243, 56}, {191, 0, 0}, {100, 64, 255}, {67, 138, 28}};
static constexpr uint16_t kInk = 0x0000, kPaper = 0xFFFF;
static constexpr uint16_t kYellow = rgb(255, 243, 56), kRed = rgb(191, 0, 0);
static constexpr uint16_t kBlue = rgb(100, 64, 255), kGreen = rgb(67, 138, 28);

// ── canvas, fonts, text ────────────────────────────────────────────────

static int W = 400, H = 600;
static constexpr int PAD = 22;
static constexpr int kHeadH = 52;  // the top bar
static constexpr int kFootH = 48;  // the button legend
static M5Canvas canvas(&M5.Display);
static lgfx::VLWfont fontSet[vlw::Count];
static lgfx::PointerWrapper fontData[vlw::Count];

static String barAgent, barState;
static Health hWifi = Down, hAgent = Down;

struct Rect {
  int x, y, w, h;
};
static std::vector<Rect> pictures;  // regions to error-diffuse in this frame

static void font(vlw::Id id) { canvas.setFont(&fontSet[id]); }
static int tw(const String& s) { return canvas.textWidth(s.c_str()); }

static void text(const String& s, int x, int y, uint16_t color, uint8_t datum = top_left) {
  canvas.setTextDatum(datum);
  canvas.setTextColor(color);
  canvas.drawString(s.c_str(), x, y);
}

static int utf8Step(uint8_t c) { return c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1; }

// Shorten to `width` px with "…" (current font), whole UTF-8 characters.
static String fit(const String& s, int width) {
  if (tw(s) <= width) return s;
  String out = s;
  while (out.length() > 1 && tw(out + "…") > width) {
    int cut = out.length() - 1;
    while (cut > 0 && (uint8_t(out[cut]) & 0xC0) == 0x80) --cut;
    out.remove(cut);
  }
  return out + "…";
}

// Word-wrap UTF-8 text to `width` px in the current font; words wider than a
// line break between characters. Paragraph breaks are kept.
static std::vector<String> wrap(const String& s, int width) {
  std::vector<String> out;
  int start = 0;
  const int n = s.length();
  while (start <= n) {
    int nl = s.indexOf('\n', start);
    String para = s.substring(start, nl < 0 ? n : nl);
    start = (nl < 0 ? n : nl) + 1;
    String line;
    int i = 0;
    while (i < (int)para.length()) {
      int sp = para.indexOf(' ', i);
      String word = para.substring(i, sp < 0 ? para.length() : sp);
      i = sp < 0 ? para.length() : sp + 1;
      String trial = line.length() ? line + " " + word : word;
      if (tw(trial) <= width) {
        line = trial;
        continue;
      }
      if (line.length()) out.push_back(line);
      line = "";
      while (word.length() && tw(word) > width) {
        int cut = 0, k = 0;
        while (k < (int)word.length()) {
          int step = utf8Step(word[k]);
          if (tw(word.substring(0, k + step)) > width) break;
          k += step;
          cut = k;
        }
        if (cut == 0) cut = utf8Step(word[0]);
        out.push_back(word.substring(0, cut));
        word = word.substring(cut);
      }
      line = word;
    }
    out.push_back(line);
    if (nl < 0) break;
  }
  while (!out.empty() && out.back().isEmpty()) out.pop_back();
  return out;
}

// At most `max` lines; the last one ends in "…" when there was more.
static std::vector<String> clamp(std::vector<String> lines, int max, int width) {
  if ((int)lines.size() <= max) return lines;
  lines.resize(max);
  lines.back() = fit(lines.back() + "…", width);
  return lines;
}

static void drawLines(const std::vector<String>& lines, int x, int y, int lineH, uint16_t color) {
  for (size_t i = 0; i < lines.size(); ++i) text(lines[i], x, y + i * lineH, color);
}

// ── the last answer, kept in flash ─────────────────────────────────────

struct Card {
  String q, a;
  time_t when = 0;
  uint8_t* jpg = nullptr;
  size_t len = 0;
  bool valid() const { return a.length() > 0; }
  void clear() {
    q = a = "";
    when = 0;
    if (jpg) free(jpg);
    jpg = nullptr;
    len = 0;
  }
};
static Card savedLast;  // from flash: the answer before this boot

static String question, pendingQuestion;
static bool restoring = false;
static String replyText;
static time_t replyWhen = 0;
static uint8_t* imgBuf = nullptr;
static size_t imgLen = 0;

#ifndef PREVIEW
static bool fsUp = false;

static void saveCard(const char* name, const Card& c) {
  if (!fsUp) return;
  String base = String("/") + name;
  if (!c.valid()) {
    LittleFS.remove(base + ".json");
    LittleFS.remove(base + ".jpg");
    return;
  }
  JsonDocument d;
  d["q"] = c.q;
  d["a"] = c.a;
  d["t"] = (uint32_t)c.when;
  File f = LittleFS.open(base + ".json", "w");
  if (f) {
    serializeJson(d, f);
    f.close();
  }
  if (c.len) {
    File j = LittleFS.open(base + ".jpg", "w");
    if (j) {
      j.write(c.jpg, c.len);
      j.close();
    }
  } else {
    LittleFS.remove(base + ".jpg");
  }
}

static void loadCard(const char* name, Card& c) {
  c.clear();
  if (!fsUp) return;
  String base = String("/") + name;
  File f = LittleFS.open(base + ".json", "r");
  if (!f) return;
  JsonDocument d;
  if (deserializeJson(d, f) == DeserializationError::Ok) {
    c.q = d["q"] | "";
    c.a = d["a"] | "";
    c.when = (time_t)(d["t"] | 0u);
  }
  f.close();
  File j = LittleFS.open(base + ".jpg", "r");
  if (j) {
    size_t n = j.size();
    uint8_t* buf = n && n <= 256 * 1024 ? (uint8_t*)ps_malloc(n) : nullptr;
    if (buf && j.read(buf, n) == n) {
      c.jpg = buf;
      c.len = n;
    } else if (buf) {
      free(buf);
    }
    j.close();
  }
}
#else
static void saveCard(const char*, const Card&) {}
static void loadCard(const char*, Card& c) { c.clear(); }
#endif

static uint8_t* copyBytes(const uint8_t* src, size_t len) {
  if (!src || !len) return nullptr;
  uint8_t* out = (uint8_t*)ps_malloc(len);
  if (out) memcpy(out, src, len);
  return out;
}

static void saveLast() {
  Card c;
  c.q = question;
  c.a = replyText;
  c.when = replyWhen;
  c.jpg = copyBytes(imgBuf, imgLen);
  c.len = c.jpg ? imgLen : 0;
  saveCard("last", c);
  c.clear();
}

// ── pictures ───────────────────────────────────────────────────────────

static bool jpegSize(const uint8_t* d, size_t len, int* w, int* h) {
  size_t i = 2;
  while (i + 9 < len) {
    if (d[i] != 0xFF) return false;
    uint8_t m = d[i + 1];
    size_t seg = (d[i + 2] << 8) | d[i + 3];
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      *h = (d[i + 5] << 8) | d[i + 6];
      *w = (d[i + 7] << 8) | d[i + 8];
      return *w > 0 && *h > 0;
    }
    i += 2 + seg;
  }
  return false;
}

// Draws a picture to fit inside the box (never larger than it was sent),
// centred; returns the height it took (0 when there's none).
static int drawPicture(const uint8_t* jpg, size_t len, int x, int y, int bw, int bh) {
  int w = 0, h = 0;
  if (!jpg || !len || !jpegSize(jpg, len, &w, &h)) return 0;
  float s = min(1.0f, min((float)bw / w, (float)bh / h));
  int dw = (int)(w * s), dh = (int)(h * s);
  int dx = x + (bw - dw) / 2;
  canvas.drawJpg(jpg, len, dx, y, dw, dh, 0, 0, s, s);
  pictures.push_back({dx, y, dw, dh});
  return dh;
}

// ── the slow refresh, off the main loop ───────────────────────────────

static volatile bool refreshing = false;

// The nearest ink, for text and shapes: greys go to black or white by
// lightness (antialiased edges stay crisp), colours to the nearest ink.
static uint16_t snap(uint16_t c) {
  int r = ((c >> 11) & 31) * 255 / 31, g = ((c >> 5) & 63) * 255 / 63, b = (c & 31) * 255 / 31;
  int mx = max(r, max(g, b)), mn = min(r, min(g, b));
  if (mx - mn < 64) return (r * 299 + g * 587 + b * 114) / 1000 < 150 ? kInk : kPaper;
  int best = 0, bestD = 1 << 30;
  for (int i = 0; i < 6; ++i) {
    int dr = r - kInkDefs[i].r, dg = g - kInkDefs[i].g, db = b - kInkDefs[i].b;
    int dd = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
    if (dd < bestD) bestD = dd, best = i;
  }
  return rgb(kInkDefs[best].r, kInkDefs[best].g, kInkDefs[best].b);
}

static inline uint16_t px(uint16_t* buf, int i) { return (uint16_t)((buf[i] >> 8) | (buf[i] << 8)); }
static inline void setPx(uint16_t* buf, int i, uint16_t c) { buf[i] = (uint16_t)((c >> 8) | (c << 8)); }

// Floyd–Steinberg (serpentine) over one picture, into the six inks.
static void diffuse(uint16_t* buf, const Rect& r) {
  const int w = r.w;
  int16_t* err = (int16_t*)calloc(2 * 3 * (w + 2), sizeof(int16_t));
  if (!err) return;
  int16_t* cur = err;
  int16_t* nxt = err + 3 * (w + 2);
  for (int y = 0; y < r.h; ++y) {
    bool ltr = (y & 1) == 0;
    memset(nxt, 0, 3 * (w + 2) * sizeof(int16_t));
    for (int k = 0; k < w; ++k) {
      int x = ltr ? k : w - 1 - k;
      int i = (r.y + y) * W + r.x + x;
      uint16_t c = px(buf, i);
      int v[3] = {((c >> 11) & 31) * 255 / 31 + cur[3 * (x + 1)] / 16,
                  ((c >> 5) & 63) * 255 / 63 + cur[3 * (x + 1) + 1] / 16,
                  (c & 31) * 255 / 31 + cur[3 * (x + 1) + 2] / 16};
      int best = 0, bestD = 1 << 30;
      for (int p = 0; p < 6; ++p) {
        int dr = v[0] - kInkDefs[p].r, dg = v[1] - kInkDefs[p].g, db = v[2] - kInkDefs[p].b;
        int dd = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
        if (dd < bestD) bestD = dd, best = p;
      }
      const InkDef& ink = kInkDefs[best];
      setPx(buf, i, rgb(ink.r, ink.g, ink.b));
      int e[3] = {constrain(v[0] - ink.r, -255, 255), constrain(v[1] - ink.g, -255, 255),
                  constrain(v[2] - ink.b, -255, 255)};
      int ahead = ltr ? 1 : -1;
      for (int ch = 0; ch < 3; ++ch) {
        cur[3 * (x + 1 + ahead) + ch] += e[ch] * 7;
        nxt[3 * (x + 1 - ahead) + ch] += e[ch] * 3;
        nxt[3 * (x + 1) + ch] += e[ch] * 5;
        nxt[3 * (x + 1 + ahead) + ch] += e[ch];
      }
    }
    std::swap(cur, nxt);
  }
  free(err);
}

// The frame in the six inks: pictures diffused, everything else snapped.
static void toInks() {
  uint16_t* buf = (uint16_t*)canvas.getBuffer();
  if (!buf) return;
  const int n = W * H;
  std::vector<uint8_t> inPicture;
  if (!pictures.empty()) {
    inPicture.assign(n, 0);
    for (auto& r : pictures) {
      Rect c = {max(0, r.x), max(0, r.y), 0, 0};
      c.w = min(W, r.x + r.w) - c.x;
      c.h = min(H, r.y + r.h) - c.y;
      if (c.w <= 0 || c.h <= 0) continue;
      diffuse(buf, c);
      for (int y = c.y; y < c.y + c.h; ++y) memset(&inPicture[y * W + c.x], 1, c.w);
    }
  }
  uint16_t lastIn = 0xFFFF, lastOut = kPaper;
  for (int i = 0; i < n; ++i) {
    if (!inPicture.empty() && inPicture[i]) continue;
    uint16_t c = px(buf, i);
    if (c != lastIn) {
      lastIn = c;
      lastOut = snap(c);
    }
    if (lastOut != c) setPx(buf, i, lastOut);
  }
}

// The panel's own BUSY line, for longer than its driver waits (20 s): a
// colour refresh can take 30, and new data sent while it runs is lost.
static void waitPanel(uint32_t ms) {
  uint32_t t0 = millis();
  while (M5.Display.displayBusy() && millis() - t0 < ms) delay(20);
}

// ── the panel controller ───────────────────────────────────────────────

#ifndef PREVIEW
// The panel's bus and pins (M5GFX keeps them protected).
struct PaperPanel : lgfx::Panel_ED2208 {
  int width() const { return _cfg.panel_width; }
  int height() const { return _cfg.panel_height; }

  void cmd(uint8_t c) { _bus->writeCommand(c, 8); }
  void data(uint8_t d) { _bus->writeData(d, 8); }
  bool idle() const { return _cfg.pin_busy < 0 || lgfx::gpio_in(_cfg.pin_busy); }
  void waitBusy(uint32_t ms) {
    _bus->wait();
    uint32_t t0 = millis();
    while (!idle() && millis() - t0 < ms) delay(10);
  }

  // A reset pulse, then the set-up. With the paper keeping its picture
  // through restarts, M5GFX starts it without one (init_without_reset), so
  // the controller ran on whatever state it had since power came; from a
  // reset every refresh is whole, and several times faster. The PaperColor's
  // EPD reset is G12 (M5GFX pulses it at start; the panel's reset pin is not it).
  void reset() {
    lgfx::gpio_hi(kResetPin);
    lgfx::pinMode(kResetPin, lgfx::pin_mode_t::output);
    lgfx::gpio_lo(kResetPin);
    delay(10);
    lgfx::gpio_hi(kResetPin);
    delay(10);
    for (uint32_t t0 = millis(); !idle() && millis() - t0 < 5000;) delay(10);
    reinit();
  }
  static constexpr int kResetPin = GPIO_NUM_12;

  // The controller's set-up as M5GFX sends it at start (its init list, then
  // the resolution).
  void reinit() {
    static constexpr uint8_t kInit[] = {
        0xAA, 6, 0x49, 0x55, 0x20, 0x08, 0x09, 0x18, 0x01, 1, 0x3F, 0x00, 2, 0x5F, 0x69,
        0x05, 4, 0x40, 0x1F, 0x1F, 0x2C, 0x08, 4, 0x6F, 0x1F, 0x1F, 0x22, 0x06, 4, 0x6F, 0x1F, 0x17, 0x17,
        0x03, 4, 0x03, 0x54, 0x00, 0x44, 0x60, 2, 0x02, 0x00, 0x30, 1, 0x08, 0x50, 1, 0x3F,
        0xE3, 1, 0x2F, 0x84, 1, 0x01, 0xFF};
    _bus->beginTransaction();
    cs_control(false);
    for (size_t i = 0; kInit[i] != 0xFF; i += 2 + kInit[i + 1]) {
      waitBusy(5000);
      cmd(kInit[i]);
      for (uint8_t k = 0; k < kInit[i + 1]; ++k) data(kInit[i + 2 + k]);
    }
    waitBusy(5000);
    cmd(0x61);  // resolution
    for (int v : {width(), height()}) {
      data(v >> 8);
      data(v & 0xFF);
    }
    _bus->wait();
    cs_control(true);
    _bus->endTransaction();
  }
};

static PaperPanel* paperPanel() {
  if (M5.getBoard() != m5::board_t::board_M5PaperColor) return nullptr;
  return static_cast<PaperPanel*>(static_cast<lgfx::Panel_ED2208*>(M5.Display.getPanel()));
}

#endif

static void pushToPanel() {
  waitPanel(45000);
  M5.Display.startWrite();
  canvas.pushSprite(&M5.Display, 0, 0);
  M5.Display.endWrite();
#ifndef PREVIEW
  if (PaperPanel* p = paperPanel()) p->reset();  // every refresh from a reset controller
#endif
  M5.Display.display();
  waitPanel(45000);
}

#ifndef PREVIEW
static TaskHandle_t paperTask = nullptr;

static void paperLoop(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    toInks();
    pushToPanel();
    refreshing = false;
  }
}

static void startRefresh() {
  refreshing = true;
  if (paperTask) xTaskNotifyGive(paperTask);
  else {  // no task (out of memory): do it here
    toInks();
    pushToPanel();
    refreshing = false;
  }
}
#else
static void startRefresh() {
  toInks();
  pushToPanel();
}
#endif

// ── what the paper may draw, and when ──────────────────────────────────

enum class Kind : uint8_t { None, Dash, Reply, Link, Setup };

#ifndef RTC_DATA_ATTR
#define RTC_DATA_ATTR  // (the preview has no deep sleep)
#endif
// What the paper shows (the panel keeps its picture with no power at all),
// and when the dashboard was last drawn, by the clock: in RTC memory across
// deep sleep, and in flash (NVS) across everything else — a restart, a
// brownout, the battery jolted loose — so a reset never redraws it.
RTC_DATA_ATTR static bool rtcShown = false;
RTC_DATA_ATTR static uint8_t rtcKind = 0;
RTC_DATA_ATTR static uint32_t rtcDashAt = 0;
static Kind shownKind = Kind::None;
static bool everShown = false;
static uint32_t lastDashMs = 0;  // the dashboard, this boot (0: not yet)
static constexpr uint32_t kDashEverySec = 5 * 60;  // on the clock's five-minute marks

#ifndef PREVIEW
static void keepShown() {
  Preferences p;
  if (!p.begin("paperb", false)) return;
  p.putUChar("kind", rtcKind);
  p.putULong("dash", rtcDashAt);
  p.end();
}

static void loadShown() {
  Preferences p;
  if (!p.begin("paperb", true)) return;  // (read-only: missing on the first boot)
  rtcKind = p.getUChar("kind", 0);
  rtcDashAt = p.getULong("dash", 0);
  p.end();
  rtcShown = rtcKind != 0;
}
#else
static void keepShown() {}
#endif

// One screen waiting its turn (the panel may be busy with the last).
struct Want {
  Kind kind = Kind::None;
  std::function<void()> draw;
  bool active = false;
};
static Want want;
static bool forceNext = false;  // drawNow(): the dashboard, now, whatever the clock says
static uint32_t forceAtMs = 0;  // (only right after: a stray drawNow() never redraws later)
static String linkShown;        // the options code on the paper (title + address)
static bool viewing = false;    // the answer is on the paper; B pages it, the last B goes back to the dashboard
static bool answerDue = false;  // a new answer waits for its voice to go on the paper (not the dashboard first)
static bool sounding = false;   // its voice is playing: the page stays as it went on until the voice ends
static uint32_t replyGen = 0, shownGen = 0;  // the answer laid out / the one on the paper
static int shownPage = -1;

static uint32_t lastUseMs = 0;  // idleSince(): the last press, word or sound
void idleSince(uint32_t ms) { lastUseMs = ms; }
void drawNow() {
  forceNext = true;
  forceAtMs = millis();
  viewing = answerDue = false;
}
bool waiting() { return false; }

static void draw(Kind kind, std::function<void()> fn) {
  want.kind = kind;
  want.draw = std::move(fn);
  want.active = true;
}

static void commit() {
  Want w = std::move(want);
  want.active = false;
  pictures.clear();
  canvas.fillScreen(kPaper);
  w.draw();
  shownKind = w.kind;
  everShown = true;
  rtcShown = true;
  rtcKind = (uint8_t)w.kind;
  if (w.kind == Kind::Dash) {
    lastDashMs = millis() ? millis() : 1;
    rtcDashAt = (uint32_t)time(nullptr);
  }
  keepShown();
  startRefresh();
}

static void dashTick();

void tick() {
  if (refreshing) return;
  if (want.active) commit();
  else dashTick();
}

bool busy() { return refreshing; }

void flush() {
#ifndef PREVIEW
  while (refreshing) delay(20);
#endif
  if (want.active) commit();
#ifndef PREVIEW
  while (refreshing) delay(20);
#endif
}

// Seconds until the dashboard is next due (0: now): on the clock's
// five-minute marks (:00, :05 … :55) — due once the mark it was last drawn
// in has passed (the clock runs through deep sleep; after a reset the RTC
// chip sets it). No clock yet (power came back): five minutes from this
// boot's drawing, or from the boot. An answer on the paper stays until B
// pages past it, or the paper goes to sleep.
uint32_t dashboardDueInSec() {
  if (!everShown) return 0;
  const uint32_t now = (uint32_t)time(nullptr);
  if (now >= 1700000000) {
    if (rtcDashAt < 1700000000 || now / kDashEverySec != rtcDashAt / kDashEverySec) return 0;
    return kDashEverySec - now % kDashEverySec;
  }
  const uint32_t since = (millis() - lastDashMs) / 1000;
  return since >= kDashEverySec ? 0 : kDashEverySec - since;
}

// Due by the clock — or at once when the paper shows a code that's no longer
// open (the options page closes when its screen is left; setup restarts).
static bool dashDue() {
  const bool code = everShown && (shownKind == Kind::Link || shownKind == Kind::Setup);
  return code || dashboardDueInSec() == 0;
}

// ── chrome ─────────────────────────────────────────────────────────────

static uint16_t healthInk(Health h) { return h == Up ? kGreen : h == Partial ? kYellow : kRed; }

static int batteryPct(bool* charging) {
  static int pct = -1;
  static bool chg = false;
  static uint32_t at = 0;
  if (pct < 0 || millis() - at > 20000) {
    at = millis();
    pct = M5.Power.getBatteryLevel();
    chg = M5.Power.isCharging() == m5::Power_Class::is_charging;
  }
  *charging = chg;
  return pct;
}

static void drawBattery(int x, int y, int pct, bool chg) {  // x = right edge, y = middle
  font(vlw::Small);
  String t = pct >= 0 ? String(pct) + "%" : String("?");
  int textW = tw(t);
  const int bw = 30, bh = 16;
  int bx = x - textW - 8 - bw;
  uint16_t fill = pct < 20 ? kRed : pct < 40 ? kYellow : kGreen;
  canvas.drawRoundRect(bx, y - bh / 2, bw - 3, bh, 3, kInk);
  canvas.drawRoundRect(bx + 1, y - bh / 2 + 1, bw - 5, bh - 2, 2, kInk);
  canvas.fillRect(bx + bw - 3, y - 4, 3, 8, kInk);
  if (pct > 0) canvas.fillRect(bx + 3, y - bh / 2 + 3, max(2, pct * (bw - 9) / 100), bh - 6, fill);
  if (chg) {
    canvas.fillTriangle(bx + 15, y - 9, bx + 8, y + 1, bx + 14, y + 1, kInk);
    canvas.fillTriangle(bx + 12, y - 1, bx + 19, y - 1, bx + 11, y + 9, kInk);
  }
  text(t, x, y, pct < 20 ? kRed : kInk, middle_right);
}

// `t` formatted in local time; false before the clock was ever set.
static bool clockText(char* out, size_t n, const char* fmt, time_t t) {
  if (t < 1700000000) return false;
  struct tm tm;
  localtime_r(&t, &tm);
  strftime(out, n, fmt, &tm);
  return true;
}

// The top bar: a title (the clock when none, else the agent's name), the
// Wi-Fi and agent dots in the middle (not asleep: nothing is connected then),
// the battery; a rule under it.
static void drawHeader(const String& title = "", bool dots = true) {
  font(vlw::Small);
  int y = kHeadH / 2;
  char clock[16];
  String name = title.length() ? title
                : clockText(clock, sizeof(clock), "%H:%M", time(nullptr)) ? String(clock)
                : barAgent.length() ? barAgent : String("Captain Claw");
  bool chg = false;
  int pct = batteryPct(&chg);
  drawBattery(W - PAD, y, pct, chg);
  int right = W - PAD - 92;  // the battery's left edge
  if (dots) {
    canvas.fillCircle(W / 2 - 9, y, 6, healthInk(hWifi));
    canvas.fillCircle(W / 2 + 9, y, 6, healthInk(hAgent));
    right = W / 2 - 24;
  }
  font(vlw::Title);
  text(fit(name, right - PAD), PAD, y + 2, kInk, middle_left);
  canvas.fillRect(0, kHeadH - 3, W, 3, kInk);
}

// A keycap: the button's letter in a small black rounded square.
static int drawKey(const char* key, int x, int y) {
  font(vlw::Small);
  canvas.fillRoundRect(x, y - 14, 28, 28, 6, kInk);
  text(key, x + 14, y + 1, kPaper, middle_center);
  return x + 28;
}

// The bottom bar: what the buttons do ("C", "talk", …), a note at the left.
static void drawFooter(std::initializer_list<std::pair<const char*, const char*>> keys, const String& left = "") {
  const int y = H - kFootH / 2;
  canvas.fillRect(0, H - kFootH, W, 2, kInk);
  font(vlw::Small);
  int total = 0;
  for (auto& k : keys) total += 28 + 6 + tw(k.second) + 16;
  total -= 16;
  int x = left.length() ? W - PAD - total : (W - total) / 2;
  if (left.length()) text(fit(left, x - PAD - 10), PAD, y, kBlue, middle_left);
  for (auto& k : keys) {
    x = drawKey(k.first, x, y) + 6;
    text(k.second, x, y + 1, kInk, middle_left);
    x += tw(k.second) + 16;
  }
}

static void drawPageDots(int count, int current, int y) {
  if (count < 2) return;
  const int gap = 18;
  int x0 = W / 2 - (count - 1) * gap / 2;
  for (int i = 0; i < count && i < 16; ++i) {
    if (i == current) canvas.fillCircle(x0 + i * gap, y, 6, kInk);
    else canvas.drawCircle(x0 + i * gap, y, 5, kInk);
  }
}

// A label in blue with a short rule before it.
static void drawLabel(const String& s, int x, int y, uint16_t color = kBlue) {
  canvas.fillRect(x, y + 4, 4, 16, color);
  font(vlw::Small);
  text(s, x + 12, y, color);
}

// ── public: basics ─────────────────────────────────────────────────────

static void makeCanvas() {
  M5.Display.setRotation(0);  // the B paper stands: portrait, always
  W = M5.Display.width();
  H = M5.Display.height();
  if (canvas.getBuffer()) canvas.deleteSprite();
  canvas.setPsram(true);
  canvas.setColorDepth(16);
  canvas.createSprite(W, H);
  canvas.setTextWrap(false);
}

void begin() {
#ifndef PREVIEW
  if (esp_reset_reason() != ESP_RST_DEEPSLEEP || !rtcShown) loadShown();  // any other reset: from flash
  if (rtcShown) {  // the panel kept its picture
    everShown = true;
    shownKind = (Kind)rtcKind;
  }
#endif
  M5.Display.setAutoDisplay(false);
  M5.Display.setEpdMode(epd_mode_t::epd_fastest);  // the frame is already in the six inks
  makeCanvas();
  for (int i = 0; i < vlw::Count; ++i) {
    size_t len = 0;
    const uint8_t* d = vlw::data((vlw::Id)i, &len);
    fontData[i].set(d, len);
    fontSet[i].loadFont(&fontData[i]);
  }
#ifndef PREVIEW
  fsUp = LittleFS.begin(true, "/littlefs", 10, "spiffs");
  xTaskCreatePinnedToCore(paperLoop, "paper", 8192, nullptr, 1, &paperTask, 0);
#endif
  loadCard("last", savedLast);
}

void setRotation(uint8_t) {}  // portrait, always
void brightness(uint8_t) {}   // paper has no backlight
bool animated() { return false; }

void setBar(const String& agent, const char* state, uint16_t) {
  barAgent = agent;
  barState = state ? state : "";
}

void setHealth(Health wifi, Health agent) {
  hWifi = wifi;
  hAgent = agent;
}

// ── the dashboard ──────────────────────────────────────────────────────

// "Monday, 19 October"; returns the height it took (0 before the clock is set).
static int drawDate(int x, int y, int w) {
  char date[40];
  if (!clockText(date, sizeof(date), "%A, %e %B", time(nullptr))) return 0;
  String d = date;
  d.replace("  ", " ");
  font(vlw::Title);
  if (tw(d) > w && clockText(date, sizeof(date), "%a %e %b", time(nullptr))) {
    d = date;
    d.replace("  ", " ");
  }
  text(fit(d, w), x, y, kInk);
  return 44;
}

// The room: temperature large, humidity at the right. Returns the height
// it took (0 on a device without the sensor).
static int drawClimate(int x, int y, int w) {
  float t, rh;
  if (!telemetry::climate(&t, &rh)) return 0;
  char buf[12];
  snprintf(buf, sizeof(buf), "%.1f", t);
  String tt = buf, hum = String((int)lroundf(rh)) + "%";
  font(vlw::Small);
  int humW = tw("humidity");
  font(vlw::Title);
  humW = max(humW, tw(hum));
  font(vlw::Digits);
  const bool big = tw(tt) + 44 + 20 + humW <= w;
  if (!big) font(vlw::Badge);
  const int tW = tw(tt);
  text(tt, x - (big ? 4 : 0), y - (big ? 10 : 2), kInk);
  const int r = big ? 7 : 5, dx = x + tW + 2;
  canvas.drawCircle(dx + r, y + r + 4, r, kInk);
  canvas.drawCircle(dx + r, y + r + 4, r - 1, kInk);
  font(vlw::Title);
  text("C", dx + 2 * r + 3, y, kInk);
  text(hum, x + w, y + (big ? 12 : 0), kBlue, top_right);
  font(vlw::Small);
  text("humidity", x + w, y + (big ? 52 : 40), kBlue, top_right);
  return big ? 96 : 66;
}

// A block: a label, a large line, a quieter line under it; where it ends.
static int drawBlock(int x, int y, int w, const String& label, const String& big, const String& small,
                     uint16_t bigInk = kInk) {
  drawLabel(label, x, y);
  y += 28;
  font(vlw::Title);
  text(fit(big, w), x, y, bigInk);
  y += 40;
  if (small.length()) {
    font(vlw::BodyS);
    text(fit(small, w), x, y, kInk);
    y += 32;
  }
  return y;
}

// The battery as words: how long it has, or until it's full.
static String batteryWords(int pct, bool chg) {
  if (chg) {
    const int m = battery::minutesToFull();
    return m == 0 ? String("full") : m > 0 ? "charging · full in ≈ " + timeSpan(m) : String("charging");
  }
  if (telemetry::onUsbPower()) return "full · on USB";
  const int m = battery::minutesLeft();
  if (m >= 0) return "≈ " + timeSpan(m) + " left";
  return pct >= 0 ? String("on battery") : String("");
}

static String clockLength(uint32_t secs) {
  char len[12];
  snprintf(len, sizeof(len), "%lu:%02lu", (unsigned long)(secs / 60), (unsigned long)(secs % 60));
  return len;
}

static Home lastHome;  // what the main loop last knew of the links (its rows: Wi-Fi, deck, agent)
static bool haveHome = false;

// The dashboard's top: the header (the clock as of this drawing, the links'
// dots, the battery), the date, the room, the battery's block. Returns where
// it ends.
static int drawDashTop(bool asleep) {
  drawHeader("", !asleep);
  const int x = PAD, w = W - 2 * PAD;
  int y = kHeadH + 14;
  y += drawDate(x, y, w);
  const int ch = drawClimate(x, y, w);
  y += ch ? ch + 12 : 4;
  bool chg = false;
  const int pct = batteryPct(&chg);
  y = drawBlock(x, y, w, "BATTERY", pct >= 0 ? String(pct) + " %" : String("?"), batteryWords(pct, chg),
                pct >= 0 && pct < 20 ? kRed : kInk);
  return y + 8;
}

// The dashboard: the clock as of this drawing, the date, the room, the
// battery, the notes kept offline, the links; drawn every five minutes and on
// request. Asleep: the same, the links replaced by how to wake it.
static void drawDashboard(bool asleep) {
  const int x = PAD, w = W - 2 * PAD;
  int y = drawDashTop(asleep);
  const int n = notes::count();
  const int roomMin = (int)(notes::room(kNoteRate) / kNoteRate / 60);
  String kept = n == 0 ? String("none waiting") : String(n) + (n == 1 ? " note waiting" : " notes waiting");
  if (n) kept += " · " + clockLength(notes::seconds());
  char gb[24];
  snprintf(gb, sizeof(gb), "%.1f GB", notes::freeBytes() / 1e9);
  const String room = !notes::ready() ? String("no room offline (no flash)")
                      : roomMin < 1    ? String("no room left for more")
                      : notes::onCard() ? "microSD card · " + String(gb) + " free"
                                        : "room for " + String(roomMin) + " min more";
  y = drawBlock(x, y, w, "VOICE NOTES", kept, room);
  y += 8;
  // The links, a row each, in the ink of their state; asleep, how to wake.
  drawLabel(asleep ? "ASLEEP" : "LINKS", x, y);
  y += 28;
  const int bottom = H - kFootH - 8;
  if (asleep) {
    font(vlw::BodyS);
    text(fit(String("Press ") + board::kTalkKey + " to wake it and talk.", w), x, y, kInk);
    text("Redrawn every five minutes.", x, y + 32, kInk);
  } else {
    font(vlw::Small);
    int labelW = 0;
    for (size_t i = 1; i < lastHome.rows.size(); ++i) labelW = max(labelW, tw(lastHome.rows[i].label));
    for (size_t i = 1; i < lastHome.rows.size() && y + 26 <= bottom; ++i) {
      const Row& r = lastHome.rows[i];
      text(r.label, x, y, kBlue);
      text(fit(r.value, w - labelW - 14), x + labelW + 14, y, snap(r.color));
      y += 26;
    }
    if (!haveHome || lastHome.rows.size() < 2) {
      text(hWifi == Up ? "Wi-Fi up" : hWifi == Partial ? "Wi-Fi connecting…" : "no Wi-Fi", x, y,
           hWifi == Up ? kGreen : hWifi == Partial ? kYellow : kRed);
      text(hAgent == Up ? "agent linked" : "no agent yet", x, y + 26, hAgent == Up ? kGreen : kYellow);
    }
  }
  drawFooter({{board::kTalkKey, "talk"}, {board::kSideKey, "options"}, {board::kBackKey, "answer"}});
}

static void askDashboard(bool asleep) { draw(Kind::Dash, [=] { drawDashboard(asleep); }); }

// On each five-minute mark, while the dashboard is on the paper: the whole
// page again (a reset controller makes a full refresh quick).
static void dashTick() {
  if (shownKind == Kind::Dash && !viewing && !answerDue && dashDue()) askDashboard(false);
}

// The dashboard on its five-minute mark, or when it was asked for (drawNow).
static void dashboardIfDue() {
  const bool now = forceNext && millis() - forceAtMs < 3000;
  forceNext = false;
  if (now || dashDue()) askDashboard(false);
}

void showHome(const Home& home) {
  lastHome = home;
  haveHome = true;
  dashboardIfDue();
}

// Going to sleep, and the timer wake while asleep: the dashboard when its
// mark has come (the sleep timer is set to land on the next), else nothing.
void showSleeping(const String&, uint8_t) {
  forceNext = false;
  if (dashDue()) askDashboard(true);
  flush();
}

// ── the answer: at the largest size that fits ──────────────────────────

static std::vector<std::vector<String>> replyPages;
static int replyPage = 0;
static bool needLayout = false;  // laid out when it's drawn, not as each word streams in
static vlw::Id replyFont = vlw::BodyS;
static int replyLine = 35;
static int pictureH = 0;  // the picture on page 1 (0: none)

struct Size {
  vlw::Id font;
  int line;
};
static const Size kSizes[] = {{vlw::BodyL, 49}, {vlw::BodyM, 41}, {vlw::BodyS, 35}, {vlw::Small, 28}};

static int replyTop() { return kHeadH + 18; }
static int replyBottom() { return H - kFootH - 34; }

// The question, one line, over the answer on its first page.
static int questionHeight() { return question.length() ? 27 + 10 : 0; }

// The largest size whose words fit on one screen (under the question and
// the picture); none fits → the smallest, over pages.
static void layoutReply() {
  needLayout = false;
  ++replyGen;
  const int keep = viewing ? replyPage : 0;  // the page being read, as more arrives
  replyPages.clear();
  replyPage = 0;
  pictureH = 0;
  if (replyText.isEmpty()) return;
  const int width = W - 2 * PAD, top = replyTop(), bottom = replyBottom();
  const int qH = questionHeight();
  int pw = 0, ph = 0;
  if (imgLen && jpegSize(imgBuf, imgLen, &pw, &ph)) {
    const int natural = (int)(ph * min(1.0f, (float)width / pw));
    pictureH = min(natural, (bottom - top - qH) * 40 / 100) + 14;
  }
  const int avail = bottom - top - qH - pictureH;
  for (const Size& s : kSizes) {
    font(s.font);
    auto lines = wrap(replyText, width);
    if ((int)lines.size() * s.line <= avail || &s == &kSizes[3]) {
      replyFont = s.font;
      replyLine = s.line;
      const int first = max(1, avail / s.line), per = max(1, (bottom - top) / s.line);
      replyPages.emplace_back(lines.begin(), lines.begin() + min(lines.size(), (size_t)first));
      for (size_t i = replyPages[0].size(); i < lines.size(); i += per) {
        replyPages.emplace_back(lines.begin() + i, lines.begin() + min(lines.size(), i + per));
      }
      replyPage = min(keep, (int)replyPages.size() - 1);
      return;
    }
  }
}

static void drawReplyPage() {
  if (replyPages.empty()) return;
  const int pages = replyPages.size(), page = min(replyPage, pages - 1);
  const int width = W - 2 * PAD;
  drawHeader("Answer");
  int y = replyTop();
  if (page == 0) {
    if (question.length()) {
      font(vlw::Small);
      text(fit(String("“") + question + "”", width), PAD, y, kBlue);
      y += questionHeight();
    }
    if (pictureH) {
      const int ph = drawPicture(imgBuf, imgLen, PAD, y, width, pictureH - 14);
      if (ph) y += ph + 14;
    }
  }
  font(replyFont);
  drawLines(replyPages[page], PAD, y, replyLine, kInk);
  drawPageDots(pages, page, H - kFootH - 18);
  char when[16] = "";
  clockText(when, sizeof(when), "%H:%M", replyWhen);
  drawFooter({{board::kTalkKey, "talk"}, {board::kBackKey, page + 1 < pages ? "next" : "done"}}, when);
}

static bool newTurn = true;  // a turn began (setQuestion): what comes next is a new answer

void setQuestion(const String& text) {
  pendingQuestion = text;
  newTurn = true;
}

// A new answer: it goes on the paper as its voice starts (answerStarted).
// (The same text again — a reconnect — changes nothing.)
void setReply(const String& t) {
  if (t == replyText && replyWhen) return;
  replyText = t;
  needLayout = true;
  newTurn = false;
  question = pendingQuestion;
  pendingQuestion = "";
  replyWhen = time(nullptr);
  if (imgBuf) free(imgBuf);  // the last answer's picture goes with it
  imgBuf = nullptr;
  imgLen = 0;
  viewing = sounding = false;
  answerDue = t.length() && !restoring;
  if (t.length() && !restoring) saveLast();
}

// More of the same answer as it's written (kept once it's whole: answerEnded).
void growReply(const String& t) {
  if (newTurn || replyText.isEmpty()) {
    setReply(t);
    return;
  }
  if (t == replyText) return;
  replyText = t;
  needLayout = true;
}
void setTextSize(uint8_t) {}  // the answer picks its own size
bool hasReply() { return !replyText.isEmpty(); }

static void ensureLayout() {
  if (needLayout) layoutReply();
}

static void askReplyPage() {
  draw(Kind::Reply, [] {
    shownGen = replyGen;
    shownPage = replyPage;
    drawReplyPage();
  });
}

// Only while it's on the paper, and only when the page there isn't this one
// already — not while its voice plays (drawn again when that ends); else the
// dashboard on its mark (not before a new answer about to be spoken).
void showReply() {
  ensureLayout();
  if (!viewing || replyPages.empty()) {
    if (!answerDue) dashboardIfDue();
    return;
  }
  if (sounding) return;
  if (shownKind == Kind::Reply && shownGen == replyGen && shownPage == replyPage) return;
  askReplyPage();
}

// The answer's voice has started (or it came with speaking off): it's on
// the paper now, its first page, without waiting for B.
void answerStarted() {
  ensureLayout();
  answerDue = false;
  if (replyPages.empty()) return;
  viewing = sounding = true;
  replyPage = 0;
  forceNext = false;
  askReplyPage();
}

// The answer is whole and its voice is over: kept in flash, and its page
// redrawn when more of it came since it went on the paper.
void answerEnded() {
  sounding = false;
  if (replyText.isEmpty()) return;
  if (!restoring) saveLast();
  ensureLayout();
  if (!viewing || replyPages.empty()) return;
  if (want.active && want.kind == Kind::Reply) return;  // a page is on its way: drawn as it is now
  if (shownKind == Kind::Reply && shownGen == replyGen) return;
  askReplyPage();
}

// B: the answer (its first page), then the next page, then the dashboard.
// False when it went back to the dashboard.
bool nextPage() {
  ensureLayout();
  if (replyPages.empty()) return false;
  answerDue = false;
  if (!viewing) {
    viewing = true;
    replyPage = 0;
  } else if (replyPage + 1 < (int)replyPages.size()) {
    ++replyPage;
  } else {
    viewing = false;
    forceNext = false;
    askDashboard(false);  // fresh readings too
    return false;
  }
  askReplyPage();
  return true;
}

bool prevPage() {
  ensureLayout();
  if (!viewing || replyPage == 0) return false;
  --replyPage;
  askReplyPage();
  return true;
}

bool onLastPage() {
  ensureLayout();
  return replyPage + 1 >= (int)replyPages.size();
}
bool onFirstPage() { return replyPage == 0; }

bool restoreAnswer() {
  if (!savedLast.valid()) return false;
  pendingQuestion = savedLast.q;
  replyText = "";
  restoring = true;
  setReply(savedLast.a);
  replyWhen = savedLast.when;  // when it was answered, not now
  if (savedLast.len) storeImage(copyBytes(savedLast.jpg, savedLast.len), savedLast.len);
  restoring = false;
  return true;
}

void storeImage(uint8_t* jpg, size_t len) {
  if (imgBuf) free(imgBuf);
  imgBuf = jpg;
  imgLen = jpg ? len : 0;
  needLayout = true;
  if (replyText.length() && !restoring) saveLast();
}

bool hasImage() { return imgLen > 0; }

// ── the options code, setting up ───────────────────────────────────────

struct QrLine {
  String text;
  vlw::Id font;
  uint16_t ink;
};
static void drawQrWithLines(const String& code, int version, int top, int bottom, const std::vector<QrLine>& lines) {
  const int width = W - 2 * PAD;
  int linesH = 0;
  for (auto& l : lines) {
    font(l.font);
    linesH += min(3, (int)wrap(l.text, width).size()) * (l.font == vlw::Small ? 28 : 36);
  }
  const int q = max(120, min(width - 60, bottom - top - linesH - 20));
  const int qx = (W - q) / 2;
  canvas.qrcode(code.c_str(), qx, top, q, version);
  int y = top + q + 18;
  for (auto& l : lines) {
    font(l.font);
    int lh = l.font == vlw::Small ? 28 : 36;
    for (auto& w : clamp(wrap(l.text, width), 3, width)) {
      if (y + lh > bottom + 6) return;
      text(w, W / 2, y, l.ink, top_center);
      y += lh;
    }
  }
}

void showSetup(const char* reason, const char* apName, const char* ip) {
  String r = reason, ap = apName, addr = ip;
  forceNext = false;
  draw(Kind::Setup, [=] {
    font(vlw::Title);
    text(fit(r, W - 2 * PAD), W / 2, 28, kRed, top_center);
    drawQrWithLines(String("WIFI:T:nopass;S:") + ap + ";;", 3, 82, H - 52,
                    {{"1. Scan the code, or join this Wi-Fi:", vlw::Small, kInk},
                     {ap, vlw::BodyS, kBlue},
                     {"2. Then open:", vlw::Small, kInk},
                     {String("http://") + addr, vlw::BodyS, kBlue}});
    font(vlw::Small);
    text(String("Saving restarts it. Press ") + board::kTalkKey + " to leave.", W / 2, H - 38, kInk, top_center);
  });
  flush();  // the portal loop doesn't tick
}

// A: the options code (the settings page on a phone on the same Wi-Fi),
// with what the paper is and what it has.
// (Asked again every 30 s while it's open: drawn once, unless it changes.)
void showLink(const char* title, const String& url, const String& note) {
  String t = title;
  forceNext = false;
  const String key = t + "\n" + url;
  if (key == linkShown && ((want.active && want.kind == Kind::Link) || shownKind == Kind::Link)) return;
  linkShown = key;
  draw(Kind::Link, [=] {
    drawHeader();
    font(vlw::Title);
    text(fit(t, W - 2 * PAD), W / 2, kHeadH + 18, kInk, top_center);
    drawQrWithLines(url, 4, kHeadH + 70, H - kFootH - 12, {{url, vlw::Small, kBlue}, {note, vlw::BodyS, kInk}});
    drawFooter({{board::kTalkKey, "back"}, {board::kSideKey, "update"}});
  });
}

// ── everything else: the lights and the voice carry it ─────────────────

void showMessage(const char*, const String&, uint16_t) {}
void showAsk(const char*, const String&, const char*, const char*) {}
int showPanel(const char*, uint16_t, const std::vector<Row>&, int) { return 1; }
int showStatus(const Status&, int) { return 1; }
void showListening(int, uint32_t) {}
void showWorking(const String&, const std::vector<String>&, const String&, uint32_t, int) {}
void showSpeaking(int) {}
void showFace(uint8_t, face::Pose, int) {}
void showCharPick(uint8_t, face::Pose, int, int) {}
void showPicker(const std::vector<Item>&, int, int, const char*) { forceNext = false; }
void showStoredImage() {}
void flashFrame(bool) {}
bool canPin() { return false; }
bool pinned() { return false; }
void pin(bool) {}
void slide(bool) {}
void fade() {}
bool setSpoken(int) { return false; }
void followVoice(int) {}
void setReplayable(bool) {}
bool replayHit(int, int) { return false; }
void setBackground(const uint8_t*, size_t) {}
bool hasBackground() { return false; }
void showBackdrop() {}
void setTapButton(TapButton) {}
bool tapButtonHit(int, int) { return false; }
int askHit(int, int) { return -1; }
void toggleImageFit() {}

}  // namespace ui
