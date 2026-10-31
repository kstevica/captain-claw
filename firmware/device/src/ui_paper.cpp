// PaperColor screen: 4" E Ink Spectra 6 colour e-paper, 400×600 (portrait;
// the menu turns it), six inks: black, white, red, yellow, blue, green.
//
// The panel takes 15–30 s to redraw (every redraw: Spectra 6 has no faster
// refresh, and its undocumented partial window takes as long and fades what
// is outside it) and can't be interrupted, so the paper is voice-first
// and redraws when it's put down: while it's in use (presses, talking, an
// answer playing) nothing is drawn — speech, sound and the two status
// lights (leds.cpp) carry it, and the menu is only read aloud. Once it has
// been left alone for kIdleMs, the paper catches up with the latest answer.
// Every show…() only *asks* for a screen; the scheduler in tick() draws it
// when the device is idle (or at once when the user asked to see something:
// drawNow()), never the same screen twice, and never the same answer again
// just because it moved between the home, answer and sleeping screens or
// its readings (battery, temperature) changed. What the paper shows is
// remembered across deep sleep, so waking and sleeping don't redraw. Text
// only: no characters.
//
// Drawing goes to an off-screen canvas. A background task turns it into the
// six inks — pictures error-diffused (Floyd–Steinberg), text and shapes
// snapped to the nearest ink so they stay crisp — and runs the slow refresh,
// so the main loop keeps the audio and the link going meanwhile.
//
// The last answer and a pinned one are kept in flash (LittleFS), so the home
// screen and the sleeping screen still show them after a deep sleep.
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
#endif

namespace ui {

// ── inks ───────────────────────────────────────────────────────────────

// All six inks. (True: black and white only — colours snap by lightness and
// pictures dither to black dots. A Spectra 6 redraw takes the same 15–30 s
// either way; black merely settles first. The paper redraws when put down,
// so the colours are back.)
static constexpr bool kMono = false;

// The panel's six inks, as its driver (M5GFX Panel_ED2208) matches them.
struct InkDef {
  uint8_t r, g, b;
};
static constexpr InkDef kInkDefs[6] = {
    {0, 0, 0}, {255, 255, 255}, {255, 243, 56}, {191, 0, 0}, {100, 64, 255}, {67, 138, 28}};
static constexpr uint16_t kInk = 0x0000, kPaper = 0xFFFF;
static constexpr uint16_t kYellow = rgb(255, 243, 56), kRed = rgb(191, 0, 0);
static constexpr uint16_t kBlue = rgb(100, 64, 255), kGreen = rgb(67, 138, 28);

// A colour from the shared code (made for dark screens) as an ink on white
// paper: white and greys are ink; red/orange → red; yellow → yellow (as text:
// ink, yellow can't be read on white); green → green; cyan/blue → blue.
static uint16_t inkOf(uint16_t c, bool text) {
  int r = ((c >> 11) & 31) * 255 / 31, g = ((c >> 5) & 63) * 255 / 63, b = (c & 31) * 255 / 31;
  int mx = max(r, max(g, b)), mn = min(r, min(g, b));
  if (mx - mn < 60) return kInk;
  float d = mx - mn, h;
  if (mx == r) h = 60.0f * fmodf((g - b) / d, 6.0f);
  else if (mx == g) h = 60.0f * ((b - r) / d + 2.0f);
  else h = 60.0f * ((r - g) / d + 4.0f);
  if (h < 0) h += 360.0f;
  if (h < 40 || h >= 330) return kRed;
  if (h < 75) return text ? kInk : kYellow;
  if (h < 170) return kGreen;
  return kBlue;
}

// ── canvas, fonts, layout ──────────────────────────────────────────────

static int W = 400, H = 600;
static constexpr int PAD = 22;
static constexpr int kHeadH = 52;  // the top bar
static constexpr int kFootH = 48;  // the button legend
static M5Canvas canvas(&M5.Display);
static M5Canvas faceSprite;
static lgfx::VLWfont fontSet[vlw::Count];
static lgfx::PointerWrapper fontData[vlw::Count];
static uint8_t rotationNow = 0;

struct TextSize {
  vlw::Id body;
  int line;
};
static const TextSize kSizes[] = {{vlw::BodyS, 35}, {vlw::BodyM, 41}, {vlw::BodyL, 49}};
static uint8_t sizeIdx = 0;
static const TextSize& ts() { return kSizes[sizeIdx]; }

static String barAgent, barState;
static uint16_t barColor = kInk;
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

// ── persisted answers (the last one, a pinned one) ─────────────────────

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
static Card pinCard;

static String question, pendingQuestion;
static bool restoring = false;  // bringing back the saved answer: it's saved already
static String replyText;
static time_t replyWhen = 0;
static std::vector<std::vector<String>> replyPages;
static int replyPage = 0;
static int pictureH = 0;  // the picture's height on page 1 of the answer (0 = none)
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

// The answer on screen now, as a card.
static void currentCard(Card& c) {
  c.clear();
  c.q = question;
  c.a = replyText;
  c.when = replyWhen;
  c.jpg = copyBytes(imgBuf, imgLen);
  c.len = c.jpg ? imgLen : 0;
}

static void saveLast() {
  Card c;
  currentCard(c);
  saveCard("last", c);
  c.clear();
}

// ── JPEG size (its SOF marker) ─────────────────────────────────────────

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
static int drawPicture(const uint8_t* jpg, size_t len, int x, int y, int bw, int bh, bool centreV = false) {
  int w = 0, h = 0;
  if (!jpg || !len || !jpegSize(jpg, len, &w, &h)) return 0;
  float s = min(1.0f, min((float)bw / w, (float)bh / h));
  int dw = (int)(w * s), dh = (int)(h * s);
  int dx = x + (bw - dw) / 2, dy = centreV ? y + (bh - dh) / 2 : y;
  canvas.drawJpg(jpg, len, dx, dy, dw, dh, 0, 0, s, s);
  pictures.push_back({dx, dy, dw, dh});
  return dh;
}

// ── the slow refresh, off the main loop ───────────────────────────────

static volatile bool refreshing = false;

// The nearest ink, for text and shapes: greys go to black or white by
// lightness (antialiased edges stay crisp instead of turning blue), colours
// to the nearest ink.
static uint16_t snap(uint16_t c) {
  int r = ((c >> 11) & 31) * 255 / 31, g = ((c >> 5) & 63) * 255 / 63, b = (c & 31) * 255 / 31;
  int mx = max(r, max(g, b)), mn = min(r, min(g, b));
  if (kMono || mx - mn < 64) return (r * 299 + g * 587 + b * 114) / 1000 < 150 ? kInk : kPaper;
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

// Floyd–Steinberg (serpentine) over one picture, into the six inks (or,
// black-and-white, two).
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
      if (kMono) {
        best = (v[0] * 299 + v[1] * 587 + v[2] * 114) / 1000 < 128 ? 0 : 1;  // black or white by lightness
      } else {
        for (int p = 0; p < 6; ++p) {
          int dr = v[0] - kInkDefs[p].r, dg = v[1] - kInkDefs[p].g, db = v[2] - kInkDefs[p].b;
          int dd = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
          if (dd < bestD) bestD = dd, best = p;
        }
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

static void pushToPanel() {
  waitPanel(45000);
  M5.Display.startWrite();
  canvas.pushSprite(&M5.Display, 0, 0);
  M5.Display.endWrite();
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

// ── the scheduler ──────────────────────────────────────────────────────

enum class Kind : uint8_t { None, Home, Reply, Picture, Working, Notice, Panel, Picker, CharPick, Link, Setup, Sleep };

// How long a screen must stay asked for before the paper draws it (a few
// quick presses become one redraw), and how often live readings may redraw
// a screen that is otherwise unchanged.
static uint32_t settleMs(Kind k) {
  switch (k) {
    case Kind::Reply: return 1500;   // a picture often follows the words
    case Kind::Picker:
    case Kind::CharPick: return 900;
    case Kind::Home: return 1500;
    default: return 600;
  }
}

struct Want {
  Kind kind = Kind::None;
  String major, minor;  // what the screen says / its live readings
  uint32_t answer = 0;  // which answer it shows (0: none)
  std::function<void()> draw;
  uint32_t dueMs = 0;
  bool forced = false;  // asked for by the user: no waiting for a quiet spell
  bool active = false;
};
static Want want;
static Kind shownKind = Kind::None;
static String shownMajor, shownMinor;
static uint32_t shownAnswer = 0;
static bool everShown = false;

static constexpr uint32_t kIdleMs = 30000;  // left alone this long: the paper catches up
static uint32_t idleFromMs = 0;
static bool forceNext = false;

#ifndef RTC_DATA_ATTR
#define RTC_DATA_ATTR  // (the preview has no deep sleep)
#endif
// What the paper shows, across deep sleep (the panel keeps its picture).
RTC_DATA_ATTR static bool rtcShown = false;
RTC_DATA_ATTR static uint8_t rtcKind = 0;
RTC_DATA_ATTR static uint32_t rtcAnswer = 0;

// The home, answer and sleeping screens are the same answer in different
// clothes: moving between them isn't worth a redraw.
static bool answerScreen(Kind k) { return k == Kind::Home || k == Kind::Reply || k == Kind::Sleep; }

// Setting up, a link to scan, a notice (it answers something just done or
// gone wrong) and going to sleep can't wait for a quiet spell.
static bool waitsForIdle(Kind k) {
  return k != Kind::Setup && k != Kind::Link && k != Kind::Notice && k != Kind::Sleep;
}

void idleSince(uint32_t ms) { idleFromMs = ms; }
void drawNow() { forceNext = true; }
bool waiting() { return want.active && !want.forced && waitsForIdle(want.kind); }

static void ask(Kind kind, const String& major, const String& minor, std::function<void()> draw,
                uint32_t answer = 0) {
  const uint32_t now = millis();
  const bool forced = forceNext;
  forceNext = false;
  if (everShown && kind == shownKind && major == shownMajor) {
    want.active = false;  // what the paper shows (or is drawing now) — fresher readings wait for the next redraw
    return;
  }
  if (!forced && everShown && answerScreen(kind) && answerScreen(shownKind) && answer == shownAnswer) {
    want.active = false;  // the same answer, dressed differently
    return;
  }
  if (want.active && kind == want.kind && major == want.major) {
    want.minor = minor;  // same screen, fresher readings: keep its turn
    want.draw = std::move(draw);
    want.forced = want.forced || forced;
    return;
  }
  want.kind = kind;
  want.major = major;
  want.minor = minor;
  want.answer = answer;
  want.draw = std::move(draw);
  want.dueMs = now + settleMs(kind);
  want.forced = forced;
  want.active = true;
}

static void applyRotation();  // a turn of the paper waits for the panel (below)

static void commit() {
  Want w = std::move(want);
  want.active = false;
  pictures.clear();
  canvas.fillScreen(kPaper);
  w.draw();
  shownKind = w.kind;
  shownMajor = w.major;
  shownMinor = w.minor;
  shownAnswer = w.answer;
  everShown = true;
  rtcShown = true;
  rtcKind = (uint8_t)w.kind;
  rtcAnswer = w.answer;
  startRefresh();
}

void tick() {
  if (refreshing) return;
  applyRotation();
  if (!want.active) return;
  if ((int32_t)(millis() - want.dueMs) < 0) return;
  if (!want.forced && waitsForIdle(want.kind) && millis() - idleFromMs < kIdleMs) return;  // still in use
  commit();
}

bool busy() { return refreshing; }

void flush() {
#ifndef PREVIEW
  while (refreshing) delay(20);
#endif
  applyRotation();
  if (want.active) commit();
#ifndef PREVIEW
  while (refreshing) delay(20);
#endif
}

// ── chrome ─────────────────────────────────────────────────────────────

static uint16_t healthInk(Health h) { return h == Up ? kGreen : h == Partial ? kYellow : h == Asleep ? kBlue : kRed; }

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

// Battery in 5 % steps, for deciding whether the paper needs a redraw.
static String batteryKey() {
  bool chg = false;
  int pct = batteryPct(&chg);
  return String(pct / 5) + (chg ? "c" : "");
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

// `t` formatted in local time; false (and nothing written) before the clock
// was ever set, or for a time that was never known (0).
static bool clockText(char* out, size_t n, const char* fmt, time_t t) {
  if (t < 1700000000) return false;
  struct tm tm;
  localtime_r(&t, &tm);
  strftime(out, n, fmt, &tm);
  return true;
}

// The top bar: the agent, health dots (not while asleep: nothing is
// connected then) and the battery. A black rule under it. No clock: the
// paper keeps a time long after it's wrong — times say what they belong to
// ("drawn 14:32", the answer's time) in the footer instead.
static void drawHeader(const String& title = "", bool dots = true) {
  font(vlw::Small);
  int y = kHeadH / 2;
  String name = title.length() ? title : (barAgent.length() ? barAgent : String("Captain"));
  // The agent's name: blue while it's asleep, red while it's switched off.
  const uint16_t nameInk = title.length() || !barAgent.length() ? kInk
                           : hAgent == Asleep                   ? kBlue
                           : hAgent == Off                      ? kRed
                                                                : kInk;
  bool chg = false;
  int pct = batteryPct(&chg);
  int right = W - PAD;
  drawBattery(right, y, pct, chg);
  right -= 92;
  if (dots) {
    canvas.fillCircle(right, y, 6, healthInk(hAgent));
    canvas.fillCircle(right - 18, y, 6, healthInk(hWifi));
    right -= 34;
  }
  font(vlw::Title);
  text(fit(name, right - PAD), PAD, y + 2, nameInk, middle_left);
  canvas.fillRect(0, kHeadH - 3, W, 3, kInk);
}

// A keycap: the button's letter in a small black rounded square.
static int drawKey(const char* key, int x, int y) {
  font(vlw::Small);
  canvas.fillRoundRect(x, y - 14, 28, 28, 6, kInk);
  text(key, x + 14, y + 1, kPaper, middle_center);
  return x + 28;
}

// The bottom bar: what the buttons do here ("A", "talk", "B", "next", …).
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

// A label in small caps-ish blue with a short rule before it.
static void drawLabel(const String& s, int x, int y, uint16_t color = kBlue) {
  canvas.fillRect(x, y + 4, 4, 16, color);
  font(vlw::Small);
  text(s, x + 12, y, color);
}

// The characters are drawn for the stick's 240×135 screen on black; on the
// paper they live in a black rounded window, scaled up.
static void drawCharacter(uint8_t kind, face::Pose pose, int cx, int cy, float scale) {
  const int ww = (int)(240 * scale) + 16, wh = (int)(135 * scale) + 16;
  canvas.fillRoundRect(cx - ww / 2, cy - wh / 2, ww, wh, 18, kInk);
  faceSprite.fillScreen(0x0000);
  face::draw(faceSprite, kind, pose, 0, 0, 120, 76);
  faceSprite.setPivot(120, 76);
  faceSprite.pushRotateZoomWithAA(&canvas, cx, cy + 4, 0, scale, scale);
  pictures.push_back({cx - ww / 2 + 8, cy - wh / 2 + 8, ww - 16, wh - 16});
}

// ── public: basics ─────────────────────────────────────────────────────

static void makeCanvas() {
  M5.Display.setRotation(rotationNow);
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
  if (esp_reset_reason() == ESP_RST_DEEPSLEEP && rtcShown) {  // the panel kept its picture
    everShown = true;
    shownKind = (Kind)rtcKind;
    shownAnswer = rtcAnswer;
  }
#endif
  M5.Display.setAutoDisplay(false);
  M5.Display.setEpdMode(epd_mode_t::epd_fastest);  // the frame is already in the six inks
  makeCanvas();
  faceSprite.setPsram(true);
  faceSprite.setColorDepth(16);
  faceSprite.createSprite(240, 135);
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
  loadCard("pin", pinCard);
}

// The canvas changes size, so a turn waits until the paper isn't drawing from
// it (tick() applies it; the main loop never blocks for a redraw).
static int pendingRotation = -1;

static void layoutReply();

static void applyRotation() {
  if (pendingRotation < 0 || refreshing) return;
  rotationNow = pendingRotation;
  pendingRotation = -1;
  makeCanvas();
  everShown = false;  // whatever is wanted next is drawn the new way round
  layoutReply();
}

void setRotation(uint8_t rotation) {
  rotation &= 3;
  pendingRotation = rotation == rotationNow ? -1 : rotation;
  applyRotation();
}

void brightness(uint8_t) {}  // paper has no backlight

bool animated() { return false; }

void setBar(const String& agent, const char* state, uint16_t stateColor) {
  barAgent = agent;
  barState = state ? state : "";
  barColor = stateColor;
}

void setHealth(Health wifi, Health agent) {
  hWifi = wifi;
  hAgent = agent;
}

static String healthKey() { return String((int)hWifi) + String((int)hAgent); }

// ── notices ────────────────────────────────────────────────────────────

// A QR code with lines beside it (landscape) or under it (portrait).
// `lines` are {text, font, ink}; returns nothing — it fills the room it has.
struct QrLine {
  String text;
  vlw::Id font;
  uint16_t ink;
};
static void drawQrWithLines(const String& code, int version, int top, int bottom, const std::vector<QrLine>& lines) {
  const bool wide = W > H;
  const int width = W - 2 * PAD;
  int linesH = 0;  // portrait: the lines under the code, wrapped as they will be
  for (auto& l : lines) {
    font(l.font);
    linesH += min(3, (int)wrap(l.text, width).size()) * (l.font == vlw::Small ? 28 : 36);
  }
  const int q = wide ? min(bottom - top, width / 2) : max(120, min(width - 60, bottom - top - linesH - 20));
  const int qx = wide ? PAD : (W - q) / 2;
  canvas.qrcode(code.c_str(), qx, top, q, version);
  const int tx = wide ? qx + q + 24 : W / 2;
  const int room = wide ? W - PAD - tx : width;
  int y = wide ? top + max(0, (q - linesH) / 2) : top + q + 18;
  for (auto& l : lines) {
    font(l.font);
    int lh = l.font == vlw::Small ? 28 : 36;
    for (auto& w : clamp(wrap(l.text, room), 3, room)) {
      if (y + lh > bottom + 6) return;
      text(w, tx, y, l.ink, wide ? top_left : top_center);
      y += lh;
    }
  }
}

void showSetup(const char* reason, const char* apName, const char* ip) {
  String r = reason, ap = apName, addr = ip;
  ask(Kind::Setup, r + ap + addr, "", [=] {
    font(vlw::Title);
    text(fit(r, W - 2 * PAD), W / 2, 28, kRed, top_center);
    drawQrWithLines(String("WIFI:T:nopass;S:") + ap + ";;", 3, 82, H - 52,
                    {{"1. Scan the code, or join this Wi-Fi:", vlw::Small, kInk},
                     {ap, vlw::BodyS, kBlue},
                     {"2. Then open:", vlw::Small, kInk},
                     {String("http://") + addr, vlw::BodyS, kBlue}});
    font(vlw::Small);
    text("Saving restarts it. Hold B to leave.", W / 2, H - 38, kInk, top_center);
  });
  flush();  // the portal loop doesn't tick
}

// Digits change while the text stays the same ("Updating 34%") — those are
// live readings, not a new notice.
static String withoutDigits(const String& s) {
  String o;
  for (int i = 0; i < (int)s.length(); ++i) {
    if (!isdigit((uint8_t)s[i])) o += s[i];
  }
  return o;
}

// A notice, or (with `yes`) a question the talk button answers.
static void notice(const char* title, const String& body, uint16_t color, const char* yes = nullptr,
                   const char* no = nullptr) {
  String t = title;
  uint16_t ink = inkOf(color, false), textInk = inkOf(color, true);
  ask(Kind::Notice, withoutDigits(t + "\n" + body), t + body, [=] {
    drawHeader();
    const bool warn = ink == kRed || ink == kYellow;
    const int width = W - 2 * PAD;
    int cy = kHeadH + (H > W ? 90 : 62);
    canvas.fillCircle(W / 2, cy, 40, ink == kInk ? kBlue : ink);
    uint16_t mark = ink == kYellow ? kInk : kPaper;
    if (warn) {
      canvas.fillRoundRect(W / 2 - 5, cy - 24, 10, 30, 5, mark);
      canvas.fillCircle(W / 2, cy + 17, 6, mark);
    } else {
      canvas.fillCircle(W / 2, cy - 19, 6, mark);
      canvas.fillRoundRect(W / 2 - 5, cy - 8, 10, 30, 5, mark);
    }
    font(vlw::Title);
    int y = cy + 60;
    for (auto& l : clamp(wrap(t, width), 2, width)) {
      text(l, W / 2, y, textInk, top_center);
      y += 42;
    }
    font(ts().body);
    y += 10;
    int maxLines = max(1, (H - kFootH - 10 - y) / ts().line);
    for (auto& l : clamp(wrap(body, width), maxLines, width)) {
      text(l, W / 2, y, kInk, top_center);
      y += ts().line;
    }
    if (yes) drawFooter({{board::kTalkKey, yes}, {board::kBackKey, no}});
    else drawFooter({{board::kSideKey, "back"}});
  });
}

void showMessage(const char* title, const String& body, uint16_t color) { notice(title, body, color); }

void showAsk(const char* title, const String& body, const char* yes, const char* no) {
  notice(title, body, kInk, yes, no);
}

void showLink(const char* title, const String& url, const String& note) {
  String t = title;
  ask(Kind::Link, t + url + note, "", [=] {
    drawHeader();
    font(vlw::Title);
    text(fit(t, W - 2 * PAD), W / 2, kHeadH + 18, kInk, top_center);
    drawQrWithLines(url, 4, kHeadH + 70, H - kFootH - 12,
                    {{url, vlw::Small, kBlue}, {note, vlw::BodyS, kInk}});
    drawFooter({{board::kTalkKey, "update"}, {board::kSideKey, "done"}});
  });
}

// ── home and sleep ─────────────────────────────────────────────────────

// The answer to keep on the home screen: the pinned one, else the newest.
static const Card* homeCard(Card& scratch) {
  if (pinCard.valid()) return &pinCard;
  if (replyText.length()) {
    scratch.q = question;
    scratch.a = replyText;
    scratch.when = replyWhen;
    scratch.jpg = imgBuf;  // borrowed, not freed
    scratch.len = imgLen;
    return &scratch;
  }
  return savedLast.valid() ? &savedLast : nullptr;
}

// Which answer a card or the current reply is, for "the paper shows that
// already" (a picture that arrives later makes it a different one).
static uint32_t answerId(time_t when, size_t picture) {
  return when ? (uint32_t)when * 31u + (uint32_t)picture : 0;
}
static uint32_t answerId(const Card* c) { return c ? answerId(c->when, c->len) : 0; }

static String cardKey(const Card* c) {
  if (!c) return "-";
  return String((uint32_t)c->when) + ":" + String(c->a.length()) + ":" + String((uint32_t)c->len) +
         (c == &pinCard ? "p" : "");
}

// "Monday, 19 October"; returns the height it took (0 before the clock is set).
static int drawDate(int x, int y, int w) {
  char date[40];
  if (!clockText(date, sizeof(date), "%A, %e %B", time(nullptr))) return 0;
  String d = date;
  d.replace("  ", " ");  // %e pads single digits
  font(vlw::Title);
  if (tw(d) > w && clockText(date, sizeof(date), "%a %e %b", time(nullptr))) {  // "Mon 19 Oct"
    d = date;
    d.replace("  ", " ");
  }
  text(fit(d, w), x, y, kInk);
  return 44;
}

// The room: temperature large, humidity at the right of the column.
// Returns the height it took (0 on a device without the sensor).
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
  const bool big = tw(tt) + 44 + 20 + humW <= w;  // else the smaller bold figures
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

// The state as a pill: a dot and what to do.
static void drawChip(int x, int y, int w, uint16_t fill, uint16_t dot, uint16_t ink, const String& hint, bool moon) {
  canvas.fillRoundRect(x, y, w, 44, 22, fill);
  if (moon) {  // a crescent
    canvas.fillCircle(x + 24, y + 22, 11, ink);
    canvas.fillCircle(x + 29, y + 18, 10, fill);
  } else {
    canvas.fillCircle(x + 24, y + 22, 8, dot);
  }
  font(vlw::BodyS);
  text(fit(hint, w - 60), x + 46, y + 23, ink, middle_left);
}

// The kept answer as a card: a coloured edge (yellow pinned, blue the
// newest), what was asked, the picture, then as much of the answer as fits.
static void drawCard(const Card& c, int x0, int top, int width, int bottom, bool isPin) {
  uint16_t edge = isPin ? kYellow : kBlue;
  canvas.fillRect(x0, top, 6, bottom - top, edge);
  const int x = x0 + 18, cw = width - 18;
  int y = top;
  char when[24] = "";
  clockText(when, sizeof(when), "%a %H:%M", c.when);
  font(vlw::Small);
  String tag = isPin ? String("PINNED") : String("LAST ANSWER");
  if (when[0]) tag += String("  ·  ") + when;
  text(fit(tag, cw), x, y, isPin ? kInk : kBlue);
  y += 32;
  if (c.q.length()) {
    auto ql = clamp(wrap(String("“") + c.q + "”", cw), 2, cw);
    drawLines(ql, x, y, 27, kBlue);
    y += ql.size() * 27 + 8;
  }
  if (c.len && bottom - y > 120) {
    int ph = drawPicture(c.jpg, c.len, x, y, cw, min(220, (bottom - y) / 2));
    if (ph) y += ph + 12;
  }
  font(ts().body);
  int maxLines = (bottom - y) / ts().line;
  if (maxLines > 0) drawLines(clamp(wrap(c.a, cw), maxLines, cw), x, y, ts().line, kInk);
}

// Until there's an answer to keep: what to do, in words.
static void drawWelcome(int x0, int top, int width, int bottom) {
  font(vlw::Title);
  auto big = clamp(wrap(String(board::kTalkToggles ? "Press the " : "Hold the ") + board::kTalkButton +
                            " and ask me anything.", width), 3, width);
  font(vlw::BodyS);
  auto small = clamp(wrap(board::kTalkToggles ? "Press it again when you're done. You'll hear the answer, and it "
                                                "stays here."
                                              : "You'll hear the answer, and it stays here.", width), 4, width);
  int y = top + max(0, (bottom - top - (int)big.size() * 44 - 14 - (int)small.size() * 34) / 2);
  font(vlw::Title);
  for (auto& l : big) {
    text(l, x0 + width / 2, y, kInk, top_center);
    y += 44;
  }
  y += 14;
  font(vlw::BodyS);
  for (auto& l : small) {
    text(l, x0 + width / 2, y, kBlue, top_center);
    y += 34;
  }
}

// Live readings for deciding when the home screen needs a redraw: the
// temperature to the half degree, humidity to 5 %.
static String climateKey() {
  float t, rh;
  if (!telemetry::climate(&t, &rh)) return "";
  return String((int)lroundf(t * 2)) + "/" + String((int)lroundf(rh / 5));
}

// The day it is: the home screen's date changes at midnight.
static String dayKey() {
  time_t now = time(nullptr);
  if (now < 1700000000) return "";
  struct tm tm;
  localtime_r(&now, &tm);
  return String(tm.tm_yday);
}

// The home screen (and, asleep, the sleeping screen): the date, the room,
// the state, and the answer it keeps — or what to do, until there is one.
// Landscape puts the answer in a second column.
static void drawHomeScreen(bool asleep, uint16_t chipFill, uint16_t chipDot, uint16_t chipInk, const String& hint,
                           bool answerKey) {
  Card scratch;
  const Card* c = homeCard(scratch);
  bool isPin = c == &pinCard;
  drawHeader("", !asleep);
  const bool wide = W > H;
  const int colW = wide ? (W - 2 * PAD) * 56 / 100 : W - 2 * PAD;
  int y = kHeadH + 18;
  y += drawDate(PAD, y, colW);
  int ch = drawClimate(PAD, y, colW);
  if (ch) y += ch + 12;
  drawChip(PAD, y, colW, chipFill, chipDot, chipInk, hint, asleep);
  y += 44 + 24;
  const int cardX = wide ? PAD + colW + 22 : PAD;
  const int cardTop = wide ? kHeadH + 22 : y;
  const int cardW = wide ? W - PAD - cardX : W - 2 * PAD;
  if (c) drawCard(*c, cardX, cardTop, cardW, H - kFootH - 14, isPin);
  else drawWelcome(cardX, cardTop, cardW, H - kFootH - 10);
  if (asleep) {
    char when[24] = "";
    clockText(when, sizeof(when), "drawn %H:%M", time(nullptr));
    drawFooter({{board::kTalkKey, "talk"}}, when);
  } else if (answerKey) {
    drawFooter({{board::kTalkKey, "talk"}, {board::kSideKey, "status"}, {board::kBackKey, "answer"}});
  } else {
    drawFooter({{board::kTalkKey, "talk"}, {board::kSideKey, "status"}});
  }
}

void setWaiting(int) {}  // the home screen's hint says it (a redraw costs ~20 s)

void showHome(const Home& home) {
  Card scratch;
  String key = cardKey(homeCard(scratch));
  String hint = home.hint;
  bool ready = home.ready;
  uint16_t chip = inkOf(home.color, false) == kRed ? kRed : kYellow;
  bool answerKey = hasReply() || savedLast.valid();
  uint32_t answer = answerId(homeCard(scratch));
  // A borrowed picture must stay valid until drawn: the home screen only
  // borrows the current answer's, which lives until the next answer.
  ask(
      Kind::Home, String(ready) + hint + barAgent + key + healthKey() + dayKey(), batteryKey() + climateKey(),
      [=] {
        drawHomeScreen(false, ready ? kInk : chip, ready ? kGreen : kInk, ready || chip == kRed ? kPaper : kInk, hint,
                       answerKey);
      },
      answer);
}

void showSleeping(const String& hint, uint8_t) {
  Card scratch;
  String h = hint;
  const Card* c = homeCard(scratch);
  ask(Kind::Sleep, h + cardKey(c), "", [=] { drawHomeScreen(true, kBlue, kPaper, kPaper, h, false); }, answerId(c));
  flush();
}

// ── the conversation ──────────────────────────────────────────────────

// Listening: the lights and the mic say it. The paper keeps what it shows,
// and anything still waiting to be drawn can wait for the answer.
void showListening(int, uint32_t) { want.active = false; }

// Working: the lights and the sounds say it; the answer is coming and is
// worth the next redraw, not a "working" screen that would hold it up.
void showWorking(const String&, const std::vector<String>&, const String&, uint32_t, int) { want.active = false; }

// How the answer sits on the paper (layoutReply): in landscape with a
// picture, the picture keeps the left column on every page and the words run
// in the right one; otherwise the picture sits under the question on page 1.
static bool replySide = false;
static int replyColX = PAD, replyColW = 0, replyPicW = 0;
static int replyTop() { return kHeadH + 20; }
static int replyBottom() { return H - kFootH - 34; }

// Lays the answer out into pages: page 1 has the question (and, portrait,
// the picture), the next pages words only.
static void layoutReply() {
  replyPages.clear();
  replyPage = 0;
  pictureH = 0;
  if (replyText.isEmpty()) return;
  const int width = W - 2 * PAD, top = replyTop(), bottom = replyBottom();
  int pw = 0, ph = 0;
  const bool pic = imgLen && jpegSize(imgBuf, imgLen, &pw, &ph);
  replySide = pic && W > H;
  replyPicW = replySide ? width * 42 / 100 : 0;
  replyColX = replySide ? PAD + replyPicW + 20 : PAD;
  replyColW = replySide ? width - replyPicW - 20 : width;
  int qH = 0;
  if (question.length()) {
    font(vlw::Small);
    qH = 32 + min(3, (int)wrap(String("“") + question + "”", replyColW).size()) * 27 + 18;
  }
  font(ts().body);
  const int line = ts().line;
  auto lines = wrap(replyText, replyColW);
  const int avail = bottom - top - qH;
  if (pic && !replySide) {
    // As big as it can be while the words still fit on page 1 — never under
    // a third of the room, nor over 55 % of it, nor larger than it came.
    int natural = (int)(ph * min(1.0f, (float)width / pw));
    int h = constrain(avail - (int)lines.size() * line - 16, avail * 33 / 100, avail * 55 / 100);
    pictureH = min(natural, h) + 16;
  }
  const int first = max(1, (avail - pictureH) / line);
  const int per = max(1, (bottom - top) / line);
  replyPages.emplace_back(lines.begin(), lines.begin() + min(lines.size(), (size_t)first));
  for (size_t i = replyPages[0].size(); i < lines.size(); i += per) {
    replyPages.emplace_back(lines.begin() + i, lines.begin() + min(lines.size(), i + per));
  }
}

void setQuestion(const String& text) { pendingQuestion = text; }

void setReply(const String& t) {
  if (t != replyText || replyWhen == 0) {
    bool fresh = t != replyText;
    replyText = t;
    if (fresh) {
      question = pendingQuestion;
      pendingQuestion = "";
      replyWhen = time(nullptr);
      if (imgBuf) free(imgBuf);  // the last answer's picture goes with it
      imgBuf = nullptr;
      imgLen = 0;
      if (t.length() && !restoring) saveLast();
    }
  }
  layoutReply();
}

void setTextSize(uint8_t size) {
  sizeIdx = size < sizeof(kSizes) / sizeof(kSizes[0]) ? size : 0;
  layoutReply();
}

bool hasReply() { return !replyPages.empty() && !replyText.isEmpty(); }

void showReply() {
  if (replyPages.empty()) return;
  String key = String(replyText.length()) + ":" + replyText.substring(0, 40) + ":" + String(replyPage) + ":" +
               String((uint32_t)imgLen) + ":" + String(sizeIdx) + healthKey();
  ask(Kind::Reply, key, batteryKey(), [] {
    if (replyPages.empty()) return;
    // The layout as it is now (a picture or a turn may have changed it).
    const int pages = replyPages.size(), page = min(replyPage, pages - 1);
    drawHeader();
    const int top = replyTop();
    int y = top;
    if (replySide) drawPicture(imgBuf, imgLen, PAD, top, replyPicW, replyBottom() - top, true);
    if (page == 0) {
      if (question.length()) {
        drawLabel("YOU ASKED", replyColX, y);
        y += 32;
        font(vlw::Small);
        auto ql = clamp(wrap(String("“") + question + "”", replyColW), 3, replyColW);
        drawLines(ql, replyColX, y, 27, kBlue);
        y += ql.size() * 27 + 18;
      }
      if (imgLen && !replySide) {
        int ph = drawPicture(imgBuf, imgLen, PAD, y, W - 2 * PAD, pictureH - 16);
        if (ph) y += ph + 16;
      }
    }
    font(ts().body);
    drawLines(replyPages[page], replyColX, y, ts().line, kInk);
    drawPageDots(pages, page, H - kFootH - 18);
    char when[16] = "";
    clockText(when, sizeof(when), "%H:%M", replyWhen);
    if (pages > 1) drawFooter({{board::kTalkKey, "talk"}, {board::kSideKey, "next"}, {board::kBackKey, "home"}}, when);
    else drawFooter({{board::kTalkKey, "talk"}, {board::kBackKey, "home"}}, when);
  }, answerId(replyWhen, imgLen));
}

bool nextPage() {
  if (replyPages.empty()) return false;
  replyPage = (replyPage + 1) % replyPages.size();
  showReply();
  return replyPage != 0;
}

bool prevPage() {
  if (replyPage == 0) return false;
  --replyPage;
  showReply();
  return true;
}

bool onLastPage() { return replyPage + 1 >= (int)replyPages.size(); }
bool onFirstPage() { return replyPage == 0; }
void slide(bool) {}  // the round screen's
void fade() {}
bool setSpoken(int) { return false; }
void followVoice(int) {}
int showStatus(const Status& s, int page) { return showPanel("", 0, statusRows(s), page); }
void growReply(const String& t) { setReply(t); }
void setReplayable(bool) {}
bool replayHit(int, int) { return false; }

bool canPin() { return replyText.length() > 0 || pinCard.valid(); }

bool pinned() { return pinCard.valid(); }

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

void pin(bool on) {
  if (on) {
    currentCard(pinCard);
    if (!pinCard.valid()) return;
  } else {
    pinCard.clear();
  }
  saveCard("pin", pinCard);
}

// ── pictures ───────────────────────────────────────────────────────────

void storeImage(uint8_t* jpg, size_t len) {
  if (imgBuf) free(imgBuf);
  imgBuf = jpg;
  imgLen = jpg ? len : 0;
  layoutReply();     // page 1 makes room for it
  if (replyText.length() && !restoring) saveLast();
}

bool hasImage() { return imgLen > 0; }

void setBackground(const uint8_t*, size_t) {}  // a round screen's (ui_round.cpp)
bool hasBackground() { return false; }
void showBackdrop() {}

void showStoredImage() {
  if (!imgLen) return;
  ask(Kind::Picture, String((uint32_t)imgLen) + ":" + String((uint32_t)replyWhen), "", [=] {
    drawHeader();
    drawPicture(imgBuf, imgLen, PAD, kHeadH + 14, W - 2 * PAD, H - kHeadH - kFootH - 28, true);
    drawFooter({{board::kTalkKey, "text"}, {board::kSideKey, "back"}});
  });
}

// ── status page, menus, characters ────────────────────────────────────

int showPanel(const char* title, uint16_t titleColor, const std::vector<Row>& rows, int page) {
  // Label (small, blue) over the value (body); as many rows as fit a page.
  const int labelH = 26, gap = 12, width = W - 2 * PAD;
  const int top = kHeadH + 18, bottom = H - kFootH - 30;
  struct Laid {
    int page, y;
    std::vector<String> lines;
  };
  std::vector<Laid> laid;
  int pg = 0, y = top + ((title && *title) ? 50 : 0);
  font(ts().body);
  for (const Row& r : rows) {
    auto lines = clamp(wrap(r.value.length() ? r.value : String("-"), width), 3, width);
    int h = labelH + (int)lines.size() * ts().line + gap;
    if (y + h > bottom && y > top) {
      ++pg;
      y = top;
    }
    laid.push_back({pg, y, lines});
    y += h;
  }
  const int pages = pg + 1;
  page = constrain(page, 0, pages - 1);
  String t = title ? title : "";
  uint16_t tink = inkOf(titleColor, true);
  // The values are live (signal, battery): redraw for a new label set or
  // page at once, for new values only now and then.
  String major = t + String(page) + String(sizeIdx), minor;
  for (auto& r : rows) major += r.label + "|", minor += r.value + "|";
  std::vector<Row> rs = rows;
  ask(Kind::Panel, major, minor, [=] {
    drawHeader();
    if (page == 0 && t.length()) {
      font(vlw::Title);
      text(t, PAD, top, tink);
    }
    for (size_t i = 0; i < laid.size(); ++i) {
      if (laid[i].page != page) continue;
      font(vlw::Small);
      text(rs[i].label, PAD, laid[i].y, kBlue);
      font(ts().body);
      drawLines(laid[i].lines, PAD, laid[i].y + labelH, ts().line, inkOf(rs[i].color, true));
    }
    drawPageDots(pages, page, H - kFootH - 18);
    drawFooter({{board::kTalkKey, "menu"}, {board::kSideKey, page + 1 < pages ? "next" : "close"}, {board::kBackKey, "back"}});
  });
  return pages;
}

// Menus and lists are read aloud row by row (main.cpp sayRow), never drawn:
// a redraw per step would take 20 s each. The paper keeps what it shows.
void showPicker(const std::vector<Item>&, int, int, const char*) {
  forceNext = false;
  want.active = false;
}

void showSpeaking(int) {}  // the lights carry it (leds.cpp)
void showFace(uint8_t, face::Pose, int) {}  // icon mode is off on paper (a character can't move)

void showCharPick(uint8_t kind, face::Pose, int, int index) {
  ask(Kind::CharPick, String(kind), "", [=] {
    drawHeader("Character");
    float scale = min(1.5f, (W - 2 * PAD - 16) / 240.0f);
    int cy = kHeadH + 30 + (int)(135 * scale / 2) + 20;
    drawCharacter(kind, face::Pose::Idle, W / 2, cy, scale);
    font(vlw::Title);
    text(face::name(kind), W / 2, cy + (int)(135 * scale / 2) + 36, kInk, top_center);
    font(vlw::Small);
    text("shown on the home screen", W / 2, cy + (int)(135 * scale / 2) + 84, kBlue, top_center);
    drawPageDots(face::KindCount, index, H - kFootH - 26);
    drawFooter({{board::kTalkKey, "pick"}, {board::kSideKey, "next"}, {board::kBackKey, "back"}});
  });
}

void flashFrame(bool) {}  // the status lights flash instead

void setTapButton(TapButton) {}  // no touch screen
bool tapButtonHit(int, int) { return false; }
int askHit(int, int) { return -1; }
void toggleImageFit() {}

}  // namespace ui
