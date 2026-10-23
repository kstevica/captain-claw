// StopWatch screen: 466×466 round AMOLED.
//
// The look follows M5Stack's own StopWatch firmware: black (an AMOLED pixel
// that is black is off), light type in Inter, a glowing scalloped badge in
// the middle with a name under it, page dots and chevrons for lists. A ring
// round the edge shows what the watch is doing: red while it listens (and how
// much of the 110 s recording is used), amber while the agent works (one
// tick per tool step), cyan while it speaks. The clock and the Wi-Fi/agent
// dots sit at the top.
//
// Text is laid out to the circle: every line gets the width of the screen at
// its height, and lines are centred.
#include <M5Unified.h>
#include <math.h>
#include <time.h>
#if defined(ESP_PLATFORM)
#include <lgfx/v1/panel/Panel_AMOLED.hpp>
#endif

#include "audio.h"
#include "board.h"
#include "fonts.h"
#include "ui.h"

namespace ui {

static constexpr int W = 466, H = 466, CX = 233, CY = 233, R = 233;
static constexpr int kMargin = 24;         // text keeps this far from the edge
static constexpr int kRing = 10;           // the status ring's width
static constexpr int kTop = 78;            // content starts under the clock
static constexpr int kBottom = 410;        // …and ends above the page dots
static constexpr int kChromeY = 42;        // clock / health dots
static constexpr int kBadgeY = 196, kBadgeR = 80, kGlowR = 128;
static constexpr int kTitleY = 322, kNoteY = 360, kDotsY = 402, kBatteryY = 438;
// Tap to talk's button: under the name (home, listening), or lower and a
// little smaller under an answer or a character, whose text then ends higher.
static constexpr int kTapY = 386, kTapR = 30, kTapLowY = 414, kTapLowR = 26, kTapReplyBottom = 352;
static constexpr int kAskY = 396;          // the yes/no pills' middle

static constexpr uint16_t kBlack = 0x0000, kWhite = 0xFFFF;
static constexpr uint16_t kText = rgb(236, 238, 244), kDim = rgb(140, 146, 160), kFaint = rgb(78, 82, 94);
static constexpr uint16_t kGreen = rgb(40, 205, 100), kRed = rgb(245, 60, 75), kAmber = rgb(255, 170, 40);
static constexpr uint16_t kCyan = rgb(60, 210, 245), kBlue = rgb(90, 150, 255);

static M5Canvas canvas(&M5.Display);
static M5Canvas faceSprite;  // characters draw at the stick's size, then scale up
static M5Canvas prevFrame;   // the screen before a slide (PSRAM)
enum class Fx : uint8_t { None, SlideOn, SlideBack, Fade };
static Fx nextFx = Fx::None;        // how the next push() brings its screen in
static uint8_t curBright = 0;       // what brightness() last set (dark or dim: no fades)
static constexpr int kSlideFrames = 4;

static lgfx::VLWfont fontSet[vlw::Count];
static lgfx::PointerWrapper fontData[vlw::Count];

// Reading text in the size the user picked.
struct TextSize {
  vlw::Id body;
  int line;
};
static const TextSize kSizes[] = {{vlw::BodyS, 35}, {vlw::BodyM, 41}, {vlw::BodyL, 49}};
static uint8_t sizeIdx = 0;
static const TextSize& ts() { return kSizes[sizeIdx]; }

static String barAgent, barState;
static uint16_t barColor = kDim;
static Health hWifi = Down, hAgent = Down;

static String replyText;
static std::vector<std::vector<String>> replyPages;
static std::vector<std::vector<std::pair<int, int>>> lineSpans;  // each line's [from, to) in replyText
static std::vector<int> pageEnds;                                  // where each page's text ends
static int replyPage = 0;
static int spokenAt = -1, litFrom = -1, litTo = -1;  // the voice's place, and the sentence round it
static bool autoFollow = true;                       // the page turns with the voice (off once paged by hand)
static bool replayable = false;                      // the small replay button under the answer
static int rpX = 0, rpY = 0, rpR = 0;                // where it is (rpR 0: none)
static bool replyCentred = false;  // one short page: the block sits mid-screen
static uint8_t* imgBuf = nullptr;
static size_t imgLen = 0;
static bool imgFit = false;        // the whole picture scaled down (a tap), else it fills the circle
static bool askShown = false;      // the yes/no pills are on the screen (askHit)
static M5Canvas backdrop;          // the background picture as it is (setBackground), PSRAM
static M5Canvas backdropDim;       // the same darkened: what the screens draw on
static bool hasBackdrop = false;
static constexpr int kBackdropDim = 108;  // of 256 (0.42): white text stays readable on it
// The characters' own backdrop while there's a picture: see-through (their
// eyes and mouths are real black).
static constexpr uint16_t kSeeThrough = 0x0821;
static TapButton tapMode = TapButton::Off;
static int tapX = 0, tapY = 0, tapR = 0;  // the button on the screen now (tapR 0: none)

// ── basics ─────────────────────────────────────────────────────────────

static void font(vlw::Id id) { canvas.setFont(&fontSet[id]); }

static uint16_t mix(uint16_t a, uint16_t b, float t) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  int ar = (a >> 11) & 31, ag = (a >> 5) & 63, ab = a & 31;
  int br = (b >> 11) & 31, bg = (b >> 5) & 63, bb = b & 31;
  return (uint16_t)(((int)(ar + (br - ar) * t) << 11) | ((int)(ag + (bg - ag) * t) << 5) | (int)(ab + (bb - ab) * t));
}

// Half the width of the screen at height y.
static int halfChord(int y) {
  int dy = abs(y - CY);
  return dy >= R ? 0 : (int)sqrtf((float)(R * R - dy * dy));
}

// Room for a line spanning [top, bottom): the narrower end of it, less the margin.
static int lineRoom(int top, int bottom) {
  return max(0, 2 * min(halfChord(top), halfChord(bottom)) - 2 * kMargin);
}

#ifdef CLAW_PERF
uint32_t g_perfDrawUs = 0, g_perfDrawMax = 0, g_perfPushUs = 0, g_perfPushMax = 0, g_perfFrames = 0;
static uint32_t frameT0 = 0;
#endif

// The screen as it is, kept for a slide.
static bool keepFrame() {
  if (!prevFrame.width()) {
    prevFrame.setPsram(true);
    prevFrame.setColorDepth(16);
    if (!prevFrame.createSprite(W, H)) return false;
  }
  memcpy(prevFrame.getBuffer(), canvas.getBuffer(), (size_t)W * H * 2);
  return true;
}

// Every screen starts here: the background picture, else black (*plain*:
// black all the same — a picture of the agent's own is shown on it).
static void beginFrame(bool plain = false) {
#ifdef CLAW_PERF
  frameT0 = micros();
#endif
  if ((nextFx == Fx::SlideOn || nextFx == Fx::SlideBack) && !keepFrame()) nextFx = Fx::None;
  if (hasBackdrop && !plain) backdropDim.pushSprite(&canvas, 0, 0);
  else canvas.fillScreen(kBlack);
  tapR = 0;
  rpR = 0;
  askShown = false;
}

static void pushNow() {
#ifdef CLAW_PERF
  uint32_t t = micros();
  uint32_t draw = t - frameT0;
  canvas.pushSprite(0, 0);
  uint32_t p = micros() - t;
  g_perfDrawUs += draw;
  g_perfPushUs += p;
  g_perfDrawMax = max<uint32_t>(g_perfDrawMax, draw);
  g_perfPushMax = max<uint32_t>(g_perfPushMax, p);
  ++g_perfFrames;
#else
  canvas.pushSprite(0, 0);
#endif
}

// The old screen leaves and the new one follows it in, eased, in a few
// frames — each a screen's worth of pixels (~50 ms on the AMOLED).
static void slideIn(bool forward) {
  const int dir = forward ? -1 : 1;
  for (int k = 1; k < kSlideFrames; ++k) {
    float t = k / (float)kSlideFrames;
    t = t * t * (3 - 2 * t);
    const int off = ((int)(W * t)) & ~1;  // even: the panel's rule
    prevFrame.pushSprite(&M5.Display, dir * off, 0);
    canvas.pushSprite(dir * (off - W), 0);
  }
  pushNow();
}

// Through black: the panel's brightness steps down, the new screen goes up
// unlit and the brightness comes back — read as a fade, at no drawing cost.
static void fadeIn() {
  const uint8_t b = curBright;
  for (int k = 2; k >= 1; --k) {
    M5.Display.setBrightness(b * k / 3);
    delay(28);
  }
  M5.Display.setBrightness(0);
  pushNow();
  for (int k = 1; k <= 3; ++k) {
    delay(28);
    M5.Display.setBrightness(b * k / 3);
  }
}

static void push() {
  const Fx fx = nextFx;
  nextFx = Fx::None;
  if (fx == Fx::SlideOn || fx == Fx::SlideBack) slideIn(fx == Fx::SlideOn);
  else if (fx == Fx::Fade && curBright > 24) fadeIn();  // (not from a dark or dimmed screen)
  else pushNow();
}

void slide(bool forward) { nextFx = forward ? Fx::SlideOn : Fx::SlideBack; }
void fade() { nextFx = Fx::Fade; }

static int tw(const String& s) { return canvas.textWidth(s.c_str()); }

static void text(const String& s, int x, int y, uint16_t color, uint8_t datum = middle_center) {
  canvas.setTextDatum(datum);
  canvas.setTextColor(color);
  canvas.drawString(s.c_str(), x, y);
}

static int utf8Step(uint8_t c) { return c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1; }

// Shorten to `width` px with "…" (in the current font), whole UTF-8 characters.
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

// Word-wrap UTF-8 text in the current font; line k gets width roomFor(k).
// Words wider than a line break between characters.
template <typename RoomFn>
static std::vector<String> wrap(const String& text, RoomFn roomFor) {
  std::vector<String> out;
  int start = 0;
  const int n = text.length();
  while (start <= n) {
    int nl = text.indexOf('\n', start);
    String para = text.substring(start, nl < 0 ? n : nl);
    start = (nl < 0 ? n : nl) + 1;
    String line;
    int i = 0;
    while (i < (int)para.length()) {
      int sp = para.indexOf(' ', i);
      String word = para.substring(i, sp < 0 ? para.length() : sp);
      i = sp < 0 ? para.length() : sp + 1;
      int room = roomFor((int)out.size());
      String trial = line.length() ? line + " " + word : word;
      if (tw(trial) <= room) {
        line = trial;
        continue;
      }
      if (line.length()) out.push_back(line);
      line = "";
      room = roomFor((int)out.size());
      while (word.length() && tw(word) > room) {
        int cut = 0, k = 0;
        while (k < (int)word.length()) {
          int step = utf8Step(word[k]);
          if (tw(word.substring(0, k + step)) > room) break;
          k += step;
          cut = k;
        }
        if (cut == 0) cut = utf8Step(word[0]);
        out.push_back(word.substring(0, cut));
        word = word.substring(cut);
        room = roomFor((int)out.size());
      }
      line = word;
    }
    out.push_back(line);
    if (nl < 0) break;
  }
  while (!out.empty() && out.back().isEmpty()) out.pop_back();
  return out;
}

// Two lines read better when they're about even ("Hold the yellow / button
// to talk", not "Hold the yellow button to / talk").
static void balanceTwo(const String& s, std::vector<String>& lines, int room0, int room1) {
  if (lines.size() != 2 || s.indexOf('\n') >= 0) return;
  int best = -1, bestW = 1 << 30;
  for (int i = s.indexOf(' '); i >= 0; i = s.indexOf(' ', i + 1)) {
    int wa = tw(s.substring(0, i)), wb = tw(s.substring(i + 1));
    if (wa > room0 || wb > room1) continue;
    if (max(wa, wb) < bestW) {
      bestW = max(wa, wb);
      best = i;
    }
  }
  if (best >= 0) lines = {s.substring(0, best), s.substring(best + 1)};
}

// Lines that fit [top, bottom) at `lineH`, centred as a block in that band.
// Returns the lines and the y of the first line's top.
static std::vector<String> wrapCentred(const String& s, int top, int bottom, int lineH, int maxLines, int* firstTop) {
  const int band = bottom - top;
  const int cap = max(1, min(maxLines, band / lineH));
  int n = cap;
  std::vector<String> lines;
  for (int pass = 0; pass < 4; ++pass) {  // fewer lines sit nearer the middle, where lines are wider
    int y0 = top + (band - n * lineH) / 2;
    lines = wrap(s, [&](int k) { return lineRoom(y0 + k * lineH, y0 + (k + 1) * lineH); });
    int got = max(1, min((int)lines.size(), cap));
    if (got == n) break;
    n = got;
  }
  if ((int)lines.size() > cap) {  // too long: keep what fits, end with "…"
    lines.resize(cap);
    int y = top + (band - cap * lineH) / 2 + (cap - 1) * lineH;
    lines.back() = fit(lines.back() + "…", lineRoom(y, y + lineH));
  } else if (lines.size() == 2) {
    int y0 = top + (band - 2 * lineH) / 2;
    balanceTwo(s, lines, lineRoom(y0, y0 + lineH), lineRoom(y0 + lineH, y0 + 2 * lineH));
  }
  *firstTop = top + (band - (int)lines.size() * lineH) / 2;
  return lines;
}

static void drawLines(const std::vector<String>& lines, int firstTop, int lineH, uint16_t color) {
  for (int i = 0; i < (int)lines.size(); ++i) text(lines[i], CX, firstTop + i * lineH + lineH / 2, color);
}

// ── chrome: clock, health dots, battery, page dots, chevrons ─────────

static uint16_t healthColor(Health h) { return h == Up ? kGreen : h == Partial ? kAmber : kRed; }

// Battery: green ≥75 %, blue 40–74 %, amber 20–39 %, red <20 %.
static uint16_t batteryColor(int pct) {
  return pct >= 75 ? kGreen : pct >= 40 ? kBlue : pct >= 20 ? kAmber : kRed;
}

// The PMIC is on I2C: read it at most every few seconds, not every frame.
static int batteryPct(bool* charging) {
  static int pct = -1;
  static bool chg = false;
  static uint32_t at = 0;
  if (pct < 0 || millis() - at > 5000) {
    at = millis();
    pct = M5.Power.getBatteryLevel();
    chg = M5.Power.isCharging() == m5::Power_Class::is_charging;
  }
  *charging = chg;
  return pct;
}

static bool clockText(char* out, size_t n) {
  time_t now = time(nullptr);
  if (now < 1700000000) return false;  // not synced yet
  struct tm tm;
  localtime_r(&now, &tm);
  strftime(out, n, "%H:%M", &tm);
  return true;
}

// "Fri 23 Oct"
static String dateText() {
  time_t now = time(nullptr);
  struct tm tm;
  localtime_r(&now, &tm);
  char wday[8], mon[8];
  strftime(wday, sizeof(wday), "%a", &tm);
  strftime(mon, sizeof(mon), "%b", &tm);
  return String(wday) + " " + String(tm.tm_mday) + " " + mon;
}

static void drawBatteryAt(int y, bool force) {
  bool chg = false;
  int pct = batteryPct(&chg);
  if (pct < 0 || (!force && pct >= 20)) return;
  font(vlw::Small);
  String t = String(pct) + "%";
  int textW = tw(t);
  int bw = 26, total = bw + 6 + textW;
  int x = CX - total / 2;
  uint16_t col = batteryColor(pct);
  canvas.drawRoundRect(x, y - 7, bw - 3, 14, 3, kDim);
  canvas.fillRect(x + bw - 3, y - 3, 3, 6, kDim);
  canvas.fillRect(x + 2, y - 5, max(1, pct * (bw - 7) / 100), 10, col);
  if (chg) {
    canvas.fillTriangle(x + 13, y - 7, x + 7, y + 1, x + 12, y + 1, kWhite);
    canvas.fillTriangle(x + 11, y - 1, x + 16, y - 1, x + 10, y + 7, kWhite);
  }
  text(t, x + bw + 6, y, col, middle_left);
}

// Clock with the Wi-Fi dot left of it and the agent dot right of it; a low
// battery shows its percentage underneath on every screen.
static void drawChrome(bool lowBattery = true, bool withClock = true) {
  char clock[8];
  font(vlw::Small);
  int half = 10;
  if (withClock && clockText(clock, sizeof(clock))) {
    text(clock, CX, kChromeY, kText);
    half = tw(clock) / 2 + 14;
  }
  canvas.fillSmoothCircle(CX - half, kChromeY, 5, healthColor(hWifi));
  canvas.fillSmoothCircle(CX + half, kChromeY, 5, healthColor(hAgent));
  if (lowBattery) drawBatteryAt(kChromeY + 24, false);
}

static void drawPageDots(int count, int current, int y) {
  if (count < 2) return;
  if (count > 20) {  // too many to show as dots
    font(vlw::Small);
    text(String(current + 1) + " / " + String(count), CX, y, kDim);
    return;
  }
  const int gap = count > 14 ? 13 : 16;
  int x0 = CX - (count - 1) * gap / 2;
  for (int i = 0; i < count; ++i) {
    if (i == current) canvas.fillSmoothCircle(x0 + i * gap, y, 5, kWhite);
    else canvas.fillSmoothCircle(x0 + i * gap, y, 3, rgb(96, 100, 112));
  }
}

static void drawChevrons(int y) {
  const uint16_t col = rgb(225, 226, 232);
  const int arm = 15;
  int lx = 30, rx = W - 30;
  canvas.drawWideLine(lx + arm, y - arm, lx, y, 1.6f, col);
  canvas.drawWideLine(lx, y, lx + arm, y + arm, 1.6f, col);
  canvas.drawWideLine(rx - arm, y - arm, rx, y, 1.6f, col);
  canvas.drawWideLine(rx, y, rx - arm, y + arm, 1.6f, col);
}

// ── the ring ───────────────────────────────────────────────────────────

// An arc of the ring, `from` degrees clockwise from 12 o'clock, `sweep` long.
static void ringArc(float from, float sweep, uint16_t color, int width = kRing, int outer = R) {
  if (sweep <= 0) return;
  if (sweep >= 360) {
    canvas.fillArc(CX, CY, outer - width, outer, 0, 360, color);
    return;
  }
  float a = fmodf(from + 270.0f, 360.0f);  // M5GFX: 0° = 3 o'clock
  float b = a + sweep;
  if (b <= 360) {
    canvas.fillArc(CX, CY, outer - width, outer, a, b, color);
  } else {
    canvas.fillArc(CX, CY, outer - width, outer, a, 360, color);
    canvas.fillArc(CX, CY, outer - width, outer, 0, b - 360, color);
  }
}

static void ringPoint(float deg, int radius, int* x, int* y) {
  float rad = (deg - 90.0f) * 0.01745329f;
  *x = CX + (int)lroundf(cosf(rad) * radius);
  *y = CY + (int)lroundf(sinf(rad) * radius);
}

static void ringTrack(uint16_t color) { ringArc(0, 360, mix(kBlack, color, 0.2f)); }

// Listening: the recording so far against the 110 s limit; the inner edge
// glows with the mic level.
static void ringListening(uint32_t elapsedMs, int level) {
  ringTrack(kRed);
  float used = min(1.0f, elapsedMs / 110000.0f);
  ringArc(0, max(4.0f, 360 * used), kRed);
  int x, y;
  ringPoint(max(4.0f, 360 * used), R - kRing / 2, &x, &y);
  canvas.fillSmoothCircle(x, y, kRing / 2 + 2, kWhite);
  if (level > 4) ringArc(0, 360, mix(kBlack, kRed, level / 160.0f), 6, R - kRing - 2);
}

// Working: an amber comet circling; tool steps as ticks inside the ring.
static void ringWorking(int steps) {
  // The track breathes while the agent works: one breath in 2.4 s.
  const float breath = 0.5f + 0.5f * sinf(millis() * (6.2832f / 2400));
  ringArc(0, 360, mix(kBlack, kAmber, 0.12f + 0.22f * breath));
  float head = fmodf(millis() * 0.225f, 360.0f);  // one turn in 1.6 s
  const int segs = 9;
  for (int i = 0; i < segs; ++i) {
    ringArc(head - (i + 1) * 9.0f, 9.5f, mix(kBlack, kAmber, 1.0f - i / (float)segs));
  }
  int x, y;
  ringPoint(head, R - kRing / 2, &x, &y);
  canvas.fillSmoothCircle(x, y, kRing / 2 + 1, rgb(255, 225, 150));
  const int n = min(steps, 30);
  for (int i = 0; i < n; ++i) {  // tool steps: ticks spreading from 6 o'clock
    ringPoint(180.0f + (i - (n - 1) / 2.0f) * 8.0f, R - kRing - 9, &x, &y);
    canvas.fillSmoothCircle(x, y, 3, kAmber);
  }
}

// Speaking: a short cyan line under the text, as long as the voice is loud.
// Only its strip is redrawn (an even x and width: the AMOLED's one rule),
// a few hundred bytes a frame instead of the whole screen.
static constexpr int kSpeakW = 160, kSpeakH = 8, kSpeakX = 152, kSpeakY = 446;

void showSpeaking(int level) {
  static M5Canvas strip;
  if (!strip.width()) {
    strip.setColorDepth(16);
    if (!strip.createSprite(kSpeakW, kSpeakH)) return;
  }
  if (hasBackdrop) backdropDim.pushSprite(&strip, -kSpeakX, -kSpeakY);  // its piece of the picture
  else strip.fillSprite(kBlack);
  int len = 14 + min(level, 100) * (kSpeakW - 14) / 100;
  strip.fillSmoothRoundRect((kSpeakW - len) / 2, 2, len, 4, 2, kCyan);
  strip.pushSprite(&M5.Display, kSpeakX, kSpeakY);
}

// Tap to talk: a record button (a red dot in a ring), or a stop button (a
// red square) while a recording that was tapped on listens. An answer's
// sound: pause (two bars) while it plays, play (a triangle) while it's held.
static void drawTapButton(int cy, int r) {
  canvas.fillSmoothCircle(CX, cy, r, mix(kBlack, kText, 0.85f));
  canvas.fillSmoothCircle(CX, cy, r - 4, kBlack);
  if (tapMode == TapButton::Stop) {
    const int side = r * 7 / 10;
    canvas.fillSmoothRoundRect(CX - side / 2, cy - side / 2, side, side, 5, kRed);
  } else if (tapMode == TapButton::Pause) {
    const int h = r * 7 / 10, w = r / 5, gap = r / 5;
    canvas.fillSmoothRoundRect(CX - gap / 2 - w, cy - h / 2, w, h, 2, kWhite);
    canvas.fillSmoothRoundRect(CX + gap / 2 + 1, cy - h / 2, w, h, 2, kWhite);
  } else if (tapMode == TapButton::Play) {
    const int h = r * 7 / 10, w = r * 6 / 10, x = CX - w / 2 + 2;  // a touch right: it looks centred
    canvas.fillTriangle(x, cy - h / 2, x, cy + h / 2, x + w, cy, kWhite);
  } else {
    canvas.fillSmoothCircle(CX, cy, r - 9, kRed);
  }
  tapX = CX, tapY = cy, tapR = r;
}

// The buttons that sit under an answer (or the character showing it).
static bool answerButton() {
  return tapMode == TapButton::Talk || tapMode == TapButton::Pause || tapMode == TapButton::Play;
}

void setTapButton(TapButton button) { tapMode = button; }

bool tapButtonHit(int x, int y) {
  const int dx = x - tapX, dy = y - tapY, reach = tapR + 30;  // a fingertip's slack round it
  return tapR && dx * dx + dy * dy <= reach * reach;
}

// ── the glowing badge ──────────────────────────────────────────────────

enum Tone : uint8_t { TBlue, TGreen, TAmber, TRed, TGrey, TCyan, TToneCount };
struct ToneDef {
  uint8_t a[3], b[3], glow[3];
};
static const ToneDef kTones[TToneCount] = {
    {{40, 72, 235}, {58, 170, 250}, {50, 120, 255}},    // blue (M5's)
    {{20, 140, 90}, {50, 205, 140}, {30, 190, 110}},    // green
    {{215, 110, 20}, {250, 185, 60}, {255, 150, 30}},   // amber
    {{200, 30, 60}, {250, 95, 95}, {255, 50, 70}},      // red
    {{58, 62, 74}, {98, 104, 120}, {50, 54, 66}},       // grey (off)
    {{20, 120, 190}, {60, 210, 235}, {40, 190, 240}},   // cyan
};
static M5Canvas* badgeCache[TToneCount] = {};

// A scalloped badge (12 soft lobes) with a diagonal gradient, a highlight
// at the top left and a glow fading into the black — drawn once per tone.
static M5Canvas* badgeSprite(Tone tone) {
  if (badgeCache[tone]) return badgeCache[tone];
  auto* s = new M5Canvas();
  s->setPsram(true);
  s->setColorDepth(16);
  const int S = 2 * kGlowR;
  if (!s->createSprite(S, S)) {
    delete s;
    return nullptr;
  }
  const ToneDef& t = kTones[tone];
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      float dx = x - S / 2 + 0.5f, dy = y - S / 2 + 0.5f;
      float d = sqrtf(dx * dx + dy * dy);
      float th = atan2f(dy, dx);
      float edge = kBadgeR + 4.5f * cosf(12 * th);
      float out[3] = {0, 0, 0};
      if (d > edge - 1) {
        float fade = (kGlowR - d) / (kGlowR - kBadgeR - 8.0f);  // to nothing at the sprite's edge
        fade = fade < 0 ? 0 : fade > 1 ? 1 : fade * fade * (3 - 2 * fade);
        float g = (0.62f * expf(-(d - edge) / 14.0f) + 0.16f * expf(-(d - edge) / 42.0f)) * fade;
        for (int k = 0; k < 3; ++k) out[k] = t.glow[k] * g;
      }
      float cov = edge - d + 0.5f;
      cov = cov < 0 ? 0 : cov > 1 ? 1 : cov;
      float rgbv[3];
      if (cov > 0) {
        float g = (dx + dy) / (2.8f * kBadgeR) + 0.5f;
        g = g < 0 ? 0 : g > 1 ? 1 : g;
        float hx = dx + kBadgeR * 0.35f, hy = dy + kBadgeR * 0.45f;
        float hl = 1.0f - sqrtf(hx * hx + hy * hy) / (kBadgeR * 1.15f);
        hl = hl < 0 ? 0 : hl * 0.32f;
        for (int k = 0; k < 3; ++k) {
          float in = t.a[k] + (t.b[k] - t.a[k]) * g;
          in += (255 - in) * hl;
          rgbv[k] = out[k] + (in - out[k]) * cov;
        }
      } else {
        for (int k = 0; k < 3; ++k) rgbv[k] = out[k];
      }
      s->drawPixel(x, y, rgb((uint8_t)min(255.0f, rgbv[0]), (uint8_t)min(255.0f, rgbv[1]),
                             (uint8_t)min(255.0f, rgbv[2])));
    }
  }
  badgeCache[tone] = s;
  return s;
}

static void drawGlyph(Glyph g, int cx, int cy) {
  const uint16_t c = kWhite;
  switch (g) {
    case Glyph::Wifi:
      for (int i = 0; i < 3; ++i) canvas.fillArc(cx, cy + 22, 16 + i * 14, 23 + i * 14, 225, 315, c);
      canvas.fillSmoothCircle(cx, cy + 20, 6, c);
      break;
    case Glyph::Back:
      canvas.drawWideLine(cx + 10, cy - 22, cx - 12, cy, 4.0f, c);
      canvas.drawWideLine(cx - 12, cy, cx + 10, cy + 22, 4.0f, c);
      break;
    case Glyph::Info:
      canvas.fillSmoothCircle(cx, cy - 22, 7, c);
      canvas.fillSmoothRoundRect(cx - 6, cy - 8, 12, 38, 5, c);
      break;
    case Glyph::Replay:  // a circle arrow round a play triangle
      canvas.fillArc(cx, cy, 25, 31, 270, 360, c);
      canvas.fillArc(cx, cy, 25, 31, 0, 200, c);
      canvas.fillTriangle(cx - 13, cy - 28, cx + 3, cy - 40, cx + 3, cy - 16, c);
      canvas.fillTriangle(cx - 7, cy - 12, cx - 7, cy + 12, cx + 13, cy, c);
      break;
    case Glyph::Gear:  // a ring with eight teeth
      for (int i = 0; i < 8; ++i) {
        const float a = i * 45 * 3.14159f / 180;
        canvas.drawWideLine(cx + 20 * cosf(a), cy + 20 * sinf(a), cx + 31 * cosf(a), cy + 31 * sinf(a), 9.0f, c);
      }
      canvas.fillArc(cx, cy, 12, 24, 0, 360, c);
      break;
    default:
      break;
  }
}

// Capitals for the badge (its font has no small accented letters): ASCII,
// Latin-1 and Latin Extended-A.
static String upper(const String& s) {
  String out;
  for (int i = 0; i < (int)s.length();) {
    uint8_t c = s[i];
    int step = utf8Step(c);
    uint32_t cp = c;
    if (step == 2 && i + 1 < (int)s.length()) cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
    if (step == 1) {
      out += (char)toupper(c);
    } else if (step == 2) {
      if (cp >= 0xE0 && cp <= 0xFE && cp != 0xF7) cp -= 0x20;
      else if (cp >= 0x100 && cp <= 0x17F) {
        bool flipped = (cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17E);
        if (flipped ? (cp % 2 == 0) : (cp % 2 == 1)) cp -= 1;
      }
      out += (char)(0xC0 | (cp >> 6));
      out += (char)(0x80 | (cp & 0x3F));
    } else {
      out += s.substring(i, i + step);
    }
    i += step;
  }
  return out;
}

// "Captain Claw" → "CC", "Jarvis" → "J".
static String initials(const String& name) {
  String out;
  bool atWord = true;
  int words = 0;
  for (int i = 0; i < (int)name.length() && words < 2;) {
    int step = utf8Step(name[i]);
    char c = name[i];
    bool sep = c == ' ' || c == '-' || c == '_' || c == '.';
    if (sep) {
      atWord = true;
    } else if (atWord) {
      out += name.substring(i, i + step);
      ++words;
      atWord = false;
    }
    i += step;
  }
  return upper(out);
}

// The badge with a short value, initials or a glyph in it. Text that's too
// wide for the badge font drops to the title font, then the small one.
static void drawBadge(Tone tone, const String& name, Glyph glyph, int cy = kBadgeY) {
  M5Canvas* s = badgeSprite(tone);
  if (s) s->pushSprite(&canvas, CX - kGlowR, cy - kGlowR, kBlack);  // its glow's black: see-through
  else canvas.fillSmoothCircle(CX, cy, kBadgeR, rgb(kTones[tone].a[0], kTones[tone].a[1], kTones[tone].a[2]));
  if (glyph != Glyph::None && glyph != Glyph::Initials) {
    drawGlyph(glyph, CX, cy);
    return;
  }
  String label = glyph == Glyph::Initials ? initials(name) : name;
  if (!label.length()) return;
  const int room = 2 * kBadgeR - 26;
  bool ascii = true;
  for (int i = 0; i < (int)label.length(); ++i) ascii &= (uint8_t)label[i] < 0x80;
  const bool badgeFont = ascii || upper(label) == label;  // it has no small accented letters
  const vlw::Id ids[3] = {vlw::Badge, vlw::Title, vlw::Small};
  for (int k = 0; k < 3; ++k) {
    if (k == 0 && !badgeFont) continue;
    font(ids[k]);
    if (tw(label) <= room || k == 2) {
      String t = k == 2 ? fit(label, room) : label;
      uint16_t shadow = rgb(kTones[tone].a[0] / 2, kTones[tone].a[1] / 2, kTones[tone].a[2] / 2);
      text(t, CX + 1, cy + 3, shadow);
      text(t, CX, cy + 1, kWhite);
      return;
    }
  }
}

// ── public ─────────────────────────────────────────────────────────────

#if defined(ESP_PLATFORM)
// M5GFX puts the StopWatch's AMOLED behind a framebuffer panel of its own:
// a second copy of every frame on its way out, and a setSleep() that does
// nothing, so before deep sleep the panel was never told to sleep and showed
// white. Our canvas already is a whole-screen framebuffer, pushed from (0,0)
// with an even width (the panel's one restriction): M5.Display draws straight
// to the AMOLED. The framebuffer panel itself stays — M5Unified's touch reads
// through its own copy of the display, which still points at it.
struct AmoledFb : lgfx::Panel_AMOLED_Framebuffer {
  lgfx::Panel_AMOLED* amoled() { return _panel; }
};

static void drawDirect() {
  if (M5.getBoard() != m5::board_t::board_M5StopWatch || !ESP.getPsramSize()) return;  // no framebuffer then
  auto* fb = static_cast<lgfx::Panel_AMOLED_Framebuffer*>(M5.Display.getPanel());
  lgfx::Panel_AMOLED* panel = static_cast<AmoledFb*>(fb)->amoled();
  if (!panel || panel->getPanelFb() != fb) return;  // not the framebuffer we know
  panel->setAutoDisplay(false);  // never send the (unused) framebuffer over our frames
  M5.Display.setPanel(panel);
}
#endif

void begin() {
#if defined(ESP_PLATFORM)
  drawDirect();
#endif
  M5.Display.setRotation(0);
  canvas.setPsram(true);
  canvas.setColorDepth(16);
  canvas.createSprite(W, H);
  canvas.setTextWrap(false);
  faceSprite.setPsram(true);
  faceSprite.setColorDepth(16);
  faceSprite.createSprite(240, 135);  // the stick's screen: characters are drawn for it
  for (int i = 0; i < vlw::Count; ++i) {
    size_t len = 0;
    const uint8_t* d = vlw::data((vlw::Id)i, &len);
    fontData[i].set(d, len);
    fontSet[i].loadFont(&fontData[i]);
  }
  badgeSprite(TBlue);  // the common one, ready before the first screen
}

void brightness(uint8_t level) {
  curBright = level;
  M5.Display.setBrightness(level);
}

bool animated() { return true; }

void tick() {}
void flush() {}
bool busy() { return false; }
void idleSince(uint32_t) {}
void drawNow() {}
bool waiting() { return false; }
void setRotation(uint8_t) {}

void setBar(const String& agent, const char* state, uint16_t stateColor) {
  barAgent = agent;
  barState = state ? state : "";
  barColor = stateColor;
}

void setHealth(Health wifi, Health agent) {
  hWifi = wifi;
  hAgent = agent;
}

void showSetup(const char* reason, const char* apName, const char* ip) {
  beginFrame();
  font(vlw::Title);
  text(reason, CX, 70, kAmber);
  const int q = 200;
  canvas.fillRoundRect(CX - q / 2 - 10, 100, q + 20, q + 20, 14, kWhite);
  String qr = String("WIFI:T:nopass;S:") + apName + ";;";
  canvas.qrcode(qr.c_str(), CX - q / 2, 110, q, 3);
  font(vlw::Small);
  text("Join this Wi-Fi", CX, 344, kDim);
  text(apName, CX, 372, kCyan);
  text(String("then open http://") + ip, CX, 404, kDim);
  push();
}

void showLink(const char* title, const String& url, const String& note) {
  beginFrame();
  font(vlw::Title);
  text(fit(title, lineRoom(52, 92)), CX, 70, kCyan);
  const int q = 190;
  canvas.fillRoundRect(CX - q / 2 - 10, 100, q + 20, q + 20, 14, kWhite);
  canvas.qrcode(url.c_str(), CX - q / 2, 110, q, 4);
  font(vlw::Small);
  String u = url.startsWith("http://") ? url.substring(7) : url;
  text(fit(u, lineRoom(326, 354)), CX, 340, kText);
  int first = 0;
  auto lines = wrapCentred(note, 358, 430, 26, 3, &first);
  drawLines(lines, first, 26, kDim);
  push();
}

void showSleeping(const String& hint, uint8_t) { showMessage("Sleeping", hint, TFT_DARKGREY); }

// Notices: a round alert mark, the title, then the text — and for a
// question, which button says what under it (two short lines: `foot`, `foot2`).
// With *yes* and *no*: two pills to tap instead of the foot lines (showAsk).
static void drawNotice(const char* title, const String& body, uint16_t color, const String& foot,
                       const String& foot2 = "", const char* yes = nullptr, const char* no = nullptr) {
  beginFrame();
  drawChrome();
  // Warnings (red, orange…) get "!", everything else (white, grey, green) "i".
  const bool warn = !(color == kWhite || color == TFT_DARKGREY || color == rgb(40, 200, 90));
  uint16_t mark = color == kWhite ? kBlue : color;
  canvas.fillSmoothCircle(CX, 128, 30, mark);
  if (foot.length() || yes) {
    font(vlw::Title);
    text("?", CX, 130, kBlack);
  } else if (warn) {
    canvas.fillSmoothRoundRect(CX - 4, 110, 8, 24, 4, kBlack);
    canvas.fillSmoothCircle(CX, 143, 4, kBlack);
  } else {
    canvas.fillSmoothCircle(CX, 114, 4, kBlack);
    canvas.fillSmoothRoundRect(CX - 4, 122, 8, 22, 4, kBlack);
  }
  font(vlw::Title);
  text(fit(title, lineRoom(170, 210)), CX, 190, color == kWhite ? kText : color);
  font(ts().body);
  int first = 0;
  auto lines = wrapCentred(body, 222, foot.length() || yes ? 362 : kBottom + 20, ts().line, 6, &first);
  drawLines(lines, first, ts().line, kDim);
  if (yes) {
    const int y = kAskY, h = 44, w = 140, gap = 16;
    canvas.fillSmoothRoundRect(CX - gap / 2 - w, y - h / 2, w, h, h / 2, rgb(40, 200, 90));
    canvas.fillSmoothRoundRect(CX + gap / 2, y - h / 2, w, h, h / 2, mix(kBlack, kText, 0.85f));
    canvas.fillSmoothRoundRect(CX + gap / 2 + 2, y - h / 2 + 2, w - 4, h - 4, h / 2 - 2, kBlack);
    font(vlw::BodyS);
    text(yes, CX - gap / 2 - w / 2, y + 1, kBlack);
    text(no, CX + gap / 2 + w / 2, y + 1, kText);
    askShown = true;
  } else if (foot.length()) {
    font(vlw::Small);
    text(fit(foot, lineRoom(366, 394)), CX, 380, kText);
    text(fit(foot2, lineRoom(394, 422)), CX, 408, kDim);
  }
  push();
}

void showMessage(const char* title, const String& body, uint16_t color) { drawNotice(title, body, color, ""); }

// "yellow button" → "Yellow"
static String buttonWord(const char* name) {
  String w = name;
  int sp = w.indexOf(' ');
  if (sp > 0) w = w.substring(0, sp);
  if (w.length()) w.setCharAt(0, toupper(w[0]));
  return w;
}

void showAsk(const char* title, const String& body, const char* yes, const char* no) {
  drawNotice(title, body, kWhite, "", "", yes, no);  // (the buttons work as before: talk = yes, side = no)
}

int askHit(int x, int y) {
  if (!askShown || abs(y - kAskY) > 34) return -1;
  return x < CX ? 1 : 0;
}

int showPanel(const char* title, uint16_t titleColor, const std::vector<Row>& rows, int page) {
  const int labelH = 26, gap = 10;
  const int line = ts().line;
  // Lay the rows out into pages: label (small, grey) over the value lines.
  struct Laid {
    int page;
    int y;
    std::vector<String> lines;
  };
  std::vector<Laid> laid;
  int pg = 0;
  int y = kTop + 6 + ((title && *title) ? 48 : 0);
  font(ts().body);
  for (const Row& r : rows) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      int vy = y + labelH;
      auto lines = wrap(r.value.length() ? r.value : String("-"),
                        [&](int k) { return lineRoom(vy + k * line, vy + (k + 1) * line); });
      int h = labelH + (int)lines.size() * line + gap;
      if (y + h > kBottom + gap && y > kTop + 6 && attempt == 0) {  // next page
        ++pg;
        y = kTop + 6;
        continue;
      }
      laid.push_back({pg, y, lines});
      y += h;
      break;
    }
  }
  const int pages = pg + 1;
  page = constrain(page, 0, pages - 1);

  beginFrame();
  drawChrome();
  if (page == 0 && title && *title) {
    font(vlw::Title);
    text(title, CX, kTop + 26, titleColor);
  }
  for (size_t i = 0; i < laid.size(); ++i) {
    if (laid[i].page != page) continue;
    font(vlw::Small);
    text(fit(rows[i].label, lineRoom(laid[i].y, laid[i].y + labelH)), CX, laid[i].y + labelH / 2, kDim);
    font(ts().body);
    int vy = laid[i].y + labelH;
    for (size_t k = 0; k < laid[i].lines.size(); ++k) {
      text(laid[i].lines[k], CX, vy + k * line + line / 2, rows[i].color);
    }
  }
  drawPageDots(pages, page, kBottom + 22);
  push();
  return pages;
}

// The agent's badge, small: the glow and all, scaled down, its initials in it.
static void drawSmallBadge(Tone tone, const String& name, int cy) {
  M5Canvas* s = badgeSprite(tone);
  const float z = 0.42f;
  if (s) s->pushRotateZoomWithAA(&canvas, CX, cy, 0, z, z, kBlack);
  else canvas.fillSmoothCircle(CX, cy, kBadgeR * z, rgb(kTones[tone].a[0], kTones[tone].a[1], kTones[tone].a[2]));
  font(vlw::Small);
  text(initials(name), CX, cy + 1, kWhite);
}

// Idle: a watch first — the date, the time large, the agent's badge small
// with its name, then the hint (or what's wrong), the battery as an arc at
// the bottom of the ring and in words. Before the clock is known, the badge
// takes the middle as it used to.
void showHome(const Home& home) {
  beginFrame();
  Tone tone = home.ready ? TBlue : barState == "offline" ? TGrey : TAmber;
  String name = barAgent.length() ? barAgent : String("Captain Claw");
  char clock[8];
  const bool face = clockText(clock, sizeof(clock));
  drawChrome(false, !face);
  if (face) {
    font(vlw::Small);
    text(dateText(), CX, 104, kDim);
    font(vlw::Digits);
    text(clock, CX, 180, kText);
    drawSmallBadge(tone, name, 282);
    font(vlw::Title);
    text(fit(name, lineRoom(324, 358)), CX, 342, kText);
  } else {
    drawBadge(tone, name, Glyph::Initials);
    font(vlw::Title);
    text(fit(name, lineRoom(kTitleY - 20, kTitleY + 20)), CX, kTitleY, kText);
  }
  if (tapMode == TapButton::Talk) {
    drawTapButton(face ? 398 : kTapY, face ? kTapLowR : kTapR);  // in place of "hold the yellow button…"
  } else if (home.hint.length()) {
    font(vlw::Small);
    int first = 0;
    const int top = face ? 364 : kNoteY - 16;
    auto lines = wrapCentred(home.hint, top, top + 62, 28, 2, &first);
    drawLines(lines, first, 28, home.ready ? kDim : home.color);
  }
  bool chg = false;
  const int pct = batteryPct(&chg);
  if (face && pct >= 0) {  // the level as an arc at the bottom of the ring
    ringArc(150, 60, mix(kBlack, kText, 0.12f), 6, R - 2);
    ringArc(150, max(2, 60 * pct / 100), batteryColor(pct), 6, R - 2);
  }
  drawBatteryAt(kBatteryY, true);
  push();
}

// Recent mic levels, newest last, for the bars under the timer.
static int levels[15];

void showListening(int level, uint32_t elapsedMs) {
  for (int i = 0; i < 14; ++i) levels[i] = levels[i + 1];
  levels[14] = level;
  beginFrame();
  ringListening(elapsedMs, level);
  drawChrome();
  font(vlw::Title);
  int labelW = tw("Listening");
  canvas.fillSmoothCircle(CX - labelW / 2 - 18, 132, 8, kRed);
  text("Listening", CX + 10, 132, kText);
  font(vlw::Digits);
  char t[12];
  snprintf(t, sizeof(t), "%lu:%02lu", (unsigned long)(elapsedMs / 60000), (unsigned long)(elapsedMs / 1000 % 60));
  text(t, CX, 228, kWhite);
  for (int i = 0; i < 15; ++i) {  // a little voice meter
    int h = 6 + min(levels[i], 100) * 54 / 100;
    int x = CX - 7 * 18 + i * 18;
    canvas.fillSmoothRoundRect(x - 4, 322 - h / 2, 8, h, 4, levels[i] > 85 ? kAmber : mix(kRed, kWhite, 0.25f));
  }
  if (tapMode == TapButton::Stop) {
    drawTapButton(kTapY, kTapR);
  } else {
    font(vlw::Small);
    text("release to send", CX, 396, kDim);
  }
  push();
}

void showWorking(const String& heard, const std::vector<String>& activity, const String& status,
                 uint32_t elapsedMs, int steps) {
  beginFrame();
  ringWorking(steps);
  drawChrome();
  if (heard.length()) {  // the end of a long sentence matters most: last two lines
    font(vlw::BodyS);
    String q = String("“") + heard + "”";
    // Only the last two lines show, on the top two rows: wrap to the narrower
    // (top) row, with room for a leading "…".
    const int room = lineRoom(84, 118) - tw("…");
    auto ls = wrap(q, [&](int) { return room; });
    int n = min(2, (int)ls.size());
    int first = (int)ls.size() - n;
    for (int i = 0; i < n; ++i) {
      String l = (first > 0 && i == 0) ? String("…") + ls[first + i] : ls[first + i];
      text(l, CX, 84 + i * 34 + 17, kCyan);
    }
  }
  font(vlw::Digits);
  char t[12];
  snprintf(t, sizeof(t), "%lu:%02lu", (unsigned long)(elapsedMs / 60000), (unsigned long)(elapsedMs / 1000 % 60));
  text(t, CX, 218, kWhite);
  font(vlw::Small);
  text(steps ? String(steps) + (steps == 1 ? " step" : " steps") : String("working"), CX, 286, kAmber);
  // The newest activity in white, the one before it grey above.
  int n = activity.size();
  if (n >= 2) {
    font(vlw::Small);
    text(fit(activity[n - 2], lineRoom(306, 334)), CX, 320, kFaint);
  }
  if (n >= 1) {
    font(vlw::BodyS);
    text(fit(activity[n - 1], lineRoom(338, 374)), CX, 356, kText);
  }
  if (status.length()) {
    font(vlw::Small);
    text(fit(status, lineRoom(374, 398)), CX, 386, kDim);
  }
  if (tapMode == TapButton::Stop) drawTapButton(424, 18);  // stop the agent (above the step ticks)
  push();
}

// Where an answer's text ends: above the button that sits under it (pause /
// play while it's spoken, Tap to talk's), reserved always so the text never
// reflows when a button comes or goes.
static int replyBottom() { return kTapReplyBottom; }

static void layoutReply() {
  replyPages.clear();
  replyCentred = false;
  font(ts().body);
  const int line = ts().line;
  const int bottom = replyBottom();
  const int per = (bottom - kTop) / line;
  auto lines = wrap(replyText, [&](int k) {
    int y = kTop + (k % per) * line;
    return lineRoom(y, y + line);
  });
  if ((int)lines.size() <= per) {  // one page: centred, where the lines are wider
    int first = 0;
    replyPages.push_back(wrapCentred(replyText, kTop, bottom, line, per, &first));
    replyCentred = true;
  } else {
    for (int i = 0; i < (int)lines.size(); i += per) {
      replyPages.emplace_back(lines.begin() + i, lines.begin() + min((int)lines.size(), i + per));
    }
  }
  if (replyPage >= (int)replyPages.size()) replyPage = 0;  // re-wrapped into fewer pages
  // Where each line sits in the text (lines are its pieces, in order), so a
  // sentence can be lit and the voice's place tells which page it's on.
  lineSpans.clear();
  pageEnds.clear();
  const char* all = replyText.c_str();
  int cursor = 0;
  for (const auto& page : replyPages) {
    lineSpans.emplace_back();
    for (const String& l : page) {
      const char* p = l.length() ? strstr(all + cursor, l.c_str()) : nullptr;
      const int from = p ? (int)(p - all) : cursor;
      cursor = from + (int)l.length();
      lineSpans.back().push_back({from, cursor});
    }
    pageEnds.push_back(cursor);
  }
}

// The sentence round character *at*: [from, to). A sentence ends at a full
// stop, ! or ? (or …) followed by a space or the end, or at a line break.
static void sentenceAt(int at, int* from, int* to) {
  const char* s = replyText.c_str();
  const int n = replyText.length();
  int start = 0;
  for (int i = 0; i < n; ++i) {
    const unsigned char c = s[i];
    bool end = c == '\n';
    if (!end && (c == '.' || c == '!' || c == '?')) {
      const char nx = i + 1 < n ? s[i + 1] : ' ';
      end = nx == ' ' || nx == '\n' || nx == '"' || nx == ')';
    } else if (!end && c == 0xE2 && i + 2 < n && (unsigned char)s[i + 1] == 0x80 && (unsigned char)s[i + 2] == 0xA6) {
      i += 2;  // …
      end = i + 1 >= n || s[i + 1] == ' ' || s[i + 1] == '\n';
    }
    if (!end) continue;
    int stop = i + 1;
    if (at < stop) {
      *from = start;
      *to = stop;
      return;
    }
    while (stop < n && (s[stop] == ' ' || s[stop] == '\n')) ++stop;
    start = stop;
    i = stop - 1;
  }
  *from = start;
  *to = n;
}

bool setSpoken(int at) {
  if (at < 0) {
    const bool was = litTo >= 0;
    spokenAt = litFrom = litTo = -1;
    return was;
  }
  spokenAt = min(at, (int)replyText.length());
  int from = 0, to = 0;
  sentenceAt(spokenAt, &from, &to);
  bool changed = from != litFrom || to != litTo;
  litFrom = from;
  litTo = to;
  if (autoFollow && replyPage + 1 < (int)replyPages.size() && replyPage < (int)pageEnds.size() &&
      spokenAt >= pageEnds[replyPage]) {
    ++replyPage;  // the voice has gone on to the next page
    nextFx = Fx::SlideOn;
    changed = true;
  }
  return changed;
}

void setReplayable(bool on) { replayable = on; }

// A small circle arrow in a ring, like the tap button's.
static void drawReplay(int cx, int cy, int r) {
  canvas.fillSmoothCircle(cx, cy, r, mix(kBlack, kText, 0.85f));
  canvas.fillSmoothCircle(cx, cy, r - 3, kBlack);
  canvas.fillArc(cx, cy, r - 11, r - 8, 270, 360, kText);
  canvas.fillArc(cx, cy, r - 11, r - 8, 0, 200, kText);
  canvas.fillTriangle(cx - 5, cy - (r - 10), cx + 2, cy - (r - 10) - 5, cx + 2, cy - (r - 10) + 5, kText);
  rpX = cx, rpY = cy, rpR = r;
}

bool replayHit(int x, int y) {
  const int dx = x - rpX, dy = y - rpY, reach = rpR + 24;
  return rpR && dx * dx + dy * dy <= reach * reach;
}

void setQuestion(const String&) {}
bool canPin() { return false; }
bool pinned() { return false; }
void pin(bool) {}
bool restoreAnswer() { return false; }

void setReply(const String& t) {
  replyText = t;
  replyPage = 0;
  spokenAt = litFrom = litTo = -1;
  autoFollow = true;
  layoutReply();
}

void growReply(const String& t) {
  replyText = t;
  layoutReply();  // the page stays (within the pages there are now)
}

void setTextSize(uint8_t size) {
  sizeIdx = size < sizeof(kSizes) / sizeof(kSizes[0]) ? size : 0;
  if (replyText.length()) layoutReply();
}

bool hasReply() { return !replyPages.empty() && !replyText.isEmpty(); }

void showReply() {
  beginFrame();
  drawChrome();
  if (!replyPages.empty()) {
    font(ts().body);
    const auto& lines = replyPages[replyPage];
    int line = ts().line;
    int top = replyCentred ? kTop + ((replyBottom() - kTop) - (int)lines.size() * line) / 2 : kTop;
    // While it's spoken, the sentence the voice is in is lit, the rest dimmed.
    const auto* spans = replyPage < (int)lineSpans.size() ? &lineSpans[replyPage] : nullptr;
    for (int i = 0; i < (int)lines.size(); ++i) {
      uint16_t col = kText;
      if (litTo >= 0 && spans && i < (int)spans->size()) {
        const auto& sp = (*spans)[i];
        col = sp.second > litFrom && sp.first < litTo ? kText : kDim;
      }
      text(lines[i], CX, top + i * line + line / 2, col);
    }
    drawPageDots(replyPages.size(), replyPage, replyBottom() + 22);
  }
  if (answerButton()) drawTapButton(kTapLowY, kTapLowR);
  if (replayable) drawReplay(answerButton() ? CX - 74 : CX, kTapLowY, 20);
  push();
}

bool nextPage() {
  autoFollow = false;  // paged by hand: the voice no longer turns the pages
  if (replyPages.empty()) return false;
  replyPage = (replyPage + 1) % replyPages.size();
  showReply();
  return replyPage != 0;
}

bool prevPage() {
  autoFollow = false;
  if (replyPage == 0) return false;
  --replyPage;
  showReply();
  return true;
}

bool onLastPage() { return replyPage + 1 >= (int)replyPages.size(); }
bool onFirstPage() { return replyPage == 0; }

// Characters are drawn for the stick's 240×135 screen (under its 16 px bar),
// then scaled up smoothly into the middle of the round one.
static void drawCharacter(uint8_t kind, face::Pose pose, int level, int cy) {
  faceSprite.fillScreen(hasBackdrop ? kSeeThrough : kBlack);
  face::draw(faceSprite, kind, pose, level, millis(), 120, 76);
  faceSprite.setPivot(120, 76);
  if (hasBackdrop) faceSprite.pushRotateZoomWithAA(&canvas, CX, cy, 0, 1.6f, 1.6f, kSeeThrough);
  else faceSprite.pushRotateZoomWithAA(&canvas, CX, cy, 0, 1.6f, 1.6f);
}

void showFace(uint8_t kind, face::Pose pose, int level) {
  beginFrame();
  drawCharacter(kind, pose, level, 214);  // first: its black corners would cover the ring
  if (pose == face::Pose::Listening) ringListening(0, level);
  else if (pose == face::Pose::Thinking) ringWorking(0);

  drawChrome();
  // The sentence being spoken, as a subtitle in place of the name: the two
  // lines round where the voice is in it; else the name.
  const bool subtitle = pose == face::Pose::Speaking && litTo > litFrom;
  if (subtitle) {
    font(vlw::BodyS);
    const int room = lineRoom(330, 390);
    auto ls = wrap(replyText.substring(litFrom, litTo), [&](int) { return room; });
    const int want = max(0, spokenAt - litFrom);
    int at = 0, seen = 0;
    for (int i = 0; i < (int)ls.size(); ++i) {
      at = i;
      seen += ls[i].length() + 1;
      if (want < seen) break;
    }
    const int first = min(at, max(0, (int)ls.size() - 2));
    const int n = min(2, (int)ls.size() - first);
    for (int i = 0; i < n; ++i) text(ls[first + i], CX, 344 + i * 30, kText);
  } else {
    font(vlw::Title);
    String name = barAgent.length() ? barAgent : String("Captain Claw");
    text(fit(name, lineRoom(342, 378)), CX, 360, kText);
  }
  // The tap button takes the state line's place: to talk, or to stop.
  const bool listening = pose == face::Pose::Listening;
  if ((answerButton() && !listening && pose != face::Pose::Thinking) ||
      (tapMode == TapButton::Stop && (listening || pose == face::Pose::Thinking))) {
    drawTapButton(kTapLowY, kTapLowR);
  } else if (!subtitle) {
    font(vlw::Small);
    text(fit(barState, lineRoom(382, 410)), CX, 396, pose == face::Pose::Idle ? kDim : barColor);
  }
  push();
}

void showCharPick(uint8_t kind, face::Pose pose, int level, int index) {
  beginFrame();
  drawCharacter(kind, pose, level, 206);
  drawChrome();
  drawChevrons(206);
  font(vlw::Title);
  text(face::name(kind), CX, 352, kText);
  font(vlw::Small);
  text("yellow = pick", CX, 386, kDim);
  drawPageDots(face::KindCount, index, 420);
  push();
}

void storeImage(uint8_t* jpg, size_t len) {
  if (imgBuf) free(imgBuf);
  imgBuf = jpg;
  imgLen = jpg ? len : 0;
  imgFit = false;  // a new picture fills the circle
}

// Width and height from a baseline JPEG's frame header.
static bool jpegSize(const uint8_t* d, size_t n, int* w, int* h) {
  size_t i = 2;
  while (i + 9 < n && d[i] == 0xFF) {
    const uint8_t m = d[i + 1];
    const size_t len = (d[i + 2] << 8) | d[i + 3];
    if (m == 0xC0 || m == 0xC1 || m == 0xC2) {
      *h = (d[i + 5] << 8) | d[i + 6];
      *w = (d[i + 7] << 8) | d[i + 8];
      return *w > 0 && *h > 0;
    }
    if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7)) i += 2;  // markers without a length
    else i += 2 + len;
  }
  return false;
}

void toggleImageFit() { imgFit = !imgFit; }

bool hasImage() { return imgLen > 0; }

void showStoredImage() {
  if (!imgLen) return;
  beginFrame(true);
  // Centred: the deck sends it filling the circle (its shorter side the
  // screen's), so the middle shows; after a tap, all of it, scaled down.
  float scale = 1.0f;
  int iw = 0, ih = 0;
  if (imgFit && jpegSize(imgBuf, imgLen, &iw, &ih) && (iw > W || ih > H)) {
    scale = min((float)W / iw, (float)H / ih);
  }
  canvas.drawJpg(imgBuf, imgLen, 0, 0, W, H, 0, 0, scale, scale, lgfx::datum_t::middle_center);
  push();
}

static constexpr int kFrost = 4;  // the blur's radius: a light frost, the picture still shows through

static void* bigAlloc(size_t n) {
#if defined(ESP_PLATFORM)
  return ps_malloc(n);
#else
  return malloc(n);
#endif
}

// A box blur along one line of *n* values *stride* apart, edges held.
static void blurLine(const uint8_t* in, uint8_t* out, int n, int stride, int r) {
  int sum = 0;
  for (int j = -r; j <= r; ++j) sum += in[constrain(j, 0, n - 1) * stride];
  for (int i = 0; i < n; ++i) {
    out[i * stride] = sum / (2 * r + 1);
    sum += in[min(i + r + 1, n - 1) * stride] - in[max(i - r, 0) * stride];
  }
}

// *dst* = *src* (W×H RGB565, byte-swapped) blurred a little (two box passes
// each way ≈ a soft Gaussian) and darkened by kBackdropDim/256. False when
// the planes can't be had.
static bool frost(const uint16_t* src, uint16_t* dst) {
  const int N = W * H;
  auto* plane = (uint8_t*)bigAlloc((size_t)N * 3);
  auto* tmp = (uint8_t*)bigAlloc((size_t)N);
  if (!plane || !tmp) {
    free(plane);
    free(tmp);
    return false;
  }
  for (int i = 0; i < N; ++i) {
    const uint16_t c = __builtin_bswap16(src[i]);
    plane[i] = (c >> 11) << 3;
    plane[N + i] = ((c >> 5) & 63) << 2;
    plane[2 * N + i] = (c & 31) << 3;
  }
  for (int ch = 0; ch < 3; ++ch) {
    uint8_t* p = plane + ch * N;
    for (int pass = 0; pass < 2; ++pass) {
      for (int y = 0; y < H; ++y) blurLine(p + y * W, tmp + y * W, W, 1, kFrost);
      for (int x = 0; x < W; ++x) blurLine(tmp + x, p + x, H, W, kFrost);
    }
  }
  for (int i = 0; i < N; ++i) {
    const int r = (plane[i] * kBackdropDim) >> 11, g = (plane[N + i] * kBackdropDim) >> 10,
              b = (plane[2 * N + i] * kBackdropDim) >> 11;
    dst[i] = __builtin_bswap16((uint16_t)(r << 11 | g << 5 | b));
  }
  free(plane);
  free(tmp);
  return true;
}

void setBackground(const uint8_t* jpg, size_t len) {
  hasBackdrop = false;
  if (!jpg || !len) {
    backdrop.deleteSprite();
    backdropDim.deleteSprite();
    return;
  }
  for (M5Canvas* s : {&backdrop, &backdropDim}) {
    if (s->width()) continue;
    s->setPsram(true);
    s->setColorDepth(16);
    if (!s->createSprite(W, H)) return;
  }
  backdrop.fillScreen(kBlack);
  if (!backdrop.drawJpg(jpg, len, 0, 0, W, H, 0, 0, 1.0f, 1.0f, lgfx::datum_t::middle_center)) return;
  // The copy the screens draw on: frosted (a light blur) and darkened, once.
  const auto* src = (const uint16_t*)backdrop.getBuffer();
  auto* dst = (uint16_t*)backdropDim.getBuffer();
  if (!frost(src, dst)) {
    // No room for the blur's planes: darkened only (the sprites keep RGB565 byte-swapped).
    for (int i = 0; i < W * H; ++i) {
      const uint16_t c = __builtin_bswap16(src[i]);
      const uint16_t r = ((c >> 11) * kBackdropDim) >> 8, g = (((c >> 5) & 63) * kBackdropDim) >> 8,
                     b = ((c & 31) * kBackdropDim) >> 8;
      dst[i] = __builtin_bswap16((uint16_t)(r << 11 | g << 5 | b));
    }
  }
  hasBackdrop = true;
}

bool hasBackground() { return hasBackdrop; }

void showBackdrop() {
  beginFrame(true);  // black when there's none
  if (hasBackdrop) backdrop.pushSprite(&canvas, 0, 0);  // as it is: nothing over it to read
  push();
}

void showPicker(const std::vector<Item>& items, int selected, int current, const char*) {
  beginFrame();
  if (!items.empty()) {
    selected = constrain(selected, 0, (int)items.size() - 1);
    const Item& it = items[selected];
    Tone tone = selected == current ? TGreen : it.on ? TBlue : TGrey;
    drawBadge(tone, it.badge, it.glyph);
    drawChevrons(kBadgeY);
    font(vlw::Title);
    text(fit(it.title, lineRoom(kTitleY - 20, kTitleY + 20)), CX, kTitleY, kText);
    if (it.note.length()) {
      font(vlw::Small);
      text(fit(it.note, lineRoom(kNoteY - 14, kNoteY + 14)), CX, kNoteY, selected == current ? kGreen : kDim);
    }
    drawPageDots(items.size(), selected, kDotsY);
  }
  drawChrome();
  push();
}

void flashFrame(bool on) {
  if (on) {
    ringArc(0, 360, kWhite, 30);
    push();
  }
}

}  // namespace ui
