// Waveshare 1.54" e-paper: 200×200, black and white. Text is efont, which is
// made of whole pixels and has č ć š ž đ (the deck spells Đ out for it).
//
// A screen is drawn into a canvas at once, as on the stick; a background task
// takes the latest one to the panel, so nothing waits for it. Only a frame
// that differs goes, and without a flash (a partial refresh, ~0.4 s). A full
// refresh (~2 s, flashing) is for what partial ones leave behind: the first
// frame after power-up, the QR codes, the sleeping screen, the home screen left
// alone a while, and at the latest after kMaxPartials. Each full refresh starts
// from a reset controller (epd_ssd1681.cpp), as on the PaperColor.
//
// What the panel shows stays in RTC memory across deep sleep and restarts, so
// a wake needs no flash to go on from it.
#include <M5Unified.h>
#include <math.h>
#include <time.h>

#include <vector>

#include "board.h"
#include "efont_subset.h"
#include "hw.h"
#include "telemetry.h"
#include "ui.h"

#ifndef PREVIEW
#include "epd_ssd1681.h"
#endif

namespace ui {

static M5Canvas canvas(&M5.Display);
static constexpr int W = board::kScreenW, H = board::kScreenH, BAR = 18, PAD = 4;
static constexpr int kFrameBytes = W * H / 8;
static constexpr uint16_t kInk = 0x0000, kPaper = 0xFFFF;
static const lgfx::U8g2font* kSmall = &efont::efont12;  // fixed size: top bar, legends

static int tw(const String& s) { return canvas.textWidth(s.c_str()); }

// Reading text in the size the user picked (the same three as the stick).
struct TextSize {
  const lgfx::U8g2font* body;
  int line;
  const lgfx::U8g2font* small;  // activity lines on the working screen
  int smallLine;
};
static const TextSize kSizes[] = {
    {&efont::efont14, 16, &efont::efont12, 15},
    {&efont::efont16, 18, &efont::efont14, 16},
    {&efont::efont24, 26, &efont::efont16, 18},
};
static uint8_t sizeIdx = 0;
static const TextSize& ts() { return kSizes[sizeIdx]; }

enum class Screen : uint8_t { Other, Home, Listening, Working, Ask, Sleeping };
static Screen current = Screen::Other;
static Screen drawing(Screen s) {
  const Screen was = current;
  current = s;
  return was;
}

static String barAgent, barState;
static Health hWifi = Down, hAgent = Down;
static int waitingCount = 0;
static uint8_t rotation = 0;
static uint32_t lastUseMs = 0;

static String replyText;
static std::vector<String> lines;
static int page = 0;
static std::vector<int> pageEnds;  // where each page's text ends: the voice's place tells its page
static int spokenAt = -1;
static bool autoFollow = true;
static uint8_t* imgBuf = nullptr;
static size_t imgLen = 0;
static TapButton tapButton = TapButton::Off;
static Home lastHome;

// ── from the canvas to the panel ───────────────────────────────────────

#ifndef RTC_DATA_ATTR
#define RTC_DATA_ATTR  // (the preview has no deep sleep)
#endif
static constexpr uint32_t kShownMagic = 0x31353421;
RTC_DATA_ATTR static uint8_t rtcFrame[kFrameBytes];  // what the panel shows, if rtcMagic says so
RTC_DATA_ATTR static uint32_t rtcMagic = 0;
RTC_DATA_ATTR static uint16_t rtcPartials = 0;       // since the last full refresh
static constexpr uint16_t kMaxPartials = 60;         // the next change is a full refresh, whatever it is
static constexpr uint16_t kCleanAfter = 20;          // home left alone this long after so many: one too
static constexpr uint32_t kCleanIdleMs = 60000;
static constexpr uint16_t kSleepCleanAfter = 5;      // going to sleep after this many: the sleeping screen is one

// Whether a canvas pixel is ink (it's byte-swapped RGB565; text and shapes
// are pure black or white, pictures are dithered to them first).
static inline bool isInk(uint16_t c) {
  if (c == kPaper) return false;
  if (c == kInk) return true;
  const uint16_t v = (uint16_t)((c >> 8) | (c << 8));
  const int r = ((v >> 11) & 31) * 255 / 31, g = ((v >> 5) & 63) * 255 / 63, b = (v & 31) * 255 / 31;
  return (r * 299 + g * 587 + b * 114) / 1000 < 128;
}

// The canvas as the panel's bits, turned the way the user has it.
static void toBits(uint8_t* out) {
  const uint16_t* buf = (const uint16_t*)canvas.getBuffer();
  memset(out, 0xFF, kFrameBytes);
  if (!buf) return;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!isInk(buf[y * W + x])) continue;
      int px = x, py = y;
      switch (rotation) {
        case 1: px = W - 1 - y, py = x; break;
        case 2: px = W - 1 - x, py = H - 1 - y; break;
        case 3: px = y, py = H - 1 - x; break;
        default: break;
      }
      out[py * (W / 8) + (px >> 3)] &= (uint8_t)~(0x80 >> (px & 7));
    }
  }
}

#ifndef PREVIEW

static uint8_t* nextFrame = nullptr;  // the latest screen
static uint8_t* workFrame = nullptr;  // the one going to the panel
static SemaphoreHandle_t frameLock = nullptr;
static TaskHandle_t panelTask = nullptr;
static volatile bool pending = false, running = false, wantFull = false;

static void refreshOnce() {
  xSemaphoreTake(frameLock, portMAX_DELAY);
  memcpy(workFrame, nextFrame, kFrameBytes);
  const bool full = wantFull;
  wantFull = false;
  running = true;
  pending = false;
  xSemaphoreGive(frameLock);
  const bool known = rtcMagic == kShownMagic;
  if (!known || full || memcmp(workFrame, rtcFrame, kFrameBytes)) {
    const bool doFull = !known || full || rtcPartials >= kMaxPartials;
    rtcMagic = 0;  // cut off halfway, the next boot starts with a full refresh
    if (doFull ? epd::full(workFrame) : epd::partial(workFrame, rtcFrame)) {
      memcpy(rtcFrame, workFrame, kFrameBytes);
      rtcMagic = kShownMagic;
      rtcPartials = doFull ? 0 : rtcPartials + 1;
    }
  }
  running = false;
}

static void panelLoop(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    while (pending) refreshOnce();
  }
}

static void push(bool full = false) {
  xSemaphoreTake(frameLock, portMAX_DELAY);
  toBits(nextFrame);
  if (full) wantFull = true;
  pending = true;
  xSemaphoreGive(frameLock);
  xTaskNotifyGive(panelTask);
}

void flush() {
  for (uint32_t t0 = millis(); (pending || running) && millis() - t0 < 10000;) delay(5);
}

bool busy() { return pending || running; }

void panelSleep() {
  flush();
  epd::sleep();
}

#else  // the preview: the frame, as the panel would show it

static uint8_t previewFrame[kFrameBytes];

static void push(bool = false) {
  toBits(previewFrame);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const bool white = previewFrame[y * (W / 8) + (x >> 3)] & (0x80 >> (x & 7));
      M5.Display.drawPixel(x, y, white ? kPaper : kInk);
    }
  }
}

void flush() {}
bool busy() { return false; }

#endif

// ── set-up ─────────────────────────────────────────────────────────────

void begin() {
  canvas.setPsram(true);
  canvas.setColorDepth(16);
  canvas.createSprite(W, H);
  canvas.setTextWrap(false);
  canvas.fillScreen(kPaper);
#ifndef PREVIEW
  nextFrame = (uint8_t*)heap_caps_malloc(kFrameBytes, MALLOC_CAP_8BIT);
  workFrame = (uint8_t*)heap_caps_malloc(kFrameBytes, MALLOC_CAP_8BIT);
  frameLock = xSemaphoreCreateMutex();
  epd::begin();
  xTaskCreatePinnedToCore(panelLoop, "paper", 8192, nullptr, 1, &panelTask, 0);
#endif
}

void brightness(uint8_t) {}  // no backlight
bool animated() { return false; }
void idleSince(uint32_t ms) { lastUseMs = ms; }
void drawNow() {}  // a change shows within the second anyway
bool waiting() { return false; }
void showSpeaking(int) {}
void flashFrame(bool) {}
void slide(bool) {}
void fade() {}

void setRotation(uint8_t r) {
  rotation = r & 3;
#ifndef PREVIEW
  hw::setTouchRotation(rotation);
#endif
}

// ── small pieces ───────────────────────────────────────────────────────

static String fit(const String& text, int width) {
  if (tw(text) <= width) return text;
  String s = text;
  while (s.length() > 1 && tw(s + "…") > width) {
    int cut = s.length() - 1;
    while (cut > 0 && (uint8_t(s[cut]) & 0xC0) == 0x80) --cut;  // whole UTF-8 characters
    s.remove(cut);
  }
  return s + "…";
}

// Word-wrap UTF-8 text to `width` (long words break between characters), in
// `font` — the reading font when none is given.
static std::vector<String> wrap(const String& text, int width, const lgfx::U8g2font* font = nullptr) {
  std::vector<String> out;
  canvas.setFont(font ? font : ts().body);
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
      String trial = line.length() ? line + " " + word : word;
      if (tw(trial) <= width) {
        line = trial;
        continue;
      }
      if (line.length()) out.push_back(line);
      line = "";
      while (tw(word) > width) {
        int cut = 0, k = 0;
        while (k < (int)word.length()) {
          int step = 1;
          uint8_t c = word[k];
          if (c >= 0xF0) step = 4; else if (c >= 0xE0) step = 3; else if (c >= 0xC0) step = 2;
          if (tw(word.substring(0, k + step)) > width) break;
          k += step;
          cut = k;
        }
        if (cut == 0) cut = 1;
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

static void text(const String& s, int x, int y, uint8_t datum = top_left, uint16_t color = kInk) {
  canvas.setTextDatum(datum);
  canvas.setTextColor(color);
  canvas.drawString(s.c_str(), x, y);
}

// A tag per link: filled = up, an outline = on its way (with a "z": the agent
// asleep), struck through = down or switched off. No colours to say it with.
static int drawHealth(int x, int y) {
  canvas.setFont(kSmall);
  const char* tags[2] = {"W", "A"};
  const Health hs[2] = {hWifi, hAgent};
  for (int i = 0; i < 2; ++i) {
    const bool up = hs[i] == Up;
    if (up) canvas.fillRoundRect(x, y, 14, 13, 2, kInk);
    else canvas.drawRoundRect(x, y, 14, 13, 2, kInk);
    text(hs[i] == Asleep ? "z" : tags[i], x + 7, y + 7, middle_center, up ? kPaper : kInk);
    if (hs[i] == Down || hs[i] == Off) canvas.drawLine(x + 1, y + 11, x + 12, y + 1, kInk);
    x += 16;
  }
  return x;
}

// The battery: an icon filled to its level, the percentage by it. (Read at
// most every few seconds, by hw::batteryMv().)
static int drawBattery(int right, int y) {
  const int pct = hw::batteryLevel();
  const bool chg = hw::charging();
  canvas.setFont(kSmall);
  const String txt = pct < 0 ? String("USB") : String(pct) + "%";
  text(txt, right, y + 7, middle_right);
  const int bx = right - tw(txt) - 21;
  canvas.drawRect(bx, y + 2, 18, 9, kInk);
  canvas.fillRect(bx + 18, y + 4, 2, 5, kInk);
  if (pct > 0) canvas.fillRect(bx + 2, y + 4, max(1, pct * 14 / 100), 5, kInk);
  if (chg) {  // a small bolt, knocked out of the cells
    canvas.fillTriangle(bx + 10, y + 2, bx + 6, y + 7, bx + 9, y + 7, kPaper);
    canvas.fillTriangle(bx + 9, y + 6, bx + 12, y + 6, bx + 8, y + 11, kPaper);
  }
  return bx;
}

void setWaiting(int count) { waitingCount = count; }

static void drawBar(bool links = true) {
  int x = links ? drawHealth(1, 2) + 3 : PAD;
  int left = drawBattery(W - 2, 2);
  canvas.setFont(kSmall);
  if (waitingCount > 0) {  // what waits in Flight Deck: a count, reversed out
    const String n = waitingCount > 99 ? String("99+") : String(waitingCount);
    const int w = max(BAR - 4, (int)tw(n) + 6);
    left -= w + 3;
    canvas.fillRoundRect(left, 2, w, BAR - 5, (BAR - 5) / 2, kInk);
    text(n, left + w / 2, 2 + (BAR - 5) / 2, middle_center, kPaper);
  }
  const int stateW = tw(barState);
  text(barState, left - 4, BAR / 2, middle_right);
  const int nameRoom = left - 4 - stateW - 5 - x;
  if (nameRoom > 12) {
    const String name = barAgent.length() ? barAgent : String("Captain");
    text(fit(name, nameRoom), x, BAR / 2, middle_left);
  }
  canvas.drawFastHLine(0, BAR - 1, W, kInk);
}

void setBar(const String& agent, const char* state, uint16_t) {
  barAgent = agent;
  barState = state ? state : "";
}

void setHealth(Health wifi, Health agent) {
  hWifi = wifi;
  hAgent = agent;
}

// What a press does or where a touch goes: a line reversed out of a black pill.
static void pill(const String& label, int x, int y, int w, int h) {
  canvas.fillRoundRect(x, y, w, h, h / 2, kInk);
  canvas.setFont(kSmall);
  text(label, x + w / 2, y + h / 2, middle_center, kPaper);
}

// The on-screen button (touch): Talk / Stop / Pause / Play, the bottom row.
static constexpr int kTapH = 24, kTapW = 96, kTapY = H - kTapH - 2;
static constexpr int kTapReserve = board::kHasTouch ? kTapH + 4 : 0;  // always kept free: text never reflows

void setTapButton(TapButton button) { tapButton = button; }

bool tapButtonHit(int x, int y) {
  return tapButton != TapButton::Off && x > (W - kTapW) / 2 - 12 && x < (W + kTapW) / 2 + 12 && y > kTapY - 10;
}

static void drawTapButton() {
  if (tapButton == TapButton::Off) return;
  const char* label = tapButton == TapButton::Talk    ? "Talk"
                      : tapButton == TapButton::Stop  ? "Stop"
                      : tapButton == TapButton::Pause ? "Pause"
                                                      : "Play";
  pill(label, (W - kTapW) / 2, kTapY, kTapW, kTapH);
}

// A title reversed out of a band across the top of the page.
static int band(const String& title, int y = BAR) {
  canvas.fillRect(0, y, W, 22, kInk);
  canvas.setFont(&efont::efont16);
  text(fit(title, W - 2 * PAD), PAD, y + 11, middle_left, kPaper);
  return y + 22;
}

static bool localNow(struct tm* t) {
  if (!telemetry::clockSet()) return false;
  const time_t now = time(nullptr);
  localtime_r(&now, t);
  return true;
}

static String roomLine() {
  float t, rh;
  if (!telemetry::climate(&t, &rh)) return "";
  char s[24];
  snprintf(s, sizeof(s), "%.0f°C  %d%%", t, (int)(rh + 0.5f));
  return s;
}

// ── notices ────────────────────────────────────────────────────────────

void showMessage(const char* title, const String& body, uint16_t) {
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  drawBar();
  const int top = band(title) + 5;
  const std::vector<String> ls = wrap(body, W - 2 * PAD);  // sets the reading font
  const int room = (H - top - 2) / ts().line;
  for (int i = 0; i < (int)ls.size() && i < room; ++i) text(ls[i], PAD, top + i * ts().line);
  push();
}

// The two answers are pills: the press that gives each is on it, and on a touch
// screen they're where to tap.
static constexpr int kAskH = 26, kAskY = H - kAskH - 3, kAskW = 95;

void showAsk(const char* title, const String& body, const char* yes, const char* no) {
  drawing(Screen::Ask);
  canvas.fillScreen(kPaper);
  drawBar();
  const int top = band(title) + 5;
  const std::vector<String> ls = wrap(body, W - 2 * PAD);
  const int room = (kAskY - top - 2) / ts().line;
  for (int i = 0; i < (int)ls.size() && i < room; ++i) text(ls[i], PAD, top + i * ts().line);
  pill(board::kHasTouch ? String(yes) : String(board::kTalkKey) + ": " + yes, 3, kAskY, kAskW, kAskH);
  pill(board::kHasTouch ? String(no) : String(board::kSideKey) + ": " + no, W - kAskW - 3, kAskY, kAskW, kAskH);
  push();
}

int askHit(int x, int y) {
  if (current != Screen::Ask || y < kAskY - 12) return -1;
  return x < W / 2 ? 1 : 0;
}

// QR codes get a full refresh: a clean picture is what scans.
void showLink(const char* title, const String& url, const String& note) {
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  canvas.setFont(kSmall);
  text(fit(title, W - 2 * PAD), W / 2, 1, top_center);
  canvas.qrcode(url.c_str(), (W - 132) / 2, 15, 132, 4);
  const String u = url.startsWith("http://") ? url.substring(7) : url;
  text(fit(u, W - 2 * PAD), W / 2, 148, top_center);
  const std::vector<String> ls = wrap(note, W - 2 * PAD, kSmall);
  for (int i = 0; i < (int)ls.size() && i < 3; ++i) text(ls[i], W / 2, 162 + i * 13, top_center);
  push(true);
}

void showSetup(const char* reason, const char* apName, const char* ip) {
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  canvas.setFont(kSmall);
  text(fit(reason, W - 2 * PAD), W / 2, 1, top_center);
  const String qr = String("WIFI:T:nopass;S:") + apName + ";;";
  canvas.qrcode(qr.c_str(), (W - 116) / 2, 15, 116, 3);
  text(apName, W / 2, 133, top_center);
  text(String("then http://") + ip, W / 2, 147, top_center);
  text("Saving restarts it", W / 2, 165, top_center);
  text(String("Hold ") + board::kSideKey + " to leave", W / 2, 179, top_center);
  push(true);
}

// ── home ───────────────────────────────────────────────────────────────

// The minute it was last drawn with: tick() redraws it when that changes.
static int homeMinute = -1;

void showHome(const Home& home) {
  lastHome = home;
  drawing(Screen::Home);
  canvas.fillScreen(kPaper);
  drawBar();
  struct tm t;
  const bool clock = localNow(&t);
  homeMinute = clock ? t.tm_min : -1;
  char buf[32];
  if (clock) strftime(buf, sizeof(buf), "%H:%M", &t);
  else strcpy(buf, "--:--");
  canvas.setFont(&fonts::FreeSansBold24pt7b);
  text(buf, W / 2, 26, top_center);
  canvas.setFont(&efont::efont14);
  String under;
  if (clock) {
    strftime(buf, sizeof(buf), "%a %-d %b", &t);
    under = buf;
  }
  const String room = roomLine();
  if (room.length()) under += (under.length() ? "  ·  " : "") + room;
  if (under.length()) text(under, W / 2, 68, top_center);
  canvas.drawFastHLine(PAD, 88, W - 2 * PAD, kInk);

  int y = 94;
  if (home.noteMode) {
    pill("Note mode · hold BOOT to send", 3, y, W - 6, 20);
    y += 25;
  } else {
    canvas.setFont(&efont::efont16);
    text(home.title, W / 2, y, top_center);
    y += 20;
  }
  const int hintBottom = H - 2 - kTapReserve;
  const std::vector<String> ls = wrap(home.hint, W - 2 * PAD, &efont::efont14);
  for (int i = 0; i < (int)ls.size() && i < 2; ++i, y += 16) text(ls[i], W / 2, y, top_center);
  if (home.battery.length() && y + 15 <= hintBottom) {
    canvas.setFont(kSmall);
    String battery = home.battery;
    battery.replace("≈ ", "~");  // (not in efont)
    text(battery, W / 2, hintBottom - 14, top_center);
  }
  drawTapButton();
  push();
}

// ── recording, working ─────────────────────────────────────────────────

// Once a second, however often it's asked: each change is a refresh.
static int listenSecond = -1;

void showListening(int level, uint32_t elapsedMs) {
  const int second = elapsedMs / 1000;
  if (drawing(Screen::Listening) == Screen::Listening && second == listenSecond) return;
  listenSecond = second;
  canvas.fillScreen(kPaper);
  drawBar();
  const bool note = barState == "note";
  canvas.fillCircle(W / 2, 54, 22, kInk);
  canvas.fillCircle(W / 2, 54, 8, kPaper);
  canvas.setFont(&fonts::FreeSansBold12pt7b);
  text(note ? "Note" : "Listening", W / 2, 84, top_center);
  char t[16];
  snprintf(t, sizeof(t), "%d:%02d", second / 60, second % 60);
  canvas.setFont(&efont::efont24);
  text(t, W / 2, 112, top_center);
  // Ten cells for the voice's level: a coarse one, so it doesn't redraw for nothing.
  const int cells = constrain((level + 5) / 10, 0, 10);
  for (int i = 0; i < 10; ++i) {
    const int x = 10 + i * 18;
    if (i < cells) canvas.fillRect(x, 142, 16, 8, kInk);
    else canvas.drawRect(x, 142, 16, 8, kInk);
  }
  if (tapButton != TapButton::Off) {
    drawTapButton();
  } else {
    canvas.setFont(kSmall);
    text(note ? "press BOOT to keep it" : "press BOOT to send", W / 2, 170, top_center);
  }
  push();
}

static uint32_t workPushMs = 0;
static String workSeen;

// The time and steps are in the top bar (main sets them as the bar state).
// What it heard and what it does draw at once; the time alone every few
// seconds, or a long answer would cost a refresh a second.
void showWorking(const String& heard, const std::vector<String>& activity, const String& status, uint32_t, int) {
  String seen = heard + status + String((int)activity.size());
  if (!activity.empty()) seen += activity.back();
  if (drawing(Screen::Working) == Screen::Working && seen == workSeen && millis() - workPushMs < 5000) return;
  workSeen = seen;
  workPushMs = millis();
  canvas.fillScreen(kPaper);
  drawBar();
  const int bottom = H - 2 - kTapReserve - (status.length() ? 15 : 0);
  int y = BAR + 3;
  if (heard.length()) {  // the end of a long sentence matters most
    const std::vector<String> ls = wrap(String("“") + heard + "”", W - 2 * PAD);
    const int maxLines = activity.empty() ? (bottom - y) / ts().line : 3;
    for (int i = max(0, (int)ls.size() - maxLines); i < (int)ls.size(); ++i, y += ts().line) {
      canvas.setFont(ts().body);
      text(ls[i], PAD, y);
    }
    y += 3;
    if (!activity.empty()) canvas.drawFastHLine(PAD, y++, W - 2 * PAD, kInk);
    y += 2;
  }
  canvas.setFont(ts().small);
  const int room = max(0, (bottom - y) / ts().smallLine);
  const int first = max(0, (int)activity.size() - room);
  for (int i = first; i < (int)activity.size(); ++i, y += ts().smallLine) {
    const bool newest = i == (int)activity.size() - 1;
    text(fit((newest ? "→ " : "  ") + activity[i], W - 2 * PAD), PAD, y);
  }
  if (status.length()) {
    canvas.setFont(kSmall);
    text(fit(status, W - 2 * PAD), PAD, bottom + 1);
  }
  drawTapButton();
  push();
}

// ── the answer ─────────────────────────────────────────────────────────

static int linesPerPage() { return (H - BAR - PAD - kTapReserve) / ts().line; }
static int pagesNow() {
  const int per = linesPerPage();
  return max(1, ((int)lines.size() + per - 1) / per);
}

// Where each page's text ends (the lines are the text's pieces, in order).
static void indexPages() {
  pageEnds.clear();
  const char* all = replyText.c_str();
  const int per = linesPerPage();
  int cursor = 0;
  for (int i = 0; i < (int)lines.size(); ++i) {
    const char* p = lines[i].length() ? strstr(all + cursor, lines[i].c_str()) : nullptr;
    cursor = (p ? (int)(p - all) : cursor) + (int)lines[i].length();
    if ((i + 1) % per == 0 || i + 1 == (int)lines.size()) pageEnds.push_back(cursor);
  }
  if (pageEnds.empty()) pageEnds.push_back(0);
}

static int pageOf(int at) {
  for (int i = 0; i < (int)pageEnds.size(); ++i) {
    if (at < pageEnds[i]) return i;
  }
  return (int)pageEnds.size() - 1;
}

void setQuestion(const String&) {}
bool canPin() { return false; }
bool pinned() { return false; }
void pin(bool) {}
bool restoreAnswer() { return false; }

void setReply(const String& t) {
  replyText = t;
  lines = wrap(t, W - 2 * PAD - 4);
  page = 0;
  spokenAt = -1;
  autoFollow = true;
  indexPages();
}

void growReply(const String& t) {  // more of it, in place: the page stays
  replyText = t;
  lines = wrap(t, W - 2 * PAD - 4);
  indexPages();
  page = min(page, pagesNow() - 1);
}

bool setSpoken(int at) {
  if (at < 0) {
    spokenAt = -1;
    return false;
  }
  spokenAt = min(at, (int)replyText.length());
  if (!autoFollow) return false;
  const int p = pageOf(spokenAt);
  if (p == page) return false;
  page = p;  // the voice is on another page
  return true;
}

void followVoice(int at) {
  autoFollow = true;
  spokenAt = min(max(0, at), (int)replyText.length());
  page = pageOf(spokenAt);
}

void setTextSize(uint8_t size) {
  sizeIdx = size < sizeof(kSizes) / sizeof(kSizes[0]) ? size : 0;
  if (!replyText.length()) return;
  lines = wrap(replyText, W - 2 * PAD - 4);  // re-wrap for the new size, keeping the place
  indexPages();
  page = autoFollow && spokenAt >= 0 ? pageOf(spokenAt) : min(page, pagesNow() - 1);
}

bool hasReply() { return !lines.empty(); }

void showReply() {
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  drawBar();
  canvas.setFont(ts().body);
  const int per = linesPerPage();
  const int pages = pagesNow();
  const int first = page * per;
  for (int i = 0; i < per && first + i < (int)lines.size(); ++i) {
    text(lines[first + i], PAD, BAR + PAD - 1 + i * ts().line);
  }
  if (pages > 1) {  // where it is, on the right edge
    const int track = H - BAR - 4 - kTapReserve;
    const int thumb = max(8, track / pages);
    const int ty = BAR + 2 + (track - thumb) * page / (pages - 1);
    canvas.drawFastVLine(W - 2, BAR + 2, track, kInk);
    canvas.fillRect(W - 3, ty, 3, thumb, kInk);
  }
  drawTapButton();
  push();
}

bool nextPage() {
  autoFollow = false;  // paged by hand: the voice no longer turns the pages
  page = (page + 1) % pagesNow();
  showReply();
  return page != 0;
}

bool prevPage() {
  autoFollow = false;
  if (page == 0) return false;
  --page;
  showReply();
  return true;
}

bool onLastPage() { return page + 1 >= pagesNow(); }
bool onFirstPage() { return page == 0; }
void setReplayable(bool) {}
bool replayHit(int, int) { return false; }

// ── panels, lists ──────────────────────────────────────────────────────

// Rows with the label over the value at the large size, beside it otherwise;
// values wrap; a row never splits across pages.
struct PanelSize {
  const lgfx::U8g2font* font;
  int line;
};
static const PanelSize kPanelSizes[] = {
    {&efont::efont12, 14},
    {&efont::efont14, 16},
    {&efont::efont16, 18},
};

int showPanel(const char* title, uint16_t, const std::vector<Row>& rows, int pageNo) {
  drawing(Screen::Other);
  const PanelSize& ps = kPanelSizes[sizeIdx];
  const bool stacked = sizeIdx == 2;
  const int top = BAR + 3, bottom = H - 2;
  const int titleH = title && *title ? 22 : 0;
  canvas.setFont(ps.font);
  int labelW = 0;
  if (!stacked) {
    for (const Row& r : rows) labelW = max(labelW, (int)tw(r.label));
    labelW = min(labelW + 6, W / 2);
  }
  const int valueW = stacked ? W - 2 * PAD : W - 2 * PAD - labelW;
  struct Laid {
    std::vector<String> lines;
    int h;
  };
  std::vector<Laid> laid;
  for (const Row& r : rows) {
    Laid l{wrap(r.value.length() ? r.value : String("-"), valueW, ps.font), 0};
    l.h = (int)l.lines.size() * ps.line + (stacked ? 18 : 2);
    laid.push_back(l);
  }
  std::vector<int> pageStart{0};
  int y = top + titleH;
  for (int i = 0; i < (int)laid.size(); ++i) {
    if (y + laid[i].h > bottom && i > pageStart.back()) {
      pageStart.push_back(i);
      y = top;
    }
    y += laid[i].h;
  }
  const int pages = pageStart.size();
  pageNo = constrain(pageNo, 0, pages - 1);

  canvas.fillScreen(kPaper);
  drawBar();
  y = top;
  if (pageNo == 0 && titleH) y = band(title) + 3;
  const int end = pageNo + 1 < pages ? pageStart[pageNo + 1] : (int)laid.size();
  for (int i = pageStart[pageNo]; i < end; ++i) {
    if (stacked) {
      canvas.setFont(kSmall);
      text(rows[i].label, PAD, y);
      y += 14;
    } else {
      canvas.setFont(ps.font);
      text(rows[i].label, PAD, y);
    }
    canvas.setFont(ps.font);
    for (const String& line : laid[i].lines) {
      if (y + ps.line > bottom + 2) break;  // a row taller than a page: cut it
      text(line, stacked ? PAD : PAD + labelW, y);
      y += ps.line;
    }
    y += stacked ? 4 : 2;
  }
  if (pages > 1) {  // page dots, bottom right
    for (int k = 0; k < pages; ++k) {
      const int x = W - 6 - (pages - 1 - k) * 8;
      if (k == pageNo) canvas.fillCircle(x, H - 5, 3, kInk);
      else canvas.drawCircle(x, H - 5, 2, kInk);
    }
  }
  push();
  return pages;
}

int showStatus(const Status& s, int pageNo) { return showPanel("", 0, statusRows(s), pageNo); }

void showPicker(const std::vector<Item>& items, int selected, int current_, const char*) {
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  drawBar();
  canvas.setFont(ts().body);
  const int rowH = ts().line + 2;
  const int per = (H - BAR - 2) / rowH;
  const int first = max(0, min(selected - per / 2, (int)items.size() - per));
  for (int i = 0; i < per && first + i < (int)items.size(); ++i) {
    const int idx = first + i;
    const int y = BAR + 1 + i * rowH;
    const bool sel = idx == selected;
    if (sel) canvas.fillRect(0, y, W, rowH, kInk);
    text(fit((idx == current_ ? "● " : "") + items[idx].label, W - 2 * PAD), PAD, y + 1, top_left, sel ? kPaper : kInk);
  }
  push();
}

// ── the character (only on the phone page's say-so; not on paper by default) ──

void showFace(uint8_t kind, face::Pose pose, int level) {
  drawing(Screen::Other);
  canvas.fillScreen(kInk);
  face::draw(canvas, kind, pose, level, millis(), W / 2, H / 2);
  push();
}

void showCharPick(uint8_t kind, face::Pose pose, int level, int) { showFace(kind, pose, level); }

// ── pictures ───────────────────────────────────────────────────────────

void storeImage(uint8_t* jpg, size_t len) {
  if (imgBuf) free(imgBuf);
  imgBuf = jpg;
  imgLen = jpg ? len : 0;
}

bool hasImage() { return imgLen > 0; }
void toggleImageFit() {}
void setBackground(const uint8_t*, size_t) {}  // a round screen's
bool hasBackground() { return false; }
void showBackdrop() {}

static bool jpegSize(const uint8_t* d, size_t len, int* w, int* h) {
  size_t i = 2;
  while (i + 9 < len) {
    if (d[i] != 0xFF) return false;
    const uint8_t m = d[i + 1];
    const size_t seg = (d[i + 2] << 8) | d[i + 3];
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      *h = (d[i + 5] << 8) | d[i + 6];
      *w = (d[i + 7] << 8) | d[i + 8];
      return *w > 0 && *h > 0;
    }
    i += 2 + seg;
  }
  return false;
}

// Floyd–Steinberg to pure black and white, on the whole canvas.
static void dither() {
  uint16_t* buf = (uint16_t*)canvas.getBuffer();
  int16_t* err = (int16_t*)calloc(2 * (W + 2), sizeof(int16_t));
  if (!buf || !err) {
    free(err);
    return;
  }
  int16_t* cur = err;
  int16_t* nxt = err + W + 2;
  for (int y = 0; y < H; ++y) {
    memset(nxt, 0, (W + 2) * sizeof(int16_t));
    const bool ltr = (y & 1) == 0;
    for (int k = 0; k < W; ++k) {
      const int x = ltr ? k : W - 1 - k;
      const uint16_t c = (uint16_t)((buf[y * W + x] >> 8) | (buf[y * W + x] << 8));
      const int grey = ((((c >> 11) & 31) * 255 / 31) * 299 + (((c >> 5) & 63) * 255 / 63) * 587 +
                        ((c & 31) * 255 / 31) * 114) / 1000;
      const int v = grey + cur[x + 1] / 16;
      const int out = v < 128 ? 0 : 255;
      buf[y * W + x] = out ? kPaper : kInk;
      const int e = constrain(v - out, -255, 255);
      const int ahead = ltr ? 1 : -1;
      cur[x + 1 + ahead] += e * 7;
      nxt[x + 1 - ahead] += e * 3;
      nxt[x + 1] += e * 5;
      nxt[x + 1 + ahead] += e;
    }
    int16_t* swap = cur;
    cur = nxt;
    nxt = swap;
  }
  free(err);
}

void showStoredImage() {
  if (!imgLen) return;
  drawing(Screen::Other);
  canvas.fillScreen(kPaper);
  int w = 0, h = 0;
  if (jpegSize(imgBuf, imgLen, &w, &h)) {
    const float s = min(1.0f, min((float)W / w, (float)H / h));
    const int dw = (int)(w * s), dh = (int)(h * s);
    canvas.drawJpg(imgBuf, imgLen, (W - dw) / 2, (H - dh) / 2, dw, dh, 0, 0, s, s);
  }
  dither();
  push();
}

// ── asleep, and what happens by itself ─────────────────────────────────

void showSleeping(const String& hint, uint8_t) {
  drawing(Screen::Sleeping);
  canvas.fillScreen(kPaper);
  drawBar(false);  // (no links asleep: Wi-Fi is off)
  // A crescent, out of two circles.
  canvas.fillCircle(W / 2, 62, 26, kInk);
  canvas.fillCircle(W / 2 + 12, 54, 24, kPaper);
  struct tm t;
  char buf[32];
  int y = 100;
  if (localNow(&t)) {
    strftime(buf, sizeof(buf), "%a %-d %b", &t);
    canvas.setFont(&efont::efont16);
    text(buf, W / 2, y, top_center);
    y += 20;
  }
  const String room = roomLine();
  if (room.length()) {
    canvas.setFont(&efont::efont14);
    text(room, W / 2, y, top_center);
  }
  pill(hint, 3, H - 28, W - 6, 24);
#ifndef PREVIEW
  push(rtcPartials >= kSleepCleanAfter);
  flush();
#else
  push();
#endif
}

void tick() {
  if (current != Screen::Home) return;
  static uint32_t lastCheckMs = 0;
  if (millis() - lastCheckMs < 1000) return;
  lastCheckMs = millis();
  struct tm t;
  if (localNow(&t) && t.tm_min != homeMinute) showHome(lastHome);
#ifndef PREVIEW
  // A quiet moment to clear what the partial refreshes left: nobody's reading.
  if (rtcPartials >= kCleanAfter && millis() - lastUseMs > kCleanIdleMs && !busy()) push(true);
#endif
}

}  // namespace ui
