// PaperColor, the "B" way: a device with two lights and a voice, and a
// screen only because it's there. The lights carry listening, working,
// speaking and trouble (leds.cpp), the deck's voice carries the answers. The
// screen is a dashboard — the clock, the room, the battery, the voice notes
// kept offline — redrawn on every five-minute mark (awake, or on
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
static M5Canvas canvas(&M5.Display);
static lgfx::VLWfont fontSet[vlw::Count];
static lgfx::PointerWrapper fontData[vlw::Count];

static String barAgent, barState;
static Health hWifi = Down, hAgent = Down;

struct Rect {
  int x, y, w, h;
};
static std::vector<Rect> pictures;  // regions to error-diffuse in this frame
struct Duo {
  Rect r;
  uint16_t a, b;
};
static std::vector<Duo> duos;  // regions that stay between two inks: text on a colour

static void font(vlw::Id id) { canvas.setFont(&fontSet[id]); }
static int tw(const String& s) { return canvas.textWidth(s.c_str()); }

static void text(const String& s, int x, int y, uint16_t color, uint8_t datum = top_left) {
  canvas.setTextDatum(datum);
  canvas.setTextColor(color);
  canvas.drawString(s.c_str(), x, y);
}

// Text on a solid block: its edges stay between the two inks, with no third one
// (a blend of white and green would snap to yellow).
static void textOn(const String& s, int x, int y, uint16_t fg, uint16_t bg, uint8_t datum = top_left) {
  canvas.setTextDatum(datum);
  canvas.setTextColor(fg, bg);
  canvas.drawString(s.c_str(), x, y);
  const int w = tw(s), h = canvas.fontHeight(), across = datum & 3, down = (datum >> 2) & 3;
  duos.push_back({{x - (across == 1 ? w / 2 : across == 2 ? w : 0) - 1, y - (down == 1 ? h / 2 : down == 2 ? h : 0) - 1, w + 2, h + 2}, fg, bg});
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

// Word-wrap UTF-8 text in the current font, widthAt(n) px for the line of that
// number; words wider than a line break between characters. Paragraph breaks are kept.
static std::vector<String> wrap(const String& s, const std::function<int(size_t)>& widthAt) {
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
      if (tw(trial) <= widthAt(out.size())) {
        line = trial;
        continue;
      }
      if (line.length()) out.push_back(line);
      line = "";
      while (word.length() && tw(word) > widthAt(out.size())) {
        int cut = 0, k = 0;
        while (k < (int)word.length()) {
          int step = utf8Step(word[k]);
          if (tw(word.substring(0, k + step)) > widthAt(out.size())) break;
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

static std::vector<String> wrap(const String& s, int width) {
  return wrap(s, [width](size_t) { return width; });
}

// At most `max` lines; the last one ends in "…" when there was more.
static std::vector<String> clamp(std::vector<String> lines, int max, int width) {
  if ((int)lines.size() <= max) return lines;
  lines.resize(max);
  lines.back() = fit(lines.back() + "…", width);
  return lines;
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
  // Text on a colour first: it's an ink after this, and the pass below leaves it.
  static const auto channels = [](uint16_t c, int* v) {
    v[0] = ((c >> 11) & 31) * 255 / 31, v[1] = ((c >> 5) & 63) * 255 / 63, v[2] = (c & 31) * 255 / 31;
  };
  for (const Duo& d : duos) {
    int a[3], b[3], v[3];
    channels(d.a, a);
    channels(d.b, b);
    const int x0 = max(0, d.r.x), x1 = min(W, d.r.x + d.r.w), y0 = max(0, d.r.y), y1 = min(H, d.r.y + d.r.h);
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        const int i = y * W + x;
        channels(px(buf, i), v);
        int toA = 0, toB = 0;
        for (int k = 0; k < 3; ++k) toA += (v[k] - a[k]) * (v[k] - a[k]), toB += (v[k] - b[k]) * (v[k] - b[k]);
        setPx(buf, i, toA <= toB ? d.a : d.b);
      }
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
// The panel's framebuffer, bus and pins (M5GFX keeps them protected).
struct PaperPanel : lgfx::Panel_ED2208 {
  int width() const { return _cfg.panel_width; }
  int height() const { return _cfg.panel_height; }
  const uint8_t* row(int y) const { return _lines_buffer[y]; }

  void cmd(uint8_t c) { _bus->writeCommand(c, 8); }
  void data(uint8_t d) { _bus->writeData(d, 8); }
  bool idle() const { return _cfg.pin_busy < 0 || lgfx::gpio_in(_cfg.pin_busy); }
  void waitBusy(uint32_t ms) {
    _bus->wait();
    uint32_t t0 = millis();
    while (!idle() && millis() - t0 < ms) delay(10);
  }
  void begin() {
    _bus->beginTransaction();
    cs_control(false);
  }
  void end() {
    _bus->wait();
    cs_control(true);
    _bus->endTransaction();
  }

  // The reset line alone (a GPIO: safe from any task — the main loop uses it
  // to cut a running waveform short, see tick()).
  static void pulse() {
    lgfx::gpio_hi(kResetPin);
    lgfx::pinMode(kResetPin, lgfx::pin_mode_t::output);
    lgfx::gpio_lo(kResetPin);
    delay(10);
    lgfx::gpio_hi(kResetPin);
    delay(10);
  }

  // A reset pulse, then the set-up. With the paper keeping its picture
  // through restarts, M5GFX starts it without one (init_without_reset), so
  // the controller ran on whatever state it had since power came; from a
  // reset every refresh is whole, and several times faster. The PaperColor's
  // EPD reset is G12 (M5GFX pulses it at start; the panel's reset pin is not it).
  void reset() {
    pulse();
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

  // A framebuffer pixel as the panel's 4-bit ink code: the nearest of the
  // six, as M5GFX's own refresh picks them (the frame is in the inks already).
  static uint8_t code(const lgfx::bgr888_t& p) {
    static constexpr uint8_t kCodes[6] = {0x0, 0x1, 0x2, 0x3, 0x5, 0x6};  // black white yellow red blue green
    uint32_t best = UINT32_MAX;
    uint8_t out = 0x1;
    for (int i = 0; i < 6; ++i) {
      const int dr = p.r - kInkDefs[i].r, dg = p.g - kInkDefs[i].g, db = p.b - kInkDefs[i].b;
      const uint32_t d = dr * dr + dg * dg + db * db;
      if (d < best) best = d, out = kCodes[i];
    }
    return out;
  }

  // The whole frame, as M5GFX's display() sends it — the pixels, power on,
  // the booster, the refresh — but with `*waveform` set while the panel runs
  // its waveform (the long part), so the main loop can cut it short with a
  // reset pulse when another screen is asked for, and the bus let go
  // meanwhile (the microSD card shares it). `white`: an all-white page
  // instead of the frame (the cleaning pass, see pushToPanel()).
  void refresh(volatile bool* waveform, bool white = false) {
    const int w = width(), h = height();
    static uint8_t line[200];
    begin();
    cmd(0x10);  // the pixels, two a byte
    if (white) memset(line, 0x11, w / 2);
    for (int y = 0; y < h; ++y) {
      if (!white) {
        const lgfx::bgr888_t* px = reinterpret_cast<const lgfx::bgr888_t*>(row(y));
        for (int x = 0; x < w; x += 2) line[x >> 1] = code(px[x]) << 4 | code(px[x + 1]);
      }
      _bus->writeBytes(line, w / 2, true, false);
    }
    _bus->wait();
    cmd(0x04);  // power on
    waitBusy(5000);
    delay(200);
    cmd(0x06);  // booster
    for (uint8_t d : {0x6F, 0x1F, 0x17, 0x27}) data(d);
    delay(200);
    cmd(0x12);  // refresh
    data(0x00);
    end();
    *waveform = true;
    for (uint32_t t0 = millis(); !idle() && millis() - t0 < 45000;) delay(10);
    *waveform = false;
    delay(200);
    begin();
    cmd(0x02);  // power off
    data(0x00);
    waitBusy(5000);
    delay(200);
    end();
  }
};

static PaperPanel* paperPanel() {
  if (M5.getBoard() != m5::board_t::board_M5PaperColor) return nullptr;
  return static_cast<PaperPanel*>(static_cast<lgfx::Panel_ED2208*>(M5.Display.getPanel()));
}

#endif

static volatile bool inWaveform = false;  // the panel is running a refresh's waveform
static volatile bool cleanNext = false;   // a white pass before this frame (commit(): cleanDue())
static volatile bool cutDone = false;     // tick() cut the waveform short
static void markCleaned();

// Every refresh starts from a reset controller, and runs as PaperPanel::
// refresh() (so it can be cut short); off the board, M5GFX's own. With
// cleanNext, the paper goes all white first, then to the frame: the inks
// driven to one end and back, so they keep their colour (see cleanDue()).
// Another screen asked for during the white pass cuts it short and skips
// the frame — the new screen draws next.
static void pushToPanel() {
  waitPanel(45000);
  M5.Display.startWrite();
  canvas.pushSprite(&M5.Display, 0, 0);
  M5.Display.endWrite();
#ifndef PREVIEW
  if (PaperPanel* p = paperPanel()) {
    cutDone = false;
    if (cleanNext) {
      cleanNext = false;
      p->reset();
      p->refresh(&inWaveform, true);
      if (cutDone) return;
      markCleaned();
    }
    p->reset();
    p->refresh(&inWaveform);
    return;
  }
#endif
  cleanNext = false;
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

enum class Kind : uint8_t { None, Dash, Reply, Link, Setup, Off };

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

// A white pass before the dashboard (pushToPanel()): six-ink paper that only
// ever goes from one picture to the next loses colour and keeps faint
// shadows of the pictures before. Once an hour — the first dashboard drawn
// on its mark in a new hour, normally :00 — and when it's powered on, with the
// first dashboard after that, whatever brought it. Not after any other restart
// (an update, a crash), and not for a dashboard someone asked for (back from
// the answer, a press): that one stays quick.
RTC_DATA_ATTR static bool rtcPowerOnPass = false;  // powered on, the pass still to come
RTC_DATA_ATTR static uint32_t rtcCleanAt = 0;      // by the clock
static constexpr uint32_t kCleanEverySec = 60 * 60;

// A new hour by the clock since the last pass. (A restart forgets that one, so it
// goes by the next new hour; with no clock there are no hours to go by.)
static bool cleanDue() {
  const uint32_t now = (uint32_t)time(nullptr);
  if (now < 1700000000) return false;
  if (rtcCleanAt < 1700000000) rtcCleanAt = now;
  return now / kCleanEverySec != rtcCleanAt / kCleanEverySec;
}

static void markCleaned() {
  rtcCleanAt = (uint32_t)time(nullptr);
  rtcPowerOnPass = false;
}

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
  bool onMark = false;  // the dashboard on its five-minute mark (not asked for)
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

static bool cutShort = false;  // a different screen was asked for while one draws

static void draw(Kind kind, std::function<void()> fn, bool onMark = false) {
  want.kind = kind;
  want.draw = std::move(fn);
  want.active = true;
  want.onMark = onMark;
  if (refreshing && kind != Kind::Reply && kind != shownKind) cutShort = true;  // (pages: askReplyPage)
}

static void commit() {
  Want w = std::move(want);
  want.active = false;
  pictures.clear();
  duos.clear();
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
  cleanNext = w.kind == Kind::Dash && (rtcPowerOnPass || (w.onMark && cleanDue()));
  keepShown();
  startRefresh();
}

static void dashTick();

void tick() {
#ifndef PREVIEW
  // Another screen wanted while one draws: its waveform is cut short with a
  // reset pulse, and the new one starts at once rather than after it. Only
  // during the waveform — mid-transfer the frame would be half-sent.
  if (refreshing && cutShort && inWaveform) {
    PaperPanel::pulse();
    cutShort = false;
    cutDone = true;
  }
#endif
  if (refreshing) return;
  cutShort = false;
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
// open (the options page closes when its screen is left; setup restarts), or
// the capsized robot, switched on again.
static bool dashDue() {
  const bool stale = everShown && (shownKind == Kind::Link || shownKind == Kind::Setup || shownKind == Kind::Off);
  return stale || dashboardDueInSec() == 0;
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

// `t` formatted in local time; false before the clock was ever set.
static bool clockText(char* out, size_t n, const char* fmt, time_t t) {
  if (t < 1700000000) return false;
  struct tm tm;
  localtime_r(&t, &tm);
  strftime(out, n, fmt, &tm);
  return true;
}

// ── what the buttons do, drawn beside them ─────────────────────────────
//
// No legend: each button's icon is on the screen where the button is — A and B
// on the left edge, C on top, in the middle. Not smoothed: a blend of two inks
// would snap to a third.

enum class Key : uint8_t { None, Talk, Answer, Options, Back, Next, Home, Update };
static constexpr int kKeyX = 34, kKeyAY = 66, kKeyBY = 141, kKeyCY = 28;
static constexpr int kKeysBottom = 172;  // what's beside A and B ends here: below it, the whole width
static constexpr int kKeysRight = 68;    // and beside them, this is free

// The other ink for something drawn on `around`.
static uint16_t inkOn(uint16_t around) { return around == kPaper || around == kYellow ? kInk : kPaper; }

// A microphone, its middle on (cx, cy), `h` px tall.
static void drawMic(int cx, int cy, int h, uint16_t ink) {
  const auto U = [h](int u) { return (int)lroundf(u * h / 79.0f); };  // the drawing is 79 tall
  const int top = cy - h / 2, r = U(17), mid = top + U(26);
  canvas.fillRoundRect(cx - r, top, 2 * r, U(51), r, ink);
  canvas.fillArc(cx, mid, U(28), U(23), 0, 180, ink);
  const int stem = max(2, U(6));
  canvas.fillRect(cx - stem / 2, top + U(59), stem, U(16), ink);
  canvas.fillRoundRect(cx - r, top + h - stem, 2 * r, stem, stem / 2, ink);
}

// Captain, in the six inks: the app icon's drawing, (cx, cy) the middle of his
// head, `s` px a unit (he's 108 wide, 86 above that middle and 40 below it).
// `around`: what he's drawn on. Asleep, his eyes are shut.
static void drawRobot(int cx, int cy, float s, uint16_t around, bool asleep = false) {
  const auto X = [&](float u) { return cx + (int)lroundf(u * s); };
  const auto Y = [&](float u) { return cy + (int)lroundf(u * s); };
  const auto L = [&](float u) { return max(1, (int)lroundf(u * s)); };
  const uint16_t edge = inkOn(around);
  canvas.fillRect(X(0) - L(1.5f), Y(-78), 2 * L(1.5f), L(12), edge);  // antenna, lamp
  canvas.fillCircle(X(0), Y(-81), L(6), around == kRed ? kYellow : kRed);
  canvas.fillRoundRect(X(-52), Y(-12), L(104), L(24), L(3), edge);    // ears
  canvas.fillRoundRect(X(-44), Y(-36), L(88), L(76), L(14), edge);    // head
  canvas.fillRoundRect(X(-41), Y(-33), L(82), L(70), L(11), kPaper);
  canvas.fillRoundRect(X(-36), Y(-28), L(72), L(58), L(9), kInk);     // face plate
  canvas.fillRoundRect(X(-46), Y(-68), L(92), L(28), L(10), kBlue);   // cap, band, badge
  canvas.fillRect(X(-46), Y(-44), L(92), L(7), kYellow);
  canvas.fillRoundRect(X(-50), Y(-38), L(100), L(6), L(3), kInk);     // (its peak)
  canvas.fillCircle(X(0), Y(-56), L(7), kYellow);
  canvas.fillCircle(X(0), Y(-56), L(3), kBlue);
  for (float x : {-26.0f, 10.0f}) {                                   // eyes, grille
    if (asleep) canvas.fillRect(X(x), Y(-10), L(16), L(3), kPaper);
    else canvas.fillRoundRect(X(x), Y(-16), L(16), L(12), L(3), kPaper);
  }
  static constexpr uint8_t kBars[7] = {5, 7, 9, 10, 9, 7, 5};
  for (int i = 0; i < 7; ++i) {
    const int h = L(asleep ? 2 : kBars[i]);
    canvas.fillRect(X(-24 + i * 7), Y(16) - h / 2, L(5), h, kPaper);
  }
}

static void drawGear(int cx, int cy, int r, uint16_t ink, uint16_t around) {
  for (int k = 0; k < 8; ++k) {
    const float a = k * 3.14159f / 4, ca = cosf(a), sa = sinf(a), w = r * 0.25f, in = r * 0.6f;
    const int x0 = cx + lroundf(ca * in - sa * w), y0 = cy + lroundf(sa * in + ca * w);
    const int x1 = cx + lroundf(ca * in + sa * w), y1 = cy + lroundf(sa * in - ca * w);
    const int x2 = cx + lroundf(ca * r + sa * w), y2 = cy + lroundf(sa * r - ca * w);
    const int x3 = cx + lroundf(ca * r - sa * w), y3 = cy + lroundf(sa * r + ca * w);
    canvas.fillTriangle(x0, y0, x1, y1, x2, y2, ink);
    canvas.fillTriangle(x0, y0, x2, y2, x3, y3, ink);
  }
  canvas.fillCircle(cx, cy, r * 3 / 4, ink);
  canvas.fillCircle(cx, cy, r * 3 / 8, around);
}

// An arrow `size` px long down the page, or to the left.
static void drawArrow(int cx, int cy, int size, bool left, uint16_t ink) {
  const int half = size / 2, head = size * 11 / 20, w = size * 3 / 20;
  if (left) {
    canvas.fillTriangle(cx - half, cy, cx - half + head, cy - head, cx - half + head, cy + head, ink);
    canvas.fillRect(cx - half + head, cy - w, size - head, 2 * w, ink);
  } else {
    canvas.fillTriangle(cx, cy + half, cx - head, cy + half - head, cx + head, cy + half - head, ink);
    canvas.fillRect(cx - w, cy - half, 2 * w, size - head, ink);
  }
}

static void drawKey(Key key, int cx, int cy, uint16_t around) {
  const uint16_t ink = inkOn(around);
  switch (key) {
    case Key::Talk: drawMic(cx, cy, 44, ink); break;
    case Key::Answer: drawRobot(cx, cy + 9, 0.38f, around); break;
    case Key::Options: drawGear(cx, cy, 17, ink, around); break;
    case Key::Back: drawArrow(cx, cy, 40, true, ink); break;
    case Key::Next: drawArrow(cx, cy, 40, false, ink); break;
    case Key::Home:
      canvas.fillTriangle(cx - 22, cy - 2, cx + 22, cy - 2, cx, cy - 22, ink);
      canvas.fillRect(cx - 15, cy - 2, 30, 22, ink);
      canvas.fillRect(cx - 5, cy + 7, 10, 13, around);
      break;
    case Key::Update:
      drawArrow(cx, cy - 4, 26, false, ink);
      canvas.fillRect(cx - 14, cy + 12, 28, 4, ink);
      break;
    case Key::None: break;
  }
}

static void drawKeys(Key a, Key b, Key c, uint16_t around = kPaper) {
  drawKey(a, kKeyX, kKeyAY, around);
  drawKey(b, kKeyX, kKeyBY, around);
  drawKey(c, W / 2, kKeyCY, around);
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
  if (esp_reset_reason() == ESP_RST_POWERON) rtcPowerOnPass = true;
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

// "Monday, 19 October", or "Mon 19 Oct" when that's wider than `w` (the current
// font); "" before the clock is set.
static String dateWords(int w) {
  char date[40];
  if (!clockText(date, sizeof(date), "%A, %e %B", time(nullptr))) return "";
  String d = date;
  d.replace("  ", " ");
  if (tw(d) > w && clockText(date, sizeof(date), "%a %e %b", time(nullptr))) {
    d = date;
    d.replace("  ", " ");
  }
  return fit(d, w);
}

// The battery as words, in `w` px of the current font: how long it has, or
// until it's full.
static String powerWords(bool chg, int w) {
  String s;
  if (chg) {
    const int m = battery::minutesToFull();
    if (m == 0) return "full";
    if (m < 0) return "charging";
    s = "full in " + timeSpan(m);
    if (tw(s) > w) s = timeSpan(m);
  } else if (telemetry::onUsbPower()) {
    return "on USB";
  } else {
    const int m = battery::minutesLeft();
    if (m < 0) return "on battery";
    s = timeSpan(m);
  }
  if (tw(s) > w) s.replace(" min", "");
  return fit(s, w);
}

static String clockLength(uint32_t secs) {
  char len[12];
  snprintf(len, sizeof(len), "%lu:%02lu", (unsigned long)(secs / 60), (unsigned long)(secs % 60));
  return len;
}

static Home lastHome;  // what the main loop last knew of the links (its rows: Wi-Fi, deck, agent)
static bool haveHome = false;
static int waitingCount = 0;  // proposals waiting in Flight Deck, as last heard (setWaiting)

// A line of handwriting: a wavering stroke, a gap, a wavering stroke. (Plain
// lines: a smoothed black on yellow would snap to green.)
static void scribble(int x, int y, int len, int seed) {
  static constexpr int kWords[] = {24, 14, 30, 18, 12, 26, 20};
  for (int done = 0, word = 0; done < len; ++word) {
    const int n = min(len - done, kWords[(seed + word) % 7]);
    int was = y + lroundf(2 * sinf((done + seed * 3) * 0.8f));
    for (int i = 3; i <= n; i += 3) {
      const int now = y + lroundf(2 * sinf((done + i + seed * 3) * 0.8f));
      for (int t = 0; t < 2; ++t) canvas.drawLine(x + done + i - 3, was + t, x + done + i, now + t, kInk);
      was = now;
    }
    done += n + 8;
  }
}

// A yellow notepad on the black tile: rings along the top, handwriting on the sheet.
static void drawNotepad(int x, int y, int w, int h) {
  const int ringH = constrain(h / 6, 10, 22), ringW = constrain(w / 16, 4, 8), top = y + ringH / 2;
  canvas.fillRoundRect(x, top, w, h - ringH / 2, 6, kYellow);
  const int margin = max(6, w / 10), rowW = w - 2 * margin;
  const int first = top + ringH / 2 + max(9, h / 7), last = y + h - max(9, h / 7);
  for (int i = 0; i < 3; ++i) scribble(x + margin, first + i * (last - first) / 2, i == 2 ? rowW * 55 / 100 : rowW, i * 2 + 1);
  const int inset = max(9, w / 8);
  for (int i = 0; i < 5; ++i) {
    const int rx = x + inset + i * (w - 2 * inset) / 4;
    canvas.fillRoundRect(rx - ringW / 2 - 2, y - 2, ringW + 4, ringH + 4, ringW / 2 + 2, kInk);
    canvas.fillRoundRect(rx - ringW / 2, y, ringW, ringH, ringW / 2, kPaper);
  }
}

// How many notes there are, on the notepad's corner: it grows to the left, over the pad.
static void drawNoteCount(int right, int cy, int notes) {
  const String n = notes > 99 ? String("99+") : String(notes);
  font(vlw::Title);
  static constexpr int kH = 46;
  const int w = max(kH, tw(n) + 22), x = right - w;
  canvas.fillRoundRect(x - 3, cy - kH / 2 - 3, w + 6, kH + 6, kH / 2 + 3, kInk);
  canvas.fillRoundRect(x, cy - kH / 2, w, kH, kH / 2, kRed);
  textOn(n, x + w / 2, cy, kPaper, kRed, middle_center);
}

// The top block's first line: the links' dots and the agent (asleep: only that
// it's asleep, nothing is connected then) up to C's icon, what waits after it.
static void drawStatusLine(int x, int w, int y, uint16_t bg, bool asleep) {
  int left = x;
  if (!asleep) {
    for (const Health h : {hWifi, hAgent}) {  // white around them: a red one still shows on red
      canvas.fillCircle(left + 8, y, 8, kPaper);
      canvas.fillCircle(left + 8, y, 6, healthInk(h));
      left += 20;
    }
    left += 4;
  }
  font(vlw::Small);
  const String name = asleep ? String("Asleep") : barAgent.length() ? barAgent : String("Captain");
  textOn(fit(name, W / 2 - 26 - left), left, y, kPaper, bg, middle_left);
  if (waitingCount > 0) {
    font(vlw::Small);
    const String t = String(waitingCount) + " waiting";
    const int pw = min(tw(t) + 18, x + w - W / 2 - 26), px = x + w - pw;
    const uint16_t fill = bg == kRed ? kPaper : kRed, fg = bg == kRed ? kRed : kPaper;
    canvas.fillRoundRect(px, y - 14, pw, 28, 14, fill);
    textOn(t, px + pw / 2, y, fg, fill, middle_center);
  }
}

// One of the three blocks: a label, a big number with its unit, a line under it.
struct StatBlock {
  int x, w;
  uint16_t fill, fg;
  const char* label;
  String big;
  const char* unit;  // "o": degrees
};

static void drawStatBlock(const StatBlock& b, int y, int h) {
  canvas.fillRoundRect(b.x, y, b.w, h, 14, b.fill);
  font(vlw::Small);
  textOn(b.label, b.x + 9, y + 7, b.fg, b.fill);
  font(vlw::Badge);
  textOn(b.big, b.x + 7, y + 28, b.fg, b.fill);
  const int ux = b.x + 7 + tw(b.big) + 2;
  if (b.unit[0] == 'o') {  // (the big font has no ° of its own)
    canvas.fillCircle(ux + 5, y + 44, 5, b.fg);
    canvas.fillCircle(ux + 5, y + 44, 2, b.fill);
  } else if (b.unit[0]) {
    font(vlw::Small);
    if (ux + tw(b.unit) <= b.x + b.w - 3) textOn(b.unit, ux, y + 36, b.fg, b.fill);
  }
}

// What the talk button does, in a big tile: the mic and whom it talks to, offline
// too, where a note is kept. Note mode: black, a notepad, how many there are and
// the room left.
static void drawTile(int x, int y, int w, int h, bool asleep, int notes, const String& kept, const String& room) {
  canvas.fillRoundRect(x, y, w, h, 16, kInk);
  if (lastHome.noteMode) {
    const int padH = min(128, h - 40), padW = padH * 3 / 4, padX = x + 16, padY = y + (h - padH) / 2;
    drawNotepad(padX, padY, padW, padH);
    drawNoteCount(padX + padW + 12, padY + padH - 22, notes);
    const int tx = padX + padW + 26, right = x + w - 10, top = y + (h - 112) / 2;
    font(vlw::Title);
    textOn("Hold", tx, top, kPaper, kInk);
    const int micX = tx + tw("Hold") + 19;
    drawMic(micX, top + 19, 36, kPaper);
    font(vlw::Title);
    textOn("to send", micX + 19, top, kPaper, kInk);
    font(vlw::BodyS);
    textOn(fit(kept, right - tx), tx, top + 46, kPaper, kInk);
    textOn(fit(room, right - tx), tx, top + 80, kPaper, kInk);
    return;
  }
  canvas.fillRoundRect(x + 4, y + 4, w - 8, h - 8, 13, kPaper);
  const float s = min(1.4f, (h - 44) / 126.0f);  // Captain's 126 units
  const int mid = y + h / 2;
  drawMic(x + w / 4 + 6, mid, (int)(126 * s * 0.92f), kInk);
  drawRobot(x + w * 3 / 4 - 6, mid + (int)(23 * s), s, kPaper, asleep);
}

// The links, on the options screen, a row each: a dot in the ink of their state,
// the name, the value.
static constexpr int kLinkRowH = 28;

static void drawLinks(int x, int y, int w) {
  font(vlw::Small);
  static constexpr int kRowH = kLinkRowH;
  auto row = [&](const String& label, const String& value, uint16_t ink) {
    canvas.fillCircle(x + 7, y + kRowH / 2, 6, ink);
    if (ink == kYellow) canvas.drawCircle(x + 7, y + kRowH / 2, 6, kInk);  // (faint on paper by itself)
    text(label, x + 22, y + kRowH / 2, kInk, middle_left);
    const int left = 22 + tw(label) + 12;
    text(fit(value, w - left), x + w, y + kRowH / 2, ink == kRed || ink == kBlue ? ink : kInk, middle_right);
    y += kRowH;
  };
  if (!haveHome || lastHome.rows.size() < 2) {
    row("Wi-Fi", hWifi == Up ? "up" : hWifi == Partial ? "connecting…" : "not connected", healthInk(hWifi));
    row("Agent", hAgent == Up ? "linked" : hAgent == Asleep ? "asleep" : hAgent == Off ? "stopped" : "no agent yet", healthInk(hAgent));
    return;
  }
  for (size_t i = 1; i < lastHome.rows.size() && i < 5; ++i) {
    const Row& r = lastHome.rows[i];
    if (String(r.label) != "Waiting") row(r.label, r.value, snap(r.color));  // (the pill on top says that)
  }
}

// The dashboard, in blocks of colour: the clock as of this drawing, the room, the
// battery, the notes, what the talk button does; drawn every five minutes and on
// request. Note mode shows at a glance: the top block turns red.
static void drawDashboard(bool asleep) {
  static constexpr int M = 6, G = 6, kFullW = 400 - 2 * M;
  int y = M;
  const uint16_t top = lastHome.noteMode ? kRed : kInk;
  static constexpr int kTopH = 224;
  canvas.fillRoundRect(M, y, kFullW, kTopH, 16, top);
  drawStatusLine(M + 16, kFullW - 32, kKeyCY, top, asleep);
  drawKeys(Key::Talk, Key::Answer, Key::Options, top);
  char clock[8] = "--:--";
  clockText(clock, sizeof(clock), "%H:%M", time(nullptr));
  font(vlw::Digits);
  textOn(clock, W / 2 + 10, y + 56, kPaper, top, top_center);  // (A's and B's icons on its left)
  font(vlw::Title);
  const String date = dateWords(kFullW - 32);
  if (date.length()) textOn(date, W / 2, y + 168, kPaper, top, top_center);
  y += kTopH + G;

  // The three blocks, each as wide as what it says.
  static constexpr int kStatH = 122, kRoomW = 102, kPowerW = 126, kNotesW = 136;
  const int powerX = M + kRoomW + G, notesX = powerX + kPowerW + G;
  float t, rh;
  const bool hasRoom = telemetry::climate(&t, &rh);
  drawStatBlock({M, kRoomW, kBlue, kPaper, "Room", hasRoom ? String((int)lroundf(t)) : String("--"), hasRoom ? "o" : ""}, y, kStatH);
  if (hasRoom) {  // a water drop, as the word doesn't fit
    const int cy = y + 105;
    canvas.fillCircle(M + 16, cy + 2, 7, kPaper);
    canvas.fillTriangle(M + 16, cy - 11, M + 10, cy - 1, M + 22, cy - 1, kPaper);
    font(vlw::Small);
    textOn(String((int)lroundf(rh)) + "%", M + 28, y + 92, kPaper, kBlue);
  }

  bool chg = false;
  const int pct = batteryPct(&chg);
  const uint16_t powerFill = pct < 0 ? kInk : pct < 20 ? kRed : pct < 40 ? kYellow : kGreen;
  const uint16_t powerFg = powerFill == kYellow ? kInk : kPaper;
  drawStatBlock({powerX, kPowerW, powerFill, powerFg, "Power", pct >= 0 ? String(pct) : String("?"), pct >= 0 ? "%" : ""}, y, kStatH);
  if (chg) {  // a bolt, in the label's row
    const int bx = powerX + kPowerW - 26, by = y + 19;
    canvas.fillTriangle(bx + 9, by - 10, bx + 1, by + 2, bx + 8, by + 2, powerFg);
    canvas.fillTriangle(bx + 6, by - 1, bx + 14, by - 1, bx + 5, by + 11, powerFg);
  }
  font(vlw::Small);
  textOn(powerWords(chg, kPowerW - 16), powerX + 8, y + 92, powerFg, powerFill);

  const int n = notes::count();
  const int roomMin = (int)(notes::room(kNoteRate) / kNoteRate / 60);
  char gb[24];
  const double freeGb = notes::freeBytes() / 1e9;
  snprintf(gb, sizeof(gb), freeGb < 10 ? "%.1f GB free" : "%.0f GB free", freeGb);
  const String room = !notes::ready() ? String("no room")
                      : roomMin < 1    ? String("full")
                      : notes::onCard() ? String(gb)
                                        : String(roomMin) + " min free";
  const uint16_t notesFill = roomMin < 2 ? kRed : kYellow, notesFg = notesFill == kRed ? kPaper : kInk;
  drawStatBlock({notesX, kNotesW, notesFill, notesFg, "Notes", String(n), ""}, y, kStatH);
  font(vlw::Small);
  textOn(fit(room, kNotesW - 16), notesX + 8, y + 92, notesFg, notesFill);
  y += kStatH + G;

  const String kept = n == 0 ? String("no notes yet") : String(n) + (n == 1 ? " note · " : " notes · ") + clockLength(notes::seconds());
  // Asleep, a line under it says that it's redrawn all the same.
  static constexpr int kAsleepLineH = 32;
  const int tileBottom = H - M - (asleep ? kAsleepLineH : 0);
  drawTile(M, y, kFullW, tileBottom - y, asleep, n, kept, room);
  if (asleep) {
    font(vlw::Small);
    text("Redrawn every five minutes", W / 2, tileBottom + kAsleepLineH / 2, kInk, middle_center);
  }
}

static void askDashboard(bool asleep, bool onMark = false) {
  draw(Kind::Dash, [=] { drawDashboard(asleep); }, onMark);
}

// On each five-minute mark, while the dashboard is on the paper: the whole
// page again (a reset controller makes a full refresh quick).
static bool held = false;  // a recording is going on (hold())

void hold(bool recording) { held = recording; }

static void dashTick() {
  if (held) return;  // drawn once the recording ends
  if (shownKind == Kind::Dash && !viewing && !answerDue && dashDue()) askDashboard(false, true);
}

// The dashboard on its five-minute mark, or when it was asked for (drawNow).
static void dashboardIfDue() {
  // (Back from the robot it's as good as asked for.)
  const bool now = (forceNext && millis() - forceAtMs < 3000) || (everShown && shownKind == Kind::Off);
  forceNext = false;
  if (now || (dashDue() && !held)) askDashboard(false, !now);
}

void setWaiting(int count) { waitingCount = count; }

#ifdef PREVIEW
void previewAsleep() { askDashboard(true); }  // (asleep it goes by the clock's marks: the Mac's real clock there)
#endif

void setHome(const Home& home) {
  lastHome = home;
  haveHome = true;
}

void showHome(const Home& home) {
  setHome(home);
  dashboardIfDue();
}

// Going to sleep, and the timer wake while asleep: the dashboard when its
// mark has come (the sleep timer is set to land on the next), else nothing.
void showSleeping(const String&) {
  forceNext = false;
  if (dashDue()) askDashboard(true, true);
  flush();
}

// ── switched off ───────────────────────────────────────────────────────

// A drawing made upright on its own canvas, then turned onto the page. Not
// smoothed: a blend of two inks would snap to a third (yellow and black: green).
template <typename Fn>
static void drawTurned(int w, int h, int x, int y, float angle, Fn fn) {
  M5Canvas c(&canvas);
  c.setPsram(true);
  c.setColorDepth(16);
  if (!c.createSprite(w, h)) return;
  c.fillScreen(kPaper);
  fn(c);
  c.pushRotateZoom(&canvas, x, y, angle, 1.0f, 1.0f, kPaper);
  c.deleteSprite();
}

// The robot capsized: feet in the air, head in the sea, its hat floating off.
static void drawCapsized(const String& hint) {
  drawTurned(240, 292, W / 2, 288, 200, [](M5Canvas& c) {
    static constexpr int cx = 120;
    // Antenna, bent by the fall.
    c.drawWideLine(cx, 36, cx + 5, 21, 2.5f, kInk);
    c.drawWideLine(cx + 5, 21, cx + 26, 13, 2.5f, kInk);
    c.fillCircle(cx + 31, 13, 8, kRed);
    // Head, ears, eyes, a surprised mouth.
    c.fillRoundRect(cx - 83, 62, 15, 36, 5, kRed);
    c.fillRoundRect(cx + 68, 62, 15, 36, 5, kRed);
    c.fillRoundRect(cx - 68, 31, 136, 99, 24, kInk);
    c.fillRoundRect(cx - 63, 36, 126, 89, 19, kPaper);
    for (int eye : {cx - 31, cx + 31}) {
      c.drawWideLine(eye - 13, 55, eye + 13, 81, 3.0f, kInk);
      c.drawWideLine(eye + 13, 55, eye - 13, 81, 3.0f, kInk);
    }
    c.fillCircle(cx, 104, 10, kInk);
    c.fillCircle(cx, 104, 5, kPaper);
    // Arms flung out, then the body over them.
    c.drawWideLine(cx - 52, 156, cx - 94, 187, 6.5f, kInk);
    c.drawWideLine(cx + 52, 156, cx + 94, 130, 6.5f, kInk);
    c.fillRoundRect(cx - 52, 135, 104, 62, 16, kInk);
    c.fillRoundRect(cx - 47, 140, 94, 52, 11, kYellow);
    c.fillCircle(cx, 166, 8, kRed);
    // Legs and feet.
    c.fillRoundRect(cx - 36, 197, 16, 47, 5, kInk);
    c.fillRoundRect(cx + 21, 197, 16, 36, 5, kInk);
    c.fillRoundRect(cx - 57, 239, 42, 16, 8, kRed);
    c.fillRoundRect(cx + 16, 228, 42, 16, 8, kRed);
  });
  drawTurned(84, 62, 335, 388, 18, [](M5Canvas& c) {
    static constexpr int cx = 42;
    // The crown, wider at the top; its ink, then again smaller in paper.
    c.fillEllipse(cx, 15, 37, 13, kInk);
    c.fillTriangle(cx - 37, 18, cx + 37, 18, cx + 18, 42, kInk);
    c.fillTriangle(cx - 37, 18, cx + 18, 42, cx - 18, 42, kInk);
    c.fillEllipse(cx, 15, 33, 9, kPaper);
    c.fillTriangle(cx - 31, 18, cx + 31, 18, cx + 13, 37, kPaper);
    c.fillTriangle(cx - 31, 18, cx + 13, 37, cx - 13, 37, kPaper);
    c.fillCircle(cx, 24, 5, kRed);  // the badge
    // Band and peak.
    c.fillRect(cx - 20, 37, 40, 11, kInk);
    c.fillEllipse(cx, 51, 26, 7, kInk);
  });
  // Bubbles.
  static constexpr int bubbles[][3] = {{72, 396, 11}, {50, 356, 8}, {74, 324, 6}};
  for (const auto& b : bubbles) {
    canvas.fillCircle(b[0], b[1], b[2], kBlue);
    canvas.fillCircle(b[0], b[1], b[2] - 3, kPaper);
  }
  // The sea, its crest hiding what's under it, and a wave in it.
  for (int x = 0; x < W; ++x) {
    const int y = 408 + (int)(14 * fabsf(sinf(x * 3.14159f / 64)));
    canvas.drawFastVLine(x, y, H - y, kBlue);
    canvas.fillRect(x, 452 + (int)(10 * fabsf(sinf((x + 24) * 3.14159f / 76))), 1, 3, kPaper);
  }
  font(vlw::BodyS);
  text(hint, W / 2, H - 58, kPaper, middle_center);
}

void showSwitchedOff(const String& hint, void (*whileDrawing)()) {
  draw(Kind::Off, [=] {
    font(vlw::Title);
    text("Switched off", PAD, 14, kInk);
    drawCapsized(hint);
  });
  do {  // (tick(): a dashboard on its way is cut short, then this one goes on)
    tick();
    if (whileDrawing) whileDrawing();
    delay(20);
  } while (refreshing || want.active);
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

// The page is a conversation: the question in a yellow bubble on the right (first
// page only), Captain's row under it, then the answer.
static constexpr int kBubbleW = 300, kBubblePadX = 14, kBubblePadY = 9, kBubbleLine = 33, kBubbleTail = 12;
static constexpr int kCaptainRowH = 40;
static constexpr int kReplyY = 46;  // under C's icon

static std::vector<String> questionLines() {
  if (!question.length()) return {};
  font(vlw::BodyS);
  return clamp(wrap(question, kBubbleW - 2 * kBubblePadX), 3, kBubbleW - 2 * kBubblePadX);
}

static int bubbleHeight(int lines) { return lines * kBubbleLine + 2 * kBubblePadY; }

// Where the answer's words start.
static int replyTop(bool firstPage) {
  const int lines = firstPage ? (int)questionLines().size() : 0;
  return kReplyY + (lines ? bubbleHeight(lines) + kBubbleTail : 0) + kCaptainRowH + 6;
}
static int replyBottom() { return H - 22; }

// Where something at height `y` starts: clear of A's and B's icons.
static int leftAt(int y) { return y < kKeysBottom ? kKeysRight : PAD; }

// Where words after `y` start: one line alone beside the icons would only look
// indented, so it goes under them.
static int wordsFrom(int y, int line) { return y < kKeysBottom && y + line >= kKeysBottom ? kKeysBottom : y; }

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
  const int top = replyTop(false), firstTop = replyTop(true), bottom = replyBottom();
  int pw = 0, ph = 0;
  if (imgLen && jpegSize(imgBuf, imgLen, &pw, &ph)) {
    const int natural = (int)(ph * min(1.0f, (float)(W - PAD - leftAt(firstTop)) / pw));
    pictureH = min(natural, (bottom - firstTop) * 40 / 100) + 14;
  }
  for (const Size& s : kSizes) {
    font(s.font);
    const int firstY = wordsFrom(firstTop + pictureH, s.line), nextY = wordsFrom(top, s.line);
    const int avail = bottom - firstY, first = max(1, avail / s.line), per = max(1, (bottom - nextY) / s.line);
    auto lines = wrap(replyText, [&, first, per](size_t n) {
      const int y = (int)n < first ? firstY + (int)n * s.line : nextY + ((int)n - first) % per * s.line;
      return W - PAD - leftAt(y);
    });
    if ((int)lines.size() * s.line <= avail || &s == &kSizes[3]) {
      replyFont = s.font;
      replyLine = s.line;
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
  int y = kReplyY;
  if (page == 0 && question.length()) {
    const std::vector<String> qs = questionLines();
    font(vlw::BodyS);
    int bw = 0;
    for (const String& l : qs) bw = max(bw, tw(l));
    bw += 2 * kBubblePadX;
    const int bh = bubbleHeight(qs.size()), bx = W - 10 - bw;
    canvas.fillRoundRect(bx, y, bw, bh, 16, kYellow);
    canvas.fillTriangle(W - 44, y + bh - 2, W - 16, y + bh - 2, W - 12, y + bh + kBubbleTail, kYellow);
    for (int i = 0; i < (int)qs.size(); ++i) textOn(qs[i], bx + kBubblePadX, y + kBubblePadY + i * kBubbleLine, kInk, kYellow);
    y += bh + kBubbleTail;
  }
  const int rowX = leftAt(y);
  drawRobot(rowX + 18, y + 26, 0.3f, kPaper);
  char when[16] = "";
  clockText(when, sizeof(when), "%H:%M", replyWhen);
  font(vlw::Small);
  text(fit(barAgent.length() ? barAgent : String("Captain"), W - PAD - rowX - 124), rowX + 46, y + 22, kInk, middle_left);
  text(when, W - PAD, y + 22, kInk, middle_right);
  y += kCaptainRowH + 6;
  if (page == 0 && pictureH) {
    const int px = leftAt(y), ph = drawPicture(imgBuf, imgLen, px, y, W - PAD - px, pictureH - 14);
    if (ph) y += ph + 14;
  }
  font(replyFont);
  y = wordsFrom(y, replyLine);
  for (const String& line : replyPages[page]) {
    text(line, leftAt(y), y, kInk);
    y += replyLine;
  }
  if (pages > 1) {  // how far through it is: a thin line, a thick stretch
    const int barY = H - 14;
    canvas.fillRect(PAD, barY + 2, width, 2, kInk);
    canvas.fillRoundRect(PAD, barY, width * (page + 1) / pages, 6, 3, kInk);
  }
  drawKeys(Key::Talk, page + 1 < pages ? Key::Next : Key::Home, Key::Options);
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
  answerDue = t.length() && !restoring && shownKind == Kind::Reply;  // (on the answer page: the new one comes there)
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
  const bool other = !(shownKind == Kind::Reply && shownPage == replyPage);  // another screen, or page
  draw(Kind::Reply, [] {
    shownGen = replyGen;
    shownPage = replyPage;
    drawReplyPage();
  });
  if (refreshing && other) cutShort = true;
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

// The answer's voice has started (or it came with speaking off): on the
// paper now, its first page — when the answer page is what the paper shows.
// From the dashboard it stays the dashboard: B shows the answer.
void answerStarted() {
  ensureLayout();
  const bool onAnswer = shownKind == Kind::Reply || (want.active && want.kind == Kind::Reply);
  answerDue = false;
  if (replyPages.empty() || !onAnswer) return;
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
    drawQrWithLines(String("WIFI:T:nopass;S:") + ap + ";;", 3, 82, H - 42,
                    {{"1. Scan the code, or join this Wi-Fi:", vlw::Small, kInk},
                     {ap, vlw::BodyS, kBlue},
                     {"2. Then open:", vlw::Small, kInk},
                     {String("http://") + addr, vlw::BodyS, kBlue}});
    font(vlw::Small);
    text("Saving restarts it.", W / 2, H - 34, kInk, top_center);
    drawKeys(Key::Back, Key::None, Key::None);
  });
  flush();  // the portal loop doesn't tick
}

static bool linkFromAnswer = false;  // the options were opened from the answer page
static int linkFromPage = 0;

// C: the options code (the settings page on a phone on the same Wi-Fi),
// with what the paper is and what it has.
// (Asked again every 30 s while it's open: drawn once, unless it changes.)
void showLink(const char* title, const String& url, const String& note) {
  String t = title;
  forceNext = false;
  const String key = t + "\n" + url;
  const bool open = (want.active && want.kind == Kind::Link) || shownKind == Kind::Link;
  if (key == linkShown && open) return;
  if (!open) {  // where back goes to
    linkFromAnswer = shownKind == Kind::Reply || (want.active && want.kind == Kind::Reply);
    linkFromPage = shownPage;
  }
  linkShown = key;
  draw(Kind::Link, [=] {
    font(vlw::Title);
    text(fit(t, W - 2 * kKeysRight), W / 2, 50, kInk, top_center);
    const int linksY = H - 10 - 3 * kLinkRowH;
    drawQrWithLines(url, 4, 92, linksY - 8, {{url, vlw::Small, kBlue}, {note, vlw::BodyS, kInk}});
    drawLinks(PAD, linksY, W - 2 * PAD);
    drawKeys(Key::Back, Key::Answer, Key::Update);
  });
}

// A on the options: back to the page they were opened from — the answer
// (the page that was showing) or the dashboard.
void leaveOptions() {
  ensureLayout();
  if (linkFromAnswer && !replyPages.empty()) {
    viewing = true;
    replyPage = min(linkFromPage, (int)replyPages.size() - 1);
    forceNext = false;
    askReplyPage();
  } else {
    drawNow();
  }
}

// The links went down a while ago (or came back after that): the paper is
// drawn again with them — the dashboard's LINKS rows, the top bar's dots. An
// answer being read keeps its page (its dots change); the options or setup
// code stays as it is.
void linksChanged() {
  if (shownKind == Kind::Link || shownKind == Kind::Setup) return;
  if (viewing && shownKind == Kind::Reply) {
    if (!sounding) askReplyPage();
    return;
  }
  askDashboard(false);
}

// A microSD card put in or taken out: the dashboard again, when it's what
// the paper shows (the notes' room: GB on a card, minutes in flash).
void notesChanged() {
  if (shownKind == Kind::Dash && !viewing && !held) askDashboard(false);
}

// ── everything else: the lights and the voice carry it ─────────────────

void showMessage(const char*, const String&, uint16_t) {}
void showAsk(const char*, const String&, const char*, const char*) {}
int showPanel(const char*, uint16_t, const std::vector<Row>&, int) { return 1; }
int showStatus(const Status&, int) { return 1; }
void showListening(int, uint32_t) {}
void showWorking(const String&, const std::vector<String>&, const String&, uint32_t, int) {}
void showSpeaking(int) {}
void showFace(face::Pose, int) {}
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
