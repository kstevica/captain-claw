// StickS3 screen: 240×135 landscape LCD. Text uses efont (trimmed efontJA, see
// efont_subset.h), which has č ć š ž đ.
#include "ui.h"
#include "board.h"
#include "efont_subset.h"

#include <M5Unified.h>

namespace ui {

static M5Canvas canvas(&M5.Display);
static constexpr int W = 240, H = 135, BAR = 16, PAD = 4;
static const lgfx::U8g2font* kBody = &efont::efont14;  // fixed size: status page, setup
static const lgfx::U8g2font* kSmall = &efont::efont12; // fixed size: top bar, panels
static constexpr int kLine = 16;

// Reading text (replies, the working screen, notices, the menu) in the size
// the user picked. efont at 14/16/24 px all have č ć š ž đ.
struct TextSize {
  const lgfx::U8g2font* body;
  int line;
  const lgfx::U8g2font* small;  // activity lines on the working screen
  int smallLine;
};
static const TextSize kSizes[] = {
    {&efont::efont14, 16, &efont::efont12, 15},  // small (the original)
    {&efont::efont16, 18, &efont::efont14, 16},  // medium
    {&efont::efont24, 26, &efont::efont16, 18},  // large
};
static uint8_t sizeIdx = 0;
static const TextSize& ts() { return kSizes[sizeIdx]; }
static String replyText;

static String barAgent = "";
static String barState = "";
static uint16_t barColor = TFT_DARKGREY;
static Health hWifi = Down, hAgent = Down;

static std::vector<String> lines;
static int page = 0;
static uint8_t* imgBuf = nullptr;
static size_t imgLen = 0;

static int linesPerPage() { return (H - BAR - PAD) / ts().line; }

void begin() {
  M5.Display.setRotation(1);
  canvas.setPsram(true);
  canvas.setColorDepth(16);
  canvas.createSprite(W, H);
  canvas.setTextWrap(false);
}

void brightness(uint8_t level) { M5.Display.setBrightness(level); }

bool animated() { return false; }

void tick() {}
void flush() {}
bool busy() { return false; }
void idleSince(uint32_t) {}
void drawNow() {}
bool waiting() { return false; }
void showSpeaking(int) {}
void setRotation(uint8_t) {}

static void push() { canvas.pushSprite(0, 0); }

static uint16_t healthColor(Health h) {
  return h == Up ? rgb(40, 200, 90) : h == Partial ? rgb(250, 200, 40) : rgb(230, 50, 50);
}

// Battery: green ≥75 %, blue 40–74 %, yellow 20–39 %, red <20 %.
static uint16_t batteryColor(int pct) {
  if (pct >= 75) return rgb(40, 200, 90);
  if (pct >= 40) return rgb(70, 150, 255);
  if (pct >= 20) return rgb(250, 200, 40);
  return rgb(230, 50, 50);
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

// Two colour blocks: W = Wi-Fi, A = agent (green up, yellow partial, red down).
static int drawHealth(int x, int y) {
  canvas.setFont(kSmall);
  canvas.setTextDatum(middle_center);
  const char* tags[2] = {"W", "A"};
  Health hs[2] = {hWifi, hAgent};
  for (int i = 0; i < 2; ++i) {
    canvas.fillRoundRect(x, y, 14, 12, 3, healthColor(hs[i]));
    canvas.setTextColor(TFT_BLACK);
    canvas.drawString(tags[i], x + 7, y + 6);
    x += 16;
  }
  return x;
}

// Battery icon + percentage, right-aligned at `right`; returns its left edge.
static int drawBattery(int right, int y) {
  bool chg = false;
  int pct = batteryPct(&chg);
  canvas.setFont(kSmall);
  canvas.setTextDatum(middle_right);
  String txt = pct < 0 ? String("--") : String(pct) + "%";
  uint16_t col = pct < 0 ? TFT_LIGHTGREY : batteryColor(pct);
  canvas.setTextColor(col);
  canvas.drawString(txt, right, y + 6);
  int bx = right - canvas.textWidth(txt) - 22;
  canvas.drawRect(bx, y + 2, 18, 9, TFT_LIGHTGREY);
  canvas.fillRect(bx + 18, y + 5, 2, 3, TFT_LIGHTGREY);
  if (pct > 0) canvas.fillRect(bx + 1, y + 3, max(1, pct * 16 / 100), 7, col);
  if (chg) {  // a small bolt over the cells
    canvas.fillTriangle(bx + 10, y + 2, bx + 6, y + 7, bx + 9, y + 7, TFT_WHITE);
    canvas.fillTriangle(bx + 9, y + 6, bx + 12, y + 6, bx + 8, y + 11, TFT_WHITE);
  }
  return bx;
}

static void drawBar() {
  canvas.fillRect(0, 0, W, BAR, rgb(24, 28, 36));
  int x = drawHealth(2, 2) + 3;
  int left = drawBattery(W - 2, 2);
  canvas.setFont(kSmall);
  int stateW = canvas.textWidth(barState);
  canvas.setTextDatum(middle_right);
  canvas.setTextColor(barColor);
  canvas.drawString(barState, left - 5, BAR / 2);
  int nameRoom = left - 5 - stateW - 6 - x;
  String name = barAgent.length() ? barAgent : String("Captain Claw");
  while (name.length() > 1 && canvas.textWidth(name) > nameRoom) name.remove(name.length() - 1);
  canvas.setTextDatum(middle_left);
  canvas.setTextColor(TFT_WHITE);
  if (nameRoom > 12) canvas.drawString(name, x, BAR / 2);
}

void setBar(const String& agent, const char* state, uint16_t stateColor) {
  barAgent = agent;
  barState = state ? state : "";
  barColor = stateColor;
}

void setHealth(Health wifi, Health agent) {
  hWifi = wifi;
  hAgent = agent;
}

static String fit(const String& text, int width) {
  if (canvas.textWidth(text) <= width) return text;
  String s = text;
  while (s.length() > 1 && canvas.textWidth(s + "…") > width) {
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
      if (canvas.textWidth(trial) <= width) {
        line = trial;
        continue;
      }
      if (line.length()) out.push_back(line);
      line = "";
      // A word wider than the screen: split it between UTF-8 characters.
      while (canvas.textWidth(word) > width) {
        int cut = 0, k = 0;
        while (k < (int)word.length()) {
          int step = 1;
          uint8_t c = word[k];
          if (c >= 0xF0) step = 4; else if (c >= 0xE0) step = 3; else if (c >= 0xC0) step = 2;
          if (canvas.textWidth(word.substring(0, k + step)) > width) break;
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

void showSetup(const char* reason, const char* apName, const char* ip) {
  canvas.fillScreen(TFT_BLACK);
  String qr = String("WIFI:T:nopass;S:") + apName + ";;";
  canvas.qrcode(qr.c_str(), 2, 2, 110, 3);
  canvas.setFont(kSmall);
  canvas.setTextDatum(top_left);
  canvas.setTextColor(TFT_ORANGE);
  canvas.drawString(reason, 118, 4);
  canvas.setTextColor(TFT_WHITE);
  canvas.drawString("1. Scan or join", 118, 26);
  canvas.setTextColor(TFT_CYAN);
  canvas.drawString(apName, 118, 42);
  canvas.setTextColor(TFT_WHITE);
  canvas.drawString("2. Open", 118, 66);
  canvas.setTextColor(TFT_CYAN);
  canvas.drawString(String("http://") + ip, 118, 82);
  canvas.setTextColor(TFT_DARKGREY);
  canvas.drawString("Save restarts", 118, 110);
  push();
}

void showLink(const char* title, const String& url, const String& note) {
  canvas.fillScreen(TFT_BLACK);
  canvas.qrcode(url.c_str(), 2, 2, 110, 4);
  canvas.setFont(kSmall);
  canvas.setTextDatum(top_left);
  canvas.setTextColor(TFT_ORANGE);
  canvas.drawString(title, 118, 4);
  canvas.setTextColor(TFT_CYAN);
  String u = url.startsWith("http://") ? url.substring(7) : url;
  canvas.drawString(fit(u, W - 120), 118, 24);
  canvas.setTextColor(TFT_WHITE);
  int y = 46;
  for (auto& l : wrap(note, W - 122)) {  // sets the reading font
    if (y > H - 14) break;
    canvas.drawString(l, 118, y);
    y += ts().line;
  }
  push();
}

void showSleeping(const String& hint, uint8_t) { showMessage("Sleeping", hint, TFT_DARKGREY); }

void showMessage(const char* title, const String& body, uint16_t color) {
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  auto ls = wrap(body, W - 2 * PAD);  // sets the reading font
  canvas.setTextDatum(top_left);
  canvas.setTextColor(color);
  canvas.drawString(title, PAD, BAR + 6);
  canvas.setTextColor(TFT_LIGHTGREY);
  for (int i = 0; i < (int)ls.size() && i < linesPerPage() - 1; ++i) {
    canvas.drawString(ls[i], PAD, BAR + 6 + ts().line * (i + 1) + 4);
  }
  push();
}

void showListening(int level, uint32_t elapsedMs) {
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  canvas.fillCircle(30, 70, 14, TFT_RED);
  canvas.setFont(&fonts::FreeSansBold12pt7b);
  canvas.setTextDatum(middle_left);
  canvas.setTextColor(TFT_WHITE);
  canvas.drawString("Listening", 56, 64);
  canvas.setFont(kSmall);
  canvas.setTextColor(TFT_DARKGREY);
  char t[16];
  snprintf(t, sizeof(t), "%lu.%lus", (unsigned long)(elapsedMs / 1000), (unsigned long)(elapsedMs / 100 % 10));
  canvas.drawString(t, 56, 88);
  canvas.drawString(board::kTalkToggles ? "press to send" : "release to send", 120, 88);
  int bar = (W - 2 * PAD) * min(level, 100) / 100;
  canvas.fillRect(PAD, H - 12, W - 2 * PAD, 6, 0x2104);
  canvas.fillRect(PAD, H - 12, bar, 6, level > 85 ? TFT_ORANGE : TFT_GREEN);
  push();
}

// The time and steps are in the top bar (main sets them as the bar state).
void showWorking(const String& heard, const std::vector<String>& activity, const String& status,
                 uint32_t, int) {
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  canvas.setTextDatum(top_left);
  int y = BAR + 3;
  if (heard.length()) {  // the end of a long sentence matters most: last two lines
    auto ls = wrap(String("“") + heard + "”", W - 2 * PAD);
    canvas.setTextColor(TFT_CYAN);
    int maxLines = activity.empty() ? (H - 16 - y) / ts().line : 2;
    for (int i = max(0, (int)ls.size() - maxLines); i < (int)ls.size(); ++i, y += ts().line) {
      canvas.drawString(ls[i], PAD, y);
    }
    y += 3;
  }
  canvas.setFont(ts().small);
  int room = max(0, (H - 16 - y) / ts().smallLine);  // keep the bottom line free
  int first = max(0, (int)activity.size() - room);
  for (int i = first; i < (int)activity.size(); ++i, y += ts().smallLine) {
    bool newest = i == (int)activity.size() - 1;
    canvas.setTextColor(newest ? TFT_WHITE : 0x8410);  // older lines grey
    canvas.drawString(fit((newest ? "→ " : "   ") + activity[i], W - 2 * PAD), PAD, y);
  }
  if (status.length()) {
    canvas.setFont(kSmall);
    canvas.setTextColor(TFT_DARKGREY);
    canvas.drawString(fit(status, W - 2 * PAD), PAD, H - 14);
  }
  push();
}

void setQuestion(const String&) {}
bool canPin() { return false; }
bool pinned() { return false; }
void pin(bool) {}
bool restoreAnswer() { return false; }

void setReply(const String& text) {
  replyText = text;
  lines = wrap(text, W - 2 * PAD - 4);
  page = 0;
}

void setTextSize(uint8_t size) {
  sizeIdx = size < sizeof(kSizes) / sizeof(kSizes[0]) ? size : 0;
  if (replyText.length()) setReply(replyText);  // re-wrap for the new size
}

bool hasReply() { return !lines.empty(); }

void showReply() {
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  canvas.setFont(ts().body);
  canvas.setTextDatum(top_left);
  canvas.setTextColor(TFT_WHITE);
  int per = linesPerPage();
  int pages = max(1, ((int)lines.size() + per - 1) / per);
  int first = page * per;
  for (int i = 0; i < per && first + i < (int)lines.size(); ++i) {
    canvas.drawString(lines[first + i], PAD, BAR + PAD + i * ts().line);
  }
  if (pages > 1) {  // scroll position on the right edge
    int track = H - BAR - 4;
    int thumb = max(6, track / pages);
    int ty = BAR + 2 + (track - thumb) * page / (pages - 1);
    canvas.fillRect(W - 3, BAR + 2, 2, track, 0x2104);
    canvas.fillRect(W - 3, ty, 2, thumb, TFT_LIGHTGREY);
  }
  push();
}

bool nextPage() {
  int per = linesPerPage();
  int pages = max(1, ((int)lines.size() + per - 1) / per);
  page = (page + 1) % pages;
  showReply();
  return page != 0;
}

bool onFirstPage() { return page == 0; }
void slide(bool) {}  // the round screen's
void fade() {}
bool setSpoken(int) { return false; }
void setReplayable(bool) {}
bool replayHit(int, int) { return false; }

bool prevPage() {
  if (page == 0) return false;
  --page;
  showReply();
  return true;
}

bool onLastPage() {
  int per = linesPerPage();
  return page + 1 >= max(1, ((int)lines.size() + per - 1) / per);
}

// Panels (status page, idle health) follow the text size too: compact rows
// with the label beside the value for small/medium, label above the value
// for large. Values wrap; a row never splits across pages.
struct PanelSize {
  const lgfx::U8g2font* font;
  int line;
};
static const PanelSize kPanelSizes[] = {
    {&efont::efont12, 14},  // small (the original)
    {&efont::efont16, 18},  // medium
    {&efont::efont24, 26},  // large
};

int showPanel(const char* title, uint16_t titleColor, const std::vector<Row>& rows, int page) {
  const PanelSize& ps = kPanelSizes[sizeIdx];
  const bool stacked = sizeIdx == 2;
  const uint16_t labelColor = rgb(140, 150, 165);
  const int top = BAR + 3;
  const int bottom = H - 2;
  // Title (only on the first page) in the reading font.
  int titleH = 0;
  if (title && *title) titleH = ts().line + 2;
  // Label column for side-by-side rows.
  canvas.setFont(ps.font);
  int labelW = 0;
  if (!stacked) {
    for (const Row& r : rows) labelW = max(labelW, (int)canvas.textWidth(r.label));
    labelW += 8;
  }
  const int valueW = stacked ? W - 2 * PAD : W - 2 * PAD - labelW;
  // Lay the rows out into pages.
  struct Laid {
    std::vector<String> lines;
    int h;
  };
  std::vector<Laid> laid;
  for (const Row& r : rows) {
    Laid l{wrap(r.value.length() ? r.value : String("-"), valueW, ps.font), 0};
    l.h = (int)l.lines.size() * ps.line + (stacked ? 14 : 0) + (stacked ? 4 : 0);
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
  page = constrain(page, 0, pages - 1);

  canvas.fillScreen(TFT_BLACK);
  drawBar();
  canvas.setTextDatum(top_left);
  y = top;
  if (page == 0 && titleH) {
    canvas.setFont(ts().body);
    canvas.setTextColor(titleColor);
    canvas.drawString(title, PAD, y);
    y += titleH;
  }
  int end = page + 1 < pages ? pageStart[page + 1] : (int)laid.size();
  for (int i = pageStart[page]; i < end; ++i) {
    const Row& r = rows[i];
    if (stacked) {
      canvas.setFont(kSmall);
      canvas.setTextColor(labelColor);
      canvas.drawString(r.label, PAD, y);
      y += 14;
    } else {
      canvas.setFont(ps.font);
      canvas.setTextColor(labelColor);
      canvas.drawString(r.label, PAD, y);
    }
    canvas.setFont(ps.font);
    canvas.setTextColor(r.color);
    for (const String& line : laid[i].lines) {
      if (y + ps.line > bottom + 2) break;  // a row taller than a page: cut it
      canvas.drawString(line, stacked ? PAD : PAD + labelW, y);
      y += ps.line;
    }
    if (stacked) y += 4;
  }
  if (pages > 1) {  // page dots, bottom right
    for (int k = 0; k < pages; ++k) {
      canvas.fillCircle(W - 6 - (pages - 1 - k) * 8, H - 5, 2, k == page ? TFT_WHITE : rgb(70, 70, 80));
    }
  }
  push();
  return pages;
}

void showFace(uint8_t kind, face::Pose pose, int level) {
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  face::draw(canvas, kind, pose, level, millis(), W / 2, BAR + (H - BAR) / 2 + 1);
  push();
}

// The bar already says "3/18  A=pick".
void showCharPick(uint8_t kind, face::Pose pose, int level, int) { showFace(kind, pose, level); }

void showHome(const Home& home) { showPanel(home.title, home.color, home.rows); }

void storeImage(uint8_t* jpg, size_t len) {
  if (imgBuf) free(imgBuf);
  imgBuf = jpg;
  imgLen = jpg ? len : 0;
}

bool hasImage() { return imgLen > 0; }

void setBackground(const uint8_t*, size_t) {}  // a round screen's (ui_round.cpp)
bool hasBackground() { return false; }
void showBackdrop() {}

void showStoredImage() {
  if (!imgLen) return;
  canvas.fillScreen(TFT_BLACK);
  canvas.drawJpg(imgBuf, imgLen, 0, 0, W, H);
  // Keep health + battery in sight over the picture.
  canvas.fillRoundRect(W - 92, 0, 92, 15, 4, rgb(0, 0, 0));
  drawHealth(W - 90, 1);
  drawBattery(W - 2, 1);
  push();
}

void showPicker(const std::vector<Item>& items, int selected, int current, const char*) {
  std::vector<String> names;
  for (const Item& it : items) names.push_back(it.label);
  canvas.fillScreen(TFT_BLACK);
  drawBar();
  canvas.setFont(ts().body);
  canvas.setTextDatum(top_left);
  int per = linesPerPage();
  int first = max(0, min(selected - per / 2, (int)names.size() - per));
  for (int i = 0; i < per && first + i < (int)names.size(); ++i) {
    int idx = first + i;
    int y = BAR + PAD + i * ts().line;
    if (idx == selected) canvas.fillRect(0, y - 1, W, ts().line, 0x3186);
    canvas.setTextColor(idx == current ? TFT_GREEN : TFT_WHITE);
    canvas.drawString(fit((idx == current ? "● " : "  ") + names[idx], W - 2 * PAD), PAD, y);
  }
  push();
}

void flashFrame(bool on) {
  if (on) {
    canvas.fillScreen(TFT_WHITE);
    push();
  }
}

void showAsk(const char* title, const String& body, const char* yes, const char* no) {
  showMessage(title, body + "  Front: " + yes + " · side: " + no, TFT_WHITE);
}

void setTapButton(TapButton) {}  // no touch screen
bool tapButtonHit(int, int) { return false; }
int askHit(int, int) { return -1; }
void toggleImageFit() {}

}  // namespace ui
