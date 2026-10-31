// Screens and machines: Matrix rain, Commodore 64, Amiga Boing, HAL 9000,
// KITT and Pac-Man — homages drawn from scratch, no copied artwork.
#include "face_common.h"
#include "efont_subset.h"

namespace face {

static uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

// ── Matrix: katakana rain, a face in dark glasses ───────────────────

namespace {
constexpr int kCols = 16, kColW = 15, kRowH = 13;
struct Drop {
  float y;
  float speed;  // rows per second
  uint8_t len;
  uint16_t seed;
};
Drop drops[kCols];
bool rainReady = false;
uint32_t rainT = 0;
uint32_t rng = 12345;

void resetDrop(Drop& d, bool anywhere) {
  d.y = anywhere ? (float)(lcg(rng) % 12) - 3 : -(float)(lcg(rng) % 6);
  d.speed = 4 + (lcg(rng) % 70) / 10.0f;
  d.len = 4 + lcg(rng) % 6;
  d.seed = lcg(rng);
}

void kana(char* out, uint32_t n) {
  if (n % 6 == 0) {  // a sprinkle of digits
    out[0] = '0' + (n / 6) % 10;
    out[1] = 0;
    return;
  }
  uint32_t cp = 0x30A1 + n % 86;  // full-width katakana ァ..ヶ
  out[0] = 0xE0 | (cp >> 12);
  out[1] = 0x80 | ((cp >> 6) & 0x3F);
  out[2] = 0x80 | (cp & 0x3F);
  out[3] = 0;
}
}  // namespace

void drawMatrix(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  c.fillRect(0, kTop, 240, kBottom - kTop, kBlack);
  if (!rainReady) {
    for (auto& d : drops) resetDrop(d, true);
    rainReady = true;
    rainT = t;
  }
  float pace = pose == Pose::Thinking ? 2.6f : pose == Pose::Listening ? 0.5f
             : pose == Pose::Speaking ? 0.8f + level / 60.0f : pose == Pose::Sad ? 0.0f : 1.0f;
  float dt = min<uint32_t>(t - rainT, 250) / 1000.0f;
  rainT = t;
  const int rows = (kBottom - kTop) / kRowH;
  bool red = pose == Pose::Sad;
  c.setFont(&efont::efont12);
  c.setTextDatum(top_left);
  char g[5];
  for (int i = 0; i < kCols; ++i) {
    Drop& d = drops[i];
    d.y += d.speed * pace * dt;
    if (d.y - d.len > rows) resetDrop(d, false);
    int head = (int)d.y;
    for (int k = 0; k < d.len; ++k) {
      int row = head - k;
      if (row < 0 || row >= rows) continue;
      uint8_t v = k == 0 ? 255 : (uint8_t)max(40, 230 - k * 30);
      uint16_t col = k == 0 ? (red ? rgb(255, 200, 200) : rgb(200, 255, 200)) : (red ? rgb(v, 0, 0) : rgb(0, v, 0));
      uint32_t flick = k == 0 ? t / 120 : 0;
      kana(g, d.seed + row * 7 + flick);
      c.setTextColor(col);
      c.drawString(g, i * kColW + 1, kTop + row * kRowH);
    }
  }
  // The face: a dark silhouette in sunglasses, lips moving with the voice.
  uint16_t edge = red ? rgb(120, 0, 0) : rgb(0, 130, 40);
  c.fillEllipse(cx, cy + 2, 32, 44, kBlack);
  c.drawEllipse(cx, cy + 2, 32, 44, edge);
  uint16_t lens = rgb(10, 25, 12), shine = red ? rgb(255, 60, 60) : rgb(90, 255, 120);
  c.fillRoundRect(cx - 27, cy - 10, 23, 13, 5, lens);
  c.fillRoundRect(cx + 4, cy - 10, 23, 13, 5, lens);
  c.drawFastHLine(cx - 4, cy - 6, 8, edge);
  c.drawLine(cx - 22, cy - 2, cx - 14, cy - 8, shine);
  c.drawLine(cx + 9, cy - 2, cx + 17, cy - 8, shine);
  int o = pose == Pose::Speaking ? openness(level, 12, t) : 0;
  c.fillEllipse(cx, cy + 24, 9, o / 2 + 1, red ? rgb(120, 0, 0) : rgb(0, 110, 30));
}

// ── Commodore 64: blue screen, typed replies, tape-loading stripes ───

static String c64Upper(const String& s) {
  String out;
  out.reserve(s.length());
  for (size_t i = 0; i < s.length();) {
    uint8_t b = s[i];
    uint32_t cp = b;
    int n = 1;
    if (b >= 0xF0) cp = b & 0x07, n = 4;
    else if (b >= 0xE0) cp = b & 0x0F, n = 3;
    else if (b >= 0xC0) cp = b & 0x1F, n = 2;
    for (int k = 1; k < n && i + k < s.length(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    i += n;
    if (cp < 0x80) out += (char)toupper((int)cp);
    else if (cp == 0x10C || cp == 0x10D || cp == 0x106 || cp == 0x107) out += 'C';
    else if (cp == 0x160 || cp == 0x161) out += 'S';
    else if (cp == 0x17D || cp == 0x17E) out += 'Z';
    else if (cp == 0x110 || cp == 0x111) out += 'D';
    else if (cp == 0x2014 || cp == 0x2013) out += '-';
    else if (cp == 0x2018 || cp == 0x2019) out += '\'';
    else if (cp == 0x201C || cp == 0x201D) out += '"';
    else if (cp == 0x2026) out += "...";
    else if (cp == 0x2022) out += '*';
  }
  return out;
}

static void c64Wrap(const String& text, int cols, std::vector<String>& out) {
  int start = 0;
  while (start < (int)text.length()) {
    int nl = text.indexOf('\n', start);
    String para = text.substring(start, nl < 0 ? text.length() : nl);
    start = nl < 0 ? text.length() : nl + 1;
    while ((int)para.length() > cols) {
      int cut = para.lastIndexOf(' ', cols);
      if (cut <= 0) cut = cols;
      out.push_back(para.substring(0, cut));
      para = para.substring(cut + (para[cut] == ' ' ? 1 : 0));
    }
    out.push_back(para);
  }
}

void drawC64(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  static const uint16_t pal[16] = {
      rgb(0, 0, 0), rgb(255, 255, 255), rgb(136, 57, 50), rgb(103, 182, 189),
      rgb(139, 63, 150), rgb(85, 160, 73), rgb(64, 49, 141), rgb(191, 206, 114),
      rgb(139, 84, 41), rgb(87, 66, 0), rgb(184, 105, 98), rgb(80, 80, 80),
      rgb(120, 120, 120), rgb(148, 224, 137), rgb(120, 105, 196), rgb(159, 159, 159)};
  const uint16_t border = pal[14], screen = pal[6], ink = pal[14];
  if (pose == Pose::Thinking) {  // tape-loading stripes in the border
    uint32_t s = t / 70;
    for (int y = kTop; y < kBottom;) {
      int h = 2 + lcg(s) % 6;
      c.fillRect(0, y, 240, h, pal[lcg(s) % 16]);
      y += h;
    }
  } else {
    c.fillRect(0, kTop, 240, kBottom - kTop, border);
  }
  const int x0 = 12, y0 = kTop + 7, w = 216, h = kBottom - 6 - y0;
  c.fillRect(x0, y0, w, h, screen);
  const int cols = w / 6, rows = h / 8;
  std::vector<String> lines;
  switch (pose) {
    case Pose::Idle:
      lines = {"      **** CAPTAIN 64 ****", "64K RAM SYSTEM  38911 BYTES FREE", "", "READY."};
      break;
    case Pose::Listening:
      lines = {"READY.", "LOAD\"VOICE\",8,1", "", "SEARCHING FOR VOICE"};
      break;
    case Pose::Thinking:
      lines = {"READY.", "LOAD\"VOICE\",8,1", "", "SEARCHING FOR VOICE", "LOADING"};
      break;
    case Pose::Sad:
      lines = {"READY.", "LOAD\"AGENT\",8,1", "", "?DEVICE NOT PRESENT  ERROR", "READY."};
      break;
    case Pose::Speaking: {
      lines = {"RUN"};
      String said = c64Upper(sayText().length() ? sayText() : String("HELLO! I AM YOUR CAPTAIN."));
      size_t shown = min<size_t>(said.length(), poseAge(t) / 55);  // typed at speaking pace
      c64Wrap(said.substring(0, shown), cols, lines);
      break;
    }
  }
  c.setFont(&fonts::Font0);
  c.setTextSize(1);
  c.setTextDatum(top_left);
  c.setTextColor(ink);
  int first = max(0, (int)lines.size() - rows + 1);  // keep the last rows, room for the cursor
  int y = y0 + 2;
  for (int i = first; i < (int)lines.size(); ++i, y += 8) c.drawString(lines[i].c_str(), x0 + 2, y);
  // Blinking cursor after the last line (on the next line after READY./LOADING).
  bool after = pose != Pose::Speaking;
  int cxl = after ? x0 + 2 : x0 + 2 + (int)lines.back().length() * 6;
  int cyl = after ? y : y - 8;
  if (pose == Pose::Listening) {  // a mic meter as reversed blocks
    int n = level * (cols - 2) / 100;
    for (int k = 0; k < n; ++k) c.fillRect(x0 + 2 + k * 6, y, 5, 7, ink);
  } else if ((t / 400) % 2) {
    c.fillRect(cxl, cyl, 6, 8, ink);
  }
}

// ── Amiga: the bouncing checkered ball; Guru Meditation when down ───

void drawAmiga(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  if (pose == Pose::Sad) {
    c.fillRect(0, kTop, 240, kBottom - kTop, kBlack);
    uint16_t red = rgb(255, 20, 20);
    if ((t / 700) % 2) {
      for (int k = 0; k < 3; ++k) c.drawRect(4 + k, kTop + 6 + k, 232 - 2 * k, 46 - 2 * k, red);
    }
    c.setFont(&fonts::Font0);
    c.setTextSize(1);
    c.setTextDatum(top_center);
    c.setTextColor(red);
    c.drawString("Software Failure.", 120, kTop + 14);
    c.drawString("Press left mouse button to continue.", 120, kTop + 24);
    c.drawString("Guru Meditation #00000004.0000AAC0", 120, kTop + 36);
    c.setTextColor(rgb(150, 150, 150));
    c.drawString("Agent not reachable", 120, kTop + 70);
    return;
  }
  const uint16_t bg = rgb(170, 170, 170), grid = rgb(170, 0, 170);
  c.fillRect(0, kTop, 240, kBottom - kTop, bg);
  for (int x = 8; x < 240; x += 16) c.drawFastVLine(x, kTop, kBottom - kTop, grid);
  for (int y = kTop + 6; y < kBottom; y += 16) c.drawFastHLine(0, y, 240, grid);
  const int R = 24, floorY = kBottom - 4 - R;
  int bx = cx;
  float hgt = 0;
  uint32_t period = 1100;
  switch (pose) {
    case Pose::Idle: hgt = 52, bx = cx + (int)(wave(t, 6000) * 70); break;
    case Pose::Thinking: hgt = 30, period = 1600, bx = cx + (int)(wave(t, 9000) * 60); break;
    case Pose::Speaking: hgt = 6 + level * 0.5f, period = 520, bx = cx + (int)(wave(t, 7000) * 40); break;
    default: hgt = 0; break;  // listening: spins in place
  }
  float ph = (t % period) / (float)period;
  int by = floorY - (int)(hgt * fabsf(sinf(3.14159f * ph)));
  c.fillCircle(bx + 10, by + 6, R, rgb(100, 100, 100));  // shadow
  float spin = (t % 100000) * (pose == Pose::Listening ? 0.006f : 0.003f);
  const float ct = 0.956f, st = 0.292f, seg = 3.14159f / 8;
  for (int dy = -R; dy <= R; ++dy) {
    for (int dx = -R; dx <= R; ++dx) {
      int d2 = dx * dx + dy * dy;
      if (d2 > R * R) continue;
      float nx = dx / (float)R, ny = dy / (float)R;
      float nz = sqrtf(max(0.0f, 1 - nx * nx - ny * ny));
      float tx = nx * ct - ny * st, ty = nx * st + ny * ct;  // tilted axis
      float lon = atan2f(tx, nz) + spin;
      float lat = asinf(constrain(ty, -1.0f, 1.0f));
      int a = (int)floorf(lon / seg), b = (int)floorf((lat + 1.5708f) / seg);
      c.drawPixel(bx + dx, by + dy, ((a + b) & 1) ? rgb(220, 20, 30) : kWhite);
    }
  }
  if (pose == Pose::Thinking) {  // the "zZ" busy pointer
    int px = 206, py = kTop + 8;
    c.fillRoundRect(px, py, 24, 16, 3, kWhite);
    c.drawRoundRect(px, py, 24, 16, 3, kBlack);
    c.setFont(&fonts::Font0);
    c.setTextSize(1);
    c.setTextDatum(middle_center);
    c.setTextColor(kBlack);
    c.drawString("zZ", px + 12, py + 8);
  }
}

// ── HAL 9000: one red eye ───────────────────────────────────────────

void drawHal(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  c.fillRect(0, kTop, 240, kBottom - kTop, kBlack);
  c.setFont(&fonts::Font0);
  c.setTextSize(1);
  c.setTextDatum(middle_center);
  c.fillRect(6, kTop + 6, 22, 12, rgb(200, 200, 205));
  c.setTextColor(kBlack);
  c.drawString("HAL", 17, kTop + 12);
  c.fillRect(28, kTop + 6, 28, 12, rgb(30, 60, 160));
  c.setTextColor(kWhite);
  c.drawString("9000", 42, kTop + 12);
  for (int r = 48; r >= 40; --r) {  // brushed bezel
    uint8_t s = 70 + (48 - r) * 14;
    c.fillCircle(cx, cy, r, rgb(s, s, s + 10));
  }
  c.fillCircle(cx, cy, 39, rgb(8, 8, 10));
  float g;
  switch (pose) {
    case Pose::Thinking: g = 13 + 6 * fabsf(wave(t, 1400)); break;
    case Pose::Listening: g = 21; break;
    case Pose::Speaking: g = 11 + level * 0.24f; break;
    case Pose::Sad: g = 9; break;
    default: g = 15 + 2 * wave(t, 2400);
  }
  float dim = pose == Pose::Sad ? 0.45f : 1.0f;
  int outer = (int)g + 12;
  for (int r = outer; r >= 1; --r) {
    uint8_t R, G, B;
    if (r > g) {
      float f = (outer - r) / 12.0f;
      R = (uint8_t)((40 + 140 * f) * dim), G = 0, B = 0;
    } else {
      float f = 1 - r / g;
      R = (uint8_t)((200 + 55 * f) * dim);
      G = (uint8_t)(230 * f * f * dim);
      B = (uint8_t)(120 * f * f * f * dim);
    }
    c.fillCircle(cx, cy, r, rgb(R, G, B));
  }
  c.fillEllipse(cx - 15, cy - 18, 7, 3, rgb(70, 70, 85));  // glint on the lens
}

// ── KITT: scanner sweep; voice box while talking ────────────────────

void drawKitt(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  c.fillRect(0, kTop, 240, kBottom - kTop, kBlack);
  if (pose == Pose::Speaking || pose == Pose::Listening) {
    c.drawRoundRect(cx - 58, kTop + 6, 116, 106, 6, rgb(70, 70, 70));
    const int rowsN = 11, segH = 6, gap = 2, top = cy - 43;
    int litC = level > 4 ? 1 + level * 5 / 100 : 0;  // rows lit each side of the middle
    int litS = litC * 2 / 3;
    struct { int x, w, lit; } colsK[3] = {{cx - 48, 22, litS}, {cx - 12, 24, litC}, {cx + 26, 22, litS}};
    for (auto& k : colsK) {
      for (int r = 0; r < rowsN; ++r) {
        bool on = abs(r - rowsN / 2) < k.lit;
        c.fillRect(k.x, top + r * (segH + gap), k.w, segH, on ? rgb(255, 40, 30) : rgb(55, 0, 0));
      }
    }
    return;
  }
  const int n = 8, segW = 22, segGap = 3, x0 = (240 - (n * segW + (n - 1) * segGap)) / 2;
  uint32_t period = pose == Pose::Thinking ? 700 : 2000;
  float f = (t % period) / (float)period;
  float p = (f < 0.5f ? f * 2 : (1 - f) * 2) * (n - 1);
  for (int i = 0; i < n; ++i) {
    float b = pose == Pose::Sad ? (i == 0 && (t / 500) % 2 ? 0.5f : 0.08f) : max(0.0f, 1 - fabsf(i - p) / 2.0f);
    c.fillRoundRect(x0 + i * (segW + segGap), cy - 7, segW, 14, 3, rgb((uint8_t)(30 + 225 * b), (uint8_t)(20 * b), 0));
  }
  c.setFont(&fonts::Font0);
  c.setTextSize(1);
  c.setTextDatum(middle_center);
  c.setTextColor(rgb(90, 90, 90));
  c.drawString("KNIGHT INDUSTRIES 2000", cx, kBottom - 14);
}

// ── Pac-Man: chomps with the voice ──────────────────────────────────

static void ghost(M5Canvas& c, int x, int y, uint16_t body) {
  c.fillCircle(x, y - 6, 14, body);
  c.fillRect(x - 14, y - 6, 28, 18, body);
  for (int i = 0; i < 3; ++i) {  // wavy skirt
    int bx = x - 14 + i * 10;
    c.fillTriangle(bx + 4, y + 12, bx + 9, y + 6, bx + 14, y + 12, kBlack);
  }
  c.fillEllipse(x - 6, y - 7, 4, 5, kWhite);
  c.fillEllipse(x + 6, y - 7, 4, 5, kWhite);
  c.fillCircle(x - 8, y - 6, 2, rgb(30, 60, 220));  // looking left, at Pac-Man
  c.fillCircle(x + 4, y - 6, 2, rgb(30, 60, 220));
}

void drawPacMan(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  const uint16_t wall = rgb(33, 33, 222), yellow = rgb(255, 230, 0), pellet = rgb(255, 184, 150);
  c.fillRect(0, kTop, 240, kBottom - kTop, kBlack);
  c.drawRoundRect(3, kTop + 3, 234, kBottom - kTop - 6, 10, wall);
  c.drawRoundRect(7, kTop + 7, 226, kBottom - kTop - 14, 8, wall);
  const int px = 74, py = cy, R = 30;
  bool moving = pose != Pose::Sad && pose != Pose::Listening;
  int offset = moving ? (int)((t / 14) % 22) : 0;
  int lastX = pose == Pose::Thinking ? 160 : 200;
  for (int x = px + 30 - offset; x < lastX; x += 22) {
    if (x > px + 4) c.fillRect(x - 2, py - 2, 4, 4, pellet);
  }
  if (pose != Pose::Thinking && (t / 300) % 2) c.fillCircle(212, py, 6, pellet);  // power pellet
  if (pose == Pose::Thinking) ghost(c, 190, py + (int)(wave(t, 900) * 4), rgb(255, 0, 0));
  float a;
  switch (pose) {
    case Pose::Speaking: a = level > 5 ? 6 + level * 0.42f : 4; break;
    case Pose::Listening: a = 6; break;
    case Pose::Thinking: a = 4 + 30 * fabsf(wave(t, 600)); break;
    case Pose::Sad: {  // the dying animation, over and over
      uint32_t age = poseAge(t) % 2600;
      a = age < 1700 ? 4 + 176.0f * age / 1700 : 180;
      break;
    }
    default: a = 4 + 36 * fabsf(wave(t, 480));
  }
  if (a < 179) c.fillArc(px, py, 0, R, a, 360 - a, yellow);
}

}  // namespace face
