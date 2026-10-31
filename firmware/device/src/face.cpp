// Captain, as the app icon draws him (build_support/icon.svg, in the firmware's
// units: his head is 88×76, centred on 0,0). What never moves (head, plate, cap)
// is rendered once per size with the icon's gradients and soft edges; his lamp,
// eyes and grille go on top each frame.
#include <math.h>

#include "face.h"

namespace face {

static constexpr uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) {
  return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

static uint16_t mix(uint16_t a, uint16_t b, float t) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const int ar = (a >> 11) & 31, ag = (a >> 5) & 63, ab = a & 31;
  const int br = (b >> 11) & 31, bg = (b >> 5) & 63, bb = b & 31;
  return (uint16_t)(((int)(ar + (br - ar) * t) << 11) | ((int)(ag + (bg - ag) * t) << 5) | (int)(ab + (bb - ab) * t));
}

static constexpr uint16_t kEye = rgb(60, 230, 255), kLamp = rgb(255, 70, 70), kWire = rgb(141, 151, 167);
static constexpr uint16_t kPlate = rgb(30, 36, 48);
static constexpr uint16_t kWave = rgb(90, 200, 255), kDots = rgb(200, 200, 200);

// ── what never moves ─────────────────────────────────────────────────

// A shape of the drawing, in the order the icon paints them: a rounded rect
// (x, y, w, h, r) or, with h == 0, an ellipse (x, y the middle, w, r the radii),
// a vertical gradient from `top` to `bottom`.
struct Shape {
  float x, y, w, h, r;
  uint8_t top[3], bottom[3];
};

static const Shape kShapes[] = {
    {-52, -12, 8, 24, 3, {153, 163, 178}, {120, 130, 143}},   // ears
    {44, -12, 8, 24, 3, {153, 163, 178}, {120, 130, 143}},
    {-44, -36, 88, 76, 14, {176, 185, 198}, {133, 144, 161}},  // head
    {-36, -28, 72, 58, 9, {38, 45, 61}, {20, 25, 36}},         // face plate
    {-46, -68, 92, 28, 10, {45, 62, 140}, {21, 30, 85}},       // cap
    {-46, -44, 92, 7, 2, {243, 207, 106}, {177, 122, 42}},     // gold band
    {0, -34, 50, 0, 5.5f, {14, 22, 64}, {14, 22, 64}},         // visor
    {0, -55, 7, 0, 7, {243, 207, 106}, {177, 122, 42}},        // badge
    {0, -55, 4, 0, 4, {24, 35, 87}, {24, 35, 87}},
    {0, -55, 1.6f, 0, 1.6f, {223, 169, 63}, {223, 169, 63}},
};

// How far (x, y) is outside the shape, in units; negative inside.
static float outside(const Shape& s, float x, float y) {
  // Most of the drawing is nowhere near most shapes: no need to measure.
  const float left = s.h == 0 ? s.x - s.w : s.x, right = s.h == 0 ? s.x + s.w : s.x + s.w;
  const float top = s.h == 0 ? s.y - s.r : s.y, bottom = s.h == 0 ? s.y + s.r : s.y + s.h;
  if (x < left - 2 || x > right + 2 || y < top - 2 || y > bottom + 2) return 2;
  if (s.h == 0) {
    const float kx = (x - s.x) / s.w, ky = (y - s.y) / s.r;
    return (sqrtf(kx * kx + ky * ky) - 1) * min(s.w, s.r);
  }
  const float qx = fabsf(x - (s.x + s.w / 2)) - s.w / 2 + s.r, qy = fabsf(y - (s.y + s.h / 2)) - s.h / 2 + s.r;
  const float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
  return sqrtf(ox * ox + oy * oy) + min(max(qx, qy), 0.0f) - s.r;
}

static constexpr int kBodyLeft = 54, kBodyTop = 70, kBodyW = 108, kBodyH = 112;  // units

static M5Canvas* renderBody(float scale) {
  auto* b = new M5Canvas();
  b->setPsram(true);
  b->setColorDepth(16);
  const int w = (int)ceilf(kBodyW * scale) + 2, h = (int)ceilf(kBodyH * scale) + 2;
  if (!b->createSprite(w, h)) {
    delete b;
    return nullptr;
  }
  for (int py = 0; py < h; ++py) {
    const float y = (py + 0.5f) / scale - kBodyTop;
    for (int px = 0; px < w; ++px) {
      const float x = (px + 0.5f) / scale - kBodyLeft;
      float out[3] = {0, 0, 0};
      for (const Shape& s : kShapes) {
        float cov = 0.5f - outside(s, x, y) * scale;
        if (cov <= 0) continue;
        if (cov > 1) cov = 1;
        const float top = s.h == 0 ? s.y - s.r : s.y, height = s.h == 0 ? 2 * s.r : s.h;
        float g = (y - top) / height;
        g = g < 0 ? 0 : g > 1 ? 1 : g;
        for (int k = 0; k < 3; ++k) out[k] += (s.top[k] + (s.bottom[k] - s.top[k]) * g - out[k]) * cov;
      }
      b->drawPixel(px, py, rgb((uint8_t)out[0], (uint8_t)out[1], (uint8_t)out[2]));
    }
  }
  return b;
}

// One rendering per size asked for: the screens use a handful.
static M5Canvas* body(float scale) {
  static constexpr int kSizes = 8;
  static float scales[kSizes] = {};
  static M5Canvas* bodies[kSizes] = {};
  for (int i = 0; i < kSizes; ++i) {
    if (scales[i] == scale) return bodies[i];
    if (scales[i] == 0) {
      scales[i] = scale;
      return bodies[i] = renderBody(scale);
    }
  }
  return nullptr;
}

// ── what moves ────────────────────────────────────────────────────────

static bool blinking(uint32_t t) { return t % 3600 < 130; }

void drawCaptain(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy, float scale, uint16_t lamp) {
  const auto X = [&](float u) { return cx + (int)lroundf(u * scale); };
  const auto Y = [&](float u) { return cy + (int)lroundf(u * scale); };
  const auto L = [&](float u) { return max(1, (int)lroundf(u * scale)); };
  const bool asleep = pose == Pose::Asleep;

  // The antenna and its lamp: pulsing while he thinks, low while he sleeps.
  c.drawWideLine(X(0), Y(-68), X(0), Y(-77), max(1.0f, 1.5f * scale), kWire);
  uint16_t lit = lamp ? lamp : kLamp;
  if (pose == Pose::Thinking) lit = mix(mix(0, lit, 0.35f), lit, 0.5f + 0.5f * sinf(t * (6.2832f / 1200)));
  else if (asleep) lit = mix(0, lit, 0.45f);
  c.fillSmoothCircle(X(0), Y(-81), L(9), mix(0, lit, 0.28f));
  c.fillSmoothCircle(X(0), Y(-81), L(5), lit);

  if (M5Canvas* b = body(scale)) b->pushSprite(&c, cx - (int)lroundf(kBodyLeft * scale), cy - (int)lroundf(kBodyTop * scale), (uint16_t)0);

  // His eyes: scanning while he thinks, wide while he listens, shut asleep.
  const uint16_t eye = pose == Pose::Sad ? mix(kPlate, kEye, 0.6f) : asleep ? mix(kPlate, kEye, 0.45f) : kEye;
  float dx = 0, dy = 0, eyeH = 11;
  if (pose == Pose::Thinking) {
    dx = 4 * sinf(t * (6.2832f / 2000));
    dy = -3;
  } else if (pose == Pose::Listening) {
    dy = -1;
    eyeH = 13;
  } else if (pose == Pose::Sad) {
    dy = 2;
    eyeH = 7;
  } else if (asleep || blinking(t)) {
    dy = 4;
    eyeH = 2;
  }
  for (float x : {-26.0f, 10.0f}) {
    if (eyeH > 3 && !(pose == Pose::Sad)) {
      c.fillSmoothRoundRect(X(x + dx - 2), Y(-14 + dy - 2), L(20), L(eyeH + 4), L(5), mix(kPlate, eye, 0.2f));
    }
    c.fillSmoothRoundRect(X(x + dx), Y(-14 + dy), L(16), L(eyeH), L(eyeH > 3 ? 3 : 1), eye);
  }

  // His grille: dancing while he speaks, following the mic while he listens.
  static const uint8_t kBars[7] = {5, 7, 9, 10, 9, 7, 5};
  for (int i = 0; i < 7; ++i) {
    float h = kBars[i];
    if (pose == Pose::Speaking && level > 5) {
      const int wobble = (int)((t / 60 + i * 37) % 7);
      h = constrain(level * 16 / 100 + wobble - 3, 3, 18);
    } else if (pose == Pose::Listening) {
      h = kBars[i] * (0.5f + min(level, 100) / 100.0f);
    } else if (pose == Pose::Sad) {
      h = 12 - kBars[i];
    } else if (asleep) {
      h = 2;
    }
    const int barH = L(h);
    c.fillSmoothRoundRect(X(-24 + i * 7), Y(16) - barH / 2, L(5), barH, L(1.5f), eye);
  }
}

// ── the stick ─────────────────────────────────────────────────────────

// "Sound coming in" arcs left of him, one per ~30% of mic level.
static void listenWaves(M5Canvas& c, int x, int y, int level) {
  for (int i = 0; i < 3; ++i) {
    const uint16_t col = level > 8 + i * 28 ? kWave : rgb(40, 60, 80);
    const int r = 10 + i * 9;
    c.fillArc(x, y, r, r + 3, 135, 225, col);
  }
}

// Three thought dots appearing one by one above-right.
static void thinkDots(M5Canvas& c, int x, int y, uint32_t t) {
  const int n = (t / 350) % 4;
  for (int i = 0; i < n; ++i) c.fillCircle(x + i * 11, y - i * 5, 3 + i, kDots);
}

void draw(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy) {
  static constexpr float kScale = 0.85f;
  // Lamp to chin centred on cy: his head's middle sits below it.
  drawCaptain(c, pose, level, t, cx, cy + (int)((kTopUnits - kBottomUnits) / 2 * kScale), kScale);
  if (pose == Pose::Listening) listenWaves(c, cx - 70, cy, level);
  if (pose == Pose::Thinking) thinkDots(c, cx + 62, cy - 30, t);
}

}  // namespace face
