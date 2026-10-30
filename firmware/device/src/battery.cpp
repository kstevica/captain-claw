#include "battery.h"

#include <M5Unified.h>
#include <time.h>

#include "hw.h"
#include "telemetry.h"

namespace battery {

struct Sample {
  uint32_t t;        // the clock (NTP time; the RTC keeps it through sleep)
  uint8_t pct;
  uint8_t charging;
};
static constexpr int kKeep = 16;
RTC_DATA_ATTR static Sample ring[kKeep];
RTC_DATA_ATTR static uint8_t ringN = 0;
static uint32_t lastPollMs = 0;

bool charging() { return hw::charging(); }

void begin() { lastPollMs = 0; }

void sample() {
  if (!telemetry::clockSet()) return;  // no clock yet: a sample would have no time
  const int pct = hw::batteryLevel();
  if (pct < 0) return;
  const uint32_t now = (uint32_t)time(nullptr);
  const uint8_t chg = charging() ? 1 : 0;
  if (ringN && ring[ringN - 1].charging != chg) ringN = 0;  // a new phase
  if (ringN && ring[ringN - 1].pct == pct && now - ring[ringN - 1].t < 600) return;  // nothing new yet
  if (ringN == kKeep) {  // full: keep the oldest (the window's long end), drop the next
    memmove(ring + 1, ring + 2, sizeof(Sample) * (kKeep - 2));
    ringN = kKeep - 1;
  }
  ring[ringN++] = {now, (uint8_t)pct, chg};
}

void poll() {
  if (lastPollMs && millis() - lastPollMs < 60000) return;
  lastPollMs = millis();
  sample();
}

// Per cent an hour over the window, signed (+ charging); 0 until it spans
// at least five minutes and two per cent.
static float ratePerHour() {
  if (ringN < 2) return 0;
  const Sample& a = ring[0];
  const Sample& b = ring[ringN - 1];
  const uint32_t dt = b.t - a.t;
  const int dp = (int)b.pct - (int)a.pct;
  if (dt < 300 || abs(dp) < 2) return 0;
  return dp * 3600.0f / dt;
}

int minutesLeft() {
  if (charging()) return -1;
  const float r = ratePerHour();
  if (r >= 0) return -1;
  const int pct = hw::batteryLevel();
  return (int)min(60.0f * 24 * 30, pct / -r * 60);
}

int minutesToFull() {
  if (!charging()) return -1;
  const int pct = hw::batteryLevel();
  if (pct >= 97) return 0;
  const float r = ratePerHour();
  if (r <= 0) return -1;
  return (int)min(60.0f * 24, (100 - pct) / r * 60);
}

}  // namespace battery
