#include "telemetry.h"

#include <M5Unified.h>
#include <WiFi.h>
#include <esp_sntp.h>
#include <math.h>
#include <time.h>

#include "board.h"
#include "hw.h"

namespace telemetry {

static constexpr time_t kSane = 1700000000;  // before this the clock was never set

// ── motion (StickS3, StopWatch: BMI270) ──────────────────────────────

static bool haveImu = false;
static float gx = 0, gy = 0, gz = 1;  // gravity, low-passed (in g)
static float shake = 0;               // recent deviation of |a| from 1 g (in g)
static uint32_t lastSampleMs = 0;
static int samples = 0;

static void sampleImu() {
  float ax, ay, az;
  if (!M5.Imu.getAccel(&ax, &ay, &az)) return;
  float mag = sqrtf(ax * ax + ay * ay + az * az);
  if (mag < 0.2f || mag > 8.0f) return;  // a glitch, not a reading
  const float k = samples < 5 ? 1.0f : 0.25f;
  gx += (ax - gx) * k;
  gy += (ay - gy) * k;
  gz += (az - gz) * k;
  float dev = fabsf(mag - 1.0f);
  shake = shake * 0.85f + dev * 0.15f;  // ~1.5 s memory at 10 Hz
  ++samples;
}

// The IMU lies parallel to the screen, so the angle between gravity and its
// z axis is how far the screen is from lying flat, whichever way the chip
// is mounted (which side is up isn't known for every board, so it isn't said).
static const char* orientation() {
  float mag = sqrtf(gx * gx + gy * gy + gz * gz);
  if (mag < 0.5f) return "";
  float flat = fabsf(gz) / mag;  // 1 = flat, 0 = upright
  return flat > 0.9f ? "lying flat" : flat < 0.4f ? "held upright" : "tilted";
}

static const char* movement() {
  return shake < 0.015f ? "still" : shake < 0.08f ? "moving gently" : shake < 0.3f ? "moving" : "shaking";
}

// The screen facing up (looked at) or away (hanging, upright), with a band
// between so a slight tilt doesn't flip it; a change with movement is a
// raise or a lowering.
static int8_t facing = 0;  // 1 up, -1 away, 0 not known yet
static bool raisedFlag = false, loweredFlag = false;

static void trackFacing() {
  const float mag = sqrtf(gx * gx + gy * gy + gz * gz);
  if (mag < 0.5f || samples < 5) return;
  const float flat = fabsf(gz) / mag;
  const int8_t now = flat > 0.75f ? 1 : flat < 0.55f ? -1 : facing;
  if (now == facing) return;
  if (facing == -1 && now == 1 && shake > 0.02f) raisedFlag = true;
  if (facing == 1 && now == -1) loweredFlag = true;
  facing = now;
}

bool raised() {
  const bool r = raisedFlag;
  raisedFlag = false;
  return r;
}

bool lowered() {
  const bool l = loweredFlag;
  loweredFlag = false;
  return l;
}

// ── room climate (PaperColor: SHT40; Waveshare 1.54": SHTC3) ─────────────────────────────────

static constexpr uint8_t kSht40 = 0x44;
static bool haveClimate = board::kHasClimate;
static float lastT = NAN, lastRh = NAN;
static uint32_t climateAt = 0;

#if defined(BOARD_EPAPER154)
static constexpr uint8_t kShtc3 = 0x70;

static bool shtc3Command(uint16_t command) {
  const uint8_t bytes[2] = {(uint8_t)(command >> 8), (uint8_t)command};
  if (!M5.In_I2C.start(kShtc3, false, 400000)) return false;
  const bool ok = M5.In_I2C.write(bytes, 2);
  M5.In_I2C.stop();
  return ok;
}

static bool readSht40(float* t, float* rh) {
  if (!shtc3Command(0x3517)) return false;  // wake up
  delay(1);
  if (!shtc3Command(0x7866)) return false;  // measure, temperature first (~12 ms)
  delay(15);
  uint8_t buf[6];
  if (!M5.In_I2C.start(kShtc3, true, 400000)) return false;
  const bool ok = M5.In_I2C.read(buf, 6);
  M5.In_I2C.stop();
  shtc3Command(0xB098);  // back to sleep
  if (!ok) return false;
  const uint16_t rawT = (buf[0] << 8) | buf[1], rawH = (buf[3] << 8) | buf[4];
  *t = -45.0f + 175.0f * rawT / 65536.0f;
  *rh = constrain(100.0f * rawH / 65536.0f, 0.0f, 100.0f);
  return true;
}
#else
static bool readSht40(float* t, float* rh) {
  uint8_t cmd = 0xFD;  // measure, high precision (~8.3 ms)
  uint8_t buf[6];
  if (!M5.In_I2C.start(kSht40, false, 400000)) return false;
  M5.In_I2C.write(&cmd, 1);
  M5.In_I2C.stop();
  delay(10);
  if (!M5.In_I2C.start(kSht40, true, 400000)) return false;
  bool ok = M5.In_I2C.read(buf, 6);
  M5.In_I2C.stop();
  if (!ok) return false;
  uint16_t rawT = (buf[0] << 8) | buf[1], rawH = (buf[3] << 8) | buf[4];
  *t = -45.0f + 175.0f * rawT / 65535.0f;
  *rh = constrain(-6.0f + 125.0f * rawH / 65535.0f, 0.0f, 100.0f);
  return true;
}
#endif

#if defined(BOARD_EPAPER154)
// This small board warms its own sensor. In a room of about 24 °C, on USB, it
// read 28.8 °C straight after a long sleep, then rose to 32.7 °C awake, with a
// time constant of 260 s. On battery nothing should heat it while it sleeps.
static constexpr float kUsbHeatC = 4.8f, kAwakeHeatC = 4.0f, kAwakeTauS = 260.0f;
RTC_DATA_ATTR static float rtcHeat = 0;     // what being awake had added when it went to sleep
RTC_DATA_ATTR static time_t rtcHeatAt = 0;
static float bootHeat = 0;                  // what's left of that at this boot (it cools as it warmed)

static float awakeHeat() { return kAwakeHeatC + (bootHeat - kAwakeHeatC) * expf(-(millis() / 1000.0f) / kAwakeTauS); }

static float saturationHpa(float c) { return 6.112f * expf(17.62f * c / (243.12f + c)); }  // Magnus

static void roomFromSensor(float* t, float* rh) {
  const float room = *t - (hw::onUsbPower() ? kUsbHeatC : 0.0f) - awakeHeat();
  *rh = constrain(*rh * saturationHpa(*t) / saturationHpa(room), 0.0f, 100.0f);  // the same air, cooler
  *t = room;
}
#endif

bool climate(float* tempC, float* humidity) {
  if (!haveClimate) return false;
  if (isnan(lastT) || millis() - climateAt > 30000) {
    climateAt = millis();
    float t, rh;
    if (readSht40(&t, &rh)) {
#if defined(BOARD_EPAPER154)
      roomFromSensor(&t, &rh);
#endif
      lastT = t;
      lastRh = rh;
    }
  }
  if (isnan(lastT)) return false;
  *tempC = lastT;
  *humidity = lastRh;
  return true;
}

// ── clock ────────────────────────────────────────────────────────────

static volatile bool ntpSynced = false;

static void onNtpSync(struct timeval*) { ntpSynced = true; }  // SNTP task: only a flag

bool clockSet() { return time(nullptr) > kSane; }

// ── battery ──────────────────────────────────────────────────────────

bool onUsbPower() {
#if defined(BOARD_EPAPER154)
  return hw::onUsbPower();
#endif
  int vbus = M5.Power.getVBUSVoltage();
  if (vbus > 0) return vbus > 4000;
  return hw::charging() || hw::batteryLevel() >= 99;
}

// ── public ───────────────────────────────────────────────────────────

void beforeSleep() {
#if defined(BOARD_EPAPER154)
  rtcHeat = awakeHeat();
  rtcHeatAt = time(nullptr);
#endif
}

void begin() {
  haveImu = M5.Imu.isEnabled();
#if defined(BOARD_EPAPER154)
  hw::clockFromRtc();
  const time_t now = time(nullptr);
  bootHeat = rtcHeatAt && now > rtcHeatAt ? rtcHeat * expf(-(float)(now - rtcHeatAt) / kAwakeTauS) : rtcHeat;
#else
  if (M5.Rtc.isEnabled()) {
    m5::rtc_datetime_t dt;
    // An RTC that kept time (set from NTP on an earlier boot) gives the clock
    // at once — before Wi-Fi, and after a deep sleep.
    if (M5.Rtc.getDateTime(&dt) && dt.date.year >= 2025 && dt.date.year < 2100) {
      // M5Unified switches TZ to GMT0 for this and means to put it back,
      // but newlib reuses the string it kept: restore our own copy.
      const char* tz = getenv("TZ");
      String keep = tz ? tz : "";
      M5.Rtc.setSystemTimeFromRtc();
      if (keep.length()) setenv("TZ", keep.c_str(), 1);
      else unsetenv("TZ");
      tzset();
    }
  }
#endif
  sntp_set_time_sync_notification_cb(onNtpSync);
  if (haveClimate) {
    float t, rh;
    haveClimate = readSht40(&t, &rh);  // not there: never ask again
    if (haveClimate) {
#if defined(BOARD_EPAPER154)
      roomFromSensor(&t, &rh);
#endif
      lastT = t, lastRh = rh;
      climateAt = millis();
    }
  }
}

void poll() {
  if (haveImu && millis() - lastSampleMs >= 100) {
    trackFacing();
    lastSampleMs = millis();
    sampleImu();
  }
  if (ntpSynced) {  // keep the RTC (UTC) in step, from the main loop's I2C
    ntpSynced = false;
#if defined(BOARD_EPAPER154)
    hw::clockToRtc();
#else
    if (M5.Rtc.isEnabled()) {
      time_t now = time(nullptr);
      struct tm utc;
      gmtime_r(&now, &utc);
      M5.Rtc.setDateTime(&utc);
    }
#endif
  }
}

void fill(JsonObject ctx) {
  JsonObject bat = ctx["battery"].to<JsonObject>();
  int pct = hw::batteryLevel();
  if (pct >= 0) bat["percent"] = pct;
  int mv = hw::batteryMv();
  if (mv > 0) bat["mv"] = mv;
  bat["charging"] = hw::charging();
  bat["usb"] = onUsbPower();

  if (WiFi.status() == WL_CONNECTED) {
    JsonObject wifi = ctx["wifi"].to<JsonObject>();
    wifi["ssid"] = WiFi.SSID();
    wifi["rssi"] = WiFi.RSSI();
  }
  if (haveImu && samples >= 5) {
    JsonObject motion = ctx["motion"].to<JsonObject>();
    const char* o = orientation();
    if (*o) motion["orientation"] = o;
    motion["movement"] = movement();
  }
  float t, rh;
  if (climate(&t, &rh)) {
    JsonObject env = ctx["environment"].to<JsonObject>();
    env["temperature_c"] = roundf(t * 10) / 10;
    env["humidity"] = roundf(rh);
  }
  if (clockSet()) {
    time_t now = time(nullptr);
    struct tm tm;
    localtime_r(&now, &tm);
    char buf[40];
    strftime(buf, sizeof(buf), "%a %Y-%m-%d %H:%M (%z)", &tm);
    ctx["local_time"] = buf;
  }
}

}  // namespace telemetry
