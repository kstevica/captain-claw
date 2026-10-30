// The Waveshare ESP32-S3-ePaper-1.54 (pins from its schematic and examples).
// Its codec is the StickS3's ES8311 at the same I2C address, so the register
// values are the ones M5Unified uses there.
#include "hw.h"

#include <Arduino.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>
#if defined(EPAPER154_TOUCH)
#include <lgfx/v1/panel/Panel_GDEW0154D67.hpp>
#include <lgfx/v1/touch/Touch_FT5x06.hpp>
#endif
#include <soc/usb_serial_jtag_struct.h>
#include <sys/time.h>
#include <time.h>

namespace hw {

static constexpr gpio_num_t kEpdPower = GPIO_NUM_6;     // low = the panel's 3.3 V on
static constexpr gpio_num_t kAudioPower = GPIO_NUM_42;  // low = codec and amplifier supply on
static constexpr gpio_num_t kAmp = GPIO_NUM_46;         // high = the speaker amplifier on
static constexpr gpio_num_t kKeepOn = GPIO_NUM_17;      // high = stay on when running from the battery
static constexpr int kBatteryAdc = 4;                   // half the battery voltage
static constexpr int kSda = 47, kScl = 48;
static constexpr int kMck = 14, kBck = 15, kWs = 38, kCodecIn = 16, kCodecOut = 45;
static constexpr uint8_t kCodec = 0x18, kRtc = 0x51;
static constexpr uint32_t kCodecHz = 100000;

// ── audio ──────────────────────────────────────────────────────────────

struct CodecReg {
  uint8_t reg, value;
};

static void codecWrite(const CodecReg* regs, size_t n) {
  for (size_t i = 0; i < n; ++i) M5.In_I2C.writeRegister8(kCodec, regs[i].reg, regs[i].value, kCodecHz);
}

static bool speakerEnabled(void*, bool on) {
  static constexpr CodecReg kOn[] = {
      {0x00, 0x80},  // start the state machine
      {0x01, 0xB5},  // the clock comes from BCLK
      {0x02, 0x18},  // ×3
      {0x0D, 0x01},  // analog power up
      {0x12, 0x00},  // DAC power up
      {0x13, 0x10},  // output drive on
      {0x32, 0xBF},  // DAC volume: 0 dB
      {0x37, 0x08},  // no DAC equalizer
  };
  if (on) {
    codecWrite(kOn, sizeof(kOn) / sizeof(kOn[0]));
    digitalWrite(kAmp, HIGH);
  } else {
    digitalWrite(kAmp, LOW);
  }
  return true;
}

static bool micEnabled(void*, bool on) {
  static constexpr CodecReg kOn[] = {
      {0x00, 0x80},  // start the state machine
      {0x01, 0xBA},  // the clock comes from BCLK
      {0x02, 0x18},  // ×3
      {0x0D, 0x01},  // analog power up
      {0x0E, 0x02},  // the mic amplifier and the ADC on
      {0x14, 0x10},  // Mic1, lowest amplifier gain
      {0x17, 0xFF},  // ADC volume: highest
      {0x1C, 0x6A},  // no ADC equalizer, DC offset cancelled
  };
  static constexpr CodecReg kOff[] = {{0x0D, 0xFC}, {0x0E, 0x6A}, {0x00, 0x00}};
  if (on) codecWrite(kOn, sizeof(kOn) / sizeof(kOn[0]));
  else codecWrite(kOff, sizeof(kOff) / sizeof(kOff[0]));
  return true;
}

// The ES8311 arms its capture path only through a write made while the I2S
// clock runs, once per power-up (M5Unified does this on the StopWatch, whose
// codec this is, but keeps that hook private; hence a call after Mic.begin()).
void micStarted() {
  static bool armed = false;
  if (armed) return;
  bool ok = false;
  for (int i = 0; i < 4; ++i) {
    delay(40);
    ok = M5.In_I2C.writeRegister8(kCodec, 0x0D, 0x01, kCodecHz);
  }
  armed = ok;
}

// M5Unified sets these per board and keeps them protected.
struct MicCallback : m5::Mic_Class {
  using m5::Mic_Class::setCallback;
};
struct SpeakerCallback : m5::Speaker_Class {
  using m5::Speaker_Class::setCallback;
};

static void audioBegin() {
  auto spk = M5.Speaker.config();
  spk.pin_mck = kMck;
  spk.pin_bck = kBck;
  spk.pin_ws = kWs;
  spk.pin_data_out = kCodecOut;
  spk.i2s_port = I2S_NUM_0;
  spk.magnification = 1;
  spk.sample_rate = 22050;
  spk.stereo = true;
  spk.buzzer = false;
  spk.use_dac = false;
  spk.dac_zero_level = 0;
  static_cast<SpeakerCallback&>(M5.Speaker).setCallback(nullptr, speakerEnabled);
  M5.Speaker.config(spk);

  auto mic = M5.Mic.config();
  mic.pin_mck = kMck;
  mic.pin_bck = kBck;
  mic.pin_ws = kWs;
  mic.pin_data_in = kCodecIn;
  mic.i2s_port = I2S_NUM_1;
  static_cast<MicCallback&>(M5.Mic).setCallback(nullptr, micEnabled);
  M5.Mic.config(mic);
}

// ── touch ──────────────────────────────────────────────────────────────

#if defined(EPAPER154_TOUCH)
// M5.Touch reads through a display object: this one never draws (the panel
// has its own driver), it only carries the touch chip and the rotation.
static lgfx::Panel_GDEW0154D67 touchPanel;
static lgfx::Touch_FT5x06 touchChip;
static lgfx::LGFX_Device touchDevice;

static void touchBegin() {
  auto c = touchChip.config();
  c.i2c_port = 1;
  c.i2c_addr = 0x38;
  c.pin_sda = kSda;
  c.pin_scl = kScl;
  c.pin_int = GPIO_NUM_21;
  c.pin_rst = GPIO_NUM_7;
  c.x_min = 0;
  c.x_max = 199;
  c.y_min = 0;
  c.y_max = 199;
  c.freq = 400000;
  touchChip.config(c);
  auto p = touchPanel.config();  // its size is what touches are scaled to
  p.panel_width = p.memory_width = board::kScreenW;
  p.panel_height = p.memory_height = board::kScreenH;
  touchPanel.config(p);
  touchPanel.setRotation(0);
  touchPanel.setTouch(&touchChip);
  touchDevice.setPanel(&touchPanel);
  if (touchPanel.initTouch()) M5.Touch.begin(&touchDevice);
  else Serial.println("[claw] touch chip not answering");
}

void setTouchRotation(uint8_t rotation) { touchPanel.setRotation(rotation & 3); }
#else
void setTouchRotation(uint8_t) {}
#endif

// ── start, buttons ─────────────────────────────────────────────────────

void begin() {
  // On battery the press that switched it on holds the power only until it
  // lets go: stay on first. (After a deep sleep the pin is still latched high.)
  pinMode(kKeepOn, OUTPUT);
  digitalWrite(kKeepOn, HIGH);
  rtc_gpio_hold_dis(kKeepOn);
  pinMode(kEpdPower, OUTPUT);
  digitalWrite(kEpdPower, LOW);
  pinMode(kAudioPower, OUTPUT);
  digitalWrite(kAudioPower, LOW);
  pinMode(kAmp, OUTPUT);
  digitalWrite(kAmp, LOW);
  pinMode(board::kTalkPin, INPUT_PULLUP);
  pinMode(board::kSidePin, INPUT_PULLUP);
  analogSetPinAttenuation(kBatteryAdc, ADC_11db);
  delay(5);  // the codec's supply
  M5.In_I2C.begin(I2C_NUM_1, kSda, kScl);
  audioBegin();
#if defined(EPAPER154_TOUCH)
  touchBegin();
#endif
  // The press that switched it on or woke it isn't a click of the side button.
  for (uint32_t t0 = millis(); digitalRead(board::kSidePin) == LOW && millis() - t0 < 3000;) delay(5);
  update();
}

void update() {
  M5.update();
  const uint32_t ms = m5gfx::millis();
  M5.BtnA.setRawState(ms, gpio_get_level(board::kTalkPin) == 0);
  M5.BtnB.setRawState(ms, gpio_get_level(board::kSidePin) == 0);
}

// ── battery ────────────────────────────────────────────────────────────

int batteryMv() {
  static int mv = 0;
  static uint32_t at = 0;
  if (!at || millis() - at > 5000) {
    at = millis();
    uint32_t sum = 0;
    for (int i = 0; i < 8; ++i) sum += analogReadMilliVolts(kBatteryAdc);
    mv = (int)(sum / 8) * 2;
  }
  return mv;
}

int batteryLevel() {
  struct Point {
    int mv, pct;
  };
  static constexpr Point kCurve[] = {{4150, 100}, {4060, 90}, {3980, 80}, {3920, 70}, {3870, 60}, {3820, 50},
                                     {3790, 40},  {3770, 30}, {3740, 20}, {3680, 10}, {3450, 5},  {3300, 0}};
  const int mv = batteryMv();
  if (mv < 2500) return -1;  // none connected
  if (mv >= kCurve[0].mv) return 100;
  for (size_t i = 1; i < sizeof(kCurve) / sizeof(kCurve[0]); ++i) {
    if (mv >= kCurve[i].mv) {
      const Point& hi = kCurve[i - 1];
      const Point& lo = kCurve[i];
      return lo.pct + (mv - lo.mv) * (hi.pct - lo.pct) / (hi.mv - lo.mv);
    }
  }
  return 0;
}

// No pin says so. A USB host sends start-of-frame packets, and a charger holds
// the battery near its 4.2 V (a battery alone falls off that within minutes).
bool onUsbPower() {
  static uint32_t lastFrame = 0, lastMs = 0;
  static bool host = false;
  if (!lastMs || millis() - lastMs >= 250) {
    lastMs = millis();
    const uint32_t frame = USB_SERIAL_JTAG.fram_num.sof_frame_index;
    host = frame != lastFrame;
    lastFrame = frame;
  }
  const int mv = batteryMv();
  return host || mv >= 4180 || mv < 2500;
}

bool charging() { return onUsbPower() && batteryMv() >= 2500 && batteryMv() < 4150; }

// ── power ──────────────────────────────────────────────────────────────

void prepareSleep() {
  digitalWrite(kAmp, LOW);
  digitalWrite(kAudioPower, HIGH);
  digitalWrite(kEpdPower, HIGH);
  digitalWrite(kKeepOn, HIGH);
  rtc_gpio_hold_en(kKeepOn);
}

void powerOff() {
  digitalWrite(kAmp, LOW);
  digitalWrite(kAudioPower, HIGH);
  digitalWrite(kEpdPower, HIGH);
  rtc_gpio_hold_dis(kKeepOn);
  digitalWrite(kKeepOn, LOW);
  // On battery the power goes as soon as PWR is let go, and this ends there.
  for (uint32_t t0 = millis(); digitalRead(board::kSidePin) == LOW && millis() - t0 < 10000;) delay(10);
  delay(300);
  // Still here: on USB.
  const uint64_t pins = (1ULL << board::kTalkPin) | (1ULL << board::kSidePin);
  for (gpio_num_t pin : {board::kTalkPin, board::kSidePin}) {
    rtc_gpio_pullup_en(pin);
    rtc_gpio_pulldown_dis(pin);
  }
  esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
  esp_sleep_enable_ext1_wakeup(pins, ESP_EXT1_WAKEUP_ANY_LOW);
  esp_deep_sleep_start();
}

// ── the clock ──────────────────────────────────────────────────────────

static uint8_t bcd(uint8_t v) { return ((v / 10) << 4) | (v % 10); }
static uint8_t unbcd(uint8_t v) { return ((v >> 4) * 10) + (v & 0x0F); }

// Days since 1970 of a civil date (the RTC keeps UTC, whatever TZ says).
static int64_t daysFromCivil(int y, int m, int d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097LL + (int64_t)doe - 719468;
}

void clockFromRtc() {
  uint8_t r[7];
  if (!M5.In_I2C.readRegister(kRtc, 0x04, r, sizeof(r), 400000)) return;
  if (r[0] & 0x80) return;  // the oscillator stopped: the time means nothing
  const int year = 2000 + unbcd(r[6]);
  if (year < 2025) return;
  const int64_t days = daysFromCivil(year, unbcd(r[5] & 0x1F), unbcd(r[3] & 0x3F));
  struct timeval tv = {};
  tv.tv_sec = (time_t)(days * 86400 + unbcd(r[2] & 0x3F) * 3600 + unbcd(r[1] & 0x7F) * 60 + unbcd(r[0] & 0x7F));
  settimeofday(&tv, nullptr);
}

void clockToRtc() {
  const time_t now = time(nullptr);
  struct tm utc;
  gmtime_r(&now, &utc);
  const uint8_t r[7] = {bcd(utc.tm_sec), bcd(utc.tm_min), bcd(utc.tm_hour), bcd(utc.tm_mday),
                        (uint8_t)utc.tm_wday, bcd(utc.tm_mon + 1), bcd(utc.tm_year % 100)};
  M5.In_I2C.writeRegister(kRtc, 0x04, r, sizeof(r), 400000);
}

}  // namespace hw
