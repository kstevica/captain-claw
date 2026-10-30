// How the firmware reaches a board's power, battery and buttons. On the
// M5Stack boards that's M5Unified; on the Waveshare board (not one M5Unified
// knows) M5.begin() is skipped, since its board detection would probe this
// board's pins, and hw_epaper154.cpp sets up M5Unified's speaker, mic, button
// and I2C classes by hand.
#pragma once

#include <M5Unified.h>

#include "board.h"

namespace hw {

#if defined(BOARD_EPAPER154)

void begin();
void update();
void micStarted();  // right after M5.Mic.begin(): what the codec needs with its clock running
int batteryLevel();  // percent, from the voltage; -1: no battery
int batteryMv();
bool charging();
bool onUsbPower();
// The touch screen turns with the picture (0..3; nothing without touch).
void setTouchRotation(uint8_t rotation);
// Just before deep sleep: on battery the board stays powered only while G17
// is kept high through it.
void prepareSleep();
// Off, on battery (only PWR brings it back). On USB the board can't lose power:
// it goes to deep sleep instead, woken by a button.
[[noreturn]] void powerOff();
// The real-time clock (PCF85063, in UTC) to the system clock, and back.
void clockFromRtc();
void clockToRtc();

#else

inline void begin() {
  auto cfg = M5.config();
  cfg.internal_mic = true;
  cfg.internal_spk = true;
  if (board::kEpaper) cfg.clear_display = false;  // the paper keeps what it showed while asleep
  M5.begin(cfg);
}
inline void update() { M5.update(); }
inline void micStarted() {}
inline int batteryLevel() { return M5.Power.getBatteryLevel(); }
inline int batteryMv() { return M5.Power.getBatteryVoltage(); }
inline bool charging() { return M5.Power.isCharging() == m5::Power_Class::is_charging; }

#endif

}  // namespace hw
