// What the device knows about itself and its surroundings: battery, Wi-Fi,
// how it's held (IMU), room temperature and humidity (PaperColor's SHT40),
// the clock. A snapshot rides along with every turn; Flight Deck turns it
// into one line of context for the agent (device_context_line()).
#pragma once

#include <Arduino.h>
#if __has_include(<ArduinoJson.h>)  // not in the Mac preview, which only draws
#include <ArduinoJson.h>
#define TELEMETRY_JSON 1
#endif

namespace telemetry {

// Sets the clock from the RTC (devices that have one) and probes the sensors.
void begin();
// Before a deep sleep (the Waveshare board: how warm it is, for the next wake's reading).
void beforeSleep();
// From loop(): samples motion a few times a second; keeps the RTC in step
// with NTP.
void poll();
#ifdef TELEMETRY_JSON
// The readings now, for {"type":"rec_end","ctx":{…}}.
void fill(JsonObject ctx);
#endif

// Room temperature (°C) and relative humidity (%), read at most every 30 s;
// false on a device without the sensor (or when it doesn't answer).
bool climate(float* tempC, float* humidity);

// Plugged in: VBUS where the PMIC reports it, else charging / full.
bool onUsbPower();

// The local time is known (NTP, or an RTC set by it earlier).
bool clockSet();

// Raise to wake: true once when the screen has just turned from hanging or
// upright to facing up, with movement (a wrist raise); lowered(): the
// reverse. Devices without an IMU never report either.
bool raised();
bool lowered();

}  // namespace telemetry
