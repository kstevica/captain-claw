// The Waveshare 1.54" e-paper's controller (SSD1681, 200×200, 1 bit), driven
// directly: M5GFX's panel can't be reset between refreshes or run these
// waveforms. A frame is 5000 bytes, a row at a time, the top-left pixel in the
// top bit, 1 = white. Called from one task only; each call blocks until the
// panel has finished.
#pragma once

#include <Arduino.h>

namespace epd {

constexpr int kWidth = 200, kHeight = 200, kFrameBytes = kWidth * kHeight / 8;

void begin();
// The whole panel flashes to the frame (1.8 s), from a reset controller.
bool full(const uint8_t* frame);
// Only the pixels that differ change, without a flash (0.6 s). `shown` is
// what the panel shows now: the controller's own copy is gone after a deep
// sleep, and this puts it back.
bool partial(const uint8_t* frame, const uint8_t* shown);
// The controller's deep sleep (the picture stays). The next refresh resets it.
void sleep();

}  // namespace epd
