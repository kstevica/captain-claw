// Captain: the robot from the app icon, drawn with shapes (no image files).
#pragma once

#include <M5Unified.h>

namespace face {

enum class Pose : uint8_t { Idle, Listening, Thinking, Speaking, Sad, Asleep };

// Milliseconds between frames that keep him moving smoothly.
constexpr uint32_t kFrameMs = 120;

// Where he ends, in units of `scale` from the middle of his head: the top of
// the antenna's lamp and his chin.
constexpr int kTopUnits = 86, kBottomUnits = 40;

// Captain at any size, the middle of his head on (cx, cy). `level` 0..100
// drives his grille, `t` is millis() for blinking and motion, `lamp` is the
// antenna's colour (0: the icon's red).
void drawCaptain(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy, float scale, uint16_t lamp = 0);

// For the stick's 240×135 screen: Captain centred on (cx, cy) in the area under
// its top bar, with sound coming in or thought dots beside him (that screen has
// no ring to show them).
void draw(M5Canvas& c, Pose pose, int level, uint32_t t, int cx, int cy);

}  // namespace face
