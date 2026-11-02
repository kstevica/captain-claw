// Mac check of the mic's ring (src/audio.cpp, as it is, on a fake mic that runs on
// the simulated clock, and its task as a real thread): the main loop held up for
// seconds, as a weak link holds it in the socket, must lose nothing of what was said.
//
//   c++ -std=c++17 -O1 -DBOARD_EPAPER154 -Itest/host/fake -Ipreview/include -Isrc -o /tmp/mic_stall test/host/mic_stall.cpp src/audio.cpp && /tmp/mic_stall
#include <chrono>
#include <cstdio>
#include <thread>

#include "M5Unified.h"
#include "audio.h"

uint32_t g_simMs = 1000;
FakeM5 M5;
void delay(uint32_t ms) {
  g_simMs += ms;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
namespace hw {
void micStarted() {}
}

static int fails = 0;
#define CHECK(c, ...)            \
  do {                           \
    if (!(c)) {                  \
      ++fails;                   \
      printf("FAIL: " __VA_ARGS__); \
      printf("\n");              \
    }                            \
  } while (0)

// What the sink got: the fake's counter, so a hole is a jump.
static uint32_t got = 0, holes = 0, lost = 0;
static int next = -1;
static void sink(const int16_t* s, size_t n) {
  for (size_t i = 0; i < n; ++i, ++got) {
    if (next >= 0 && s[i] != next) {
      ++holes;
      lost += (s[i] - next + FakeMic::kPeriod) % FakeMic::kPeriod;
    }
    next = (s[i] + 1) % FakeMic::kPeriod;
  }
}

// `ms` of the simulated clock, a little faster than real time (the feeding task runs
// on real time, every 8 ms). `poll`: the loop is running; else it's held up.
static void run(uint32_t ms, bool poll) {
  for (uint32_t t = 0; t < ms; t += 16) {
    g_simMs += 16;
    if (poll) audio::micPoll(sink);
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
  }
}

static void reset() {
  got = holes = lost = 0;
  next = -1;
}

int main() {
  audio::begin(200, 0);

  // Held up for 1 s and for 3 s in the middle of a recording: all of it, no holes.
  reset();
  audio::micStart();
  run(1000, true);
  run(1000, false);
  CHECK(audio::micBehind(), "after 1 s away the loop should have blocks to take");
  run(1000, true);
  CHECK(!audio::micBehind(), "the loop should have caught up");
  run(3000, false);
  run(1000, true);
  audio::micStop(sink);
  const uint32_t expect = (7000 - 200) * 16;  // (the last blocks may still be open when it stops)
  CHECK(holes == 0, "held up 1 s + 3 s: %u holes, %u samples lost", holes, lost);
  CHECK(got >= expect, "held up 1 s + 3 s: got %u samples, wanted at least %u", got, expect);
  printf("held up 1 s + 3 s: %u samples, %u holes\n", got, holes);

  // Held up for longer than the ring: a hole, but what's kept is intact and in order.
  reset();
  audio::micStart();
  run(500, true);
  run(6000, false);
  run(1000, true);
  audio::micStop(sink);
  CHECK(holes == 1, "held up 6 s: %u holes (wanted the one)", holes);
  CHECK(lost > 0 && lost <= 2500 * 16, "held up 6 s: %u samples lost (the 4 s ring keeps the rest)", lost);
  printf("held up 6 s: %u samples, %u hole, %u lost\n", got, holes, lost);

  // The loop keeping up all along.
  reset();
  audio::micStart();
  run(2000, true);
  audio::micStop(sink);
  CHECK(holes == 0, "keeping up: %u holes", holes);
  printf("keeping up: %u samples, %u holes\n", got, holes);

  printf(fails ? "%d FAILED\n" : "all good\n", fails);
  return fails ? 1 : 0;
}
