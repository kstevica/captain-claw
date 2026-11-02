// Mac check of the answer's ring (src/audio.cpp, as it is, on a fake
// speaker): a long answer sent far faster than it's spoken, the deck holding
// when asked; a long pause; a deck that doesn't hold; replay's rules.
//
//   c++ -std=c++17 -O1 -DBOARD_EPAPER154 -Itest/host/fake -Ipreview/include -Isrc -o /tmp/audio_ring test/host/audio_ring.cpp src/audio.cpp && /tmp/audio_ring
#include <cstdio>
#include <vector>

#include "M5Unified.h"
#include "audio.h"

uint32_t g_simMs = 1000;
FakeM5 M5;
void delay(uint32_t ms) {
  g_simMs += ms;
  M5.Speaker.advance();
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

static constexpr uint32_t kRate = 24000;
static int16_t sampleAt(size_t i) { return (int16_t)((i * 7919u) % 60001u) - 30000; }

// Gaps or repeats in what was heard against what was sent: how many places
// it doesn't follow on (0 = exactly what was sent, in order).
static int breaks(const std::vector<int16_t>& heard, size_t* firstMiss) {
  size_t k = 0;
  int n = 0;
  *firstMiss = SIZE_MAX;
  for (size_t i = 0; i < heard.size(); ++i, ++k) {
    if (heard[i] == sampleAt(k)) continue;
    ++n;
    if (*firstMiss == SIZE_MAX) *firstMiss = i;
    // find where it picks up again (a jump forward or back within 2 s)
    size_t j = k > 48000 ? k - 48000 : 0;
    for (; j < k + 48000 && heard[i] != sampleAt(j); ++j) {}
    k = j;
  }
  return n;
}

struct Run {
  std::vector<int16_t> heard;
  bool clipCut, hasClip;
  uint32_t ms;
};

// `seconds` of answer, the deck sending `perStep` 4 KB frames every 10 ms
// (3: 1.2 MB/s, 25x faster than it's spoken — faster than the voice is made
// or Wi-Fi carries it); `honourHold`: it stops while the device holds, 300 ms
// after it asks (what's on its way by then still lands);
// paused from `pauseAt` for `pauseFor` s.
static Run play(double seconds, int perStep, bool honourHold, double pauseAt = -1, double pauseFor = 0) {
  M5.Speaker.heard.clear();
  audio::newClip();
  audio::playBegin(kRate, 0);
  const size_t total = (size_t)(seconds * kRate);
  size_t sent = 0;
  bool deckHeld = false, asked = false;
  uint32_t askedAt = 0, t0 = g_simMs;
  bool paused = false;
  while (audio::playing()) {
    // the deck
    if (asked != deckHeld && g_simMs - askedAt >= 300) deckHeld = asked;
    if (!(honourHold && deckHeld)) {
      for (int f = 0; f < perStep && sent < total; ++f) {
        std::vector<uint8_t> frame;
        for (int s = 0; s < 2048 && sent < total; ++s, ++sent) {
          const int16_t v = sampleAt(sent);
          frame.push_back((uint8_t)v);
          frame.push_back((uint8_t)(v >> 8));
        }
        audio::playFeed(frame.data(), frame.size());
      }
      if (sent == total) audio::playEnd();
    }
    // the device's loop
    audio::playPoll();
    const bool want = audio::holdWanted();
    if (want != asked) {
      asked = want;
      askedAt = g_simMs;
    }
    const double t = (g_simMs - t0) / 1000.0;
    if (pauseAt >= 0 && !paused && t >= pauseAt) {
      audio::pause();
      paused = true;
    }
    if (paused && pauseFor > 0 && t >= pauseAt + pauseFor) {
      audio::resume();
      pauseFor = 0;
    }
    delay(10);
    if (g_simMs - t0 > (seconds + 600) * 1000) break;  // (stuck)
  }
  return {M5.Speaker.heard, audio::clipCut(), audio::hasClip(), g_simMs - t0};
}

// The link falling behind the voice: `seconds` of answer arriving at `pace` of
// the speed it's spoken. How many times it ran out, and how long the sound lasted.
static int slowLink(int seconds, float pace, size_t* heard) {
  M5.Speaker.heard.clear();
  audio::newClip();
  audio::playBegin(kRate);
  const size_t total = seconds * kRate;
  size_t sent = 0;
  float owed = 0;
  std::vector<uint8_t> frame;
  for (int guard = 0; guard < 400000 && audio::playing(); ++guard) {
    owed += kRate * pace / 100;
    frame.clear();
    for (; owed >= 1 && sent < total; owed -= 1, ++sent) {
      const int16_t v = sampleAt(sent);
      frame.push_back((uint8_t)v);
      frame.push_back((uint8_t)(v >> 8));
    }
    if (!frame.empty()) audio::playFeed(frame.data(), frame.size());
    if (sent >= total) audio::playEnd();
    audio::playPoll();
    delay(10);
  }
  *heard = M5.Speaker.heard.size();
  return audio::speechDryRuns();
}

int main() {
  audio::begin(200, 0);
  size_t miss;

  // Slower than it's spoken: a few real gaps with a second of sound after each,
  // not one every tenth of a second. Every sample still comes, in order.
  size_t heard = 0;
  int dry = slowLink(20, 0.8f, &heard);
  CHECK(heard == 20 * kRate, "slow link: heard %zu of %u samples", heard, 20 * kRate);
  CHECK(breaks(M5.Speaker.heard, &miss) == 0, "slow link: what was heard isn't what was sent (first at %zu)", miss);
  CHECK(dry >= 1 && dry <= 8, "slow link: ran dry %d times", dry);
  printf("slow link (0.8x): ran dry %d times in 20 s of speech\n", dry);
  // Faster than it's spoken: never.
  dry = slowLink(10, 3.0f, &heard);
  CHECK(heard == 10 * kRate, "fast link: heard %zu of %u samples", heard, 10 * kRate);
  CHECK(dry == 0, "fast link: ran dry %d times", dry);
  M5.Speaker.heard.clear();

  // A 3-minute answer, sent ~170x faster than it's spoken: all of it, in order.
  Run a = play(180, 3, true);
  CHECK(a.heard.size() == 180 * kRate, "long: heard %zu of %u samples", a.heard.size(), 180 * kRate);
  CHECK(breaks(a.heard, &miss) == 0, "long: what was heard isn't what was sent (first at %zu)", miss);
  CHECK(a.clipCut && !a.hasClip, "long: longer than the ring — a replay comes from the deck");
  printf("long answer: %.1f s of sound in %.1f s, cut=%d has=%d\n", a.heard.size() / (double)kRate, a.ms / 1000.0,
         a.clipCut, a.hasClip);

  // Paused for 2 minutes in the middle of a 3-minute answer.
  Run b = play(180, 3, true, 40, 120);
  const int bb = breaks(b.heard, &miss);
  CHECK(bb <= 1, "pause: %d breaks (first at %zu)", bb, miss);
  CHECK(b.heard.size() + 48 >= 180 * kRate, "pause: heard %zu of %u", b.heard.size(), 180 * kRate);
  printf("long answer, 2 min pause: %.1f s of sound, %d break(s) at the pause, in %.1f s\n",
         b.heard.size() / (double)kRate, bb, b.ms / 1000.0);

  // A short answer: replayed from here.
  Run c = play(10, 3, true);
  CHECK(breaks(c.heard, &miss) == 0 && c.heard.size() == 10 * kRate, "short: not as sent");
  CHECK(!c.clipCut && c.hasClip, "short: the whole clip is here to replay");
  M5.Speaker.heard.clear();
  audio::replay();
  for (int i = 0; i < 2000 && audio::playing(); ++i) {
    audio::playPoll();
    delay(10);
  }
  CHECK(M5.Speaker.heard.size() == 10 * kRate && breaks(M5.Speaker.heard, &miss) == 0, "replay: not the clip again");
  printf("short answer: replayed %.1f s from the ring\n", M5.Speaker.heard.size() / (double)kRate);

  // A deck that doesn't hold: as much as the ring takes, then a clean end
  // (as before the ring), and a replay from the deck.
  Run d = play(180, 3, false);
  const int db = breaks(d.heard, &miss);
  CHECK(d.clipCut && !d.hasClip, "no hold: the clip is cut");
  CHECK(db == 0 && d.heard.size() >= 1600 * 1024 && d.heard.size() < 1700 * 1024,
        "no hold: %zu heard, %d breaks", d.heard.size(), db);
  printf("deck that doesn't hold: %.1f s heard (%d break), cut=%d\n", d.heard.size() / (double)kRate, db, d.clipCut);

  printf(fails ? "%d FAILURES\n" : "all good\n", fails);
  return fails ? 1 : 0;
}
