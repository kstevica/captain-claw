// Mac stand-in for M5Unified's speaker and mic, as audio.cpp uses them. The
// speaker plays its queued pieces on the simulated clock, back to back, and
// reads each piece's samples only as they're played — so a sample written
// over before it sounded shows up in what was "heard".
#pragma once
#include <deque>
#include <mutex>
#include <vector>

#include "Arduino.h"

struct FakeSpeaker {
  struct Piece {
    const int16_t* ptr;
    size_t n;
    uint32_t rate;
    uint32_t startMs;  // when it began to sound
    size_t done;       // samples heard
  };
  std::deque<Piece> q;
  std::vector<int16_t> heard;
  bool enabled = false;
  bool begin() { return enabled = true; }
  void end() { enabled = false; q.clear(); }
  bool isEnabled() const { return enabled; }
  void setVolume(uint8_t) {}
  void tone(float, uint32_t, int = -1, bool = true) {}
  size_t isPlaying(uint8_t ch) const { return ch == 0 ? q.size() : 0; }
  void stop(uint8_t ch) {
    if (ch != 0) return;
    advance();
    q.clear();
  }
  bool playRaw(const int16_t* p, size_t n, uint32_t rate, bool, uint32_t, int ch, bool) {
    if (ch != 0) return true;
    if (q.size() >= 2) return false;
    q.push_back({p, n, rate, q.empty() ? g_simMs : 0, 0});
    return true;
  }
  // Up to the clock: what's due of the front piece is heard, then the next.
  void advance() {
    while (!q.empty()) {
      Piece& f = q.front();
      const size_t due = (size_t)((uint64_t)(g_simMs - f.startMs) * f.rate / 1000);
      const size_t upTo = due < f.n ? due : f.n;
      for (; f.done < upTo; ++f.done) heard.push_back(f.ptr[f.done]);
      if (f.done < f.n) return;
      const uint32_t endMs = f.startMs + (uint32_t)((uint64_t)f.n * 1000 / f.rate);
      q.pop_front();
      if (!q.empty()) q.front().startMs = endMs;
    }
  }
};

// Fills what it's given on the simulated clock, 16 samples a millisecond, from a
// counter that never stops: a hole in what's heard is a jump in the numbers.
struct FakeMic {
  struct Rec {
    int16_t* ptr;
    size_t n, done;
  };
  std::mutex lock;
  std::deque<Rec> q;
  uint32_t lastMs = 0;
  static constexpr uint32_t kPeriod = 30011;  // (prime: no hole a whole number of periods long)
  uint32_t heard = 0;  // samples spoken since begin()
  bool begin() {
    std::lock_guard<std::mutex> g(lock);
    q.clear();
    lastMs = g_simMs;
    heard = 0;
    return true;
  }
  void end() {
    std::lock_guard<std::mutex> g(lock);
    q.clear();
  }
  void advance() {
    const uint32_t now = g_simMs;
    for (; lastMs < now; ++lastMs) {
      for (int i = 0; i < 16; ++i, ++heard) {
        if (q.empty()) continue;  // nobody asked: said, and gone
        Rec& f = q.front();
        f.ptr[f.done++] = (int16_t)(heard % kPeriod);
        if (f.done == f.n) q.pop_front();
      }
    }
  }
  size_t isRecording() {
    std::lock_guard<std::mutex> g(lock);
    advance();
    return q.size();
  }
  bool record(int16_t* p, size_t n, uint32_t) {
    std::lock_guard<std::mutex> g(lock);
    advance();
    if (q.size() >= 2) return false;
    q.push_back({p, n, 0});
    return true;
  }
};

struct FakeM5 {
  FakeSpeaker Speaker;
  FakeMic Mic;
};
extern FakeM5 M5;
