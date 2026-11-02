#include "audio.h"

#include <M5Unified.h>

#include "hw.h"

namespace audio {

// ── microphone: a ring of blocks in PSRAM ──
// M5Unified fills up to two blocks at once, so whoever hands it the next can't be
// away for more than about 64 ms, and the main loop often is: a frame on the round
// screen, or the socket on a weak link. What it misses is a hole in what was said.
// So a small task keeps the mic supplied, and the loop takes the blocks as it gets
// to them, a few seconds late if it must.
static constexpr size_t kBlock = 512;      // 32 ms at 16 kHz
static constexpr uint32_t kRing = 125;     // blocks: 4 s
static constexpr uint32_t kCatchUp = 8;    // blocks a pass of the loop takes at most (it has other work)
static constexpr uint32_t kSettle = 10;    // blocks: the mic's start-up, 0.3 s
static int16_t* micBuf = nullptr;          // kRing * kBlock samples
// Blocks since micStart(), counted up: given to the mic, filled by it, taken by the loop.
static volatile uint32_t blocksAsked = 0, blocksDone = 0, blocksTaken = 0;
static volatile bool feedOn = false, feeding = false;
static bool micOn = false;
static int lastLevel = 0;
static uint32_t voiceMs = 0;
static int floorPeak = 0, maxPeak = 0;  // this recording's quietest and loudest block so far
static int loudRun = 0;  // loud blocks in a row (kVoiceRun of them = a voice)

// ── speaker: the answer's clip in a PSRAM ring ──
// Positions count samples from the clip's start and only grow; a sample sits
// at pos % kPlayCap. No limit on an answer's length: the deck is asked to hold
// (holdWanted) while the ring has as much still to play as it can keep, and
// to go on once some has played. While the whole clip still fits (nothing
// written over), replay() plays it again from here; past that, from the deck.
static constexpr size_t kPlayCap = 1600 * 1024;  // samples (~68 s at 24 kHz)
static int16_t* playBuf = nullptr;
static size_t writePos = 0, playPos = 0;  // in samples; the clip is [0, writePos)
static size_t pieceStart[2];              // where the speaker's queued pieces begin (they point into the ring)
static int pieces = 0;
static bool dropped = false;              // some of the clip didn't fit (a deck that doesn't hold)
static bool holding = false;              // holdWanted()'s last answer
static uint32_t playRate = 24000;
static bool streamOpen = false, streamEnded = false, started = false;
static bool freshClip = true;             // the next stream starts a new clip
static bool cut = false;                  // a stream of this clip stopped before its end
static uint32_t holdUntilMs = 0;
static uint32_t soundStartMs = 0;         // when the speaker last started from silence…
static size_t soundStartPos = 0;          // …and from which sample
static int carry = -1;                    // odd byte waiting for its pair
static bool replaying = false;            // the clip is playing again (replay())
static bool onHold = false;               // pause(): the stream stays open, nothing is fed
static size_t firstEnd = 0;               // where the clip's speech ended (markSpeechEnd)…
static bool firstEnded = false;           // …once it has
static bool finished = false;             // it played to its end by itself (takeFinished())
static bool refilling = false;            // it ran dry: waiting for more before going on
static int dryRuns = 0;                   // how many times, in this clip

// ── a line spoken aside: its own small buffer and speaker channel ──
static constexpr size_t kAsideCap = 6 * 24000;  // samples (6 s at 24 kHz)
static constexpr uint8_t kAsideChannel = 2;     // 0 = the answer, 1 = beeps
static int16_t* asideBuf = nullptr;
static size_t asideLen = 0;
static uint32_t asideRate = 24000;
static bool asideOn = false;
static int asideCarry = -1;

// Speech boost: gain, then a soft knee above half scale so loud syllables
// round off instead of clipping (TTS peaks near -3 dBFS, averages ~-20 dBFS).
static float boostGain = 1.0f;

static inline int16_t shape(int16_t s) {
  if (boostGain == 1.0f) return s;
  constexpr float kKnee = 0.5f;
  float x = s * boostGain * (1.0f / 32768.0f);
  float a = fabsf(x);
  if (a > kKnee) a = kKnee + (1.0f - kKnee) * tanhf((a - kKnee) / (1.0f - kKnee));
  return (int16_t)(copysignf(a, x) * 32767.0f);
}

void setBoost(uint8_t db) { boostGain = powf(10.0f, db / 20.0f); }

void begin(uint8_t volume, uint8_t boostDb) {
  micBuf = (int16_t*)heap_caps_malloc(kRing * kBlock * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  playBuf = (int16_t*)heap_caps_malloc(kPlayCap * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  M5.Speaker.setVolume(volume);
  setBoost(boostDb);
  M5.Speaker.end();
  M5.Mic.end();
}

void setVolume(uint8_t volume) { M5.Speaker.setVolume(volume); }

void beep() {
  if (micOn) return;
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  M5.Speaker.tone(880, 120, 1);
}

void cue() {
  if (micOn) return;
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  M5.Speaker.tone(1046, 110, 1, true);   // C6
  M5.Speaker.tone(1568, 160, 1, false);  // then G6
}

void cueEnd() {
  if (micOn || streamOpen) return;
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  M5.Speaker.tone(1568, 110, 1, true);   // G6
  M5.Speaker.tone(1046, 160, 1, false);  // then C6
}

void chime() {
  if (micOn || streamOpen) return;
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  // The first tone long enough to outlast the codec waking (it swallows
  // the first tens of ms).
  M5.Speaker.tone(1319, 140, 1, true);   // E6
  M5.Speaker.tone(2093, 220, 1, false);  // then C7
}

void talkStart() {
  if (micOn) return;
  playStop();
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  M5.Speaker.tone(1760, board::kTalkStartCueMs, 1, true);  // A6
  // The blip first; the mic (which takes the codec, ending the speaker) only
  // once it's over. A fixed wait: right after tone() the speaker can still
  // report nothing playing, and waiting on that ended the blip unheard.
  delay(board::kTalkMicAfterMs);
}

void talkEnd() {
  if (micOn || streamOpen) return;
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  M5.Speaker.tone(880, board::kTalkEndCueMs, 1, true);  // A5
}

void click(bool low) {
  if (micOn || streamOpen) return;  // never over the mic or a voice
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
  // 60 ms: the codec, just woken, swallows the start of a sound (a 14 ms tick
  // was never heard); a note apart from the recording blips (A6 / A5).
  M5.Speaker.tone(low ? 1047 : 1319, 60, 1);  // C6 / E6
}

// The mic's next blocks, while it has room for them and the ring holds no more
// than the loop can still take.
static void feedMic() {
  size_t inflight = M5.Mic.isRecording();
  blocksDone = blocksAsked - inflight;
  while (inflight < 2 && blocksAsked - blocksTaken < kRing &&
         M5.Mic.record(micBuf + blocksAsked % kRing * kBlock, kBlock, kMicRate)) {
    blocksAsked = blocksAsked + 1;
    ++inflight;
  }
}

static void micFeedLoop(void*) {
  while (feedOn) {
    feedMic();
    vTaskDelay(pdMS_TO_TICKS(8));  // a quarter of a block
  }
  feeding = false;
  vTaskDelete(nullptr);
}

static void stopFeeding() {
  feedOn = false;
  for (uint32_t t0 = millis(); feeding && millis() - t0 < 200;) delay(1);
}

static void drainDone(void (*sink)(const int16_t*, size_t), uint32_t most) {
  while (blocksTaken != blocksDone && most--) {
    const int16_t* blk = micBuf + blocksTaken % kRing * kBlock;
    int peak = 0;
    for (size_t i = 0; i < kBlock; ++i) peak = max(peak, abs((int)blk[i]));
    lastLevel = min(100, peak * 100 / 12000);
    // A voice is a block clearly above this recording's own quiet: louder
    // than kVoicePeak and, once something loud has been heard, a third of
    // the way from the quietest block to the loudest — a mic whose floor sits
    // above kVoicePeak (the StickS3's) would otherwise never go quiet.
    const bool settled = blocksTaken >= kSettle;  // (not the mic's start-up)
    if (settled) {
      floorPeak = floorPeak ? min(floorPeak, peak) : peak;
      maxPeak = max(maxPeak, peak);
    }
    const int thr = max((int)kVoicePeak, floorPeak + (maxPeak - floorPeak) * 3 / 10);
    loudRun = peak >= thr ? loudRun + 1 : 0;
    if (loudRun >= kVoiceRun && settled) voiceMs = millis();
    sink(blk, kBlock);
    if (!micOn) return;  // the send found the link gone, and that stopped the mic (it took the rest)
    blocksTaken = blocksTaken + 1;
  }
}

void micStart() {
  stopFeeding();
  playStop();
  M5.Speaker.end();
  M5.Mic.begin();
  hw::micStarted();
  blocksAsked = blocksDone = blocksTaken = 0;
  lastLevel = 0;
  voiceMs = 0;
  loudRun = 0;
  floorPeak = maxPeak = 0;
  micOn = true;
  feedMic();
  // Above the loop, on its core: a loop busy with a frame or a TLS record gives way.
  feedOn = feeding = true;
  if (xTaskCreatePinnedToCore(micFeedLoop, "micfeed", 4096, nullptr, 3, nullptr, 1) != pdPASS) {
    feedOn = feeding = false;  // (micPoll feeds it, as well as it can)
  }
}

int micPoll(void (*sink)(const int16_t*, size_t)) {
  if (!micOn) return 0;
  if (!feeding) feedMic();
  drainDone(sink, kCatchUp);
  return lastLevel;
}

void micStop(void (*sink)(const int16_t*, size_t)) {
  if (!micOn) return;
  stopFeeding();
  uint32_t t0 = millis();
  while (M5.Mic.isRecording() && millis() - t0 < 200) delay(1);
  blocksDone = blocksAsked - M5.Mic.isRecording();
  drainDone(sink, UINT32_MAX);
  M5.Mic.end();
  micOn = false;
}

bool micActive() { return micOn; }

uint32_t micVoiceMs() { return voiceMs; }

bool micBehind() { return micOn && blocksDone - blocksTaken > 2; }

// The codec does one thing at a time: the speaker takes over from the mic.
static void speakerOn() {
  if (micOn) micStop([](const int16_t*, size_t) {});
  M5.Mic.end();
  if (!M5.Speaker.isEnabled()) M5.Speaker.begin();
}

bool playBegin(uint32_t rate, uint32_t holdMs) {
  rate = rate ? rate : 24000;
  bool fresh = freshClip || rate != playRate;
  // The answer's previous stream has all arrived and is still sounding:
  // carry straight on, so the two play back to back.
  if (!fresh && streamOpen && streamEnded) {
    streamEnded = false;
    carry = -1;
    return true;
  }
  playStop();
  speakerOn();
  if (fresh) {
    writePos = 0;
    cut = dropped = false;
    freshClip = false;
    firstEnded = false;
    firstEnd = 0;
    dryRuns = 0;
  }
  refilling = false;
  playRate = rate;
  playPos = writePos;
  carry = -1;
  replaying = false;
  streamOpen = true;
  streamEnded = false;
  started = false;
  holdUntilMs = millis() + holdMs;
  return false;
}

static inline size_t ringAt(size_t pos) { return pos % kPlayCap; }

// The oldest sample still needed: the start of the piece the speaker is
// playing (its pieces point into the ring), else where playing goes on from.
static size_t keepFrom() {
  const int queued = M5.Speaker.isEnabled() ? (int)M5.Speaker.isPlaying(0) : 0;
  while (pieces > queued) {  // played out, oldest first
    pieceStart[0] = pieceStart[1];
    --pieces;
  }
  return pieces ? pieceStart[0] : playPos;
}

void playFeed(const uint8_t* data, size_t len) {
  if (!streamOpen || !playBuf) return;
  const size_t from = keepFrom();
  auto put = [&](int16_t s) {
    // Full of what's still to play — a deck that didn't hold: the rest of
    // this clip is dropped (a clean end, as an old deck's answer had, not
    // speech that skips ahead as room frees up). Its replay comes from the deck.
    if (dropped || writePos - from >= kPlayCap) {
      dropped = true;
      return;
    }
    playBuf[ringAt(writePos++)] = shape(s);
  };
  size_t i = 0;
  if (carry >= 0 && len) {
    put((int16_t)(carry | (data[0] << 8)));
    carry = -1;
    i = 1;
  }
  for (; i + 1 < len; i += 2) put((int16_t)(data[i] | (data[i + 1] << 8)));  // little-endian
  if (i < len) carry = data[i];
}

bool holdWanted() {
  if (!streamOpen || !playBuf) return holding = false;
  const size_t ahead = writePos - keepFrom();
  if (ahead + (size_t)playRate * 15 >= kPlayCap) holding = true;        // 15 s short of full
  else if (ahead + (size_t)playRate * 30 <= kPlayCap) holding = false;  // 30 s of room again
  return holding;
}

void playEnd() {
  if (streamOpen) streamEnded = true;
}

void markSpeechEnd() {
  if (firstEnded) return;
  firstEnded = true;
  firstEnd = writePos;
}

size_t speechTotal() { return firstEnded ? firstEnd : writePos; }
size_t clipSamples() { return writePos; }
bool streamArrived() { return !streamOpen || streamEnded; }
bool speechEnded() { return firstEnded; }
uint32_t playRateHz() { return playRate; }

size_t speechAt() {
  const size_t total = speechTotal();
  if (!streamOpen) return total;
  if (onHold || !started) return min(playPos, total);
  const size_t at = soundStartPos + (size_t)((millis() - soundStartMs) * (uint64_t)playRate / 1000);
  return min(min(at, playPos), total);
}

void playStop() {
  if (M5.Speaker.isEnabled()) {
    M5.Speaker.stop(0);
    M5.Speaker.stop(kAsideChannel);  // a menu line gives way too
  }
  pieces = 0;
  if (streamOpen && !streamEnded) cut = true;  // the rest of it won't arrive
  streamOpen = false;
  streamEnded = false;
  started = false;
  onHold = false;
}

void playPoll() {
  if (!streamOpen || onHold) return;
  size_t avail = writePos - playPos;
  size_t queued = M5.Speaker.isPlaying(0);
  if (!started) {
    if ((int32_t)(millis() - holdUntilMs) < 0) return;  // the cue plays first
    // Buffer a little first so a slow network doesn't stutter the start.
    if (avail < playRate * 3 / 10 && !streamEnded) return;
    started = true;
  }
#ifdef CLAW_PERF
  {
    extern uint32_t g_perfUnderruns;
    if (queued == 0 && avail > 0) ++g_perfUnderruns;
  }
#endif
  // Run dry with more to come: the link fell behind the voice. A good second of it
  // first, not the next tenth: one gap, not a stutter until the link catches up.
  if (queued == 0 && avail < playRate / 10 && !streamEnded && !refilling) {
    refilling = true;
    ++dryRuns;
  }
  if (refilling) {
    if (avail < playRate && !streamEnded) return;
    refilling = false;
  }
  if (queued < 2 && avail > 0 && (avail >= playRate / 10 || streamEnded)) {
    // Up to 2 s a piece, two pieces queued: a main loop held up by the
    // network for a second or two still finds the speaker busy. A piece
    // never runs past the ring's end (the next starts at its beginning).
    size_t n = min(avail, (size_t)(playRate * 2));
    n = min(n, kPlayCap - ringAt(playPos));
    if (queued == 0) {  // from silence: outputLevel() counts from here
      soundStartMs = millis();
      soundStartPos = playPos;
    }
    keepFrom();  // (the pieces it has played go from the list)
    M5.Speaker.playRaw(playBuf + ringAt(playPos), n, playRate, false, 1, 0, false);
    if (pieces < 2) pieceStart[pieces++] = playPos;
    playPos += n;
  }
  if (streamEnded && playPos == writePos && M5.Speaker.isPlaying(0) == 0) {
    streamOpen = false;  // stream finished and fully played
    finished = !replaying;
  }
}

bool playing() { return streamOpen; }

int speechDryRuns() { return dryRuns; }

bool sounding() { return streamOpen && started && !onHold; }

void pause() {
  if (!sounding()) return;
  // Where the speaker is now (gapless since it last started from silence):
  // the pieces queued past it are dropped, and resume() goes on from there.
  size_t at = soundStartPos + (size_t)((millis() - soundStartMs) * (uint64_t)playRate / 1000);
  M5.Speaker.stop(0);
  pieces = 0;
  playPos = min(at, playPos);
  onHold = true;
}

void resume() {
  if (!onHold) return;
  speakerOn();  // the mic may have had the codec meanwhile
  onHold = false;
  holdUntilMs = millis();
}

bool paused() { return streamOpen && onHold; }

bool takeFinished() {
  bool f = finished;
  finished = false;
  return f;
}

int outputLevel() {
  // Where the speaker is now, assuming it has played without gaps since it
  // last started from silence (good enough to move a mouth).
  if (!sounding() || !playBuf) return 0;
  size_t at = soundStartPos + (size_t)((millis() - soundStartMs) * (uint64_t)playRate / 1000);
  if (at >= playPos) return 0;
  size_t win = playRate / 50;  // 20 ms
  size_t from = at > soundStartPos + win ? at - win : soundStartPos;
  int peak = 0;
  for (size_t i = from; i < at; i += 2) peak = max(peak, abs((int)playBuf[ringAt(i)]));
  return min(100, peak * 100 / 20000);
}

void asideBegin(uint32_t rate) {
  asideOn = false;
  if (micOn || streamOpen) return;  // never over the mic or an answer
  if (!asideBuf) asideBuf = (int16_t*)heap_caps_malloc(kAsideCap * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!asideBuf) return;
  if (M5.Speaker.isEnabled()) M5.Speaker.stop(kAsideChannel);  // the buffer is about to change
  asideLen = 0;
  asideRate = rate ? rate : 24000;
  asideCarry = -1;
  asideOn = true;
}

void asideFeed(const uint8_t* data, size_t len) {
  if (!asideOn) return;
  size_t i = 0;
  if (asideCarry >= 0 && len) {
    if (asideLen < kAsideCap) asideBuf[asideLen++] = shape((int16_t)(asideCarry | (data[0] << 8)));
    asideCarry = -1;
    i = 1;
  }
  for (; i + 1 < len && asideLen < kAsideCap; i += 2) {
    asideBuf[asideLen++] = shape((int16_t)(data[i] | (data[i + 1] << 8)));
  }
  if (i < len && asideLen < kAsideCap) asideCarry = data[i];
}

void asideEnd() {
  if (!asideOn) return;
  asideOn = false;
  if (!asideLen || micOn || streamOpen) return;
  speakerOn();
  M5.Speaker.playRaw(asideBuf, asideLen, asideRate, false, 1, kAsideChannel, true);
}

void newClip() { freshClip = true; }

// Here to play again: the whole clip still in the ring.
bool hasClip() { return playBuf && writePos > 0 && !freshClip && !dropped && writePos <= kPlayCap; }

// Not all of it here: cut short, or longer than the ring (the deck sends it again).
bool clipCut() { return (cut || dropped || writePos > kPlayCap) && !freshClip; }

void replay() {
  if (!hasClip()) return;
  bool arriving = streamOpen && !streamEnded;  // its last stream is still coming in
  if (M5.Speaker.isEnabled()) M5.Speaker.stop(0);
  speakerOn();
  pieces = 0;
  playPos = 0;
  refilling = false;
  replaying = true;
  onHold = false;
  if (!arriving) carry = -1;
  streamOpen = true;
  streamEnded = !arriving;
  started = false;
  holdUntilMs = millis();
}

}  // namespace audio
