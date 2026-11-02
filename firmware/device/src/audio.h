// Microphone capture and streamed speaker playback.
// The StickS3's ES8311 codec does one at a time: starting either stops the other.
#pragma once

#include <Arduino.h>

namespace audio {

static constexpr uint32_t kMicRate = 16000;
// A voice, for the 2.5 s silence stop: kVoiceRun blocks of kVoiceBlock
// samples (32 ms) in a row, each peaking at kVoicePeak or louder (about a
// fifth of the level meter) — ~0.1 s of sound. Room noise and hum stay
// under it; a click, a tap on the case or a rustle is too short.
static constexpr int kVoicePeak = 2400;
static constexpr size_t kVoiceBlock = 512;
static constexpr int kVoiceRun = 3;

void begin(uint8_t volume, uint8_t boostDb);
void setVolume(uint8_t volume);
// Extra gain on speech with a soft limiter (peaks are rounded, not clipped).
void setBoost(uint8_t db);
void beep();  // short tone at the current volume (feedback for volume changes)
void cue();   // two-tone "about to speak" chime
void cueEnd(); // the same two tones the other way round: "done speaking"
void chime();  // two rising tones, wider apart: something new waits in Flight Deck
// Recording sounds — single notes, unlike the two-note speech chimes:
void talkStart(); // a short high blip as a recording begins; returns 0.25 s later, the blip played
void talkEnd();   // a short lower blip when it's let go (sent, or kept offline)
// A short beep for a button press (the paper changes slowly): `low` for the
// second button, so the two sound apart.
void click(bool low = false);

// ── microphone ──
void micStart();
// Each finished block of 16 kHz mono samples goes to `sink` (called from loop();
// a task keeps the mic recording meanwhile, so a loop held up for up to 4 s
// loses nothing). Returns the last block's peak level 0..100 (for the on-screen meter).
int micPoll(void (*sink)(const int16_t* samples, size_t count));
// Stops recording and flushes the blocks still in flight to `sink`.
void micStop(void (*sink)(const int16_t* samples, size_t count));
bool micActive();
// When the mic last heard a voice (kVoiceRun loud blocks in a row, past its
// first 0.3 s), in millis(); 0 = not since micStart().
uint32_t micVoiceMs();
bool micBehind();  // the loop has blocks still to take: micVoiceMs() isn't up to date

// ── speaker (one stream at a time, PCM s16le mono) ──
// The streams of one answer (its speech, then any sound the agent made) are
// kept back to back as the answer's clip, in a ring (~68 s at 24 kHz) with no
// limit on the length: holdWanted() says when the deck should stop sending
// for now. replay() plays the clip again while all of it is still there.
//
// A new stream; sound starts no earlier than holdMs from now — room for the
// beep/flash cue before speech. Returns true when it carries straight on from
// the answer's previous stream, which is still sounding (no cue then).
bool playBegin(uint32_t rate, uint32_t holdMs = 0);
void playFeed(const uint8_t* data, size_t len);
void playEnd();                            // no more data for this stream
void playStop();                           // stop now (the clip is kept)
// The deck should hold the sound (true): the ring has as much still to play
// as it can keep (15 s short of full); false again with 30 s of room.
bool holdWanted();
void playPoll();                           // keep the speaker fed; call from loop()
bool playing();                            // a stream is buffered or sounding (or held by pause())
bool sounding();                           // the speaker is actually outputting it
int speechDryRuns();                       // how many times this clip ran out before its end: the link fell behind
// The answer's speech (its sounds up to markSpeechEnd(), which the deck's
// speech_end asks for): the sample in it the speaker is at now (the gapless
// estimate the mouth uses), how many it has (so far, or in all once it has
// ended), whether it has all arrived, its rate.
size_t speechAt();
size_t speechTotal();
bool speechEnded();
void markSpeechEnd();
size_t clipSamples();                      // samples in the answer's clip so far (where the next sound starts)
bool streamArrived();                      // the sound now playing has all arrived (or none is)
uint32_t playRateHz();
void pause();                              // hold the sound where it is (the stream keeps arriving)
void resume();                             // carry on from there
bool paused();                             // held by pause()
// True once each time the answer's sound has played to its end by itself
// (not stopped, not a replay): the moment for the closing chime.
bool takeFinished();
int outputLevel();                         // 0..100 loudness of what's sounding now

// ── a line spoken aside (the e-paper's menu rows) ──
// Kept apart from the answer's clip (a replay never plays it), played once
// it has all arrived; skipped while the mic or an answer is on.
void asideBegin(uint32_t rate);
void asideFeed(const uint8_t* data, size_t len);
void asideEnd();

// ── the last answer's clip ──
void newClip();   // a new answer: its first stream replaces the clip
bool hasClip();   // all of the clip is here to replay
bool clipCut();   // not all of it is here: cut short (stopped, link lost) or longer than the ring
void replay();    // play the clip again from its start, at once (no cue)

}  // namespace audio
