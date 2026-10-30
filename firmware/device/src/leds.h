// Status lights (PaperColor: two RGB LEDs). The paper takes ~20 s to redraw,
// so the lights say what's happening now: while it records you the left one
// blinks red (a REC light, brighter when you're louder) and the right one
// says how long it's been — green, then yellow after 20 s, red after 40; a
// blink for every step
// the deck reports while the agent works — light blue for a tool, dark blue
// for narration and progress, purple for the model thinking — green with the voice
// while it speaks, a soft white glow while the paper redraws, purple blinks
// left and right in turn, one a second, while an update downloads. Starting up,
// the left light shows Wi-Fi and the right one the agent; offline (the first
// minute), the left one blinks blue with no Wi-Fi, the right one with Wi-Fi
// but no Flight Deck or agent; idle, a short blink every 10 s says how close
// sleep is; while proposals wait in Flight Deck, that blink is magenta on
// the right. Devices without lights ignore all of it. The Waveshare 1.54" has
// one green light, so it says these with rhythm instead (leds.cpp, monoLevel).
#pragma once

#include <Arduino.h>

namespace leds {

enum class State : uint8_t { Off, NoWifi, NoAgent, Listening, Working, Speaking, Redrawing, Updating, LowBattery,
                             Startup, Idle };

void begin();
// 0 off, 1 low, 2 bright (settings.ledLevel).
void setLevel(uint8_t level);
// What the device is doing; `level` 0..100 is the mic or the voice.
void set(State state, int level = 0);
// Recording (State::Listening): how long it has been going — the right
// light's colour.
void recordFor(uint32_t ms);
// Idle: a short blink every 10 s — green for the first 30% of the way to
// sleep, blue for the next 50%, yellow for the last 20%.
void setIdle(uint32_t idleMs, uint32_t sleepMs);

// Starting up: both lights yellow; then the left one is Wi-Fi and the right
// one the agent, each green once it's up or red when it can't be reached.
// Both green: they stay so for 6 s, then the start-up show is over.
enum class Link : uint8_t { Trying, Up, Down };
void startup();
void links(Link wifi, Link agent);  // the latest, every loop pass
bool startingUp();

// The cue before speech: three quick white blinks.
void flash();
// The deck reported a step of the work: one blink, coloured by what it was.
enum class Step : uint8_t { Tool, Narration, Model };
void action(Step step);
// Something went wrong: two red blinks.
void alert();
// Something new waits in Flight Deck (a proposal, a nudge): magenta blinks
// on both lights for two seconds.
void notify();
// Proposals wait in Flight Deck: the idle blink's right light is magenta.
void setWaiting(bool waiting);
// From loop(): animates the lights (at most ~30 times a second).
void tick();
// Off now (before sleep).
void off();

}  // namespace leds
