#include "leds.h"

#include <M5Unified.h>
#include <esp32-hal-rmt.h>
#include <esp_timer.h>
#include <math.h>

#include "board.h"

namespace leds {

// The two lights are WS2812-style LEDs on one data pin (M5Unified's pin
// table: G21 on the PaperColor). M5Unified's own LED driver only works on
// ESP-IDF 5 (its RMT v1 path is empty), so they're driven here with the
// Arduino core's RMT, the way its neopixelWrite() does it: 100 ns ticks,
// 24 bits per LED, green-red-blue.
//
// The Waveshare 1.54" board has one green LED we can drive (PWM, active low): see
// monoLevel(). The red one beside it is the charger chip's, wired to its STAT pin
// with no GPIO on the way, so nothing here can change or read it.
static rmt_obj_t* rmt = nullptr;
static bool ready = false;
static uint8_t maxBright = 48;
#if defined(BOARD_EPAPER154)
static constexpr uint8_t kMonoPwm = 7;  // an LEDC channel nothing else uses
#endif

static void write(uint32_t a, uint32_t b) {
#if defined(BOARD_EPAPER154)
  (void)b;  // a = the duty, 0..255
  ledcWrite(kMonoPwm, 255 - min<uint32_t>(a, 255));
  return;
#endif
  if (!rmt) return;
  rmt_data_t bits[48];
  int i = 0;
  for (uint32_t rgb : {a, b}) {
    uint32_t grb = ((rgb >> 8) & 0xFF) << 16 | ((rgb >> 16) & 0xFF) << 8 | (rgb & 0xFF);
    for (int k = 23; k >= 0; --k, ++i) {
      bool one = (grb >> k) & 1;
      bits[i].level0 = 1;
      bits[i].duration0 = one ? 8 : 4;  // 0.8 / 0.4 µs high
      bits[i].level1 = 0;
      bits[i].duration1 = one ? 4 : 8;  // 0.4 / 0.8 µs low
    }
  }
  rmtWriteBlocking(rmt, bits, 48);
}

static State state = State::Off;
static int level = 0;
static uint32_t recMs = 0;  // State::Listening: how long the recording has been going
static uint32_t flashUntil = 0, alertUntil = 0, notifyUntil = 0, actionUntil = 0, lastMs = 0;
static bool waitingOn = false;  // proposals wait in Flight Deck
static Step lastStep = Step::Narration;
static uint32_t idleMs = 0, idleAt = 0, sleepMs = 1;  // State::Idle: how long, as of when
static bool inStartup = false;
static Link linkWifi = Link::Trying, linkAgent = Link::Trying;
static uint32_t startupMs = 0, bothUpMs = 0;
static uint32_t lastColor[2] = {1, 1};  // what the lights show (forces the first write)

#if defined(BOARD_EPAPER154)
static void monoTick(void*);
#endif

void begin() {
  if (!board::kHasLeds) return;
#if defined(BOARD_EPAPER154)
  ledcSetup(kMonoPwm, 5000, 8);
  ledcWrite(kMonoPwm, 255);
  ledcAttachPin(board::kMonoLed, kMonoPwm);
  ready = true;
  // A timer of its own: the loop can stall for seconds (Wi-Fi, the deck's TLS),
  // and a light that stops breathing looks hung.
  esp_timer_create_args_t args = {};
  args.callback = monoTick;
  args.name = "led";
  esp_timer_handle_t timer;
  if (esp_timer_create(&args, &timer) == ESP_OK) esp_timer_start_periodic(timer, 33000);
  return;
#else
  int pin = M5.getPin(m5::pin_name_t::rgb_led);
  if (pin >= 0 && (rmt = rmtInit(pin, RMT_TX_MODE, RMT_MEM_64)) != nullptr) rmtSetTick(rmt, 100);
  Serial.printf("[claw] status lights on G%d: %s\n", pin, rmt ? "ready" : "no driver");
  ready = rmt != nullptr;
#endif
  off();
}

void setLevel(uint8_t l) { maxBright = l == 0 ? 0 : l == 1 ? 48 : 200; }

void set(State s, int lvl) {
  state = s;
  level = constrain(lvl, 0, 100);
}

void recordFor(uint32_t ms) { recMs = ms; }

void setIdle(uint32_t idle, uint32_t sleep) {
  state = State::Idle;
  idleMs = idle;
  idleAt = millis();
  sleepMs = sleep ? sleep : 1;
}

void startup() {
  inStartup = true;
  startupMs = millis();
  bothUpMs = 0;
  linkWifi = linkAgent = Link::Trying;
}

void links(Link wifi, Link agent) {
  if (!inStartup) return;
  linkWifi = wifi;
  linkAgent = agent;
  const uint32_t now = millis();
  if (wifi == Link::Up && agent == Link::Up) {
    if (!bothUpMs) bothUpMs = now ? now : 1;
    if (now - bothUpMs >= 6000) inStartup = false;  // all good, said for 6 s: dark
  } else {
    bothUpMs = 0;
  }
  if (now - startupMs > 120000) inStartup = false;  // a red light, but not for ever
}

bool startingUp() { return inStartup; }

void flash() { flashUntil = millis() + 700; }

void alert() { alertUntil = millis() + 900; }

void notify() { notifyUntil = millis() + 2000; }

void setWaiting(bool waiting) { waitingOn = waiting; }

void action(Step step) {
  lastStep = step;
  actionUntil = millis() + 500;
  if (!actionUntil) actionUntil = 1;
}

// A step's colour: light blue a tool, dark blue narration, purple the model.
static void stepColor(Step s, uint8_t* r, uint8_t* g, uint8_t* b) {
  switch (s) {
    case Step::Tool: *r = 40, *g = 190, *b = 255; break;
    case Step::Model: *r = 170, *g = 0, *b = 255; break;
    default: *r = 0, *g = 30, *b = 255; break;
  }
}

static uint32_t scale(uint8_t r, uint8_t g, uint8_t b, float k) {
  k = k < 0 ? 0 : k > 1 ? 1 : k;
  k *= maxBright / 255.0f;
  return ((uint32_t)(r * k) << 16) | ((uint32_t)(g * k) << 8) | (uint32_t)(b * k);
}

// A start-up light: yellow while it tries, green when up, red when not.
static uint32_t linkColor(Link l) {
  return l == Link::Up ? scale(0, 255, 40, 1) : l == Link::Down ? scale(255, 0, 0, 1) : scale(255, 170, 0, 1);
}

static void show(uint32_t a, uint32_t b) {
  if (a == lastColor[0] && b == lastColor[1]) return;
  lastColor[0] = a;
  lastColor[1] = b;
  write(a, b);
}

void off() {
  if (!ready) return;
#if defined(BOARD_EPAPER154)
  ready = false;  // for good: before sleep, and the timer would light it again
#endif
  lastColor[0] = lastColor[1] = 1;
  show(0, 0);
}

#if defined(BOARD_EPAPER154)
// `count` short blinks at the start of each period.
static float blips(uint32_t at, uint32_t period, int count) {
  at %= period;
  return at < (uint32_t)count * 300 && at % 300 < 120 ? 1.0f : 0.0f;
}

// One colour, so the rhythm says what colours say on the PaperColor. It's also
// the one thing on this board that answers at once, and the only way to tell
// awake from asleep: the panel keeps its picture either way.
static float monoLevel(uint32_t now) {
  if (gpio_get_level(board::kTalkPin) == 0 || gpio_get_level(board::kSidePin) == 0) return 1;  // the press landed
  if ((int32_t)(flashUntil - now) > 0) return ((flashUntil - now) / 100) % 2;
  if ((int32_t)(alertUntil - now) > 0) return ((alertUntil - now) / 150) % 2;
  if ((int32_t)(notifyUntil - now) > 0) return ((notifyUntil - now) / 250) % 2;
  switch (state) {
    case State::Listening:  // steady, brighter with the voice
      return 0.6f + 0.4f * level / 100.0f;
    case State::Working:  // breathing, with a flick per step the deck reports
      if ((int32_t)(actionUntil - now) > 0) return 1;
      return 0.15f + 0.55f * (0.5f + 0.5f * sinf(now / 1000.0f * 3.14f));
    case State::Speaking:  // with the voice
      return level > 8 ? 0.3f + 0.7f * level / 100.0f : 0.05f;
    case State::NoWifi:  // offline, its first minute
      return now % 2000 < 1000;
    case State::NoAgent:
      return blips(now, 2000, 2);
    case State::Updating:
      return now % 200 < 100;
    case State::LowBattery:
      return blips(now, 5000, 3);
    case State::Startup:  // quick while it tries; both up, steady for its last 6 s
      if (linkWifi == Link::Up && linkAgent == Link::Up) return 1;
      if (linkWifi == Link::Down) return now % 2000 < 1000;
      if (linkAgent == Link::Down) return blips(now, 2000, 2);
      return now % 400 < 200;
    case State::Idle: {  // awake: a blink every 10 s; something waits in Flight Deck: two every 5 s
      const uint32_t idle = idleMs + (now - idleAt);
      if (waitingOn) return blips(idle, 5000, 2);
      return idle >= 10000 && idle % 10000 < 150;
    }
    default:
      return 0;
  }
}

static void monoTick(void*) {
  if (!ready) return;
  const float k = maxBright ? monoLevel(millis()) : 0;
  show((uint32_t)(k * k * (maxBright >= 200 ? 255 : 128)), 0);  // squared for the eye; Low is half of Bright
}
#endif

void tick() {
#if defined(BOARD_EPAPER154)
  return;  // monoTick
#endif
  if (!ready) return;
  const uint32_t now = millis();
  if (now - lastMs < 33) return;
  lastMs = now;
  if (maxBright == 0) {
    show(0, 0);
    return;
  }
  if ((int32_t)(flashUntil - now) > 0) {  // white, on/off every 100 ms
    uint32_t c = ((flashUntil - now) / 100) % 2 ? scale(255, 255, 255, 1) : 0;
    show(c, c);
    return;
  }
  if ((int32_t)(alertUntil - now) > 0) {
    uint32_t c = ((alertUntil - now) / 150) % 2 ? scale(255, 0, 0, 1) : 0;
    show(c, c);
    return;
  }
  if ((int32_t)(notifyUntil - now) > 0) {  // magenta, on/off every 250 ms
    uint32_t c = ((notifyUntil - now) / 250) % 2 ? scale(255, 0, 170, 1) : 0;
    show(c, c);
    return;
  }
  const float t = now / 1000.0f;
  switch (state) {
    case State::Listening: {
      // Left: red, blinking like a REC light, brighter with the voice. Right,
      // steady: how long it's been — green for the first 20 s, yellow for the
      // next 20, red after that.
      const uint32_t rec = fmodf(t, 0.8f) < 0.45f ? scale(255, 0, 0, 0.45f + 0.55f * level / 100.0f) : 0;
      const uint32_t since = recMs < 20000 ? scale(0, 255, 40, 0.8f)
                            : recMs < 40000 ? scale(255, 170, 0, 0.8f) : scale(255, 0, 0, 0.8f);
      show(rec, since);
      break;
    }
    case State::Working: {  // a blink per step the deck reports, in its colour; a faint glow between
      uint8_t r, g, b;
      stepColor(lastStep, &r, &g, &b);
      uint32_t c = (int32_t)(actionUntil - now) > 0 ? scale(r, g, b, 1) : scale(r, g, b, 0.06f);
      show(c, c);
      break;
    }
    case State::Speaking: {  // green, blinking with the voice
      uint32_t c = scale(0, 255, 40, level > 8 ? 0.25f + 0.75f * level / 100.0f : 0.04f);
      show(c, c);
      break;
    }
    case State::NoWifi:  // offline, blue blinking: the left light, no Wi-Fi
    case State::NoAgent: {  // the right light, Wi-Fi but no Flight Deck or no agent
      const uint32_t c = fmodf(t, 1.0f) < 0.5f ? scale(40, 80, 255, 0.8f) : 0;
      if (state == State::NoWifi) show(c, 0);
      else show(0, c);
      break;
    }
    case State::Updating: {  // downloading an update: left, then right, a purple blink a second each
      const uint32_t c = now % 1000 < 600 ? scale(200, 0, 255, 1) : 0;
      if ((now / 1000) % 2) show(0, c);
      else show(c, 0);
      break;
    }
    case State::Redrawing: {  // a slow, soft white glow: the paper is changing
      uint32_t c = scale(255, 255, 255, 0.05f + 0.2f * (0.5f + 0.5f * sinf(t * 2.0f)));
      show(c, 0);
      break;
    }
    case State::LowBattery: {  // an orange blip every 5 s
      uint32_t c = fmodf(t, 5.0f) < 0.15f ? scale(255, 80, 0, 1) : 0;
      show(c, 0);
      break;
    }
    case State::Startup:  // left: Wi-Fi, right: the agent
      show(linkColor(linkWifi), linkColor(linkAgent));
      break;
    case State::Idle: {  // a blink 10 s into the quiet and every 10 s after, coloured by how near sleep is
                         // (the right one magenta while proposals wait)
      const uint32_t idle = idleMs + (now - idleAt);
      const float toSleep = (float)idle / sleepMs;
      uint32_t c = 0, right = 0;
      if (idle >= 10000 && idle % 10000 < 150) {
        c = toSleep < 0.3f ? scale(0, 255, 40, 1) : toSleep < 0.8f ? scale(0, 60, 255, 1) : scale(255, 170, 0, 1);
        right = waitingOn ? scale(255, 0, 170, 1) : c;  // something waits in Flight Deck
      }
      show(c, right);
      break;
    }
    default:
      show(0, 0);
      break;
  }
}

}  // namespace leds
