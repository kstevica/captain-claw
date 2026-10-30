// What differs between the devices this firmware builds for. Everything else
// (the Flight Deck protocol, settings, setup portal, audio, characters) is
// shared; each device has its own screen layer (ui_stick.cpp / ui_round.cpp /
// ui_paper.cpp).
//
//   -DBOARD_STICKS3     M5Stack StickS3: 240×135 LCD, front + side button
//   -DBOARD_STOPWATCH   M5Stack StopWatch: 466×466 round AMOLED, two buttons,
//                       vibration motor
//   -DBOARD_PAPERCOLOR  M5Stack PaperColor: 4" Spectra 6 colour e-paper
//                       (400×600, ~20 s to redraw), three buttons, two RGB
//                       LEDs, SHT40 temperature/humidity, RTC
//   -DBOARD_EPAPER154   Waveshare ESP32-S3-ePaper-1.54 (SKU 32298): 200×200
//                       black-and-white e-paper (1.8 s full, 0.6 s partial), BOOT
//                       + PWR buttons, one green LED, SHTC3, RTC, microSD;
//                       with -DEPAPER154_TOUCH the Touch version (SKU 34212)
#pragma once

#include <driver/gpio.h>

namespace board {

#if defined(BOARD_STOPWATCH)

constexpr const char* kModel = "StopWatch";   // portal title, hotspot name
constexpr const char* kHw = "stopwatch";      // build kind for over-the-air updates (= PlatformIO env)
constexpr const char* kNoun = "watch";        // "the watch restarts…"
constexpr const char* kIdPrefix = "watch";    // device id: watch-a1b2c3
constexpr const char* kTalkButton = "yellow button";
constexpr const char* kSideButton = "other button";
constexpr const char* kTalkHint = "Press the yellow button to talk";  // the home screen's call to action
// Hold to talk, let go to send (false), or press to start and press again to
// send (true) — or 2.5 s of quiet sends it, as the on-screen button does.
constexpr bool kTalkToggles = true;
constexpr bool kTalkFromSleep = false;  // a press that wakes it (or lights a dark screen) only wakes it
// Which of M5Unified's buttons (A/B/C) talks, goes next and goes back (0: none).
constexpr char kTalkM5 = 'A', kSideM5 = 'B', kBackM5 = 0;
constexpr const char* kTalkKey = "A";         // the letters a screen prints for them
constexpr const char* kSideKey = "B";
constexpr const char* kBackKey = "";
// M5.BtnA / M5.BtnB, read raw by the 20 ms sampler; the talk button also
// wakes the device from deep sleep (G2 is an RTC GPIO).
constexpr gpio_num_t kTalkPin = GPIO_NUM_2;   // KEYA (yellow)
constexpr gpio_num_t kSidePin = GPIO_NUM_1;   // KEYB
constexpr gpio_num_t kBackPin = GPIO_NUM_NC;  // no third button
constexpr int kScreenW = 466, kScreenH = 466;
constexpr bool kRound = true;
constexpr bool kEpaper = false;
constexpr bool kHasMotor = true;
constexpr bool kHasLeds = false;
constexpr bool kHasClimate = false;
// Voice notes recorded offline are kept in internal flash and sent when the
// deck is back and the user says yes (notes.cpp).
constexpr bool kOfflineNotes = true;
constexpr bool kHasSdCard = false;
// A touch screen: "tap to talk" puts a record button on it (an option).
constexpr bool kHasTouch = true;
constexpr bool kLoudPeakRestarts = false;  // not seen (yet) on the StopWatch
// Pictures bigger than one WebSocket frame arrive in parts (0x03 … 0x02).
constexpr bool kChunkedImages = true;

#elif defined(BOARD_PAPERCOLOR)

constexpr const char* kModel = "PaperColor";
constexpr const char* kHw = "papercolor";
constexpr const char* kNoun = "PaperColor";
constexpr const char* kIdPrefix = "paper";
constexpr bool kTalkToggles = true;  // press to start, press again to send: no holding on
constexpr bool kTalkFromSleep = true;  // a press that wakes it starts a recording
// M5Unified: BtnA = KEY3 (G10, top left), BtnB = KEY2 (G9, under it),
// BtnC = KEY1 (G1, on top). All three are RTC GPIOs, so any of them wakes
// the device from deep sleep.
#if defined(PAPER_B)
// The B paper: A talks (and goes back from the options), C opens the
// options (hold: Wi-Fi & settings), B shows the answer.
constexpr const char* kTalkButton = "A button";
constexpr const char* kSideButton = "top button";
constexpr const char* kTalkHint = "Press A to talk";
constexpr char kTalkM5 = 'A', kSideM5 = 'C', kBackM5 = 'B';
constexpr const char* kTalkKey = "A";
constexpr const char* kSideKey = "C";
constexpr const char* kBackKey = "B";
constexpr gpio_num_t kTalkPin = GPIO_NUM_10;  // A: talk / back from the options
constexpr gpio_num_t kSidePin = GPIO_NUM_1;   // C (top): options
constexpr gpio_num_t kBackPin = GPIO_NUM_9;   // B: the answer
#else
// The top one (C) talks; A goes next, B back.
constexpr const char* kTalkButton = "top button";
constexpr const char* kSideButton = "A button";
constexpr const char* kTalkHint = "Top button to talk";  // fits the home screen's chip
constexpr char kTalkM5 = 'C', kSideM5 = 'A', kBackM5 = 'B';
constexpr const char* kTalkKey = "C";
constexpr const char* kSideKey = "A";
constexpr const char* kBackKey = "B";
constexpr gpio_num_t kTalkPin = GPIO_NUM_1;   // C (top): talk / choose
constexpr gpio_num_t kSidePin = GPIO_NUM_10;  // A: next / status / Options
constexpr gpio_num_t kBackPin = GPIO_NUM_9;   // B: previous / back / home
#endif
constexpr int kScreenW = 400, kScreenH = 600;  // portrait; the menu can turn it
constexpr bool kRound = false;
constexpr bool kEpaper = true;
constexpr bool kHasMotor = false;
constexpr bool kHasLeds = true;
constexpr bool kHasClimate = true;
constexpr bool kOfflineNotes = true;
constexpr bool kHasSdCard = true;  // microSD slot (SPI, shared with the panel): notes go there when a card is in
constexpr bool kHasTouch = false;
constexpr bool kLoudPeakRestarts = false;
constexpr bool kChunkedImages = true;

#elif defined(BOARD_EPAPER154)

// Both are the V2 board (ESP32-S3-PICO-1-N8R8, "V2" on the silkscreen); a V1
// (4 MB flash, 2 MB PSRAM) is too small for this firmware.
#if defined(EPAPER154_TOUCH)
constexpr const char* kModel = "TouchPaper154";
constexpr const char* kHw = "epaper154_touch";
constexpr const char* kIdPrefix = "tpaper";
constexpr bool kHasTouch = true;  // FT6336: tap to talk, swipes page
#else
constexpr const char* kModel = "ePaper154";
constexpr const char* kHw = "epaper154";
constexpr const char* kIdPrefix = "epaper";
constexpr bool kHasTouch = false;
#endif
constexpr const char* kNoun = "device";
constexpr const char* kTalkButton = "BOOT button";
constexpr const char* kSideButton = "PWR button";
constexpr const char* kTalkHint = "Press BOOT to talk";
constexpr bool kTalkToggles = true;
constexpr bool kTalkFromSleep = true;  // the paper looks awake when it sleeps: a press talks
constexpr char kTalkM5 = 'A', kSideM5 = 'B', kBackM5 = 0;  // hw::update() feeds M5.BtnA / M5.BtnB
constexpr const char* kTalkKey = "BOOT";
constexpr const char* kSideKey = "PWR";
constexpr const char* kBackKey = "";
constexpr gpio_num_t kTalkPin = GPIO_NUM_0;   // BOOT: a strapping pin, read only at power-on and reset
constexpr gpio_num_t kSidePin = GPIO_NUM_18;  // PWR: on battery it also switches the board on
constexpr gpio_num_t kBackPin = GPIO_NUM_NC;
constexpr gpio_num_t kMonoLed = GPIO_NUM_3;   // green, active low
constexpr int kScreenW = 200, kScreenH = 200;
constexpr bool kRound = false;
constexpr bool kEpaper = true;
constexpr bool kHasMotor = false;
constexpr bool kHasLeds = true;  // the green one (leds.cpp); the red one is the charger's
constexpr bool kHasClimate = false;  // there's an SHTC3, but the board heats it well past the room
constexpr bool kOfflineNotes = true;
constexpr bool kHasSdCard = true;  // microSD on its own SDMMC pins
constexpr bool kLoudPeakRestarts = false;
constexpr bool kChunkedImages = true;

#else  // BOARD_STICKS3

constexpr const char* kModel = "StickS3";
constexpr const char* kHw = "sticks3";
constexpr const char* kNoun = "stick";
constexpr const char* kIdPrefix = "stick";
constexpr const char* kTalkButton = "front button";
constexpr const char* kSideButton = "side button";
constexpr const char* kTalkHint = "Press the front button to talk";
constexpr bool kTalkToggles = true;  // press to start, press again to send
constexpr bool kTalkFromSleep = false;  // a press that wakes it only wakes it; the next one talks
constexpr char kTalkM5 = 'A', kSideM5 = 'B', kBackM5 = 0;
constexpr const char* kTalkKey = "A";
constexpr const char* kSideKey = "B";
constexpr const char* kBackKey = "";
constexpr gpio_num_t kTalkPin = GPIO_NUM_11;  // front (A)
constexpr gpio_num_t kSidePin = GPIO_NUM_12;  // side (B)
constexpr gpio_num_t kBackPin = GPIO_NUM_NC;
constexpr int kScreenW = 240, kScreenH = 135;
constexpr bool kRound = false;
constexpr bool kEpaper = false;
constexpr bool kHasMotor = false;
constexpr bool kHasLeds = false;
constexpr bool kHasClimate = false;
constexpr bool kOfflineNotes = true;  // in flash: the 1.5 MB data partition holds ~5½ min
constexpr bool kHasSdCard = false;
constexpr bool kHasTouch = false;
constexpr bool kLoudPeakRestarts = true;   // M5Stack's note: >75% volume on battery
constexpr bool kChunkedImages = false;

#endif

// The PaperColor built the "B" way (env papercolor_b): the screen is a
// dashboard redrawn every half hour, the lights and the voice carry the
// rest; C talks, A shows the options code, B shows the last answer.
#if defined(PAPER_B)
constexpr bool kPaperB = true;
#else
constexpr bool kPaperB = false;
#endif

// Note-taking mode (hold the talk button): every recording is kept as a voice
// note, written to flash (or a card) as it's recorded, and they all go to the
// agent once the mode is turned off. The B paper and the StopWatch.
#if defined(PAPER_B) || defined(BOARD_STOPWATCH) || defined(BOARD_EPAPER154)
constexpr bool kNoteMode = true;
#else
constexpr bool kNoteMode = false;
#endif

// A screen that takes ~20 s to redraw (Spectra 6): the menu is read aloud
// instead of drawn, and only what must be on it goes on it. kEpaper alone is
// what every e-paper shares: it keeps its picture unpowered, no backlight.
#if defined(BOARD_PAPERCOLOR)
constexpr bool kSlowPaper = true;
#else
constexpr bool kSlowPaper = false;
#endif

// Tap to talk's first setting, until it's changed on the phone page.
#if defined(BOARD_EPAPER154)
constexpr bool kTapTalkDefault = true;
#else
constexpr bool kTapTalkDefault = false;
#endif

// A double-click of the side button goes to the home screen, from wherever it
// is. Elsewhere it replays an answer (here the talk button's double press does).
#if defined(BOARD_EPAPER154)
constexpr bool kDoubleHome = true;
#else
constexpr bool kDoubleHome = false;
#endif

// A toggled recording also ends after 2.5 s of quiet. Not here: this mic's level
// stopped it mid-sentence, so only a press ends one (or the length limits).
#if defined(BOARD_EPAPER154)
constexpr bool kSilenceStop = false;
#else
constexpr bool kSilenceStop = true;
#endif

// Fast enough to keep a clock going asleep: "Redraw asleep" can be every minute.
#if defined(BOARD_EPAPER154)
constexpr bool kSleepClock = true;
#else
constexpr bool kSleepClock = false;
#endif

}  // namespace board
