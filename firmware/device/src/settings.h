// Persistent device settings (NVS): saved Wi-Fi networks, the Flight Deck
// address and token, voice options. Edited from the setup portal.
#pragma once

#include <Arduino.h>

static constexpr int kMaxNetworks = 5;
static constexpr const char* kDefaultTz = "CET-1CEST,M3.5.0,M10.5.0/3";  // Central Europe

// The paper's power-off choices, in idle minutes (0 = never), and how they read.
static constexpr uint8_t kPowerOffMins[] = {1, 3, 10, 30, 60, 120, 0};
bool powerOffMinValid(int minutes);
String powerOffText(uint8_t minutes);  // "3 min", "1 hour", "never"

// How often the sleeping screen is redrawn, in minutes (0 = never). Under 30 only
// where the panel is quick enough for a clock (board::kSleepClock).
static constexpr uint8_t kAsleepMins[] = {1, 2, 5, 10, 30, 60, 180, 0};
bool asleepMinValid(int minutes);

struct WifiNet {
  String ssid;
  String pass;
};

struct Settings {
  WifiNet nets[kMaxNetworks];
  int netCount = 0;
  String server;        // ws://192.168.1.10:25080  or  wss://deck.example.com
  String token;         // cc_pat_… personal access token ("" on an auth-off deck)
  String agent;         // preferred agent slug ("" = first running)
  String lang;          // voice language for spoken replies ("" = server default)
  bool speak = true;    // read replies aloud
  uint8_t volume = 191; // 0..255 (M5 docs: above ~75% on battery may brown out)
  uint8_t boostDb = 6;  // extra speech gain with a soft limiter: 0, 6 or 10 dB
  uint8_t screenOffS = 15;  // screen goes dark after this many idle seconds: 5, 10 or 15
  uint8_t powerOffS = 60;   // deep sleep (front button wakes) after: 15, 30, 60 or 180 s
  bool femaleVoice = false; // spoken replies: male (Adrian) / female (Emma) on the deck
  bool iconMode = false;    // screen: text stages, or Captain himself
  bool cueBeep = true;      // chime before speech starts
  bool talkSounds = true;   // a blip when recording starts, a lower one when it's let go
  bool cueFlash = true;     // flash the screen before speech starts
  bool cueBuzz = true;      // buzz before speech starts (devices with a motor)
  uint8_t cueLeadS = 1;     // the cue comes this many seconds (1 or 2) before speech
  bool tapTalk = false;     // touch screens: a record button — tap to talk, tap again to send
  uint16_t talks = 0;       // recordings sent so far: the home screen's "hold … to talk" goes after a few
  bool autoUpdate = false;  // install offered updates by itself (idle + USB power); the B paper defaults to on
  bool noteMode = false;    // the B paper's note-taking mode: recordings are kept, sent when it's turned off
  String lastFw;            // the version that last ran (shows "Updated to …" once)
  uint8_t textSize = 0;     // reading text: 0 small, 1 medium, 2 large
  String tz;                // POSIX time zone for the clock ("" = kDefaultTz)
  // PaperColor (e-paper):
  uint8_t rotation = 0;     // 0 portrait, 1 landscape, 2 portrait flipped, 3 landscape flipped
  uint8_t ledLevel = 1;     // status lights: 0 off, 1 low, 2 bright
  bool keyClicks = true;    // a click for every button press (the paper changes slowly)
  uint8_t powerOffMin = 3;  // deep sleep after this many idle minutes: kPowerOffMins (0 = never)
  uint8_t asleepRefreshMin = 60;  // redraw the sleeping screen: kAsleepMins
  bool speakMenu = true;    // read the highlighted menu row aloud (the paper shows it ~20 s later)

  // Idle time before deep sleep, for this device.
  uint32_t powerOffMs() const;

  void load();
  void save() const;
  // Add or update a network (most recent first); false when the list is full.
  bool addNetwork(const String& ssid, const String& pass);
  void removeNetwork(int index);
  bool configured() const { return netCount > 0 && server.length() > 0; }
};

extern Settings settings;

// "stick-a1b2c3" / "watch-a1b2c3" — stable per device (from the Wi-Fi MAC).
String deviceId();
