// Renders the Waveshare 1.54" screens (the real ui_epaper154.cpp and its
// black-and-white frames) on a Mac, one PPM per screen. preview/render.py
// epaper154 turns them into PNGs on the panel's paper tone.
//
//   pio run -e preview_epaper154 -e preview_epaper154_touch
//   uv run --no-project --with pillow python preview/render.py epaper154
#include <M5Unified.h>

#include <string>
#include <vector>

#include "../src/hw.h"
#include "../src/telemetry.h"
#include "../src/ui.h"

uint32_t g_previewMillis = 0;
PreviewM5 M5;

// ── stand-ins for the device-only parts ─────────────────────────────────

namespace telemetry {
bool clockSet() { return true; }
bool onUsbPower() { return false; }
}  // namespace telemetry

namespace hw {
int batteryLevel() { return 76; }
int batteryMv() { return 3950; }
bool charging() { return false; }
void setTouchRotation(uint8_t) {}
}  // namespace hw

// ── output ─────────────────────────────────────────────────────────────

static std::string gOut;

static void save(const char* name) {
  ui::flush();
  const int w = M5.Display.width(), h = M5.Display.height();
  std::vector<uint8_t> rgb(w * h * 3);
  M5.Display.readRectRGB(0, 0, w, h, rgb.data());
  std::string path = gOut + "/" + name + ".ppm";
  FILE* f = fopen(path.c_str(), "wb");
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  fwrite(rgb.data(), 1, rgb.size(), f);
  fclose(f);
  printf("%s\n", path.c_str());
}

static uint8_t* readFile(const std::string& path, size_t* len) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  *len = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* buf = (uint8_t*)malloc(*len);
  fread(buf, 1, *len, f);
  fclose(f);
  return buf;
}

// ── the screens ────────────────────────────────────────────────────────

static const char* kAnswer =
    "Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at 16:00. "
    "I moved the dentist to Friday at 15:00 — the clinic confirmed it.";
static const char* kLong =
    "Sutra imaš tri sastanka: dnevni sastanak u 9:30, ručak s Anom u 12:00 i zubara u 16:00. Zubara sam "
    "premjestio na petak u 15:00. Za ručak sam rezervirao stol u Čakovcu, a šećer i kava su već naručeni.\n\n"
    "Ako želiš, mogu poslati podsjetnik svima sat vremena prije, ili dodati bilješke s prošlog sastanka u "
    "pozivnicu. Prognoza za petak je sunčano, 19 °C, pa je terasa slobodna za kavu nakon zubara.";

int main(int argc, char** argv) {
  gOut = argc > 1 ? argv[1] : "preview/out/epaper154";
  setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3", 1);
  tzset();

  M5.Display.setColorDepth(16);
  M5.Display.createSprite(200, 200);
  ui::begin();
  g_previewMillis = 123456;
  ui::setHealth(ui::Up, ui::Up);
  ui::setBar("Captain", "ready", 0);
  ui::setTextSize(0);
  ui::setTapButton(board::kHasTouch ? ui::TapButton::Talk : ui::TapButton::Off);

  ui::Home home{true, "All good", 0, board::kTalkHint, {}};
  home.battery = "≈ 2 h 10 min left";
  ui::showHome(home);
  save("01_home");
  ui::setTextSize(1);
  ui::showHome(home);
  save("01b_home_medium");
  ui::setTextSize(2);
  ui::showHome(home);
  save("01c_home_large");
  ui::setTextSize(0);

  ui::setWaiting(3);
  ui::setHealth(ui::Up, ui::Asleep);
  ui::setBar("Researcher", "asleep", 0);
  ui::showHome(home);
  save("02_home_waiting_agent_asleep");
  ui::setWaiting(0);

  ui::setHealth(ui::Down, ui::Down);
  ui::setBar("", "offline", 0);
  ui::Home off{false, "Offline", 0, "Talk to keep a voice note", {}};
  ui::showHome(off);
  save("03_home_offline");

  ui::setHealth(ui::Up, ui::Up);
  ui::setBar("Captain", "ready", 0);
  home.noteMode = true;
  ui::showHome(home);
  save("04_home_note_mode");
  home.noteMode = false;

  ui::setTapButton(board::kHasTouch ? ui::TapButton::Stop : ui::TapButton::Off);
  ui::setBar("Captain", "listening", 0);
  ui::showListening(62, 12400);
  save("05_listening");
  ui::setBar("Captain", "note", 0);
  ui::showListening(30, 75000);
  save("06_note");

  ui::setBar("Captain", "0:14 · 3", 0);
  ui::showWorking("What's on my calendar tomorrow, and can you move the dentist to Friday afternoon?",
                  {"Reading calendar", "Searching mail for the clinic", "Moving the dentist"}, "Checking Friday…", 14000, 3);
  save("07_working");

  ui::setTapButton(board::kHasTouch ? ui::TapButton::Pause : ui::TapButton::Off);
  ui::setBar("Captain", "speaking", 0);
  ui::setReply(kAnswer);
  ui::showReply();
  save("08_answer_small");

  ui::setTextSize(1);
  ui::setReply(kLong);
  ui::showReply();
  save("09_answer_medium_p1");
  ui::nextPage();
  save("10_answer_medium_p2");

  ui::setTextSize(2);
  ui::setReply(kAnswer);
  ui::showReply();
  save("11_answer_large");
  ui::setTextSize(0);
  ui::setTapButton(ui::TapButton::Off);

  ui::setBar("Captain", "menu", 0);
  std::vector<ui::Item> menu;
  for (const char* row : {"Replay last answer", "Status", "Voice notes: 3", "Agent: Captain", "Volume: 75%", "Text size: small",
                          "Tap to talk: on", "More…", "Software 0.19.2 · check", "Wi-Fi & settings…"}) {
    ui::Item it;
    it.label = row;
    menu.push_back(it);
  }
  ui::showPicker(menu, 4, 3);
  save("12_menu");

  ui::setBar("Captain", "1/2", 0);
  ui::Status s;
  s.wifi = true;
  s.ssid = "Harbour-5G";
  s.ip = "192.168.1.48";
  s.rssi = -58;
  s.deck = true;
  s.deckHost = "glasses.captain-claw.com";
  s.rttMs = 41;
  s.agent = "Captain";
  s.agentLinked = true;
  s.batteryPct = 76;
  s.batteryMv = 3950;
  s.minutesLeft = 130;
  s.uptime = "1 h 4 min";
  s.wake = "PWR 240s  timer 12s  restart";
  s.turns = 12;
  s.lastAnswerMin = 3;
  s.voice = "female";
  s.volumePct = 75;
  s.speak = true;
  s.deviceId = "epaper-a1b2c3";
  s.fw = "0.19.2";
  s.freeKb = 7600;
  s.notes = 3;
  s.notesRoomMin = 5;
  s.notesLength = "1:24";
  ui::showStatus(s, 0);
  save("13_status_p1");
  ui::showStatus(s, 1);
  save("14_status_p2");

  ui::setBar("Captain", "ready", 0);
  ui::showMessage("Saved offline", "Kept on the device · 3 waiting. Back online, it asks to send them.", 0);
  save("15_notice");

  ui::showAsk("Voice notes", "There are 3 saved voice notes. Send them to the agent?", "Send", "Later");
  save("16_ask");

  ui::showLink("Phone settings", "http://192.168.1.48/s?k=7f3a91",
               "Scan it with your phone (same Wi-Fi). Saving restarts the device.");
  save("17_phone_settings");

  ui::showSetup("First-time setup", "Captain-ePaper154-b2c3", "192.168.4.1");
  save("18_setup");

  size_t len = 0;
  if (uint8_t* jpg = readFile(gOut + "/sample.jpg", &len)) {
    ui::storeImage(jpg, len);
    ui::showStoredImage();
    save("19_picture");
  }

  ui::setBar("Captain", "asleep", 0);
  ui::showSleeping("Press BOOT to wake", 0);
  save("20_asleep");

  ui::setBar("Captain", "off", 0);
  ui::showSwitchedOff("Press PWR to switch on");
  save("21_switched_off");
  return 0;
}
