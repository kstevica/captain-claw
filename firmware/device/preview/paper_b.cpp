// Renders the PaperColor "B" screens (the real ui_paper_b.cpp and its six-ink
// conversion) on a Mac, one PPM per screen. preview/render.py paper_b turns
// them into PNGs with the panel's muted inks.
//
//   pio run -e preview_paper_b
//   uv run --no-project --with pillow python preview/render.py paper_b
#include <M5Unified.h>

#include <string>
#include <vector>

#include "../src/fonts.h"
#include "../src/telemetry.h"
#include "../src/ui.h"

uint32_t g_previewMillis = 0;
PreviewM5 M5;

// ── stand-ins for the device-only parts ─────────────────────────────────

static bool gClimate = true, gUsb = false, gOnCard = false;
static int gMinutesLeft = 130, gMinutesToFull = -1, gNotes = 3, gRoomMin = 14;

namespace telemetry {
bool climate(float* t, float* rh) {
  *t = 22.6f;
  *rh = 44.0f;
  return gClimate;
}
bool clockSet() { return true; }
bool onUsbPower() { return gUsb; }
}  // namespace telemetry

namespace battery {
int minutesLeft() { return gMinutesLeft; }
int minutesToFull() { return gMinutesToFull; }
}  // namespace battery

namespace notes {
bool ready() { return true; }
bool onCard() { return gOnCard; }
uint64_t freeBytes() { return gOnCard ? 12400000000ULL : 3400000; }
int count() { return gNotes; }
uint32_t seconds() { return gNotes * 28; }
size_t room(uint32_t rate) { return (size_t)rate * 60 * gRoomMin; }
}  // namespace notes

namespace ui {
void previewAsleep();
}

static std::string gRoot = ".";

namespace vlw {
const uint8_t* data(Id id, size_t* len) {
  static const char* names[Count] = {"small", "body_s", "body_m", "body_l", "title", "badge", "digits"};
  static std::vector<uint8_t> files[Count];
  if (files[id].empty()) {
    std::string path = gRoot + "/fonts/" + names[id] + ".vlw";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      fprintf(stderr, "missing %s\n", path.c_str());
      exit(1);
    }
    fseek(f, 0, SEEK_END);
    files[id].resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    fread(files[id].data(), 1, files[id].size(), f);
    fclose(f);
  }
  *len = files[id].size();
  return files[id].data();
}
}  // namespace vlw

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

static constexpr uint16_t kGreen = ui::rgb(40, 200, 90);

static const char* kAsked = "What's on my calendar tomorrow, and can you move the dentist to Friday afternoon?";
static const char* kShort = "Done. The dentist is on Friday at 15:00 — the clinic confirmed it.";
static const char* kAnswer =
    "Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at 16:00. "
    "I moved the dentist to Friday at 15:00 — the clinic confirmed it.";
static const char* kLong =
    "Sutra imaš tri sastanka: dnevni sastanak u 9:30, ručak s Anom u 12:00 i zubara u 16:00. Zubara sam "
    "premjestio na petak u 15:00 — Đuro je potvrdio termin. Za ručak sam rezervirao stol u Čakovcu, a šećer "
    "i kava su već naručeni.\n\nAko želiš, mogu poslati podsjetnik svima sat vremena prije, ili dodati "
    "bilješke s prošlog sastanka u pozivnicu. Prognoza za petak je sunčano, 19 °C, pa je terasa slobodna "
    "za kavu nakon zubara. Ponedjeljak je mirniji: samo pregled projekta u 11:00 i poziv s Markom u 15:30, "
    "za koji sam već pripremio sažetak zadnjih brojki i tri pitanja koja je postavio prošli tjedan. "
    "Utorak i srijeda su za sada prazni, a u četvrtak u 10:00 je sastanak odbora za koji još čekamo dnevni red.";

int main(int argc, char** argv) {
  for (const char* name : {"Captain Claw", "captain-claw", "CAPTAIN_CLAW", "Captain"}) {
    if (ui::agentShownName(name) != "Captain") fprintf(stderr, "agentShownName(%s) wrong\n", name), exit(1);
  }
  if (ui::agentShownName("Jarvis") != "Jarvis" || ui::agentShownName("Captain Claw 2") != "Captain 2") {
    fprintf(stderr, "agentShownName changes the wrong names\n");
    exit(1);
  }
  gOut = argc > 1 ? argv[1] : "preview/out/paper_b";
  gRoot = argc > 2 ? argv[2] : ".";
  setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3", 1);
  tzset();

  M5.Display.setColorDepth(16);
  M5.Display.createSprite(400, 600);
  ui::begin();
  g_previewMillis = 123456;
  ui::setHealth(ui::Up, ui::Up);
  ui::setBar(ui::agentShownName("Captain Claw"), "ready", kGreen);

  static constexpr uint16_t kRed = ui::rgb(230, 50, 50), kYellow = ui::rgb(250, 200, 40);
  ui::Home home{true, "All good", kGreen, "", {}};
  home.rows = {{"Paper", "ready — press A", kGreen},
               {"Wi-Fi", "Harbour-5G", kGreen},
               {"Deck", "deck.example.com", kGreen},
               {"Agent", ui::agentShownName("Captain Claw"), kGreen}};
  ui::showHome(home);
  save("01_dashboard");

  ui::setQuestion(kAsked);
  ui::setReply(kShort);
  ui::nextPage();  // B: the answer, large
  save("02_answer_short");

  ui::setQuestion(kAsked);
  ui::setReply(kAnswer);
  ui::nextPage();
  save("03_answer_medium");

  ui::setQuestion("Što imam sutra?");
  ui::setReply(kLong);
  ui::nextPage();
  save("04_answer_long_p1");
  ui::nextPage();
  save("05_answer_long_p2");
  ui::nextPage();  // past the last page: the dashboard again
  save("06_dashboard_again");

  size_t jpgLen = 0;
  if (uint8_t* jpg = readFile(gOut + "/sample.jpg", &jpgLen)) {
    ui::setQuestion("Show me the harbour on Friday");
    ui::setReply(kShort);
    ui::storeImage(jpg, jpgLen);
    ui::nextPage();
    save("06b_answer_picture");
    ui::nextPage();
  }
  ui::setQuestion("");
  ui::setReply("Ana replied about the contract and wants to meet on Friday.");
  ui::nextPage();
  save("06c_answer_no_question");
  ui::nextPage();
  ui::setQuestion("What's on my calendar for the rest of this week and next week, and can you move everything on Friday afternoon to Monday morning, and tell everyone?");
  ui::setReply("Done.");
  ui::nextPage();
  save("06d_answer_long_question");
  ui::nextPage();

  ui::showLink("Options · 1.0.16", "http://192.168.1.48/", "Scan it with your phone (same Wi-Fi): every setting is there.");
  save("07_options");

  ui::showSetup("First-time setup", "Captain-PaperColor-b2c3", "192.168.4.1");
  save("08_setup");

  ui::setHealth(ui::Down, ui::Down);
  home.ready = false;
  home.hint = "Offline · talk to keep a note";
  home.rows = {{"Paper", "not ready", kYellow},
               {"Wi-Fi", "not connected", kRed},
               {"Deck", "connecting…", kRed},
               {"Agent", "-", kYellow}};
  ui::drawNow();
  ui::showHome(home);  // asked for (B held): the dashboard now
  save("09_dashboard_links_down");
  home.ready = true;
  home.hint = "";

  ui::setHealth(ui::Up, ui::Up);
  home.rows = {{"Paper", "ready — press A", kGreen},
               {"Wi-Fi", "Harbour-5G", kGreen},
               {"Deck", "deck.example.com", kGreen},
               {"Agent", ui::agentShownName("Captain Claw"), kGreen}};
  home.noteMode = true;  // hold A: note-taking mode
  ui::drawNow();
  ui::showHome(home);
  save("10_dashboard_note_mode");

  // Proposals wait in Flight Deck: a row after the device's (main.cpp homeNow()).
  home.noteMode = false;
  home.rows.insert(home.rows.begin() + 1, {"Waiting", "2 in Flight Deck", kRed});
  ui::setWaiting(2);
  ui::drawNow();
  ui::showHome(home);
  save("11_dashboard_waiting");

  // The agent asleep (Flight Deck put it to sleep for being idle): its dot
  // and name blue — talking wakes it. Switched off: red.
  static constexpr uint16_t kBlue = ui::rgb(70, 150, 255);
  home.rows.erase(home.rows.begin() + 1);
  ui::setWaiting(0);
  ui::setHealth(ui::Up, ui::Asleep);
  home.rows.back() = {"Agent", ui::agentShownName("Captain Claw") + " · asleep", kBlue};
  ui::drawNow();
  ui::showHome(home);
  save("12_dashboard_agent_asleep");
  ui::setHealth(ui::Up, ui::Off);
  home.rows.front() = {"Paper", "not ready", kYellow};
  home.rows.back() = {"Agent", ui::agentShownName("Captain Claw") + " · stopped", kRed};
  ui::drawNow();
  ui::showHome(home);
  save("13_dashboard_agent_stopped");

  // The readings at their edges.
  ui::setHealth(ui::Up, ui::Up);
  home.rows.front() = {"Paper", "ready — press A", kGreen};
  home.rows.back() = {"Agent", ui::agentShownName("Captain Claw"), kGreen};
  auto again = [&](const char* name) {
    g_previewMillis += 30000;  // (the battery's read at most every 20 s)
    ui::drawNow();
    ui::showHome(home);
    save(name);
  };
  M5.Power.level = 31;
  gMinutesLeft = 48;
  again("15_battery_31");
  M5.Power.level = 12;
  gMinutesLeft = 19;
  again("16_battery_12");
  M5.Power.level = 64;
  M5.Power.charging = true;
  gMinutesToFull = 45;
  again("17_charging");
  gMinutesToFull = 95;
  again("18_charging_long");
  M5.Power.charging = false;
  gUsb = true;
  M5.Power.level = 100;
  again("19_on_usb");
  gUsb = false;
  M5.Power.level = 82;
  gMinutesLeft = 130;
  gClimate = false;
  again("20_no_room_reading");
  gClimate = true;
  gNotes = 0;
  again("21_no_notes");
  gNotes = 128;
  gRoomMin = 1;
  again("22_many_notes_nearly_full");
  home.noteMode = true;
  gRoomMin = 0;
  again("23_note_mode_full");
  gNotes = 1;
  gRoomMin = 90;
  gOnCard = true;
  again("24_note_mode_on_card");
  gOnCard = false;
  gNotes = 3;
  gRoomMin = 14;
  ui::setHealth(ui::Down, ui::Off);
  ui::setWaiting(2);
  again("25_note_mode_links_down_waiting");
  ui::setHealth(ui::Up, ui::Up);
  home.noteMode = false;
  ui::setHealth(ui::Up, ui::Up);
  ui::setBar("A very long agent name indeed for this one", "ready", kGreen);
  again("26_long_names");
  ui::setBar(ui::agentShownName("Captain Claw"), "ready", kGreen);
  ui::setWaiting(0);
  ui::previewAsleep();
  save("27_asleep");
  ui::setWaiting(1);
  home.noteMode = true;
  ui::setHome(home);
  ui::previewAsleep();
  save("28_asleep_note_mode");
  ui::setWaiting(0);

  ui::showSwitchedOff("Power button to switch on");
  save("14_switched_off");

  // (Asleep: the dashboard goes by the clock's five-minute marks — the Mac's
  // real clock here, so a jump of the stand-in millis() doesn't make it due.)
  return 0;
}
