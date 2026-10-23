// Renders the StopWatch screens (the real ui_round.cpp and characters) on a
// Mac, one PPM per screen, so the round UI can be designed before — and
// without — flashing the watch. preview/render.py turns them into PNGs.
//
//   pio run -e preview && .pio/build/preview/program preview/out
//   uv run --no-project --with pillow python preview/render.py preview/out
#include <M5Unified.h>

#include <functional>
#include <string>
#include <vector>

#include "../src/audio.h"
#include "../src/fonts.h"
#include "../src/ui.h"

uint32_t g_previewMillis = 0;
PreviewM5 M5;

// ── stand-ins for the device-only parts ─────────────────────────────────

static bool gPlaying = false;
static int gLevel = 0;

namespace audio {
bool playing() { return gPlaying; }
bool sounding() { return gPlaying; }
int outputLevel() { return gLevel; }
}  // namespace audio

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

static std::vector<uint8_t> readFile(const std::string& path) {
  std::vector<uint8_t> out;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return out;
  fseek(f, 0, SEEK_END);
  out.resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  fread(out.data(), 1, out.size(), f);
  fclose(f);
  return out;
}

// ── the screens ────────────────────────────────────────────────────────

static constexpr uint16_t kGreen = ui::rgb(40, 200, 90), kYellow = ui::rgb(250, 200, 40);
static constexpr uint16_t kRed = ui::rgb(230, 50, 50), kCyan = ui::rgb(60, 210, 240);
static constexpr uint16_t kOrange = ui::rgb(255, 150, 40), kText = ui::rgb(230, 230, 230);

static ui::Item card(const char* title, const String& badge, const char* note, bool on = true,
                     ui::Glyph glyph = ui::Glyph::None) {
  ui::Item it;
  it.title = title;
  it.label = title;
  it.badge = badge;
  it.note = note;
  it.on = on;
  it.glyph = glyph;
  return it;
}

static std::vector<ui::Item> menu() {
  return {
      card("Replay answer", "", "yellow = hear it again", true, ui::Glyph::Replay),
      card("Status", "", "Wi-Fi, deck, battery…", true, ui::Glyph::Info),
      card("Agent", "Captain Claw", "Captain Claw · change", true, ui::Glyph::Initials),
      card("Volume", "75%", "yellow = change"),
      card("Settings", "", "voice, display, sounds, timers…", true, ui::Glyph::Gear),
      card("Speech boost", "off", "yellow = change", false),
      card("Voice", "female", "yellow = change"),
      card("Display", "text", "yellow = change"),
      card("Text size", "Aa", "medium · change"),
      card("Character", "Captain", "yellow = browse"),
      card("Beeps around speech", "on", "yellow = change"),
      card("Flash before speech", "on", "yellow = change"),
      card("Buzz before speech", "on", "yellow = change"),
      card("Cue lead", "1 s", "how early the cue comes"),
      card("Screen off", "15 s", "after this long idle"),
      card("Power off", "60 s", "after this long idle"),
      card("Wi-Fi & settings", "", "opens the setup hotspot", true, ui::Glyph::Wifi),
  };
}

static void bar(const char* state, uint16_t color, ui::Health wifi = ui::Up, ui::Health agent = ui::Up) {
  ui::setHealth(wifi, agent);
  ui::setBar("Captain Claw", state, color);
}

int main(int argc, char** argv) {
  gOut = argc > 1 ? argv[1] : "preview/out";
  gRoot = argc > 2 ? argv[2] : ".";
  setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3", 1);
  tzset();

  M5.Display.setColorDepth(16);
  M5.Display.createSprite(466, 466);
  ui::begin();
  ui::setTextSize(1);
  g_previewMillis = 123456;

  bar("ready", kGreen);
  ui::Home home{true, "All good", kGreen, "Hold the yellow button to talk", {}};
  ui::showHome(home);
  save("01_home");

  bar("ready", kYellow, ui::Up, ui::Partial);
  ui::showHome({false, "Getting ready", kYellow, "Waiting for the agent", {}});
  save("02_home_waiting");

  bar("listening", kRed);
  int lv[] = {10, 30, 55, 80, 62, 40, 70, 90, 66, 35, 20, 48, 75, 58, 30};
  for (int l : lv) ui::showListening(l, 4300);
  save("03_listening");

  bar("12s · 3 steps", kYellow);
  g_previewMillis = 200400;
  ui::showWorking("What's on my calendar tomorrow, and can you move the dentist to Friday afternoon?",
                  {"Reading the calendar", "google_calendar · list events", "Looking for a free slot on Friday"},
                  "Checking Friday afternoon", 12400, 3);
  save("04_working");

  bar("speaking", kCyan);
  gPlaying = true;
  gLevel = 70;
  ui::setReply("Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at "
               "16:00. I moved the dentist to Friday at 15:00.");
  ui::showReply();
  ui::showSpeaking(70);  // the speaking line, drawn over the page
  save("05_reply_speaking");
  gPlaying = false;

  bar("ready", kGreen);
  ui::setReply("Sutra imaš tri sastanka: dnevni sastanak u 9:30, ručak s Anom u 12:00 i zubara u 16:00. Zubara sam "
               "premjestio na petak u 15:00 — Đuro je potvrdio termin. Za ručak sam rezervirao stol u Čakovcu, "
               "a šećer i kava su već naručeni. Ako želiš, mogu poslati podsjetnik svima sat vremena prije, "
               "ili dodati bilješke s prošlog sastanka u pozivnicu.");
  ui::showReply();
  save("06_reply_long_p1");
  ui::nextPage();
  save("07_reply_long_p2");
  ui::prevPage();
  ui::setSpoken(120);  // the voice in the second sentence: lit, the rest dimmed
  bar("speaking", kCyan);
  ui::showReply();
  save("07b_reply_lit");
  ui::showFace(1, face::Pose::Speaking, 70);  // the character: that sentence as a subtitle
  save("07c_icon_subtitle");
  ui::setSpoken(-1);
  bar("ready", kGreen);
  ui::setReplayable(true);  // after the speech: hear it again
  ui::showReply();
  save("07d_reply_replay");
  ui::setReplayable(false);

  auto items = menu();
  bar("menu", kText);
  ui::showPicker(items, 3, -1);
  save("08_menu_volume");
  ui::showPicker(items, 4, -1);
  save("09_menu_settings");
  ui::showPicker(items, 2, -1);
  save("10_menu_agent");
  ui::showPicker(items, 15, -1);
  save("11_menu_wifi");
  ui::showPicker(items, 0, -1);
  save("11b_menu_replay");

  std::vector<ui::Item> agents = {card("Captain Claw", "Captain Claw", "current agent", true, ui::Glyph::Initials),
                                  card("Research desk", "Research desk", "yellow = choose", true, ui::Glyph::Initials),
                                  card("Back", "", "to the menu", true, ui::Glyph::Back)};
  ui::showPicker(agents, 0, 0);
  save("12_agents_current");
  ui::showPicker(agents, 1, 0);
  save("13_agents_other");

  bar("status 1/2", kText);
  std::vector<ui::Row> rows = {
      {"Wi-Fi", "Kuharski-5G  -54 dBm", kGreen},
      {"IP", "192.168.1.48", kText},
      {"Deck", "glasses.captain-claw.com  38 ms", kGreen},
      {"Agent", "Captain Claw", kGreen},
      {"Battery", "82%  3.98 V", kGreen},
      {"Voice", "female  vol 75%  +6 dB", kText},
      {"Usage", "7 turns  last 2m ago  up 1h 12m", kText},
      {"Device", "watch-a1b2c3  fw 0.7.0  7480k free", kText},
  };
  int pages = ui::showPanel("", kText, rows, 0);
  save("14_status_p1");
  if (pages > 1) {
    ui::showPanel("", kText, rows, 1);
    save("15_status_p2");
  }

  bar("ready", kGreen);
  ui::showMessage("Hmm", "The agent didn't answer in time. Try again in a moment.", kOrange);
  save("16_notice");

  ui::setReply("");
  bar("ready", kGreen);
  ui::showFace(0, face::Pose::Idle, 0);
  save("17_icon_captain");
  bar("speaking", kCyan);
  gPlaying = true;
  g_previewMillis = 300000;
  ui::showFace(1, face::Pose::Speaking, 70);
  save("18_icon_bot_speaking");
  gPlaying = false;
  bar("12s · 3 steps", kYellow);
  ui::showFace(8, face::Pose::Thinking, 0);
  save("19_icon_c64_thinking");
  bar("listening", kRed);
  ui::showFace(11, face::Pose::Listening, 60);
  save("20_icon_dog_listening");

  bar("1/18  A=pick", kCyan);
  ui::showCharPick(17, face::Pose::Idle, 0, 17);
  save("21_character_pick");

  ui::showSetup("First-time setup", "Captain-StopWatch-b2c3", "192.168.4.1");
  save("22_setup");

  bar("speaking", kCyan);
  ui::setReply("On my way.");
  ui::showReply();
  ui::flashFrame(true);
  save("23_flash");

  auto jpg = readFile(gOut + "/sample.jpg");
  if (!jpg.empty()) {
    uint8_t* buf = (uint8_t*)malloc(jpg.size());
    memcpy(buf, jpg.data(), jpg.size());
    ui::storeImage(buf, jpg.size());
    ui::showStoredImage();
    save("24_picture");
    ui::toggleImageFit();  // a tap: all of it
    ui::showStoredImage();
    save("24b_picture_fit");
    ui::toggleImageFit();
  }

  M5.Power.level = 14;
  g_previewMillis = 900000;  // past the screen's 5 s battery cache
  bar("ready", kGreen);
  ui::showHome(home);
  save("25_home_low_battery");
  M5.Power.level = 82;
  g_previewMillis = 1000000;

  // Tap to talk: the record button, the stop button, under an answer.
  ui::setTapButton(ui::TapButton::Talk);
  bar("ready", kGreen);
  ui::showHome(home);
  save("26_tap_home");
  ui::setTapButton(ui::TapButton::Stop);
  bar("listening", kRed);
  for (int l : lv) ui::showListening(l, 7300);
  save("27_tap_listening");
  ui::setTapButton(ui::TapButton::Talk);
  bar("ready", kGreen);
  ui::setReply("Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at "
               "16:00. I moved the dentist to Friday at 15:00.");
  ui::showReply();
  save("28_tap_reply");
  ui::setTapButton(ui::TapButton::Pause);  // the answer's sound plays: pause it
  bar("speaking", kCyan);
  gPlaying = true;
  gLevel = 70;
  ui::showReply();
  ui::showSpeaking(70);
  save("28b_tap_reply_pause");
  gPlaying = false;
  ui::setTapButton(ui::TapButton::Play);  // held: play on
  bar("paused", kText);
  ui::showReply();
  save("28c_tap_reply_play");
  ui::setTapButton(ui::TapButton::Talk);
  ui::setReply("");
  ui::showFace(0, face::Pose::Idle, 0);
  save("29_tap_icon");
  ui::setTapButton(ui::TapButton::Off);

  bar("ready", kGreen);
  ui::showAsk("Saved voice notes", "3 voice notes recorded offline (1:24). Send them to the agent?", "send", "later");
  save("30_ask_notes");

  // A background picture behind every screen (flight_deck/device_backgrounds.py).
  auto bgJpg = readFile(gOut + "/background.jpg");
  if (!bgJpg.empty()) {
    ui::setBackground(bgJpg.data(), bgJpg.size());
    bar("ready", kGreen);
    ui::showHome(home);
    save("31_bg_home");
    bar("12s · 3 steps", kYellow);
    ui::showWorking("What's on my calendar tomorrow, and can you move the dentist to Friday afternoon?",
                    {"Reading the calendar", "google_calendar · list events", "Looking for a free slot on Friday"},
                    "Checking Friday afternoon", 12400, 3);
    save("32_bg_working");
    bar("speaking", kCyan);
    gPlaying = true;
    ui::setReply("Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at "
                 "16:00. I moved the dentist to Friday at 15:00.");
    ui::showReply();
    ui::showSpeaking(70);
    save("33_bg_reply_speaking");
    gPlaying = false;
    bar("menu", kText);
    ui::showPicker(items, 2, -1);
    save("34_bg_menu");
    ui::setReply("");
    bar("ready", kGreen);
    ui::showFace(0, face::Pose::Idle, 0);
    save("35_bg_icon");
    ui::showMessage("Answer lost", "The connection dropped while the agent was working. Ask again.", kOrange);
    save("36_bg_notice");
    ui::showBackdrop();  // a double tap on home: the picture alone
    save("37_bg_picture_only");
    ui::setBackground(nullptr, 0);
  }
  return 0;
}
