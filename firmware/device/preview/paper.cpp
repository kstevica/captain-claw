// Renders the PaperColor screens (the real ui_paper.cpp, its six-ink
// conversion and the characters) on a Mac, one PPM per screen, so the paper
// UI can be designed before — and without — the device. preview/render.py
// paper turns them into PNGs with the panel's muted inks.
//
//   pio run -e preview_paper
//   uv run --no-project --with pillow python preview/render.py paper
#include <M5Unified.h>

#include <string>
#include <vector>

#include "../src/fonts.h"
#include "../src/telemetry.h"
#include "../src/ui.h"

uint32_t g_previewMillis = 0;
PreviewM5 M5;

// ── stand-ins for the device-only parts ─────────────────────────────────

static bool gClimate = true;

namespace telemetry {
bool climate(float* t, float* rh) {
  if (!gClimate) return false;
  *t = 22.6f;
  *rh = 44.0f;
  return true;
}
bool clockSet() { return true; }
}  // namespace telemetry

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
  ui::flush();  // whatever was asked last, drawn now (no settling on the Mac)
  const int w = M5.Display.width(), h = M5.Display.height();
  std::vector<uint8_t> rgb(w * h * 3);
  M5.Display.readRectRGB(0, 0, w, h, rgb.data());
  std::string path = gOut + "/" + name + ".ppm";
  FILE* f = fopen(path.c_str(), "wb");
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  fwrite(rgb.data(), 1, rgb.size(), f);
  fclose(f);
  printf("%s\n", path.c_str());
  ui::drawNow();  // the gallery shows every screen as if the user asked for it
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

static void picture(const char* file) {
  size_t len = 0;
  uint8_t* jpg = readFile(gOut + "/" + file, &len);
  if (jpg) ui::storeImage(jpg, len);
}

// ── the screens ────────────────────────────────────────────────────────

static constexpr uint16_t kGreen = ui::rgb(40, 200, 90), kYellow = ui::rgb(250, 200, 40);
static constexpr uint16_t kRed = ui::rgb(230, 50, 50), kCyan = ui::rgb(60, 210, 240);
static constexpr uint16_t kOrange = ui::rgb(255, 150, 40), kText = ui::rgb(230, 230, 230);

static void bar(ui::Health wifi = ui::Up, ui::Health agent = ui::Up) {
  ui::setHealth(wifi, agent);
  ui::setBar("Captain Claw", "ready", kGreen);
}

static ui::Home home(bool ready) {
  ui::Home h{ready, ready ? "All good" : "Getting ready", ready ? kGreen : kYellow,
             ready ? "Top button to talk" : "Waiting for the agent", {}};
  return h;
}

static const char* kAsked = "What's on my calendar tomorrow, and can you move the dentist to Friday afternoon?";
static const char* kAnswer =
    "Tomorrow you have three meetings: stand-up at 9:30, lunch with Ana at 12:00 and the dentist at 16:00. "
    "I moved the dentist to Friday at 15:00 — the clinic confirmed it.";
static const char* kLong =
    "Sutra imaš tri sastanka: dnevni sastanak u 9:30, ručak s Anom u 12:00 i zubara u 16:00. Zubara sam "
    "premjestio na petak u 15:00 — Đuro je potvrdio termin. Za ručak sam rezervirao stol u Čakovcu, a šećer "
    "i kava su već naručeni.\n\nAko želiš, mogu poslati podsjetnik svima sat vremena prije, ili dodati "
    "bilješke s prošlog sastanka u pozivnicu. Prognoza za petak je sunčano, 19 °C, pa je terasa slobodna "
    "za kavu nakon zubara. Ponedjeljak je mirniji: samo pregled projekta u 11:00 i poziv s Markom u 15:30, "
    "za koji sam već pripremio sažetak zadnjih brojki i tri pitanja koja je postavio prošli tjedan.";

int main(int argc, char** argv) {
  gOut = argc > 1 ? argv[1] : "preview/out/paper";
  gRoot = argc > 2 ? argv[2] : ".";
  setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3", 1);
  tzset();

  M5.Display.setColorDepth(16);
  M5.Display.createSprite(400, 600);
  ui::begin();
  ui::setTextSize(0);
  g_previewMillis = 123456;

  bar();
  ui::showHome(home(true));
  save("01_home_welcome");

  bar(ui::Up, ui::Partial);
  ui::showHome(home(false));
  save("02_home_waiting");

  ui::setQuestion("How warm is it in here?");
  ui::setReply("It's 22.6 °C with 44% humidity — comfortable. The battery is at 82%.");
  ui::showReply();
  save("04_reply_short");

  ui::setQuestion("Draw me a sunset over the mountains");
  ui::setReply("Here's a sunset over the Velebit: warm sky, the sea below and the first stars coming out.");
  picture("sample.jpg");
  ui::showReply();
  save("05_reply_picture");

  ui::showStoredImage();
  save("06_picture_full");

  ui::setQuestion(kAsked);
  ui::setReply(kLong);
  ui::showReply();
  save("07_reply_long_p1");
  ui::nextPage();
  save("08_reply_long_p2");

  ui::setQuestion(kAsked);
  ui::setReply(kAnswer);
  ui::showHome(home(true));
  save("09_home_last_answer");

  ui::setQuestion("Draw me a sunset over the mountains");
  ui::setReply("Here's a sunset over the Velebit: warm sky, the sea below and the first stars coming out.");
  picture("sample.jpg");
  ui::pin(true);
  ui::showHome(home(true));
  save("10_home_pinned");


  std::vector<ui::Row> rows = {
      {"Wi-Fi", "Harbour-5G  -54 dBm", kGreen},
      {"IP", "192.168.1.48", kText},
      {"Deck", "deck.example.com  38 ms", kGreen},
      {"Agent", "Captain Claw", kGreen},
      {"Battery", "82%  3.98 V", kGreen},
      {"Room", "22.6 °C  44% humidity", kText},
      {"Voice", "female  vol 75%  +6 dB", kText},
      {"Usage", "7 turns  last 2m ago  up 1h 12m", kText},
      {"Device", "paper-a1b2c3  fw 0.18.1  7480k free", kText},
  };
  ui::showPanel("", kText, rows, 0);
  save("14_status");

  ui::showMessage("Hmm", "The agent didn't answer in time. Try again in a moment.", kOrange);
  save("15_notice");

  ui::showAsk("Saved voice notes", "3 voice notes recorded offline (1:24). Send them to the agent?", "send", "later");
  save("15b_ask_notes");

  ui::showLink("Options · 0.18.1", "http://192.168.1.48/?k=3fa91c07be22",
               "Scan it with your phone (same Wi-Fi): every setting is there.");
  save("16_options");

  ui::showSetup("First-time setup", "Captain-PaperColor-b2c3", "192.168.4.1");
  save("17_setup");

  ui::pin(false);
  ui::showSleeping("Top button to talk");
  save("19_sleeping");

  ui::showMessage("Battery empty", "Charge it over USB-C, then press the power button.", kRed);
  save("20_battery_empty");

  ui::setRotation(1);
  ui::setQuestion("Draw me a sunset over the mountains");
  ui::setReply("Here's a sunset over the Velebit: warm sky, the sea below and the first stars coming out.");
  picture("sample.jpg");
  ui::showReply();
  save("21_landscape_reply");
  ui::showHome(home(true));
  save("22_landscape_home");
  return 0;
}
