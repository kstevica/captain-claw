#include "settings.h"

#include <Preferences.h>
#include <WiFi.h>

#include "board.h"

Settings settings;

static const char* kNs = "claw";

void Settings::load() {
  Preferences p;
  p.begin(kNs, true);
  netCount = min<int>(p.getInt("n", 0), kMaxNetworks);
  for (int i = 0; i < netCount; ++i) {
    nets[i].ssid = p.getString(("s" + String(i)).c_str(), "");
    nets[i].pass = p.getString(("p" + String(i)).c_str(), "");
  }
  server = p.getString("server", "");
  token = p.getString("token", "");
  agent = p.getString("agent", "");
  lang = p.getString("lang", "");
  speak = p.getBool("speak", true);
  volume = p.getUChar("vol", 191);
  boostDb = p.getUChar("boost", 6);
  screenOffS = p.getUChar("scroff", 15);
  powerOffS = p.getUChar("pwroff", 60);
  if (screenOffS != 5 && screenOffS != 10 && screenOffS != 15) screenOffS = 15;
  if (powerOffS != 15 && powerOffS != 30 && powerOffS != 60 && powerOffS != 180) powerOffS = 60;
  femaleVoice = p.getBool("female", false);
  iconMode = p.getBool("icons", false);
  character = p.getUChar("char", 0);
  if (!p.isKey("charv")) {  // v1 numbering: 0 smiley, 1 bot, 2 captain, 3 bird
    character = character == 1 ? 1 : character == 3 ? 2 : 0;
  }
  cueBeep = p.getBool("cuebeep", true);
  talkSounds = p.getBool("talksnd", true);
  cueFlash = p.getBool("cueflash", true);
  cueBuzz = p.getBool("cuebuzz", true);
  cueLeadS = constrain(p.getUChar("cuelead", 1), 1, 2);
  tapTalk = board::kHasTouch && p.getBool("taptalk", board::kTapTalkDefault);
  talks = p.getUShort("talks", 0);
  textSize = min<uint8_t>(p.getUChar("tsize", 0), 2);
  autoUpdate = p.getBool("autoupd", board::kPaperB);  // the B paper: on unless turned off on the phone page
  noteMode = p.getBool("notemode", false);
  lastFw = p.getString("lastfw", "");
  tz = p.getString("tz", kDefaultTz);
  if (tz.isEmpty()) tz = kDefaultTz;
  rotation = p.getUChar("rot", 0) & 3;
  ledLevel = min<uint8_t>(p.getUChar("leds", 1), 2);
  keyClicks = p.getBool("clicks", true);
  powerOffMin = p.getUChar("pwroffm", 3);
  if (!powerOffMinValid(powerOffMin)) powerOffMin = 3;
  asleepRefreshMin = p.getUChar("slpref", 60);
  if (asleepRefreshMin != 0 && asleepRefreshMin != 30 && asleepRefreshMin != 60 && asleepRefreshMin != 180) {
    asleepRefreshMin = 60;
  }
  speakMenu = p.getBool("saymenu", true);
  if (board::kEpaper) speakMenu = true;  // the paper's menu is never drawn: it's read aloud
  if (board::kEpaper) iconMode = false;  // a character can't move on paper
  p.end();
}

uint32_t Settings::powerOffMs() const {
  if (board::kEpaper) return powerOffMin ? powerOffMin * 60000UL : UINT32_MAX;  // 0: never
  return powerOffS * 1000UL;
}

bool powerOffMinValid(int minutes) {
  for (uint8_t m : kPowerOffMins)
    if (m == minutes) return true;
  return false;
}

String powerOffText(uint8_t minutes) {
  if (!minutes) return "never";
  if (minutes < 60) return String(minutes) + " min";
  return minutes == 60 ? String("1 hour") : String(minutes / 60) + " hours";
}

void Settings::save() const {
  Preferences p;
  p.begin(kNs, false);
  p.putInt("n", netCount);
  for (int i = 0; i < kMaxNetworks; ++i) {
    String sk = "s" + String(i), pk = "p" + String(i);
    if (i < netCount) {
      p.putString(sk.c_str(), nets[i].ssid);
      p.putString(pk.c_str(), nets[i].pass);
    } else {
      p.remove(sk.c_str());
      p.remove(pk.c_str());
    }
  }
  p.putString("server", server);
  p.putString("token", token);
  p.putString("agent", agent);
  p.putString("lang", lang);
  p.putBool("speak", speak);
  p.putUChar("vol", volume);
  p.putUChar("boost", boostDb);
  p.putUChar("scroff", screenOffS);
  p.putUChar("pwroff", powerOffS);
  p.putBool("female", femaleVoice);
  p.putBool("icons", iconMode);
  p.putUChar("char", character);
  p.putUChar("charv", 2);
  p.putBool("cuebeep", cueBeep);
  p.putBool("talksnd", talkSounds);
  p.putBool("cueflash", cueFlash);
  p.putBool("cuebuzz", cueBuzz);
  p.putBool("taptalk", tapTalk);
  p.putUShort("talks", talks);
  p.putUChar("cuelead", cueLeadS);
  p.putUChar("tsize", textSize);
  p.putBool("autoupd", autoUpdate);
  p.putBool("notemode", noteMode);
  p.putString("lastfw", lastFw);
  p.putString("tz", tz);
  p.putUChar("rot", rotation);
  p.putUChar("leds", ledLevel);
  p.putBool("clicks", keyClicks);
  p.putUChar("pwroffm", powerOffMin);
  p.putUChar("slpref", asleepRefreshMin);
  p.putBool("saymenu", speakMenu);
  p.end();
}

bool Settings::addNetwork(const String& ssid, const String& pass) {
  if (ssid.isEmpty()) return false;
  for (int i = 0; i < netCount; ++i) {
    if (nets[i].ssid == ssid) {
      if (pass.length()) nets[i].pass = pass;  // blank keeps the saved password
      WifiNet hit = nets[i];
      for (int j = i; j > 0; --j) nets[j] = nets[j - 1];
      nets[0] = hit;
      return true;
    }
  }
  if (netCount >= kMaxNetworks) netCount = kMaxNetworks - 1;  // drop the oldest
  for (int j = netCount; j > 0; --j) nets[j] = nets[j - 1];
  nets[0] = {ssid, pass};
  ++netCount;
  return true;
}

void Settings::removeNetwork(int index) {
  if (index < 0 || index >= netCount) return;
  for (int j = index; j < netCount - 1; ++j) nets[j] = nets[j + 1];
  --netCount;
  nets[netCount] = {};
}

String deviceId() {
  uint8_t mac[6];
  WiFi.macAddress(mac);
  char buf[16];
  snprintf(buf, sizeof(buf), "%s-%02x%02x%02x", board::kIdPrefix, mac[3], mac[4], mac[5]);
  return String(buf);
}
