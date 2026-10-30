#include "notes.h"

#include <FS.h>
#include <LittleFS.h>
#include <M5Unified.h>
#if defined(BOARD_EPAPER154)
#include <SD_MMC.h>  // its slot has its own pins (one data line)
#define CARD SD_MMC
#else
#include <SD.h>
#define CARD SD
#endif
#include <SPI.h>

#include <freertos/stream_buffer.h>

#include <algorithm>

#include "adpcm.h"
#include "board.h"

namespace notes {

// A note is "<id>.adp": a 16-byte header ("CCA1", rate, samples, 0), then
// two 4-bit ADPCM codes a byte, low nibble first. Older firmware kept 16 kHz
// PCM WAVs ("<id>.wav"); those are read as they are.
static constexpr const char* kDir = "/captain-notes";
static constexpr const char* kOldDir = "/claw-notes";  // before 1.0.15: its notes move to kDir
static constexpr size_t kHead = 16;
static constexpr size_t kWavHead = 44;
static constexpr size_t kKeepFree = 128 * 1024;  // room left for the paper's answer cards
static bool mounted = false;
static fs::FS* store = nullptr;  // the card, or the flash
static bool card = false;
static bool flashUp = false;     // the flash (LittleFS) is mounted
static int cachedCount = 0;
static uint32_t cachedMs = 0;
static uint64_t cachedFree = 0;
static File reader;
static bool readerPacked = false;
static size_t readerLeft = 0;  // samples still to come
static adpcm::State readerState;

static uint64_t freeNow() {
  if (!mounted) return 0;
  return card ? CARD.totalBytes() - CARD.usedBytes() : LittleFS.totalBytes() - LittleFS.usedBytes();
}

static String pathOf(const String& id, const char* ext) { return String(kDir) + "/" + id + ext; }

// What a recording at `rate` is kept at: speech needs no more than 8 kHz.
static uint32_t keptRate(uint32_t rate) { return rate >= 16000 ? rate / 2 : rate; }

static void put32(uint8_t* p, uint32_t v) {
  p[0] = v, p[1] = v >> 8, p[2] = v >> 16, p[3] = v >> 24;
}

static uint32_t get32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }

struct Entry {
  String id;
  size_t size;
  bool packed;  // .adp (else an old .wav)
};

static Note noteOf(const Entry& e) {
  Note n;
  n.id = e.id;
  n.when = (time_t)strtoul(e.id.c_str(), nullptr, 10);
  if (e.packed) {
    n.rate = keptRate(16000);
    n.samples = (e.size - kHead) * 2;
  } else {
    n.rate = 16000;
    n.samples = (e.size - kWavHead) / 2;
  }
  return n;
}

// A note's file as the device names it — "<10-digit epoch>-<8 hex>" and its
// kind (.adp; .wav from older firmware). Nothing else in the folder — a file
// someone put on the card — is ever read, sent or removed.
static bool deviceNote(const String& name, const char* ext) {
  if (name.length() != 23 || name[10] != '-' || !name.endsWith(ext)) return false;
  for (int i = 0; i < 19; ++i) {
    if (i != 10 && !(i < 10 ? isdigit((unsigned char)name[i]) : isxdigit((unsigned char)name[i]))) return false;
  }
  return true;
}

// Each kept note; broken leftovers (a write cut short) are removed.
static std::vector<Entry> scan() {
  std::vector<Entry> out;
  std::vector<String> broken;
  File dir = mounted ? store->open(kDir) : File();
  if (!dir || !dir.isDirectory()) return out;
  for (File f = dir.openNextFile(); f; f = dir.openNextFile()) {
    String name = f.name();
    int slash = name.lastIndexOf('/');
    if (slash >= 0) name = name.substring(slash + 1);
    size_t size = f.size();
    f.close();
    bool packed = deviceNote(name, ".adp");
    if (!packed && !deviceNote(name, ".wav")) continue;
    if (size <= (packed ? kHead : kWavHead)) broken.push_back(name);
    else out.push_back({name.substring(0, name.length() - 4), size, packed});
  }
  dir.close();
  for (auto& name : broken) store->remove(String(kDir) + "/" + name);
  std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.id < b.id; });
  return out;
}

static void recount() {
  cachedCount = 0;
  cachedMs = 0;
  for (auto& e : scan()) {
    Note n = noteOf(e);
    ++cachedCount;
    cachedMs += (uint64_t)n.samples * 1000 / n.rate;
  }
  cachedFree = freeNow();
}

// Notes kept in flash before a card was put in: onto the card, then gone
// from the flash (a copy that fails stays where it was).
static void moveToCard() {
  File dir = LittleFS.open(kDir);
  if (!dir || !dir.isDirectory()) return;
  std::vector<String> names;
  for (File f = dir.openNextFile(); f; f = dir.openNextFile()) {
    String name = f.name();
    f.close();
    int slash = name.lastIndexOf('/');
    if (slash >= 0) name = name.substring(slash + 1);
    if (deviceNote(name, ".adp") || deviceNote(name, ".wav")) names.push_back(name);
  }
  dir.close();
  static uint8_t buf[4096];
  int moved = 0;
  for (auto& name : names) {
    const String path = String(kDir) + "/" + name;
    File in = LittleFS.open(path, FILE_READ);
    File out = in ? CARD.open(path, FILE_WRITE) : File();
    bool ok = in && out;
    for (size_t got; ok && (got = in.read(buf, sizeof(buf))) > 0;) ok = out.write(buf, got) == got;
    if (in) in.close();
    if (out) out.close();
    if (ok) {
      LittleFS.remove(path);
      ++moved;
    } else {
      CARD.remove(path);
    }
  }
  if (moved) Serial.printf("[claw] offline notes: %d moved from flash to the card\n", moved);
}

// Notes in the folder's old name ("claw-notes") move into kDir, on the same
// store (a rename); the old folder goes once it's empty.
static void moveOldNotes(fs::FS& fs) {
  if (!fs.exists(kOldDir)) return;
  if (!fs.exists(kDir)) fs.mkdir(kDir);
  File dir = fs.open(kOldDir);
  std::vector<String> names;
  bool other = false;  // something that isn't a note: the old folder stays
  for (File f = dir ? dir.openNextFile() : File(); f; f = dir.openNextFile()) {
    String name = f.name();
    f.close();
    int slash = name.lastIndexOf('/');
    if (slash >= 0) name = name.substring(slash + 1);
    if (deviceNote(name, ".adp") || deviceNote(name, ".wav")) names.push_back(name);
    else other = true;
  }
  if (dir) dir.close();
  for (auto& name : names) fs.rename(String(kOldDir) + "/" + name, String(kDir) + "/" + name);
  if (!other) fs.rmdir(kOldDir);
  if (!names.empty()) Serial.printf("[claw] offline notes: %d moved from %s to %s\n", (int)names.size(), kOldDir, kDir);
}

// The card mounted (FAT32 or FAT16: this SD library has no exFAT). The
// panel's driver set the shared bus up as Arduino's SPI (and locks it for
// each of its transfers): the card only needs its chip select.
static bool mountCard() {
#if defined(BOARD_EPAPER154)
  CARD.end();  // (after a try that failed, or a card that went)
  CARD.setPins(39, 41, 40);  // CLK, CMD, D0
  return CARD.begin("/sd", true, false, SDMMC_FREQ_DEFAULT, 2);
#else
  const int cs = M5.getPin(m5::pin_name_t::sd_spi_ss);
  if (cs < 0 || cs >= 255) return false;
  CARD.end();  // (after a try that failed, or a card that went)
  return CARD.begin(cs, SPI, 10000000, "/sd", 2);
#endif
}

static bool cardThere() {
  File d = CARD.open(kDir);
  const bool there = d && d.isDirectory();
  if (d) d.close();
  return there;
}

// The card is in: notes go there, and those in flash move over.
static void useCard() {
  store = &CARD;
  card = mounted = true;
  moveOldNotes(CARD);
  if (!CARD.exists(kDir)) CARD.mkdir(kDir);
  if (flashUp && LittleFS.exists(kDir)) moveToCard();
  recount();
}

// The card is gone: back to the flash.
static void useFlash() {
  closeRead();
  if (card) CARD.end();
  card = false;
  store = flashUp ? &LittleFS : nullptr;
  mounted = store != nullptr;
  if (mounted && !store->exists(kDir)) store->mkdir(kDir);
  recount();
}

bool begin() {
  if (!board::kOfflineNotes) return false;
  flashUp = LittleFS.begin(true, "/littlefs", 10, "spiffs");
  if (flashUp) moveOldNotes(LittleFS);
  if (board::kHasSdCard && mountCard()) useCard();  // no card: the flash
  else useFlash();
  if (!mounted) return false;
  Serial.printf("[claw] offline notes on %s: %d waiting, %llu KB free (~%lu min)\n", card ? "the card" : "flash",
                cachedCount, (unsigned long long)(cachedFree / 1024), (unsigned long)(room(16000) / 16000 / 60));
  return true;
}

bool ready() { return mounted; }

bool onCard() { return card; }

static bool noteOn = false;  // a note is being written as it's recorded

bool poll() {
  if (!board::kOfflineNotes || !board::kHasSdCard || noteOn) return false;
  if (card) {
    if (cardThere()) return false;
    useFlash();
    Serial.println("[claw] microSD card taken out: offline notes in flash");
    return true;
  }
  if (!mountCard()) return false;
  useCard();
  Serial.printf("[claw] microSD card in: %d offline notes on it\n", cachedCount);
  return true;
}

uint64_t freeBytes() { return cachedFree; }

size_t room(uint32_t rate) {
  const uint64_t perSecond = keptRate(rate) / 2;  // bytes: two codes a byte
  if (!rate || cachedFree < kKeepFree + kHead + perSecond) return 0;  // under a second
  return (size_t)min<uint64_t>((cachedFree - kKeepFree - kHead) * rate / perSecond, 0x7FFFFFFF);
}

static bool write(const int16_t* pcm, size_t samples, uint32_t rate, time_t when);

// Kept on the card — or, when the card was taken out since it was last looked
// for, in flash.
bool save(const int16_t* pcm, size_t samples, uint32_t rate, time_t when) {
  if (write(pcm, samples, rate, when)) return true;
  if (!card || cardThere()) return false;  // (a full card stays the card)
  useFlash();
  return write(pcm, samples, rate, when);
}

static bool write(const int16_t* pcm, size_t samples, uint32_t rate, time_t when) {
  if (!mounted || !samples || samples > room(rate)) return false;
  const bool halve = keptRate(rate) != rate;
  const size_t kept = halve ? samples / 2 : samples;
  char path[64];
  snprintf(path, sizeof(path), "%s/%010lu-%08lx.adp", kDir, (unsigned long)(when > 0 ? when : 0),
           (unsigned long)esp_random());
  File f = store->open(path, FILE_WRITE);
  if (!f) return false;
  uint8_t h[kHead] = {'C', 'C', 'A', '1'};
  put32(h + 4, keptRate(rate));
  put32(h + 8, kept);
  bool ok = f.write(h, kHead) == kHead;
  static uint8_t piece[4096];  // a flash sector a write: flash writes stall the cache
  size_t fill = 0;
  adpcm::State st;
  auto at = [&](size_t i) { return halve ? adpcm::halved(pcm, samples, i) : pcm[i]; };
  for (size_t i = 0; ok && i < kept; i += 2) {
    uint8_t b = adpcm::encode(st, at(i));
    if (i + 1 < kept) b |= adpcm::encode(st, at(i + 1)) << 4;
    piece[fill++] = b;
    if (fill == sizeof(piece)) {
      ok = f.write(piece, fill) == fill;
      fill = 0;
    }
  }
  if (ok && fill) ok = f.write(piece, fill) == fill;
  f.close();
  if (!ok) {
    store->remove(path);
    cachedFree = freeNow();
    return false;
  }
  ++cachedCount;
  cachedMs += (uint64_t)kept * 1000 / keptRate(rate);
  cachedFree = freeNow();
  return true;
}

std::vector<Note> list(size_t max) {
  std::vector<Note> out;
  for (auto& e : scan()) {
    if (out.size() >= max) break;
    out.push_back(noteOf(e));
  }
  return out;
}

int count() { return cachedCount; }

uint32_t seconds() { return cachedMs / 1000; }

bool openRead(const Note& note, uint32_t* rate) {
  closeRead();
  if (!mounted) return false;
  readerPacked = store->exists(pathOf(note.id, ".adp"));
  reader = store->open(pathOf(note.id, readerPacked ? ".adp" : ".wav"), FILE_READ);
  uint8_t h[kWavHead];
  const size_t head = readerPacked ? kHead : kWavHead;
  if (!reader || reader.read(h, head) != head || memcmp(h, readerPacked ? "CCA1" : "RIFF", 4) != 0) {
    closeRead();
    return false;
  }
  *rate = get32(h + (readerPacked ? 4 : 24));
  if (*rate < 8000 || *rate > 48000) *rate = readerPacked ? 8000 : 16000;
  readerLeft = readerPacked ? get32(h + 8) : 0;
  // A note written as recorded whose end never came (the power went): its
  // header still says 0 — all that's there, then.
  if (readerPacked && !readerLeft) readerLeft = (reader.size() - kHead) * 2;
  readerState = adpcm::State();
  return true;
}

size_t read(uint8_t* buf, size_t max) {
  if (!reader) return 0;
  if (!readerPacked) return reader.read(buf, max & ~(size_t)1);
  // Decoded as it's read: whole bytes (two samples) but for a note's odd last one.
  const size_t want = min(max / 2 & ~(size_t)1, readerLeft);
  uint8_t packed[512];
  size_t done = 0;
  while (done < want) {
    size_t got = reader.read(packed, min((want - done + 1) / 2, sizeof(packed)));
    if (!got) break;
    for (size_t i = 0; i < got && done < want; ++i) {
      for (int half = 0; half < 2 && done < want; ++half, ++done) {
        int16_t s = adpcm::decode(readerState, half ? packed[i] >> 4 : packed[i] & 15);
        buf[2 * done] = (uint8_t)s;
        buf[2 * done + 1] = (uint8_t)(s >> 8);
      }
    }
  }
  readerLeft -= done;
  return done * 2;
}

void closeRead() {
  if (reader) reader.close();
}

void remove(const String& id) {
  if (!mounted) return;
  if (!store->remove(pathOf(id, ".adp"))) store->remove(pathOf(id, ".wav"));
  recount();
}

void quarantine(const String& id) {
  if (!mounted) return;
  closeRead();
  if (!store->rename(pathOf(id, ".adp"), pathOf(id, ".bad"))) store->rename(pathOf(id, ".wav"), pathOf(id, ".bad"));
  recount();
}

// ── a note written as it's recorded ──────────────────────────────────────
//
// Each mic block is halved (16 → 8 kHz) and coded here, on the main loop
// (adpcm::StreamCoder), into a stream buffer in PSRAM; a writer task drains
// it to the file. The mic has only two blocks queued (64 ms), and a flash
// erase or a card write can take longer than that.

static constexpr size_t kStreamBytes = 64 * 1024;  // ~16 s of coded voice
static StreamBufferHandle_t stream = nullptr;
static File noteFile;
static String notePath;
static uint32_t noteRate = 16000;
static size_t noteMaxBytes = 0;          // the room it had when it began
static size_t noteBytes = 0;             // coded bytes handed to the writer
static volatile size_t written = 0;      // …and put in the file by it
static volatile bool writeFailed = false;
static adpcm::StreamCoder coder;

static void writerLoop(void*) {
  static uint8_t buf[4096];  // a flash sector a write
  for (;;) {
    const size_t n = xStreamBufferReceive(stream, buf, sizeof(buf), portMAX_DELAY);
    if (!n) continue;
    if (!noteFile || noteFile.write(buf, n) != n) writeFailed = true;
    written += n;
  }
}

// Coded bytes to the writer.
static void toWriter(const uint8_t* bytes, size_t n) {
  const size_t sent = xStreamBufferSend(stream, bytes, n, pdMS_TO_TICKS(20));
  if (sent < n) writeFailed = true;  // the writer is 16 s behind: the store is stuck
  noteBytes += sent;
}

bool beginNote(uint32_t rate, time_t when) {
  if (!mounted || noteOn) return false;
  if (!stream) {  // the first note: its buffer and its writer
    static StaticStreamBuffer_t holder;
    uint8_t* mem = (uint8_t*)ps_malloc(kStreamBytes + 1);
    if (!mem) return false;
    stream = xStreamBufferCreateStatic(kStreamBytes, 1, mem, &holder);
    if (!stream || xTaskCreatePinnedToCore(writerLoop, "notes", 4096, nullptr, 1, nullptr, 0) != pdPASS) {
      stream = nullptr;
      return false;
    }
  }
  const uint64_t perSecond = keptRate(rate) / 2;
  if (cachedFree < kKeepFree + kHead + perSecond * 3) return false;  // under 3 s of room
  noteMaxBytes = (size_t)min<uint64_t>(cachedFree - kKeepFree - kHead, 0x7FFFFFFF);
  char path[64];
  snprintf(path, sizeof(path), "%s/%010lu-%08lx.adp", kDir, (unsigned long)(when > 0 ? when : 0),
           (unsigned long)esp_random());
  noteFile = store->open(path, FILE_WRITE);
  if (!noteFile) return false;
  uint8_t h[kHead] = {'C', 'C', 'A', '1'};
  put32(h + 4, keptRate(rate));  // its length goes in at the end
  if (noteFile.write(h, kHead) != kHead) {
    noteFile.close();
    store->remove(path);
    return false;
  }
  notePath = path;
  noteRate = rate;
  coder = adpcm::StreamCoder();
  coder.halve = keptRate(rate) != rate;
  noteBytes = 0;
  written = 0;
  writeFailed = false;
  xStreamBufferReset(stream);
  noteOn = true;
  return true;
}

void addNote(const int16_t* pcm, size_t samples) {
  if (noteOn) coder.push(pcm, samples, toWriter);
}

bool noteFull() { return noteOn && (writeFailed || noteBytes + 4096 >= noteMaxBytes); }

// The writer caught up with what it was given (or 10 s went by).
static void drained() {
  for (uint32_t t0 = millis(); written < noteBytes && millis() - t0 < 10000;) delay(5);
}

bool endNote() {
  if (!noteOn) return false;
  coder.finish(toWriter);
  drained();
  noteOn = false;
  const uint64_t kept = coder.samples;
  bool ok = !writeFailed && written >= noteBytes && kept >= keptRate(noteRate);  // a second at least
  if (ok) {
    uint8_t n[4];
    put32(n, (uint32_t)kept);
    ok = noteFile.seek(8) && noteFile.write(n, 4) == 4;
  }
  noteFile.close();
  if (!ok) {
    store->remove(notePath);
    cachedFree = freeNow();
    return false;
  }
  ++cachedCount;
  cachedMs += kept * 1000 / keptRate(noteRate);
  cachedFree = freeNow();
  return true;
}

void dropNote() {
  if (!noteOn) return;
  drained();
  noteOn = false;
  noteFile.close();
  store->remove(notePath);
  cachedFree = freeNow();
}

}  // namespace notes
