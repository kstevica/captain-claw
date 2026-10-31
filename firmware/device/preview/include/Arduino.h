// Just enough of Arduino for the screen code (ui_round.cpp, face*.cpp) to
// build on a Mac for the preview renderer. Not used on the devices.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using std::max;
using std::min;

#define constrain(amt, low, high) ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt)))

// The preview sets the clock so every frame renders the same.
extern uint32_t g_previewMillis;
inline uint32_t millis() { return g_previewMillis; }
inline void delay(uint32_t) {}
inline void* ps_malloc(size_t n) { return malloc(n); }
inline void* ps_realloc(void* p, size_t n) { return realloc(p, n); }

class String {
 public:
  String() = default;
  String(const char* s) : s_(s ? s : "") {}
  String(const std::string& s) : s_(s) {}
  explicit String(char c) : s_(1, c) {}
  String(int v) : s_(std::to_string(v)) {}
  String(unsigned v) : s_(std::to_string(v)) {}
  String(long v) : s_(std::to_string(v)) {}
  String(unsigned long v) : s_(std::to_string(v)) {}

  unsigned length() const { return (unsigned)s_.size(); }
  bool isEmpty() const { return s_.empty(); }
  const char* c_str() const { return s_.c_str(); }
  char operator[](unsigned i) const { return i < s_.size() ? s_[i] : 0; }
  char charAt(unsigned i) const { return (*this)[i]; }
  void setCharAt(unsigned i, char c) { if (i < s_.size()) s_[i] = c; }
  String substring(unsigned from) const { return from >= s_.size() ? String() : String(s_.substr(from)); }
  String substring(unsigned from, unsigned to) const {
    if (from > to) std::swap(from, to);
    if (from >= s_.size()) return String();
    return String(s_.substr(from, std::min<size_t>(to, s_.size()) - from));
  }
  int indexOf(char c, unsigned from = 0) const {
    size_t p = s_.find(c, from);
    return p == std::string::npos ? -1 : (int)p;
  }
  int indexOf(const String& t, unsigned from = 0) const {
    size_t p = s_.find(t.s_, from);
    return p == std::string::npos ? -1 : (int)p;
  }
  int lastIndexOf(char c) const {
    size_t p = s_.rfind(c);
    return p == std::string::npos ? -1 : (int)p;
  }
  int lastIndexOf(char c, unsigned from) const {
    size_t p = s_.rfind(c, from);
    return p == std::string::npos ? -1 : (int)p;
  }
  void reserve(unsigned n) { s_.reserve(n); }
  void remove(unsigned index) { if (index < s_.size()) s_.erase(index); }
  void toLowerCase() { for (char& c : s_) c = (char)tolower((unsigned char)c); }
  void remove(unsigned index, unsigned count) { if (index < s_.size()) s_.erase(index, count); }
  bool startsWith(const String& p) const { return s_.rfind(p.s_, 0) == 0; }
  bool endsWith(const String& p) const {
    return s_.size() >= p.s_.size() && s_.compare(s_.size() - p.s_.size(), p.s_.size(), p.s_) == 0;
  }
  void trim() {
    size_t a = s_.find_first_not_of(" \t\r\n"), b = s_.find_last_not_of(" \t\r\n");
    s_ = a == std::string::npos ? "" : s_.substr(a, b - a + 1);
  }
  long toInt() const { return atol(s_.c_str()); }
  void replace(const String& from, const String& to) {
    if (from.s_.empty()) return;
    for (size_t p = s_.find(from.s_); p != std::string::npos; p = s_.find(from.s_, p + to.s_.size())) {
      s_.replace(p, from.s_.size(), to.s_);
    }
  }

  String& operator+=(const String& o) { s_ += o.s_; return *this; }
  String& operator+=(const char* o) { s_ += o; return *this; }
  String& operator+=(char c) { s_ += c; return *this; }
  friend String operator+(const String& a, const String& b) { return String(a.s_ + b.s_); }
  friend String operator+(const String& a, const char* b) { return String(a.s_ + b); }
  friend String operator+(const char* a, const String& b) { return String(std::string(a) + b.s_); }
  friend String operator+(const String& a, char c) { return String(a.s_ + c); }
  bool operator==(const String& o) const { return s_ == o.s_; }
  bool operator==(const char* o) const { return s_ == o; }
  bool operator!=(const String& o) const { return s_ != o.s_; }

 private:
  std::string s_;
};
