// Mac stand-in for what audio.cpp takes from Arduino: a simulated clock.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

extern uint32_t g_simMs;
inline uint32_t millis() { return g_simMs; }
void delay(uint32_t ms);

template <class A, class B> inline std::common_type_t<A, B> min(A a, B b) { return a < b ? a : b; }
template <class A, class B> inline std::common_type_t<A, B> max(A a, B b) { return a > b ? a : b; }

// FreeRTOS, as audio.cpp uses it: a task is a thread.
#include <chrono>
#include <thread>
#define pdPASS 1
#define pdMS_TO_TICKS(ms) (ms)
inline void vTaskDelay(uint32_t ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
inline void vTaskDelete(void*) {}  // (the task's function ends right after)
inline int xTaskCreatePinnedToCore(void (*fn)(void*), const char*, uint32_t, void* arg, int, void*, int) {
  std::thread(fn, arg).detach();
  return pdPASS;
}

#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_8BIT 2
inline void* heap_caps_malloc(size_t n, int) { return malloc(n); }
