/*
 * \file timer.h
 * \brief The timer
 */
#pragma once

#include <stdint.h>

#include <sys/time.h>
#include <time.h>

namespace blaze {

static inline double GetTime() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

class Timer {
 public:
  Timer();

  void Start();
  void ReStart();

  void Stop();
  void Reset();

  double GetTotalTime();
  double GetElapsedTime();

 protected:
  double total_;
  double start_;
};

}  // namespace blaze
