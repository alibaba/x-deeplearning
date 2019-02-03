/*
 * \file timer.cc
 * \brief The timer
 */
#include "blaze/common/timer.h"

namespace blaze {

Timer::Timer() {
  start_ = 0;
  total_ = 0;
}

void Timer::Start() {
  start_ = GetTime();
  total_ = 0;
}

void Timer::ReStart() {
  start_ = GetTime();
}

void Timer::Stop() {
  total_ += (GetTime() - start_);
}

void Timer::Reset() {
  start_ = 0;
  total_ = 0;
}

double Timer::GetTotalTime() {
  return total_;
}

double Timer::GetElapsedTime() {
  return GetTime() - start_;
}

}  // namespace blaze
