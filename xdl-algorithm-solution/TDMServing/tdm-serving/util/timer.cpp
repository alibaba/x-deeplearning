/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "util/timer.h"

namespace tdm_serving {
namespace util {

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

}  // namespace util
}  // namespace tdm_serving
