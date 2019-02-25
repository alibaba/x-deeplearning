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

#ifndef TDM_SERVING_UTIL_TIMER_H_
#define TDM_SERVING_UTIL_TIMER_H_

#include <stdint.h>

#include <sys/time.h>
#include <time.h>

namespace tdm_serving {
namespace util {

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

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_TIMER_H_
