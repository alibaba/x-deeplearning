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

#ifndef PS_COMMON_TIME_UTILS_H
#define PS_COMMON_TIME_UTILS_H

#include <sys/time.h>
#include <time.h>
#include <cstdint>

namespace ps {

class TimeUtils {
 public:
  static uint64_t NowMicros() {
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }
};

} //ps

#endif  // PS_COMMON_TIME_UTILS_H
