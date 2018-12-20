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

#include "ps-plus/profiler/profiler.h"

static int aa;
static int bb;

PROFILE(sample, 32, 1000).Init([](size_t threads){
  bb = threads;
}).TestCase([](size_t thread_id, bool run){
  // Initialize
  int xx = 0;
  int yy = 0;
  for (int i = 0; i < 1000 + 1000 * bb; i++) {
    xx = xx * 3 + 1;
    yy += xx;
  }
  if (run) {
    // Real Profiling Code
    for (int i = 0; i < 1000 + 100 * bb; i++) {
      xx = xx * 3 + 1;
      yy += xx;
    }
  }
  // Release
  aa = yy;
});
