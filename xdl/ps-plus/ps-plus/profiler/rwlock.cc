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
#include "ps-plus/common/qrw_lock.h"
#include <atomic>

ps::QRWLock rwlock;

PROFILE(rwlock, 32, 1000).Init([](size_t threads){
}).TestCase([](size_t thread_id, bool run){
  if (run) {
    for (int i = 0; i < 100; i++) {
      ps::QRWLocker(rwlock, ps::QRWLocker::kSimpleRead);
    }
  }
});
