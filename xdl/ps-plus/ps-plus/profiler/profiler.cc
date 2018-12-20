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

#include <iostream>
#include <thread>
#include <vector>

namespace ps {
namespace profiler {

void Profiler::Profile() {
  std::cout << "-------------------------------Test[" << name_ << "]Begin------------------------------" << std::endl;
  for (size_t thread = 1; thread <= thread_; thread++) {
    size_t test, base;
    {
      init_(thread);
      std::vector<std::thread> threads;
      size_t beg = now();
      for (size_t i = 0; i < thread; i++) {
        threads.emplace_back([i, this]{
          for (size_t k = 0; k < repeat_; k++) {
            testcase_(i, true);
          }
        });
      }
      for (size_t i = 0; i < thread; i++) {
        threads[i].join();
      }
      test = now() - beg;
    }
    {
      init_(thread);
      std::vector<std::thread> threads;
      size_t beg = now();
      for (size_t i = 0; i < thread; i++) {
        threads.emplace_back([i, this]{
          for (size_t k = 0; k < repeat_; k++) {
            testcase_(i, false);
          }
        });
      }
      for (size_t i = 0; i < thread; i++) {
        threads[i].join();
      }
      base = now() - beg;
    }
    std::cout << "Test For [name=" << name_ << ",thread=" << thread << ",repeat=" << repeat_ << "] base=" << base << " test=" << test << " avg=" << double(double(test) - base) / repeat_ << " qps=" << (int64_t)(repeat_ * 1000000 * thread / ((double)test - base)) << std::endl;
  }
  std::cout << "-------------------------------Test[" << name_ << "]End--------------------------------" << std::endl << std::endl;
}

}
}

// You can use ./ps_profiler to run all profiler
// Or use ./ps_profiler profiler_name to run any profiler
int main(int argc, char *argv[]) {
  char* profiler_case = argv[argc - 1];
  auto& map = ps::profiler::ProfilerRegister::Map();
  if (map.find(profiler_case) == map.end()) {
    for (auto& item : map) {
      item.second->Profile();
    }
  } else {
    map[profiler_case]->Profile();
  }
};
