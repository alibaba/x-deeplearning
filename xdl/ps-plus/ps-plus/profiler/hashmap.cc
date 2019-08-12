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
#include "ps-plus/common/hashmap.h"
#include <iostream>
#include <vector>
#include <random>

using ps::HashMap;

std::unique_ptr<HashMap> hashmap;
const int hint = 1 << 10;
std::vector<std::unique_ptr<std::mt19937>> rands;
std::random_device rd;

#if 0
PROFILE(hashmap, 32, 100000).Init([](size_t threads){
  rands.clear();
  for (size_t i = 0; i < threads; i++) {
    rands.emplace_back(new std::mt19937(rd()));
  }
  std::srand(std::time(0));
  hashmap.reset(new HashMap(hint));
}).TestCase([](size_t thread_id, bool run){
  int hint = (1<<10);
  std::vector<int64_t> ids;
  std::vector<int64_t> reused_ids;
  const int keys_size = 500;
  int64_t keys[keys_size * 2];
  int ret = 0;

  for (int i = 0; i < keys_size; i++) {
    int rv = (*rands[thread_id])() % hint;
    rv = (rv + hint) % hint;
    keys[i*2] = rv;
    keys[i*2+1] = rv+1;
  }

  if (run) {
    ret = hashmap->Get(keys, keys_size, 2, &ids, &reused_ids); 
    if (ret != 0) {
      std::cout << "ERROR" << std::endl;
    }
  }
});
#endif
