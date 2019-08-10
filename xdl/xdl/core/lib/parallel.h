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

#ifndef XDL_CORE_LIB_PARALLEL_H_
#define XDL_CORE_LIB_PARALLEL_H_

#include <vector>
#include <tuple>

namespace xdl {
namespace common {

  /* Generate splits for parallelly running */
  __inline__
  void parallel_split_generate(size_t nr_ele, size_t threads,
                   std::vector<std::tuple<size_t, size_t, size_t>>* vec) {
    size_t step = nr_ele / threads;
    for (size_t index = 0; index < threads; index++) {
      size_t begin, end;
      begin = index * step;
      if (index == (threads - 1)) {
        end = nr_ele;
      } else {
        end = (index + 1) * step;
      }
      vec->push_back(std::make_tuple(index, begin, end));
    }
  }

}
}

#endif
