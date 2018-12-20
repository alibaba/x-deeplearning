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

#ifndef PS_PLUS_COMMON_SLICE_H_
#define PS_PLUS_COMMON_SLICE_H_

#include <cstdint>
#include "ps-plus/server/variable.h"

namespace ps {
namespace server {

struct Slices {
  size_t slice_size;
  std::vector<size_t> slice_id;
  // -1 means dense, 0+ means [slice_size] + dim[dim_part:]
  int dim_part;
  Variable* variable;
  bool writable;
};

struct TensorSlices {
  size_t slice_size;
  std::vector<size_t> slice_id;
  // -1 means dense, 0+ means [slice_size] + dim[dim_part:]
  int dim_part;
  Tensor tensor;
};

}
}

#endif

