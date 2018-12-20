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

#ifndef PS_COMMON_STREAMING_MODEL_INFO_
#define PS_COMMON_STREAMING_MODEL_INFO_

#include "ps-plus/common/tensor.h"
#include <vector>
#include <string>

namespace ps {

struct DenseVarNames {
  std::vector<std::string> names;
};

struct DenseVarValues {
  struct DenseVarValue {
    std::string name;
    size_t offset;
    Tensor data;
  };
  std::vector<DenseVarValue> values;
};

} // namespace ps

#endif // PS_COMMON_STREAMING_MODEL_INFO_
