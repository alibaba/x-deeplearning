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

#ifndef PS_COMMON_INITIALIZER_TRUNCATED_NORMAL_INITIALIZER_H
#define PS_COMMON_INITIALIZER_TRUNCATED_NORMAL_INITIALIZER_H

#include "ps-plus/common/initializer.h"
#include "ps-plus/common/tensor_shape.h"

namespace ps {
namespace initializer {

class TruncatedNormalInitializer: public Initializer {
 public:
  TruncatedNormalInitializer(int seed, 
                             float mean,
                             float stdev);

  bool Accept(DataType type) override;
  void Init(void* data, DataType type, size_t size) override;
  Initializer* Clone() override;

 private:
  int seed_;
  float mean_;
  float stddev_;
};

} //namespace initializer
} //ps

#endif  // PS_COMMON_INITIALIZER_TRUNCATED_NORMAL_INITIALIZER_H
