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

#include "normal_initializer.h"

#include <string>
#include <cstring>
#include <cmath>
#include <memory>

#include "ps-plus/common/types.h"
#include "random/random.h"
#include "random/random_ops.h"

namespace ps {
namespace initializer {

NormalInitializer::NormalInitializer(
    int seed, float mean, float stddev)
  : seed_(seed)
  , mean_(mean)
  , stddev_(stddev) {}

bool NormalInitializer::Accept(DataType type) {
  if (type == DataType::kFloat || 
      type == DataType::kDouble) {
    return true;
  }

  return false;
}

void NormalInitializer::Init(void* data, 
                                      DataType type, 
                                      size_t size) {
  int seed1 = 0;
  int seed2 = 0;
  int* seed = seed_ == -1? nullptr: &seed_;
  Random::GetSeed(seed, &seed1, &seed2);
  std::unique_ptr<PhiloxRandomOp> op(new PhiloxRandomOp(seed1, seed2));
  if (type == DataType::kFloat) {
    op->Fill<NormalDistribution<PhiloxRandom,float> >(
        size, data);
  } else {
    op->Fill<NormalDistribution<PhiloxRandom,double> >(
        size, data);
  }

  CASES(type, do {
    T* base = reinterpret_cast<T*>(data);
    for (T* ptr = base; ptr < base + size; ++ptr) {
      *ptr = *ptr * stddev_ + mean_;
    };
  } while (0));
}

Initializer* NormalInitializer::Clone() {
  return new NormalInitializer(seed_, 
                                        mean_, 
                                        stddev_);
}

} //namespace initializer
} //ps

