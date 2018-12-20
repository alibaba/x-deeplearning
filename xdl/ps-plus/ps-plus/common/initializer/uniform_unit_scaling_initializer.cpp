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

#include "uniform_unit_scaling_initializer.h"

#include <string>
#include <cstring>
#include <cmath>
#include <memory>

#include "ps-plus/common/types.h"
#include "random/random.h"
#include "random/random_ops.h"

namespace ps {
namespace initializer {

UniformUnitScalingInitializer::UniformUnitScalingInitializer(
    const TensorShape& shape, int seed, float factor)
  : shape_(shape)
  , seed_(seed)
  , factor_(factor) {
}

UniformUnitScalingInitializer::UniformUnitScalingInitializer(
    TensorShape&& shape, int seed, float factor)
  : shape_(std::move(shape))
  , seed_(seed)
  , factor_(factor) {
}

bool UniformUnitScalingInitializer::Accept(DataType type) {
  if (type == DataType::kFloat || 
      type == DataType::kDouble) {
    return true;
  }

  return false;
}

void UniformUnitScalingInitializer::Init(void* data, 
                                         DataType type, 
                                         size_t size) {
  int seed1 = 0;
  int seed2 = 0;
  int* seed = seed_ == -1? nullptr: &seed_;
  Random::GetSeed(seed, &seed1, &seed2);
  std::unique_ptr<PhiloxRandomOp> op(new PhiloxRandomOp(seed1, seed2));
  if (type == DataType::kFloat) {
    op->Fill<UniformDistribution<PhiloxRandom, float> >(
        size, data);
  } else {
    op->Fill<UniformDistribution<PhiloxRandom, double> >(
        size, data);
  }

  size_t input_size = 1;
  const std::vector<size_t>& dims = shape_.Dims();
  for (int i = 0; i < (int)(dims.size() - 1); ++i) {
    input_size *= dims[i];
  }

  float max = sqrt(3.0 / input_size) * factor_;
  CASES(type, do {
    T* base = reinterpret_cast<T*>(data);
    for (T* ptr = base; ptr < base + size; ++ptr) {
      *ptr = *ptr * 2 * max - max;
    };
  } while (0));
}

Initializer* UniformUnitScalingInitializer::Clone() {
  return new UniformUnitScalingInitializer(shape_, 
                                           seed_, 
                                           factor_);
}

} //namespace initializer
} //ps

