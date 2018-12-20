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

#include "variance_scaling_initializer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "random/random.h"
#include "random/random_ops.h"

namespace ps {
namespace initializer {

VarianceScalingInitializer::VarianceScalingInitializer(
    const TensorShape& full_shape,
    int seed, 
    double scale,
    const std::string& mode,
    const std::string& distribution)
  : full_shape_(full_shape)
  , seed_(seed)
  , scale_(scale)
  , mode_(mode)
  , distribution_(distribution) {
}

VarianceScalingInitializer::VarianceScalingInitializer(
    TensorShape&& full_shape,
    int seed, 
    double scale,
    const std::string& mode,
    const std::string& distribution)
  : full_shape_(std::move(full_shape))
  , seed_(seed)
  , scale_(scale)
  , mode_(mode)
  , distribution_(distribution) {
}

bool VarianceScalingInitializer::Accept(DataType type) {
  if (type == DataType::kFloat || 
      type == DataType::kDouble) {
    return true;
  }

  return false;
}

void VarianceScalingInitializer::Init(void* data, 
                                      DataType type, 
                                      size_t size) {
  double fan_in, fan_out;
  ComputeFans(full_shape_, &fan_in, &fan_out);
  if (mode_ == "fan_in") {
    scale_ /= fan_in < 1 ? 1 : fan_in;
  } else if (mode_ == "fan_out") {
    scale_ /= fan_out < 1 ? 1 : fan_out;
  } else {
    const double fan = (fan_in + fan_out) / 2;
    scale_ /= fan < 1 ? 1 : fan;
  }

  int seed1 = 0;
  int seed2 = 0;
  int* seed = seed_ == -1? nullptr: &seed_;
  Random::GetSeed(seed, &seed1, &seed2);
  std::unique_ptr<PhiloxRandomOp> op(new PhiloxRandomOp(seed1, seed2));
  if (distribution_ == "normal") {
    if (type == DataType::kFloat){
      op->Fill<TruncatedNormalDistribution<SingleSampleAdapter<PhiloxRandom>, float>>(
          size, data);
    } else {
      op->Fill<TruncatedNormalDistribution<SingleSampleAdapter<PhiloxRandom>, double>>(
          size, data);
    }

    const double stddev = sqrt(scale_);
    CASES(type, do {
      T* beg = reinterpret_cast<T*>(data);
      T* end = beg + size;
      for (T* p = beg; p < end; p++) {
        *p *= stddev;
      }
    } while (0));
  } else {
    if (type == DataType::kFloat) {
      op->Fill<UniformDistribution<PhiloxRandom, float>>(size, data);
    } else {
      op->Fill<UniformDistribution<PhiloxRandom, double>>(size, data);
    }

    const double limit = sqrt(3 * scale_);
    CASES(type, do {
      T* beg = reinterpret_cast<T*>(data);
      T* end = beg + size;
      for (T* p = beg; p < end; p++) {
        *p = *p * 2 * limit - limit;
      }
    } while (0));
  }
}

void VarianceScalingInitializer::ComputeFans(const TensorShape& shape,
                                             double* fan_in,
                                             double* fan_out) {
    const auto& dims = shape.Dims();
    const size_t size = dims.size();
    if (size < 1) {
        *fan_in = *fan_out = 1.0;
        return;
    }

    if (size == 1) {
        *fan_in = *fan_out = dims[0];
        return;
    }

    if (size == 2) {
        *fan_in = dims[0];
        *fan_out = dims[1];
        return;
    }

    double receptive_field_size = 1;
    size_t i;
    const size_t n = size - 2;
    for (i = 0; i < n; i++) {
        receptive_field_size *= dims[i];
    }

    *fan_in = dims[i++] * receptive_field_size;
    *fan_out = dims[i++] * receptive_field_size;
}

Initializer* VarianceScalingInitializer::Clone() {
  return new VarianceScalingInitializer(full_shape_,
                                        seed_,
                                        scale_,
                                        mode_,
                                        distribution_);
}

} // namespace initializer
} // namespace ps
