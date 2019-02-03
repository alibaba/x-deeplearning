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

#ifndef XDL_TEST_TEST_UTILS_H_
#define XDL_TEST_TEST_UTILS_H_

#include <string>

#include "xdl/core/framework/tensor.h"
#include "xdl/core/backend/device_singleton.h"

namespace xdl {
  
class TestUtils {
 public:
  template <typename T>
  static Tensor MakeTensor(const std::vector<T>& data,
                           const std::vector<size_t>& dims);
  
  template <typename T>
  static bool TensorEqual(const Tensor& src,
                          const std::vector<T>& dst,
                          const std::vector<size_t>& shape);
};

template <typename T>
Tensor TestUtils::MakeTensor(const std::vector<T>& data,
                              const std::vector<size_t>& dims) {
  Tensor ret(DeviceSingleton::CpuInstance(), 
             TensorShape(dims), 
             DataTypeToEnum<T>::value);
  T* base = ret.Raw<T>();
  for (size_t i = 0; i < ret.Shape().NumElements(); ++i) {
    *(base + i) = data[i];
  }

  return ret;
}

template <typename T>
bool TestUtils::TensorEqual(const Tensor& src, 
                            const std::vector<T>& dst,
                            const std::vector<size_t>& shape) {
  if (src.Type() != DataTypeToEnum<T>::value) {
    return false;
  }

  if (src.Shape().Size() != shape.size()) {
    return false;
  }

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != src.Shape()[i]) {
      return false;
    }
  }

  T* base = src.Raw<T>();
  if (src.Type() == DataType::kFloat ||
      src.Type() == DataType::kDouble) {
    for (size_t i = 0; i < dst.size(); ++i) {
      if (abs(base[i] - dst[i]) > 1e-6) {
        return false;
      }
    }
  } else {
    for (size_t i = 0; i < dst.size(); ++i) {
      if (base[i] != dst[i]) {
        return false;
      }
    }
  }

  return true;
}

}

#endif // HDFS_LAUNCHER_H_
