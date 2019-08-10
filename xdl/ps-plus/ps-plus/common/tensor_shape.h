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

#ifndef PS_PLUS_COMMON_TENSOR_SHAPE_H_
#define PS_PLUS_COMMON_TENSOR_SHAPE_H_

#include <vector>
#include <memory>

namespace ps {

class TensorShape {
 public:
  TensorShape()
    : num_elements_(1), dims_(0) {}
  explicit TensorShape(const std::vector<size_t>& dims)
      : dims_(dims) {
    ComputeNumElements();
  }

  explicit TensorShape(std::vector<size_t>&& dims)
      : dims_(std::move(dims)) {
    ComputeNumElements();
  }

  explicit TensorShape(std::initializer_list<size_t> dims)
      : dims_(dims) {
    ComputeNumElements();
  }

  TensorShape(const TensorShape& rhs)
      : dims_(rhs.dims_) {
    ComputeNumElements();
  }

  const std::vector<std::size_t>& Dims() const {
    return dims_;
  }

  size_t Size() const {
    return dims_.size();
  }

  size_t NumElements() const {
    return num_elements_;
  }

  bool IsScalar() const {
    return dims_.empty();
  }

  void Set(size_t id, size_t dim) {
    dims_[id] = dim;
    ComputeNumElements();
  }

  size_t operator[](size_t id) const {
    return dims_[id];
  }

  bool operator==(const TensorShape& rhs) const {
    if (dims_.size() != rhs.dims_.size()) {
      return false;
    }
    for (size_t i = 0; i < dims_.size(); i++) {
      if (dims_[i] !=  rhs.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const TensorShape& rhs) const {
    return !(*this == rhs);
  }

  std::string ToString() const {
    std::string s;
    for (auto& item : dims_) {
      s += std::to_string(item) + ",";
    }
    if (!s.empty()) { s.pop_back(); }
    return s;
  }

 private:
  void ComputeNumElements() {
    num_elements_ = 1;
    for (size_t dim : dims_) {
      num_elements_ *= dim;
    }
  }
  size_t num_elements_;
  std::vector<size_t> dims_;
};

}

#endif

