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


#pragma once

#include <stdint.h>
#include <vector>

namespace xdl {
namespace io {

struct InternalValue {
  int64_t key_;
  float value_;

  inline bool has_key() const { return key_ >= 0; }
  inline int64_t key() const { return key_; }
  inline float value() const { return value_; }
};

struct InternalFeature {
  std::vector<InternalValue> values_;

  inline const std::vector<InternalValue> &values() const { return values_; }
  inline int values_size() const { return (int) values_.size(); }
  inline const InternalValue &values(int index) const { return values_[index]; }

  inline void clear() { values_.clear(); }

  inline void reserve(int capacity) { values_.reserve(capacity); }
  inline int capacity() const { return values_.capacity(); }

  inline void push_back(int64_t key, float value) {
    const InternalValue internal_value = { key, value };
    values_.push_back(std::move(internal_value));
  }
  inline void swap(InternalFeature &internal_feature) {
    values_.swap(internal_feature.values_);
  }
};

}  // namespace io
}  // namespace xdl