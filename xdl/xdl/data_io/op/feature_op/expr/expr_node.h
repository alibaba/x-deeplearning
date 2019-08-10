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

enum FeaOpType {
  kSourceFeatureOp = 0,
  kSingleFeatureOp = 1,
  kMultiFeatureOp  = 2,
};

struct ExprNode {
  FeaOpType type;
  int8_t table_id = -1;
  bool output = false;
  bool internal;
  void *op;
  void *result;
  std::vector<int> pres;
  std::vector<const ExprNode *> pre_nodes;

 public:
  void clear();

  void reserve(int capacity);
  int capacity() const;

  void add(int64_t key, float value);
  void get(int index, int64_t &key, float &value) const;
  int64_t key(int index) const;
  float value(int index) const;

  int values_size() const;
  int BinarySearch(int64_t key, float &value) const;

  void InitResult();
  void ReleaseResult();
  void ReleaseOp();

  inline void InitInternal() {
    internal = !output && type != FeaOpType::kSourceFeatureOp;
  }

 private:
  template <typename T>
  inline int values_size(const T *feature) const {
    return feature->values_size();
  }

  template <typename T>
  inline int BinarySearch(const T *feature, int64_t key, float &value) const {
    int left = 0, right = feature->values_size() - 1, mid;
    while (left <= right) {
      mid = (left + right) / 2;
      if (!feature->values(mid).has_key()) {
        right = mid - 1;
        continue;
      }
      const int64_t mid_key = feature->values(mid).key();
      if (mid_key == key) {
        value = feature->values(mid).value();
        return mid;
      } else if (mid_key < key) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return -1;
  }
};

}  // namespace io
}  // namespace xdl