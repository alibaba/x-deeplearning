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

#include "test/data_io/op/feature_op/op_test_tool.h"
#include "xdl/proto/sample.pb.h"

using xdl::io::FeatureLine;
using xdl::io::Feature;
using xdl::io::FeatureValue;

class FeatureLinePerformanceTestData {
 public:
  void Generate(xdl::io::FeatureLine &feature_line,
                std::vector<const xdl::io::Feature *> &features,
                int size_range, int name_range,
                int key_range, float value_range,
                int feature_num) {
    for (int i = 0; i < feature_num; ++i) {
      Feature *feature = feature_line.add_features();
      char name[8];
      OpTestTool::Rand(name, 8, name_range);
      feature->set_name(name);
      GenerateFeatureValue(feature, size_range, key_range, value_range);
      features.push_back(feature);
    }
  }

  void GenerateFixed(xdl::io::FeatureLine &feature_line,
                     std::vector<const xdl::io::Feature *> &features,
                     std::vector<int> sizes,
                     int key_range, float value_range,
                     int feature_num, char begin = 'A') {
    assert(sizes.size() == (size_t) feature_num);
    for (int i = 0; i < feature_num; ++i) {
      Feature *feature = feature_line.add_features();
      char name[8];
      name[7] = 0;
      for (int offset = i, j = 6; j >= 0; --j) {
        name[j] = begin + offset % 26;
        offset /= 26;
      }
      feature->set_name(name);
      GenerateFeatureValueFixed(feature, sizes[i], key_range, value_range);
      features.push_back(feature);
    }
  }

 protected:
  inline void GenerateFeatureValue(Feature *feature, int size_range,
                                   int key_range, float value_range) const {
    int size = OpTestTool::Rand(size_range + 1);
    int *keys = new int[size];
    for (int i = 0; i < size; ++i)  keys[i] = OpTestTool::Rand(key_range);
    std::sort(keys, keys + size);
    for (int i = 0; i < size; ++i) {
      FeatureValue *feature_value = feature->add_values();
      feature_value->set_key(keys[i]);
      feature_value->set_value(OpTestTool::Rand(value_range));
    }
    delete keys;
  }

  inline void GenerateFeatureValueFixed(Feature *feature, int size,
                                        int key_range, float value_range) const {
    int *keys = new int[size];
    for (int i = 0; i < size; ++i)  keys[i] = OpTestTool::Rand(key_range);
    std::sort(keys, keys + size);
    for (int i = 0; i < size; ++i) {
      FeatureValue *feature_value = feature->add_values();
      feature_value->set_key(keys[i]);
      feature_value->set_value(OpTestTool::Rand(value_range));
    }
    delete keys;
  }
};
