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

#include "gtest/gtest.h"
#include "xdl/proto/sample.pb.h"

using xdl::io::FeatureLine;
using xdl::io::Feature;
using xdl::io::FeatureValue;

class FeatureLineTestData {
 public:
  virtual ~FeatureLineTestData() = default;

  virtual void Generate(xdl::io::FeatureLine &feature_line,
                        std::vector<const xdl::io::Feature *> &features,
                        std::string name_arr[], int size_arr[],
                        std::vector<int64_t> key_arr[], std::vector<float> value_arr[],
                        int feature_num, bool is_sort = false, bool is_check = true) {
    is_sort_ = is_sort;
    for (int i = 0; i < feature_num; ++i) {
      Feature *feature = feature_line.add_features();
      feature->set_name(name_arr[i]);
      GenerateFeatureValue(feature, size_arr[i], key_arr[i], value_arr[i]);
      features.push_back(feature);
    }
    if (is_check) {
      GenerateExpectedFeature(feature_line, name_arr, feature_num);
    }
  }

  virtual void Check(xdl::io::Feature &result_feature) const {
    for (int i = 0; i < result_feature.values_size(); ++i) {
      ExpectKeyEq(result_feature.values(i).key(), expected_key_arr_[i], i);
      ExpectValueNear(result_feature.values(i).value(), expected_value_arr_[i], i);
    }
  }

  inline void ExpectKeyEq(int64_t a, int64_t b, int i) const {
    if (a != b)  PrintError(a, b, i, "key");
  }
  inline void ExpectValueNear(float a, float b, int i) const {
    float err = fabs(a - b), base = fabs(b);
    if (base < 0.000001) {
      if (err >= 0.001)  PrintError(a, b, i, "value");
    } else {
      if (err / base >= 0.001)  PrintError(a, b, i, "value");
    }
  }

 protected:
  template <typename T>
  inline void PrintError(T a, T b, int i, const char *title) const {
    printf("\033[1;31;40m%s check failed:\033[0m [%d] ", title, i);
    std::cout << a << " != " << b << std::endl;
  }

  inline void GenerateFeatureValue(Feature *feature, int size,
                                   std::vector<int64_t> &key, std::vector<float> &value) const {
    for (int i = 0; i < size; ++i) {
      FeatureValue *feature_value = feature->add_values();
      feature_value->set_key(key[i]);
      feature_value->set_value(value[i]);
    }
  }

  virtual void GenerateExpectedFeature(xdl::io::FeatureLine &feature_line,
                                       std::string name_arr[], int feature_num) {
    if (expected_key_arr_.size() != 0)  std::vector<int64_t>().swap(expected_key_arr_);
    if (expected_value_arr_.size() != 0)  std::vector<float>().swap(expected_value_arr_);

    if (is_sort_)  std::sort(name_arr, name_arr + feature_num);

    for (int i = 0; i < feature_num; ++i) {
      for (int j = 0; j < feature_line.features_size(); ++j) {
        const Feature &feature = feature_line.features(j);
        assert(feature.has_name());
        if (feature.values_size() == 0)  continue;
        if (name_arr[i] == feature.name()) {
          if (i == 0) {
            for (int n = 0; n < feature.values_size(); ++n) {
              expected_key_arr_.push_back(feature.values(n).key());
              expected_value_arr_.push_back(feature.values(n).value());
            }
          } else {
            if (is_sort_)  assert(name_arr[i-1] < name_arr[i]);
            std::vector<int64_t> tmp_key_arr;
            std::vector<float> tmp_value_arr;
            for (size_t m = 0; m < expected_key_arr_.size(); ++m) {
              for (int n = 0; n < feature.values_size(); ++n) {
                Combine(feature.values(n), m, tmp_key_arr, tmp_value_arr);
              }
            }
            expected_key_arr_.swap(tmp_key_arr);
            expected_value_arr_.swap(tmp_value_arr);
          }
          break;
        }
      }
    }
  }

  virtual void Combine(const FeatureValue &feature_value, size_t m,
                       std::vector<int64_t> &tmp_key_arr, std::vector<float> &tmp_value_arr) { }

 protected:
  bool is_sort_ = false;
  std::vector<int64_t> expected_key_arr_;
  std::vector<float> expected_value_arr_;
};
