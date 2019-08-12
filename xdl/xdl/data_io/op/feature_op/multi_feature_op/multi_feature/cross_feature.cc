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


#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/cross_feature.h"

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/data_io/op/feature_op/expr/internal_feature.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

bool CrossFeature::Combine(std::vector<const ExprNode *> &source_nodes, ExprNode *result_node) {
  InternalFeature result_feature;
  for (const ExprNode *source_node : source_nodes) {
    const int source_values_size = source_node->values_size();
    if (source_values_size == 0)  return false;
    const int result_values_size = result_feature.values_size();
    if (result_values_size == 0) {
      int64_t source_key;
      float source_value;
      for (int k = 0; k < source_values_size; ++k) {
        source_node->get(k, source_key, source_value);
        if (source_key < 0)  continue;
        result_feature.push_back(source_key, source_value);
      }
    } else {
      InternalFeature tmp_feature;
      int64_t source_key;
      float source_value;
      for (int j = 0; j < result_values_size; ++j) {
        for (int k = 0; k < source_values_size; ++k) {
          source_node->get(k, source_key, source_value);
          if (source_key < 0)  continue;
          const int64_t combined_key = combine_key_func_(result_feature.values(j).key(), source_key);
          const float combined_value = combine_value_func_(result_feature.values(j).value(), source_value);
          tmp_feature.push_back(combined_key, combined_value);
        } // end for feature->values(k)
      } // end for result_feature.values(j)
      result_feature.swap(tmp_feature);
    }
  } // end for source_nodes
  if (result_node->output) {
    Feature *result = reinterpret_cast<Feature *>(result_node->result);
    for (const InternalValue &internal_value : result_feature.values()) {
      FeatureValue *feature_value = result->add_values();
      if (internal_value.has_key())  feature_value->set_key(internal_value.key());
      feature_value->set_value(internal_value.value());
    }
  } else {
    InternalFeature *result = reinterpret_cast<InternalFeature *>(result_node->result);
    result->swap(result_feature);
  }
  return true;
}

}  // namespace io
}  // namespace xdl