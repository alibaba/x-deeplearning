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


#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature/merge_feature.h"

#include "xdl/data_io/op/feature_op/expr/expr_node.h"

namespace xdl {
namespace io {

bool MergeFeature::Combine(std::vector<const ExprNode *> &source_nodes, ExprNode *result_node) {
  const size_t source_nodes_size = source_nodes.size();
  if (source_nodes_size == 2) {
    const int values_size0 = source_nodes[0]->values_size();
    const int values_size1 = source_nodes[1]->values_size();
    if (values_size0 == 0) {
      return false;
    } else if (values_size0 == 1) {
      if (values_size1 == 0)  return false;
      return CombineOne(source_nodes[1], source_nodes[0], result_node);
    } else if (values_size1 == 1) {
      return CombineOne(source_nodes[0], source_nodes[1], result_node);
    }
  } else if (source_nodes_size < 2) {
    return false;
  }

  std::unordered_map<int64_t, float> key_value_map;
  for (const ExprNode *source_node : source_nodes) {
    //const Feature *feature; // = features[i];
    const int source_values_size = source_node->values_size();
    if (source_values_size == 0) {
      return false;
    }
    if (key_value_map.size() == 0) {
      for (int k = 0; k < source_values_size; ++k) {
        int64_t source_key;
        float source_value;
        source_node->get(k, source_key, source_value);
        key_value_map.insert(std::make_pair(source_key, source_value));
      }
    } else {
      std::unordered_map<int64_t, float> tmp_key_value_map;
      for (int k = 0; k < source_values_size; ++k) {
        int64_t source_key;
        float source_value;
        source_node->get(k, source_key, source_value);
        if (source_key < 0)  continue;
        const auto &iter = key_value_map.find(source_key);
        if (iter == key_value_map.end())  continue;
        const int64_t combined_key = combine_key_func_(iter->first, source_key);
        const float combined_value = combine_value_func_(iter->second, source_value);
        tmp_key_value_map.insert(std::make_pair(combined_key, combined_value));
      } // end for feature->values(k)
      key_value_map.swap(tmp_key_value_map);
    }
  } // end for feature
  if (key_value_map.size() == 0) {
    return false;
  }
  for (auto &iter : key_value_map) {
    result_node->add(iter.first, iter.second);
  }
  return true;
}

bool MergeFeature::CombineOne(const ExprNode *node0, const ExprNode *node1, ExprNode *result_node) {
  if (node0->values_size() == 0)  return false;
  int64_t key1;
  float value1;
  node1->get(0, key1, value1);
  if (key1 < 0)  return false;

  // Assume that node0->result is ordered by its key.
  float value0;
  const int mid = node0->BinarySearch(key1, value0);
  if (mid < 0)  return false;
  result_node->add(key1, combine_value_func_(value0, value1));
  return true;
}

}  // namespace io
}  // namespace xdl