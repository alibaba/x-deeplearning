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


#include "xdl/data_io/op/feature_op/feature_op.h"

#include <xdl/core/utils/logging.h>
#include <set>

#include "xdl/data_io/op/feature_op/expr/expr_graph.h"
#include "xdl/data_io/op/feature_op/string_util.h"
#include "xdl/data_io/schema.h"

namespace xdl {
namespace io {

bool FeatureOP::Init(const std::vector<std::string> &dsl_arr,
                     const std::vector<FeatureNameVec> &feature_name_vecs) {
  if (inited_)  return true;
  inited_ = true;

  expr_graph_ = ExprGraph::Get(dsl_arr, false);
  feature_name_vecs_ = feature_name_vecs;
  InitFeatureNameStore();
  return true;
}

bool FeatureOP::Init(const std::map<std::string, std::string> &params) {
  if (inited_)  return true;
  inited_ = true;

  XDL_CHECK(schema_ != nullptr);
  const std::vector<FeatureOptionMap> &feature_table = schema_->feature_table();
  const size_t feature_table_size = feature_table.size();
  std::vector<std::string> dsl_arr;
  for (size_t table_id = 0; table_id < feature_table_size; ++table_id) {
    const FeatureOptionMap &feature_option_map = feature_table[table_id];
    for (const auto &iter : feature_option_map) {
      const FeatureOption *feature_option = iter.second;
      if (!feature_option->has_dsl())  continue;
      dsl_arr.push_back(feature_option->dsl());
    }
  }

  expr_graph_ = ExprGraph::Get(dsl_arr, false);
  feature_name_vec_sizes_.reserve(feature_table_size);
  feature_name_vecs_.resize(feature_table_size);
  const FeatureNameVec &feature_name_vec = expr_graph_->feature_name_vec();
  for (const std::string &feature_name : feature_name_vec) {
    size_t table_id;
    for (table_id = 0; table_id < feature_table_size; ++table_id) {
      const FeatureOptionMap &feature_option_map = feature_table[table_id];
      if (feature_option_map.find(feature_name) != feature_option_map.end())  break;
    }
    XDL_CHECK(table_id < feature_table_size) << "Some feature in dsl not found in schema!";
    feature_name_vecs_[table_id].push_back(feature_name);  // TODO: sort
  }
  InitFeatureNameStore();
  return true;
}

void FeatureOP::InitFeatureNameStore() {
  const size_t feature_name_vecs_size = feature_name_vecs_.size();
  for (size_t table_id = 0; table_id < feature_name_vecs_size; ++table_id) {
    const FeatureNameVec &feature_name_vec = feature_name_vecs_[table_id];
    feature_name_vec_sizes_.push_back(feature_name_vec.size());
    for (const std::string &feature_name : feature_name_vec) {
      feature_name_map_.insert(std::make_pair(feature_name, table_id));
    }
  }
  expr_graph_->set_feature_name_map(&feature_name_map_);
}

bool FeatureOP::Run(SampleGroup *sample_group) {
  const int tables_size = sample_group->feature_tables_size();
  bool is_clear_result_feature, is_first = true;
  google::protobuf::int32 last_refer = -1;
  std::vector<std::vector<FeatureMap>> feature_maps_arr(tables_size);  // {0,1,2}
  for (int feature_table_id = tables_size - 1; feature_table_id >= 0; --feature_table_id) {  // for (2, 1, 0)
    if (feature_name_vec_sizes_[feature_table_id] == 0)  continue;
    FeatureTable *feature_table = sample_group->mutable_feature_tables(feature_table_id);
    size_t feature_maps_size = feature_table_id == 0 ? 0 : feature_table->feature_lines_size();
    std::vector<FeatureMap> feature_maps(feature_maps_size);

    for (int i = 0, j = 0; j < feature_table->feature_lines_size(); ++j) {
      FeatureLine *feature_line = feature_table->mutable_feature_lines(j);
      const FeatureNameVec &feature_name_vec = feature_name_vecs_[feature_table_id];
      // bind
      FeatureMap feature_map(feature_name_vec.size());
      int begin = 0;
      for (const std::string &feature_name : feature_name_vec) {
        const int index = BinarySearch(feature_line, feature_name, begin);
        if (index >= 0) {
          feature_map.insert(std::make_pair(feature_name, feature_line->mutable_features(index)));
        } else if (begin < 0) {
          break;
        }
      }
      if (feature_table_id != 0) {
        feature_maps[i++] = std::move(feature_map);
        continue;
      }
      std::vector<const FeatureMap *> feature_map_arr(tables_size, nullptr);  // {0,1,2}
      int table_id = feature_table_id;
      feature_map_arr[table_id] = &feature_map;
      for (const FeatureLine *fline = feature_line;
           fline->has_refer() && ++table_id < tables_size;
           fline = &sample_group->feature_tables(table_id).feature_lines(fline->refer())) {
        feature_map_arr[table_id] = &feature_maps_arr[table_id][fline->refer()];
      }
      // execute
      if (is_first) {
        is_first = false;
        is_clear_result_feature = true;
        if (feature_line->has_refer())  last_refer = feature_line->refer();
      } else if (!feature_line->has_refer()) {
        is_clear_result_feature = true;
      } else if (feature_line->refer() != last_refer) {
        is_clear_result_feature = true;
        last_refer = feature_line->refer();
      } else {
        is_clear_result_feature = false;
      }
      expr_graph_->Execute(feature_map_arr, feature_line, is_clear_result_feature);
    }
    if (feature_table_id != 0) {
      feature_maps_arr[feature_table_id] = std::move(feature_maps);
    }
  }
  return true;
}

int FeatureOP::BinarySearch(const FeatureLine *feature_line, const std::string &feature_name, int &begin) {
  int left = begin, right = feature_line->features_size() - 1, mid;
  while (left <= right) {
    mid = (left + right) / 2;
    const Feature &mid_feature = feature_line->features(mid);
    if (!mid_feature.has_name()) {
      right = mid - 1;
      continue;
    }
    const int compare = mid_feature.name().compare(feature_name);
    if (compare == 0) {
      //begin = mid + 1;  // 对于单次查找，反而更慢了
      return mid;
    } else if (compare < 0) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  //if (left == feature_line->features_size())  begin = -1;
  //else  begin = left;
  return -1;
}

XDL_REGISTER_IOP(FeatureOP);

}  // namespace io
}  // namespace xdl
