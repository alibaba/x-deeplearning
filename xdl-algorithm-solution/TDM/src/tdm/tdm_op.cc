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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tdm/tdm_op.h"

#include <map>
#include <string>

namespace tdm {

TDMOP::TDMOP(): selector_(NULL), layer_counts_sum_(0),
                unit_id_fn_("") {
}

bool TDMOP::Init(const std::map<std::string, std::string>& params) {
  unit_id_fn_ = "train_unit_id";
  if (params.find("unit_id_fn") != params.end()) {
    unit_id_fn_ = params.find("unit_id_fn")->second;
  }

  std::string selector_name = "by_layerwise";
  if ( params.end() != params.find("selector_name") ) {
    selector_name = params.find("selector_name")->second;
  }

  std::string select_config = "";
  int start_sample_layer = -1;
  if ( params.end() != params.find("start_sample_layer")) {
     select_config += "start_sample_layer=" 
		+ std::to_string(atoi(params.find(
		    "start_sample_layer")->second.c_str()));
  }
  
  if ( params.end() != params.find("with_prob")) {
    if (select_config.length() >= 1) {
	  select_config += ";";
    }
    select_config += "with_prob=" + params.find("with_prob")->second; 
  }

  auto tree = &tdm::DistTree::GetInstance();
  tree_ = tree;
  selector_ = tdm::SelectorMapper::GetSelector(selector_name);
  selector_->set_dist_tree(tree);
  selector_->Init(select_config);

  if (params.end() !=  params.find("layer_counts")) {
    auto layer_counts_content = Split(params.find(
                                        "layer_counts")->second, ",");
    assert(layer_counts_content.size() >= tree->max_level() - 1);
    for (int i = 0; i < tree->max_level(); ++i) {
      auto count =  atoi(layer_counts_content.at(i).c_str());
      layer_counts_sum_ += count + 1;
      layer_counts_.push_back(count);
      printf("[INFO] level %d, layer_counts.push_back: %d\n", i, count);
    }
  } else {
    printf("[ERROR] Not specified layer count in parameters!\n");
    return false;
  }

  expected_count_ = false;
  if (params.end() != params.find("use_expected_count") &&
      params.find("use_expected_count")->second == "true") {
    expected_count_ = true;
  }

  for (int level = 0; level < tree->max_level(); ++level) {
    auto level_itr = tree_->LevelIterator(level);
    auto level_end = tree_->LevelEnd(level);
    int64_t now_level_sample_sum_ = 0;
    do {
      Node node;
      if (node.ParseFromString(level_itr->value)) {
        now_level_sample_sum_ += node.probality();
      } 
      ++level_itr;
    } while (level_itr != level_end);

    level_sample_sum_.push_back(now_level_sample_sum_);
  }
  
  return true;
}

bool TDMOP::Run(xdl::io::SampleGroup *sg) {
  auto re = TDMExpandSample(sg);
  return re;
}

bool TDMOP::TDMExpandSample(xdl::io::SampleGroup *sg) {
  std::vector<int64_t> target_ids(sg->labels_size());
  GetTargetFeatureIds(sg, unit_id_fn_, 0, &target_ids);

  // 获取对应的扩展结果
  std::vector<int64_t> output_ids(target_ids.size() * layer_counts_sum_);
  std::vector<float> weights(target_ids.size() * layer_counts_sum_);
  std::vector<std::vector<int64_t> > features;

  selector_->Select(target_ids, features, layer_counts_,
                   output_ids.data(), weights.data());

  // 将扩展结果写入到sg中
  xdl::io::FeatureTable* new_feature_table = InsertNewFeatureTable(sg, 0);

  // 清空label
  sg->clear_labels();

  // 构建当前的的feature table，同时，增加对应的label。
  auto sample_num = 0;
  for (int i = 0; i < target_ids.size(); ++i) {
    for (int j = 0; j < layer_counts_sum_; ++j) {
      // 如果id不为0，有效的情况下，才添加新的样本
      if (0 != output_ids.at(i * layer_counts_sum_ + j)) {
        ++sample_num;
        xdl::io::FeatureLine* new_feature_line =
            new_feature_table->add_feature_lines();
        new_feature_line->set_refer(i);

        xdl::io::Feature* new_feature = new_feature_line->add_features();
        new_feature->set_type(xdl::io::FeatureType::kSparse);
        new_feature->set_name("unit_id_expand");
        xdl::io::FeatureValue* new_feature_value = new_feature->add_values();
        new_feature_value->set_key(output_ids.at(i * layer_counts_sum_ + j));
        new_feature_value->set_value(1);
		
		//增加对应的dense 特征
        if (true == expected_count_) {
          xdl::io::Feature* new_dense_feature =
              new_feature_line->add_features();
          new_dense_feature->set_type(xdl::io::FeatureType::kDense);
          new_dense_feature->set_name("unit_id_expand_weight");
          xdl::io::FeatureValue* new_dense_feature_value =
              new_dense_feature->add_values();
          auto id = output_ids.at(i * layer_counts_sum_ + j);
          auto node = tree_->NodeById(id);
          auto level = tree_->NodeLevel(tree_->KeyNo(node.key));
		  tdm::Node t_n;
          assert(t_n.ParseFromString(node.value));
          new_dense_feature_value->set_value(
				t_n.probality() * 1.0 / level_sample_sum_.at(level) 
				* (layer_counts_.at(level) + 1));
        }

        xdl::io::Label *new_label = sg->add_labels();
        new_label->add_values(1 - weights.at(i * layer_counts_sum_ + j));
        new_label->add_values(weights.at(i * layer_counts_sum_ + j));
      }
    }
  }

  if (0 == sample_num) {
    printf("[ERROR] Appear sample_num zero\n");
    return false;
  }

  return true;
}

void TDMOP::GetTargetFeatureIds(xdl::io::SampleGroup* sg,
                                const std::string& target_feature_name,
                                int feature_table_position,
                                std::vector<int64_t>* feature_ids) {
  int idx = 0;
  const auto& target_feature_table =
      sg->feature_tables(feature_table_position);
  for (int i = 0; i < target_feature_table.feature_lines_size(); ++i) {
    const auto& target_feature_line = target_feature_table.feature_lines(i);
    for (int j = 0; j < target_feature_line.features_size(); ++j) {
      const auto& feature = target_feature_line.features(j);
      if (feature.name() == target_feature_name) {
        for (int k = 0; k < feature.values_size(); ++k) {
          feature_ids->at(idx++) = (feature.values(k).key());
        }
      }
    }
  }
  assert(idx <= feature_ids->size());
  feature_ids->resize(idx);
}

xdl::io::FeatureTable*
TDMOP::InsertNewFeatureTable(xdl::io::SampleGroup* sg, int index) {
  (void) index;
  const xdl::io::FeatureTable feature_table = sg->feature_tables(0);
  sg->clear_feature_tables();
  auto ret = sg->add_feature_tables();
  sg->add_feature_tables()->CopyFrom(feature_table);
  return ret;
}

XDL_REGISTER_IOP(TDMOP);

}  // namespace tdm
