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

#include "tdm/tdm_predict_op.h"

#include <iostream>

namespace tdm {

template <typename T1, typename T2>
bool pair_cmp_large(std::pair<T1, T2> i, std::pair<T1, T2> j) {
  return (i.second > j.second);
}

TDMPREDICTOP::TDMPREDICTOP(): tree_(NULL) {
}

std::map<std::string, std::string> 
TDMPREDICTOP::URun(const std::map<std::string, std::string> &params) {
   std::map<std::string, std::string> result;
   if (params.find("get_level_ids") != params.end()) {
  	  int target_level = atoi(params.find(
		"get_level_ids")->second.c_str());
	  target_level = (target_level + tree_->max_level()) % tree_->max_level();
      std::vector<int64_t> ids;
      GetAllLevelId(&ids, target_level);
      for (int i = 0; i < ids.size(); ++i) {
        result.insert(std::make_pair(std::to_string(ids.at(i)), "1"));
      }
	  return result;
   } 
   
   if (params.find("finish_predict") != params.end()) {
     if (predict_result_file_ != "" 
		 && predict_result_file_stream_.is_open()) {
        printf("[INFO] close predict result save\n");
        predict_result_file_stream_.close();
     }

     result.insert(std::make_pair("global_sample_num", 
								  std::to_string(global_sample_num_))); 
     result.insert(std::make_pair("global_r_num", 
								  std::to_string(global_r_num_))); 
     result.insert(std::make_pair("global_gt_num", 
								  std::to_string(global_gt_num_))); 
     result.insert(std::make_pair("global_p_num", 
								  std::to_string(global_p_num_))); 
     return result;
   }
   return result;
}

bool TDMPREDICTOP::Init(const std::map<std::string, std::string> &params) {
  gt_id_fn_ = "test_unit_id";
  if (params.find("test_unit_id") != params.end()) {
    gt_id_fn_ = params.find("test_unit_id")->second;
  }

  pred_id_fn_ = "pred_id";

  unit_id_expand_fn_ = "unit_id_expand";
  if (params.end() != params.find("unit_id_expand")) {
    unit_id_expand_fn_ = params.find("unit_id_expand")->second;
  }

  expand_mode_ = "tdm";
  if (params.end() != params.find("expand_mode")) {
    expand_mode_ = params.find("expand_mode")->second;
  }

  start_predict_layer_ = 1;
  if (params.end() != params.find("start_predict_layer")) {
    start_predict_layer_ = atoi(params.find(
          "start_predict_layer")->second.c_str());
  }


  level_topk_ = 20;
  if (params.end() != params.find("pr_test_each_layer_retrieve_num")) {
    level_topk_ = atoi(params.find(
        "pr_test_each_layer_retrieve_num")->second.c_str());
  }

  final_topk_ = 10;
  if (params.end() != params.find("pr_test_final_layer_retrieve_num")) {
    final_topk_ = atoi(params.find(
        "pr_test_final_layer_retrieve_num")->second.c_str());
  }

  predict_result_file_ = "";
  if (params.end() != params.find("predict_result_file"))
  {
	predict_result_file_ = params.find(
		"predict_result_file")->second;
    predict_result_file_stream_.open(predict_result_file_,
                                     std::ios::out|std::ios::app);
  }

  tree_ = &tdm::DistTree::GetInstance();

  global_sample_num_ = 0;
  global_r_num_ = 0;
  global_gt_num_ = 0;
  global_p_num_ = 0;
  avg_r_sum_ = 0;
  avg_p_sum_ = 0;

  return true;
}

bool TDMPREDICTOP::Run(xdl::io::SampleGroup *sg) {
  auto re = false;
  if ("tdm" == expand_mode_) {
	re = TDMExpandSample(sg);
  } else if ("vector" == expand_mode_) {
    re = VectorReExpandSample(sg);
  } else {
    printf("[ERROR] undefined expand mode\n");
  }
  return re;
}

bool TDMPREDICTOP::TDMExpandSample(xdl::io::SampleGroup *sg) {
  std::vector<int64_t> unit_expand_ids =
      GetTargetFeatureIds(sg, unit_id_expand_fn_, 0);

  std::vector<float> unit_expand_props;
  GetProbs(sg, &unit_expand_props, 1);

  std::vector<std::pair<int64_t, float> > unit_expand_ids_props;
  for (int i = 0; i < unit_expand_ids.size(); ++i) {
    unit_expand_ids_props.push_back(
        std::make_pair(unit_expand_ids.at(i), unit_expand_props.at(i)));
  }

  std::sort(unit_expand_ids_props.begin(),
            unit_expand_ids_props.end(), pair_cmp_large<int64_t, float>);

  // 如果topk结果内容为0个，则默认用户第一次来
  // 将起始层的所有节点加入到候选节点中。
  bool first_predict_tag = false;
  if (0 == unit_expand_ids_props.size()) {
    first_predict_tag = true; 
    auto level_itr = tree_->LevelIterator(start_predict_layer_ - 1);
    auto level_end = tree_->LevelEnd(start_predict_layer_ - 1);
    for (; level_itr != level_end; ++level_itr) {
      tdm::Node node;
      node.ParseFromString(level_itr->value);
      unit_expand_ids_props.push_back(std::make_pair(node.id(), 1));
    }
  }

  std::vector<int64_t> child_ids;
  std::vector<std::pair<int64_t, float> > leaf_ids;

  // 实际已经选取的Topk数量
  int winner_num = 0;
  for (int i = 0;
       i < unit_expand_ids_props.size() 
       && (winner_num < level_topk_ || first_predict_tag);
       ++i) {
    auto unit_id = unit_expand_ids_props.at(i).first;
    auto node = tree_->NodeById(unit_id);
    auto childs = tree_->Children(node);

    if (0 == childs.size()) {
      leaf_ids.push_back(std::make_pair(unit_expand_ids_props.at(i).first,
                                        unit_expand_ids_props.at(i).second));
    } else {
      for (int j = 0; j < childs.size(); ++j) {
        tdm::Node node;
        node.ParseFromString(childs.at(j).value);
        child_ids.push_back(node.id());
      }
      ++winner_num;
    }
  }

  if (0 == child_ids.size()) {
    std::vector<int64_t> gt_ids =
        GetTargetFeatureIds(sg, gt_id_fn_, sg->feature_tables_size() - 1);
    std::vector<std::pair<int64_t, float> > pred_ids =
        GetTargetFeatureIdsWithValue(sg, pred_id_fn_,
                                     sg->feature_tables_size() - 1);
    pred_ids.insert(pred_ids.end(), leaf_ids.begin(), leaf_ids.end());
    std::sort(pred_ids.begin(), pred_ids.end(),
              pair_cmp_large<int64_t, float>);

    std::vector<int64_t> final_top_ids;
    for (int i = 0; i < final_topk_; ++i) {
      final_top_ids.push_back(pred_ids.at(i).first);
    }

    std::set<int64_t> pred_ids_set(final_top_ids.begin(),
                                   final_top_ids.end());
    int com_num = 0;
    for (int i = 0; i < gt_ids.size(); ++i) {
      auto it = pred_ids_set.find(gt_ids.at(i));
      if (it != pred_ids_set.end()) {
        com_num += 1;
      }
    }

    float avg_r = 0;
    if(0 != gt_ids.size()) {
      avg_r = com_num * 1.0 / gt_ids.size();
    }

    float avg_p = 0;
    if (0 != pred_ids_set.size()) {
      avg_p = com_num * 1.0 / pred_ids_set.size();
    }

    {
      std::unique_lock<std::mutex> lck(mutex_);

      global_sample_num_ += 1;
      global_r_num_ += com_num;
      global_gt_num_ += gt_ids.size();
      global_p_num_ += pred_ids_set.size();
      avg_r_sum_ += avg_r;
      avg_p_sum_ += avg_p;

      printf("predict result:\n");
      printf("\tglobal_sample_num: %d\n", global_sample_num_);
      printf("\tglobal_r_num: %d\n", global_r_num_);
      printf("\tglobal_gt_num: %d\n", global_gt_num_);
      printf("\tglobal_p_num: %d\n", global_p_num_);
      printf("\tglobal_r: %f\n", global_r_num_ * 1.0 / global_gt_num_);
      printf("\tglobal_p: %f\n", global_r_num_ * 1.0 / global_p_num_);
      printf("\tavg_r: %f\n", avg_r_sum_ * 1.0 / global_sample_num_);
      printf("\tavg_p: %f\n", avg_p_sum_ * 1.0 / global_sample_num_);

	  //save predict result
	  if (predict_result_file_ != ""
	      && predict_result_file_stream_.is_open()) {
	  	predict_result_file_stream_ << sg->sample_ids(0) << ":";
	  	for (int i = 0; i < pred_ids.size() && i < final_topk_; ++i)
	  	{
			predict_result_file_stream_ << pred_ids.at(i).first << "," 
									    << pred_ids.at(i).second << ";";
      	} 
	  	predict_result_file_stream_ << "\n";
	  } 
    }
    return false;
  }

  AddFeatureIds(sg, sg->feature_tables_size() - 1, 0,
                pred_id_fn_, &leaf_ids, 1);

  if (1 == sg->feature_tables_size()) {
    InsertNewFeatureTable(sg, 0);
  }

  auto feature_table = sg->mutable_feature_tables(0);
  feature_table->Clear();

  sg->mutable_props()->Clear();
  auto new_feature_table = sg->mutable_feature_tables(0);

  sg->mutable_labels()->Clear();

  //构建当前的的featureTable，同时，增加对应的label。
  for (int i = 0; i < child_ids.size(); ++i) {
    auto new_feature_line = new_feature_table->add_feature_lines();
    new_feature_line->set_refer(0);

    auto new_feature = new_feature_line->add_features();
    new_feature->set_type(xdl::io::FeatureType::kSparse);
    new_feature->set_name("unit_id_expand");

    auto new_feature_value = new_feature->add_values();
    new_feature_value->set_key(child_ids.at(i));
    new_feature_value->set_value(1);

    auto new_label = sg->add_labels();
    new_label->add_values(0);
    new_label->add_values(1);
  }

  return true;

}

bool TDMPREDICTOP::VectorReExpandSample(xdl::io::SampleGroup *sg) {

  if (1 == sg->feature_tables_size()) {
    InsertNewFeatureTable(sg, 0);
  }

  auto feature_table = sg->mutable_feature_tables(0);
  feature_table->Clear();

  auto new_feature_table = sg->mutable_feature_tables(0);

  sg->mutable_labels()->Clear();

  auto new_feature_line = new_feature_table->add_feature_lines();
  new_feature_line->set_refer(0);

  auto new_feature = new_feature_line->add_features();
  new_feature->set_type(xdl::io::FeatureType::kSparse);
  new_feature->set_name("unit_id_expand");

  auto new_feature_value = new_feature->add_values();
  int mock_id = 1234;
  new_feature_value->set_key(1234);
  new_feature_value->set_value(1);

  auto new_label = sg->add_labels();
  new_label->add_values(0);
  new_label->add_values(1);

  return true;
}

std::vector<int64_t> TDMPREDICTOP::GetTargetFeatureIds(
    xdl::io::SampleGroup *sg,
    std::string target_feature_name,
    int feature_table_position) {
  std::vector<int64_t> feature_ids;

  auto target_feature_table = &sg->feature_tables(feature_table_position);
  for (int i = 0; i < target_feature_table->feature_lines_size(); ++i) {
    auto target_feature_line = &target_feature_table->feature_lines(i);
    for(int j = 0; j < target_feature_line->features_size(); ++j) {
      auto feature = &target_feature_line->features(j);
      if (feature->name() == target_feature_name) {
        for (int k = 0; k < feature->values_size(); ++k) {
          feature_ids.push_back(feature->values(k).key());
        }
      }
    }
  }
  return feature_ids;
}

std::vector<std::pair<int64_t, float> >
TDMPREDICTOP::GetTargetFeatureIdsWithValue(
    xdl::io::SampleGroup *sg,
    std::string target_feature_name,
    int feature_table_position) {
  std::vector<std::pair<int64_t, float> > feature_ids;

  auto target_feature_table = &sg->feature_tables(feature_table_position);
  for (int i = 0; i < target_feature_table->feature_lines_size(); ++i) {
    auto target_feature_line = &target_feature_table->feature_lines(i);
    for(int j = 0; j < target_feature_line->features_size(); ++j) {
      auto feature = &target_feature_line->features(j);
      if (feature->name() == target_feature_name) {
        for (int k = 0; k < feature->values_size(); ++k) {
          feature_ids.push_back(std::make_pair(
              feature->values(k).key(), feature->values(k).value()));
        }
      }
    }
  }
  return feature_ids;
}

bool TDMPREDICTOP::AddFeatureIds(
    xdl::io::SampleGroup *sg,
    int feature_table_index,
    int feature_line_index,
    std::string target_id_fn,
    std::vector<std::pair<int64_t, float> > *add_ids,
    int default_value) {
  if (NULL == sg ||
      feature_table_index >= sg->feature_tables_size() ||
      feature_line_index >= sg->feature_tables(feature_table_index)
      .feature_lines_size()) {
    return false;
  }

  if (0 == add_ids->size()) {
    return true;
  }

  xdl::io::Feature *target_feature = NULL;

  auto target_feature_line = sg->mutable_feature_tables(feature_table_index)
                             ->mutable_feature_lines(feature_line_index);
  for (int i = 0; i < target_feature_line->features_size(); ++i) {
    if (target_feature_line->features(i).name() == target_id_fn) {
      target_feature = target_feature_line->mutable_features(i);
    }
  }

  if (NULL == target_feature) {
    target_feature = sg->mutable_feature_tables(feature_table_index)
                     ->mutable_feature_lines(feature_line_index)
                     ->add_features();
  }

  for (int j = 0; j < add_ids->size(); ++j) {
    auto new_values = target_feature->add_values();
    new_values->set_key(add_ids->at(j).first);
    new_values->set_value(add_ids->at(j).second);
  }

  return true;
}

void TDMPREDICTOP::GetProbs(xdl::io::SampleGroup *sg,
                            std::vector<float> * probs,
                            int index) {
  for (int i = 0; i < sg->props_size(); ++i) {
    auto prop = sg->props(i);
    probs->push_back(prop.values(index));
  }
}

void TDMPREDICTOP::GetAllLevelId(std::vector<int64_t> *ids, int level) {
  assert(level <= tree_->max_level() - 1);
  auto level_itr = tree_->LevelIterator(level);
  auto level_end = tree_->LevelEnd(level);
  for(; level_itr != level_end; ++level_itr) {
    tdm::Node node;
    node.ParseFromString(level_itr->value);
    ids->push_back(node.id());
  } 
}

xdl::io::FeatureTable* TDMPREDICTOP::InsertNewFeatureTable(
    xdl::io::SampleGroup *sg, int index) {
  assert(sg->feature_tables_size() == 1);
  const xdl::io::FeatureTable feature_table = sg->feature_tables(0);
  sg->clear_feature_tables();
  auto ret = sg->add_feature_tables();
  sg->add_feature_tables()->CopyFrom(feature_table);
  return ret;
}

XDL_REGISTER_IOP(TDMPREDICTOP);

}  // namespace tdm
