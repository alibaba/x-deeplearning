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

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include "api/search_manager.h"
#include "util/str_util.h"
#include "util/log.h"
#include "proto/search.pb.h"

namespace tdm_serving {

bool SampleToRequest(const std::string& sample,
                     SearchParam* req,
                     std::map<std::string, float>* gt_kv_map) {
  LOG_DEBUG << "sample str: " << sample;

  FeatureGroupList* user_feature =
      req->mutable_user_info()->mutable_user_feature();

  std::vector<std::string> split_1;
  util::StrUtil::Split(sample, '|', true, &split_1);
  if (split_1.size() < 3) {
    LOG_ERROR << "sample field size < 3";
    return false;
  }

  std::string sample_key = split_1[0];
  std::string sample_group = split_1[1];

  std::vector<std::string> split_2;
  util::StrUtil::Split(split_1[2], ';', true, &split_2);
  for (size_t i = 0; i < split_2.size(); i++) {
    std::vector<std::string> split_3;
    util::StrUtil::Split(split_2[i], '@', true, &split_3);
    if (split_3.size() < 2) {
      LOG_ERROR << "feature field size < 2";
      return false;
    }

    std::string feature_name = split_3[0];

    // add feature to req
    FeatureGroup* feature_group = NULL;
    if (feature_name != "test_unit_id") {
      feature_group = user_feature->add_feature_group();
      feature_group->set_feature_group_id(feature_name);
    }

    std::vector<std::string> split_4;
    util::StrUtil::Split(split_3[1], ',', true, &split_4);
    for (size_t j = 0; j < split_4.size(); j++) {
      std::vector<std::string> split_5;
      util::StrUtil::Split(split_4[j], ':', true, &split_5);

      uint64_t id;
      std::string key;
      float value = 0.0;

      if (split_5.size() == 2) {
        key = split_5[0];
        if (!util::StrUtil::StrConvert<uint64_t>(split_5[0].c_str(), &id)) {
          LOG_WARN << "id of feature " << split_5[1] << " is not number";
          return false;
        }
        if (!util::StrUtil::StrConvert<float>(split_5[1].c_str(), &value)) {
          LOG_WARN << "value of feature "
                        << split_5[1] << " is not float";
          return false;
        }
      } else if (split_5.size() == 1) {
        key = split_5[0];
        if (!util::StrUtil::StrConvert<uint64_t>(split_5[0].c_str(), &id)) {
          LOG_WARN << "id of feature " << split_5[1] << " is not number";
          return false;
        }
        value = 1.0;
      } else {
        LOG_ERROR << "feature_eneity field size is not 1 or 2";
        return false;
      }

      if (feature_name != "test_unit_id") {
        FeatureEntity* feature_entity = feature_group->add_feature_entity();
        feature_entity->set_id(id);
        feature_entity->set_value(value);
      } else { // ground truth data
        (*gt_kv_map)[key] += value;
      }
    }
  }

  return true;
}

void DoEvaluate(const SearchResult& res,
              uint32_t topn,
              const std::map<std::string, float>& gt_kv_map,
              float* precision,
              float* recall,
              float* f1_score) {
  float gt_num = 0;
  for (auto item : gt_kv_map) {
    gt_num += item.second;
  }

  float hit_num = 0;
  for (int i = 0; i < res.result_unit_size(); i++) {
    uint64_t item_id = res.result_unit(i).id();
    std::string id_str = util::ToString(item_id);

    LOG_DEBUG << "id_str: " << id_str;

    auto it = gt_kv_map.find(id_str);
    if (it != gt_kv_map.end()) {
      hit_num += it->second;
    }
  }

  *precision = topn > 0 ? hit_num / topn : 0.0;
  *recall = gt_num > 0 ? hit_num / gt_num : 0.0;
  *f1_score = 0.0;
  if (*precision + *recall > 0) {
    *f1_score = 2 * *precision * *recall / (*precision + *recall);
  }

  LOG_INFO << "gt_num: " << gt_num;
  LOG_INFO << "hit_num: " << hit_num;

  LOG_INFO << "precision: " << *precision;
  LOG_INFO << "recall: " << *recall;
  LOG_INFO << "f1_score: " << *f1_score;
}


void SearchEvaluate(uint32_t topn,
                    uint32_t loop_num) {
  bool ret = false;

  // Init SearchManager
  SearchManager search_manager;
  ret = search_manager.Init("./eval_data/conf/index.conf",
                            "./eval_data/conf/model.conf");
  if (ret != true) {
    LOG_ERROR << "Init SearchManager failed";
    return;
  }

  std::string sample_file_path = "./eval_data/userbehavoir_test_sample.dat";
  std::ifstream sample_file_handler(sample_file_path.c_str());

  if (!sample_file_handler) {
    LOG_ERROR << "open " << sample_file_path << " failed";
    return;
  }

  float total_precision = 0;
  float total_recall = 0;
  float total_f1_score = 0;

  size_t loop_idx = 0;
  std::string line;

  while (std::getline(sample_file_handler, line)) {
    SearchParam req;
    std::map<std::string, float> gt_kv_map;

    // Make Search Request
    if (!SampleToRequest(line, &req, &gt_kv_map)) {
      LOG_ERROR << "parse sample [" << line << "] to request failed";
      return;
    }

    req.set_topn(topn);
    req.set_index_name("item_tree_index");

    LOG_DEBUG << "tdm reqeust: " << req.DebugString();

    // Search
    SearchResult res;
    if (!search_manager.Search(req, &res)) {
      LOG_ERROR << "search failed";
      return;
    }

    if (res.result_unit_size() > static_cast<int>(topn)) {
      LOG_ERROR << "result size > topn";
      return;
    }

    // Evaluate
    float precision;
    float recall;
    float f1_score;

    DoEvaluate(res, topn, gt_kv_map, &precision, &recall, &f1_score);
    total_precision += precision;
    total_recall += recall;
    total_f1_score += f1_score;

    loop_idx++;
    LOG_INFO << "loop: " << loop_idx;
    if (loop_idx == loop_num) {
      break;
    }
  }

  // Evaluate
  float avg_precision = total_precision / loop_num;
  float avg_recall = total_recall / loop_num;
  float avg_f1_score = total_f1_score / loop_num;

  LOG_INFO << "avg_precision: " << avg_precision;
  LOG_INFO << "avg_recall: " << avg_recall;
  LOG_INFO << "avg_f1_score: " << avg_f1_score;

  return;
}

}  // namespace tdm_serving

int main(int /*argc*/, char** /*argv*/) {
  LOG_CONFIG("tdm_evaluation", ".", 0);

  uint32_t search_topn = 200;
  uint32_t loop_num = 10000;

  tdm_serving::SearchEvaluate(search_topn, loop_num);

  return 0;
}


