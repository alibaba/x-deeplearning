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

#include "search_manager.h"
#include "search.pb.h"

#include "index/item.h"
#include "biz/filter.h"

#include "util/str_util.h"
#include "util/log.h"

namespace tdm_serving {

// user defined filter
class MyFilter : public Filter {
 public:
  virtual bool Init(const std::string& /*section*/,
                    const util::ConfParser& /*conf_parser*/) {
    return true;
  }

  virtual bool IsFiltered(const FilterInfo* /*filter_info*/,
                          const Item& item) {
    if (item.item_id() % 2 != 0) {
      LOG_INFO << "Item: " << item.item_id() << " is filtered";
    }
    return false;
  }
};
REGISTER_FILTER(my_filter_type, MyFilter);

bool SampleToRequest(const std::string& sample,
                     SearchParam* req,
                     std::map<std::string, float>* gt_kv_map) {
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

bool MakeRequest(SearchParam* req, uint32_t topn) {
  std::string sample_file_path = "./test_data/sample.dat";
  std::ifstream sample_file_handler(sample_file_path.c_str());

  if (!sample_file_handler) {
    LOG_ERROR << "open " << sample_file_path << " failed";
    return false;
  }

  std::string line;
  std::getline(sample_file_handler, line);

  std::map<std::string, float> gt_kv_map;
  if (!SampleToRequest(line, req, &gt_kv_map)) {
    LOG_ERROR << "parse sample [" << line << "] to request failed";
    return false;
  }

  req->set_topn(topn);
  req->set_index_name("item_tree_index");

  return true;
}

void SearchExample(uint32_t topn) {
  bool ret = false;

  // Step1: Init SearchManager
  SearchManager search_manager;
  ret = search_manager.Init("./test_data/conf/index_with_filter.conf",
                            "./test_data/conf/model.conf",
                            "./test_data/conf/filter.conf");
  if (ret != true) {
    LOG_ERROR << "Init SearchManager failed";
    return;
  }

  // Step2: Make Request
  SearchParam req;
  if (!MakeRequest(&req, topn)) {
    LOG_ERROR << "Make requests failed";
    return;
  }
  LOG_INFO << "Request: " << req.DebugString();

  // Step3: Search
  SearchResult res;
  if (!search_manager.Search(req, &res)) {
    LOG_ERROR << "search failed";
    return;
  }

  if (res.result_unit_size() > static_cast<int>(topn)) {
    LOG_ERROR << "result size > topn";
    return;
  }
  LOG_INFO << "Response: " << res.DebugString();

  return;
}

}  // namespace tdm_serving

int main(int /*argc*/, char** /*argv*/) {
  LOG_CONFIG("tdm_example", ".", 0);

  uint32_t search_topn = 200;

  tdm_serving::SearchExample(search_topn);

  return 0;
}


