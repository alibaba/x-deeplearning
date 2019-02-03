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

#ifndef TDM_SERVING_INDEX_TREE_TREE_INDEX_CONF_H_
#define TDM_SERVING_INDEX_TREE_TREE_INDEX_CONF_H_

#include <tr1/unordered_map>
#include "index/index_conf.h"

namespace tdm_serving {

class TreeIndexConf : public IndexConf {
 public:
  TreeIndexConf();
  virtual ~TreeIndexConf();

  virtual bool Init(
      const std::string& section,
      const util::ConfParser& conf_parser);

  uint32_t tree_level_topn(uint32_t level) const;

  void set_item_feature_group_id(const std::string& item_feature_group_id) {
    item_feature_group_id_ = item_feature_group_id;
  }

  const std::string& item_feature_group_id() const {
    return item_feature_group_id_;
  }

 private:
  bool ParseTreeLevelTopN(const std::string& conf_str);

 private:
  std::tr1::unordered_map<uint32_t, uint32_t> level_to_topn_;
  uint32_t model_batch_num_;
  std::string item_feature_group_id_;
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_INDEX_CONF_H_
